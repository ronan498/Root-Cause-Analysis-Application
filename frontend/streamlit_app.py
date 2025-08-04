# frontend/streamlit_app.py
import os
import requests
import streamlit as st
from typing import List, Dict, Any

# -------------------- Config --------------------
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MAX_RESULTS = 10
MIN_SIMILARITY = 0.48

st.set_page_config(page_title="RCA Demo", layout="wide")

# -------------------- Global styles --------------------
st.markdown(
    """
<style>
/* Hide Streamlit's "Press Ctrl+Enter to apply" hints */
[data-testid="stTextArea"] div[aria-live="polite"] { display: none !important; }
[data-testid="stTextInput"] div[aria-live="polite"] { display: none !important; }
div[data-baseweb="textarea"] + div p { display: none !important; }
div[data-baseweb="input"] + div p { display: none !important; }

/* Right rail */
.rail-title { font-weight: 700; margin: 0 0 .5rem 0; display:flex; align-items:center; gap:.5rem; }
.rail-title .dot { width:8px; height:8px; border-radius:50%; background:#4f46e5; display:inline-block; }
.rail-q { font-size:.95rem; margin:.25rem 0 .5rem 0; }
.rail-empty { color:#5f6368; font-size:.9rem; padding:.25rem 0 .5rem 0; }

/* Result meta + anchor spacing */
.result-meta { color:#666; font-size:0.85rem; }
.anchor { scroll-margin-top: 80px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Scroll helpers (using st.query_params) --------------------
def _do_rerun():
    st.rerun()

def set_scroll_anchor(anchor: str):
    st.query_params["scroll"] = anchor

def render_scroll_script():
    target = st.query_params.get("scroll")
    if target:
        st.components.v1.html(
            f"""
<script>
const anchor = "{target}";
const el = parent.document.querySelector(`#${{anchor}}`);
if (el) {{
  el.scrollIntoView({{ behavior: "instant", block: "start" }});
}}
</script>
""",
            height=0,
        )
        try:
            st.query_params.pop("scroll")
        except Exception:
            pass

# -------------------- Session state --------------------
if "hide_uploader" not in st.session_state:
    st.session_state.hide_uploader = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "narrow" not in st.session_state:
    st.session_state.narrow = None
if "diag" not in st.session_state:
    st.session_state.diag = {"has_results": False, "query": "", "filtered": []}

# -------------------- Header --------------------
st.title("Root Cause Analysis (RCA) Demo")
st.caption("Describe a fault and get likely causes and recommended corrective actions.")
render_scroll_script()

# -------------------- Sidebar (left): settings + upload --------------------
with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("API URL", value=DEFAULT_BACKEND_URL, help="FastAPI backend base URL")
    st.caption(f"Requesting up to {MAX_RESULTS} results; showing those with similarity ≥ {MIN_SIMILARITY:.2f}.")
    st.divider()

    st.header("Add data")
    if st.session_state.hide_uploader:
        st.success("Upload complete.")
        if st.button("Add another file", use_container_width=True):
            st.session_state.hide_uploader = False
            _do_rerun()
    else:
        st.write("Upload a CSV with columns: **component**, **fault_description**, **root_cause**, **corrective_action**; optional **model**.")
        up = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key=f"uploader_{st.session_state.uploader_key}",
            help="This updates the knowledge base and the index."
        )
        add_clicked = st.button("Add to knowledge base", use_container_width=True)
        if up and add_clicked:
            files = {"file": (up.name, up.getvalue(), "text/csv")}
            try:
                r = requests.post(f"{backend_url}/ingest", files=files, timeout=120)
                if r.ok:
                    data = r.json()
                    st.success(f"Added {data.get('added', 0)} rows.")
                    st.session_state.hide_uploader = True
                    st.session_state.uploader_key += 1
                    _do_rerun()
                else:
                    st.error(f"Upload failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------- Helpers --------------------
def fetch_components(api_base: str) -> List[str]:
    try:
        resp = requests.get(f"{api_base}/components", timeout=10)
        if resp.ok:
            comps = resp.json().get("components", [])
            return sorted({str(c).strip().lower() for c in comps if str(c).strip()})
    except Exception:
        pass
    return []

def fetch_models(api_base: str, component: str) -> List[str]:
    try:
        resp = requests.get(f"{api_base}/models", params={"component": component}, timeout=10)
        if resp.ok:
            models = resp.json().get("models", [])
            cleaned = {str(m).strip().lower() for m in models if str(m).strip() and str(m).strip().lower() != "nan"}
            return sorted(cleaned)
    except Exception:
        pass
    return []

# ============================================================
# Main two-column content area (inputs + results left; narrowing right)
# ============================================================
left, right = st.columns([2.6, 1.0], gap="large")

# -------------------- Right rail (title + placeholder shown immediately) --------------------
with right:
    st.markdown('<div class="rail-title"><span class="dot"></span><span>Narrow further</span></div>', unsafe_allow_html=True)
    rail_body = st.empty()  # placeholder we can overwrite later
    # Default placeholder (visible on first load)
    with rail_body.container():
        st.markdown(
            """
<div class="rail-empty">
No follow-up question yet.<br/>
<b>Tips:</b>
<ul style="margin:.25rem 0 .25rem 1rem; padding:0;">
  <li>Select a component/model for more targeted questions.</li>
  <li>Add specific symptoms (smell, sound, vibration, error codes).</li>
  <li>Try a slightly longer fault description.</li>
</ul>
</div>
""",
            unsafe_allow_html=True,
        )

# -------------------- Left: Inputs (width limited by the right column) --------------------
with left:
    components = fetch_components(backend_url)
    component_display = st.selectbox("Component (optional filter)", options=["All"] + [c.title() for c in components], index=0)

    model_display = None
    models: List[str] = []
    if component_display != "All":
        selected_component = component_display.lower()
        models = fetch_models(backend_url, selected_component)
        model_options = ["All models"] + [m.title() for m in models] if models else ["All models"]
        model_display = st.selectbox("Model (optional)", options=model_options, index=0)

    query = st.text_area(
        "Describe the fault",
        placeholder="e.g. Motor (ABB M3BP 160MLA 4) making a high-pitched squeal and smells burnt…",
        height=120,
    )

    if st.button("Diagnose", type="primary"):
        if not query.strip():
            st.warning("Please enter a fault description.")
        else:
            payload: Dict[str, Any] = {"query": query, "top_k": MAX_RESULTS}
            if component_display != "All":
                payload["component"] = component_display.lower()
                if model_display and model_display != "All models":
                    payload["model"] = model_display.lower()

            with st.spinner("Analysing…"):
                try:
                    r = requests.post(f"{backend_url}/diagnose", json=payload, timeout=120)
                except Exception as e:
                    r = None
                    st.error(f"Request failed: {e}")

            if r is not None:
                if r.ok:
                    raw_results = r.json() or []
                    try:
                        filtered = [it for it in raw_results if float(it.get("similarity", 0.0)) >= MIN_SIMILARITY]
                    except Exception:
                        filtered = raw_results

                    st.session_state.diag = {"has_results": True, "query": query, "filtered": filtered}
                    st.session_state.narrow = {
                        "candidates": filtered,
                        "step": 0,
                        "question": None,
                        "keywords": [],
                        "base_query": query,
                        "done": False,
                        "asked": [],
                    }
                    st.success(f"Matches found: {len(filtered)}")
                    set_scroll_anchor("results_anchor")
                    _do_rerun()
                else:
                    st.error(f"API error: {r.status_code} {r.text}")

# -------------------- Results + Narrowing logic --------------------
diag = st.session_state.diag
if diag.get("has_results"):
    filtered = diag.get("filtered", [])

    nar = st.session_state.narrow
    if (nar is None) or (nar.get("base_query") != diag["query"]):
        nar = {
            "candidates": filtered,
            "step": 0,
            "question": None,
            "keywords": [],
            "base_query": diag["query"],
            "done": False,
            "asked": [],
        }
        st.session_state.narrow = nar

    NARROW_MAX_STEPS = 6
    GAP_GOOD = 0.15
    SHORTLIST_COUNT = 1

    def should_stop(cands: List[Dict[str, Any]], step: int) -> bool:
        if len(cands) <= SHORTLIST_COUNT: return True
        if step >= NARROW_MAX_STEPS: return True
        if len(cands) >= 2:
            try:
                s0 = float(cands[0].get("similarity", 0.0))
                s1 = float(cands[1].get("similarity", 0.0))
                if (s0 - s1) >= GAP_GOOD and step >= 1: return True
            except Exception:
                pass
        return False

    # Auto-fetch a narrowing question if needed
    if (not nar["done"]) and (nar["question"] is None) and (len(nar["candidates"]) > SHORTLIST_COUNT) and (nar["step"] < NARROW_MAX_STEPS):
        try:
            rq = requests.post(
                f"{backend_url}/narrow/next",
                json={"query": diag["query"], "candidates": nar["candidates"][:5], "asked": nar.get("asked", [])},
                timeout=60,
            )
            if rq.ok:
                data = rq.json()
                nar["question"] = data.get("question")
                nar["keywords"] = data.get("keywords", [])
                already = {kk for a in nar.get("asked", []) for kk in (a.get("keywords") or [])}
                if nar["keywords"] and all((kw in already) for kw in nar["keywords"]):
                    nar["question"] = None
                    nar["keywords"] = []
            else:
                nar["done"] = True
        except Exception as e:
            st.error(f"Narrowing error: {e}")
            nar["done"] = True

    # -------- Left column: results list (inline) --------
    with left:
        st.markdown('<span id="results_anchor" class="anchor"></span>', unsafe_allow_html=True)

        list_to_show = nar["candidates"] if (nar.get("step", 0) > 0 and nar.get("candidates")) else filtered
        st.metric("Matches shown", len(list_to_show))

        if not list_to_show:
            st.info("No strong matches found. Try rephrasing or loosen the filters.")
        else:
            for i, item in enumerate(list_to_show, start=1):
                comp = (item.get('component') or '').title()
                model_raw = (item.get('model') or '').strip()
                model_txt = model_raw if model_raw and model_raw.lower() != "nan" else ""
                title = " – ".join([t for t in (comp, model_txt.title() if model_txt else "") if t]) or "Result"

                st.markdown(f"### {i}. {title}")
                st.markdown(f"**Matched fault:** {item.get('matched_fault_description','')}")
                st.markdown(f"**Root cause:** {item.get('root_cause','')}")
                st.markdown(f"**Corrective action:** {item.get('corrective_action','')}")
                try:
                    st.caption(f"Similarity: {float(item.get('similarity', 0.0)):.3f}")
                except Exception:
                    pass
                st.divider()

    # -------- Right column: overwrite placeholder with question (if any) --------
    with right:
        q = nar.get("question")
        kws = nar.get("keywords", [])
        if q and not nar["done"]:
            rail_body.empty()
            with rail_body.container():
                st.markdown(f'<div class="rail-q">{q}</div>', unsafe_allow_html=True)
                with st.form(key=f"narrow_form_{abs(hash(q)) % (10**8)}", clear_on_submit=True):
                    choice = st.radio("Answer", options=["Yes", "No", "Skip"], horizontal=True, index=0)
                    submitted = st.form_submit_button("Apply")

                if submitted:
                    if choice == "Skip":
                        nar.setdefault("asked", []).append({"question": q, "keywords": kws, "answer": None})
                        nar["question"] = None
                        nar["keywords"] = []
                        set_scroll_anchor("results_anchor")
                        _do_rerun()
                    else:
                        ans = (choice == "Yes")
                        try:
                            rr = requests.post(
                                f"{backend_url}/narrow/answer",
                                json={"answer": ans, "keywords": kws, "candidates": nar["candidates"]},
                                timeout=60,
                            )
                            if rr.ok:
                                nar["candidates"] = rr.json().get("candidates", [])
                                nar.setdefault("asked", []).append({"question": q, "keywords": kws, "answer": ans})
                                nar["question"] = None
                                nar["keywords"] = []
                                nar["step"] += 1

                                # Progressive pruning
                                if len(nar["candidates"]) > 5 and nar["step"] >= 1:
                                    nar["candidates"] = nar["candidates"][:5]
                                if len(nar["candidates"]) > 3 and nar["step"] >= 2:
                                    nar["candidates"] = nar["candidates"][:3]
                                if len(nar["candidates"]) > 1 and (nar["step"] >= 3 or should_stop(nar["candidates"], nar["step"])):
                                    nar["candidates"] = nar["candidates"][:1]

                                nar["done"] = len(nar["candidates"]) <= 1 or should_stop(nar["candidates"], nar["step"])
                                set_scroll_anchor("results_anchor")
                                _do_rerun()
                            else:
                                st.error("Could not apply the answer.")
                        except Exception as e:
                            st.error(f"Apply error: {e}")
                            nar["done"] = True
        else:
            # Keep the initial tips visible (rail_body already holds them)
            if nar.get("done"):
                rail_body.empty()
                with rail_body.container():
                    st.markdown(
                        '<div class="rail-empty">Narrowing complete. Review the top match on the left.</div>',
                        unsafe_allow_html=True,
                    )

# -------------------- Footer --------------------
with st.expander("About this demo", expanded=False):
    st.write(
        "This demo uses OpenAI embeddings for similarity search over fault descriptions, "
        "FastAPI as the backend, and Streamlit for the UI. You can optionally filter by component and model."
    )
