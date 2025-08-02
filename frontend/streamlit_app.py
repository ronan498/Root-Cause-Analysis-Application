# frontend/streamlit_app.py
import os
import requests
import streamlit as st

# -------------------- Config --------------------
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MAX_RESULTS = 10       # request up to this many from the API
MIN_SIMILARITY = 0.45  # only display results with similarity >= this value

st.set_page_config(page_title="RCA Demo", layout="centered")

# Hide Streamlit's "Press Ctrl+Enter to apply"
st.markdown("""
<style>
[data-testid="stTextArea"] div[aria-live="polite"] { display: none !important; }
[data-testid="stTextInput"] div[aria-live="polite"] { display: none !important; }
div[data-baseweb="textarea"] + div p { display: none !important; }
div[data-baseweb="input"] + div p { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- Session state --------------------
if "hide_uploader" not in st.session_state:
    st.session_state.hide_uploader = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -------------------- Header --------------------
st.title("Root Cause Analysis (RCA) Demo")
st.caption("Describe a fault and get likely causes and recommended corrective actions.")

# -------------------- Sidebar: settings + upload --------------------
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
            st.experimental_rerun()
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
                    st.experimental_rerun()
                else:
                    st.error(f"Upload failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------- Helpers --------------------
def fetch_components(api_base: str):
    try:
        resp = requests.get(f"{api_base}/components", timeout=10)
        if resp.ok:
            comps = resp.json().get("components", [])
            return sorted({str(c).strip().lower() for c in comps if str(c).strip()})
    except Exception:
        pass
    return []

def fetch_models(api_base: str, component: str):
    try:
        resp = requests.get(f"{api_base}/models", params={"component": component}, timeout=10)
        if resp.ok:
            models = resp.json().get("models", [])
            return sorted({str(m).strip().lower() for m in models if str(m).strip()})
    except Exception:
        pass
    return []

# -------------------- Inputs --------------------
components = fetch_components(backend_url)
component_display = st.selectbox("Component (optional filter)", options=["All"] + [c.title() for c in components], index=0)

# Model dropdown appears only when a specific component is selected
model_display = None
models = []
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

# -------------------- Diagnose button --------------------
if st.button("Diagnose", type="primary"):
    if not query.strip():
        st.warning("Please enter a fault description.")
    else:
        payload = {"query": query, "top_k": MAX_RESULTS}
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
                    filtered = [
                        it for it in raw_results
                        if float(it.get("similarity", 0.0)) >= MIN_SIMILARITY
                    ]
                except Exception:
                    filtered = raw_results

                n = len(filtered)
                st.metric("Matches found", n)

                if n == 0:
                    st.info("No strong matches found. Try rephrasing or loosen the filters.")
                else:
                    for i, item in enumerate(filtered, start=1):
                        title_bits = [
                            (item.get('component') or '').title(),
                            (item.get('model') or '').title()
                        ]
                        title = " – ".join([b for b in title_bits if b])
                        st.markdown(f"### {i}. {title}")
                        st.markdown(f"**Matched fault:** {item.get('matched_fault_description','')}")
                        st.markdown(f"**Root cause:** {item.get('root_cause','')}")
                        st.markdown(f"**Corrective action:** {item.get('corrective_action','')}")
                        try:
                            st.caption(f"Similarity: {float(item.get('similarity', 0.0)):.3f}")
                        except Exception:
                            pass
                        st.divider()
            else:
                st.error(f"API error: {r.status_code} {r.text}")

# -------------------- Footer --------------------
with st.expander("About this demo", expanded=False):
    st.write(
        "This demo uses OpenAI embeddings for similarity search over fault descriptions, "
        "FastAPI as the backend, and Streamlit for the UI. You can optionally filter by component and model."
    )
