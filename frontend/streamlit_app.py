# frontend/streamlit_app.py
import os
import requests
import streamlit as st

# -------------------- Config --------------------
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MAX_RESULTS = 10       # request up to this many from the API
MIN_SIMILARITY = 0.45  # only display results with similarity >= this value

st.set_page_config(page_title="RCA Demo", layout="centered")

# Hide Streamlit's "Press Ctrl+Enter to apply" hint under text inputs/areas
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
    st.session_state.uploader_key = 0  # used to force-clear the file_uploader selection

# -------------------- Header --------------------
st.title("Root Cause Analysis Demo")
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
        # Small link to show the uploader again if needed
        if st.button("Add another file", use_container_width=True):
            st.session_state.hide_uploader = False
            st.experimental_rerun()
    else:
        st.write("Upload a CSV with columns: **component**, **fault_description**, **root_cause**, **corrective_action**.")
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
                    # Hide controls and clear selected file on next render
                    st.session_state.hide_uploader = True
                    st.session_state.uploader_key += 1  # changing the key clears file_uploader
                    st.experimental_rerun()
                else:
                    st.error(f"Upload failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------- Fetch components for dropdown --------------------
def fetch_components(api_base: str):
    try:
        resp = requests.get(f"{api_base}/components", timeout=10)
        if resp.ok:
            comps = resp.json().get("components", [])
            return sorted({str(c).strip().lower() for c in comps if str(c).strip()})
    except Exception:
        pass
    return []

components = fetch_components(backend_url)

# -------------------- Inputs --------------------
display_components = ["All"] + [c.title() for c in components]
component_display = st.selectbox("Component (optional filter)", options=display_components, index=0)
query = st.text_area(
    "Describe the fault",
    placeholder="e.g. Motor making a high-pitched squeal and smells burnt…",
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

        with st.spinner("Analysing…"):
            try:
                r = requests.post(f"{backend_url}/diagnose", json=payload, timeout=120)
            except Exception as e:
                r = None
                st.error(f"Request failed: {e}")

        if r is not None:
            if r.ok:
                # --- Filter to show 'up to' MAX_RESULTS based on similarity threshold ---
                raw_results = r.json() or []
                try:
                    filtered = [
                        it for it in raw_results
                        if float(it.get("similarity", 0.0)) >= MIN_SIMILARITY
                    ]
                except Exception:
                    filtered = raw_results  # fallback if parsing similarity fails

                n = len(filtered)
                st.metric("Matches found", n)

                if n == 0:
                    st.info(
                        "No strong matches found. Try rephrasing the description or remove the component filter."
                    )
                else:
                    for i, item in enumerate(filtered, start=1):
                        st.markdown(f"### {i}. {item.get('component','').title()}")
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
        "FastAPI as the backend, and Streamlit for the UI. Upload new CSV rows to add components "
        "and update the knowledge base."
    )
