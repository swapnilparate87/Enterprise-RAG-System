"""
Streamlit UI for Enterprise RAG System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Enterprise RAG System")
st.caption("100% FREE · Ollama + ChromaDB + FastAPI")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")

    if st.button("🔄 Check Health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                st.success("✅ API is running!")
            else:
                st.error(f"❌ API returned {r.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach API\n{e}")

    if st.button("📊 Get Stats"):
        try:
            r = requests.get(f"{API_BASE}/api/v1/stats", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.metric("Documents in DB", data.get("total_documents", 0))
                st.info(f"Version: {data.get('app_version', 'N/A')}")
            else:
                st.error(f"Error: {r.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.button("🗑️ Clear Database", type="secondary"):
        if st.session_state.get("confirm_clear"):
            try:
                r = requests.delete(f"{API_BASE}/api/v1/clear", timeout=30)
                if r.status_code == 200:
                    st.success("✅ Database cleared!")
                    st.session_state["confirm_clear"] = False
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.session_state["confirm_clear"] = True
            st.warning("Click again to confirm!")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "📄 Upload File", "📝 Ingest Text", "🌐 Ingest URL"])

# ── Tab 1: Query ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("💬 Ask a Question")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is LangChain? What does the document say about...",
        height=100
    )

    k = st.slider("Number of source chunks to retrieve", min_value=1, max_value=10, value=5)

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking... (this may take 10-30s with local LLM)"):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/v1/query",
                        json={"question": question, "k": k},
                        timeout=120
                    )
                    if r.status_code == 200:
                        data = r.json()

                        st.success("✅ Answer received!")

                        # Answer
                        st.markdown("### 💡 Answer")
                        st.markdown(data["answer"])

                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Confidence", f"{data['confidence_score']:.0%}")
                        col2.metric("Retrieval", f"{data['retrieval_time']:.2f}s")
                        col3.metric("Generation", f"{data['generation_time']:.2f}s")
                        col4.metric("Total", f"{data['total_time']:.2f}s")

                        # Sources
                        if data.get("sources"):
                            st.markdown("### 📚 Sources")
                            for src in data["sources"]:
                                with st.expander(f"Source {src['id']} — {src['metadata'].get('source', 'Unknown')}"):
                                    st.write(src["content"])
                                    st.json(src["metadata"])
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The LLM is still generating — try again or reduce k.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 2: Upload File ────────────────────────────────────────────────────────
with tab2:
    st.subheader("📄 Upload a Document")
    st.caption("Supported: PDF, TXT, DOCX")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx"]
    )

    if st.button("⬆️ Upload & Ingest", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/v1/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=120
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ {data['message']}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Documents", data["num_documents"])
                        col2.metric("Chunks", data["num_chunks"])
                        col3.metric("Time", f"{data['ingestion_time']:.2f}s")
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 3: Ingest Text ────────────────────────────────────────────────────────
with tab3:
    st.subheader("📝 Ingest Raw Text")

    source_name = st.text_input("Source name", value="manual_input.txt")
    text_input = st.text_area(
        "Paste your text here",
        placeholder="Paste any text content you want to add to the knowledge base...",
        height=250
    )

    if st.button("➕ Ingest Text", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Ingesting text..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/v1/ingest-text",
                        json={"text": text_input, "source_name": source_name},
                        timeout=60
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ {data['message']}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Documents", data["num_documents"])
                        col2.metric("Chunks", data["num_chunks"])
                        col3.metric("Time", f"{data['ingestion_time']:.2f}s")
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 4: Ingest URL ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("🌐 Ingest a Web Page")

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/article"
    )

    if st.button("🌐 Fetch & Ingest", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner(f"Fetching '{url_input}'..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/v1/ingest-url",
                        json={"url": url_input},
                        timeout=60
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ {data['message']}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Documents", data["num_documents"])
                        col2.metric("Chunks", data["num_chunks"])
                        col3.metric("Time", f"{data['ingestion_time']:.2f}s")
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Error: {e}")