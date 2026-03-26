"""
Enterprise RAG System - Streamlit UI
Features: Multi-model, Streaming, Chat history, Dark theme, Export, Document management
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime

API_BASE = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    .user-msg {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px; margin: 8px 0; margin-left: 20%;
        color: white; font-size: 15px;
    }
    .assistant-msg {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px; margin: 8px 0; margin-right: 20%;
        color: #e6edf3; font-size: 15px;
    }
    .msg-meta { font-size: 11px; color: #8b949e; margin-top: 6px; }
    .source-card {
        background: #161b22; border-left: 3px solid #1f6feb;
        border-radius: 4px; padding: 10px 14px; margin: 6px 0;
        font-size: 13px; color: #8b949e;
    }
    .model-badge {
        display: inline-block; background: #21262d;
        border: 1px solid #30363d; border-radius: 12px;
        padding: 2px 10px; font-size: 11px; color: #58a6ff;
        margin-left: 8px;
    }
    .stButton > button {
        border-radius: 8px; border: 1px solid #30363d;
        background: #21262d; color: #e6edf3; transition: all 0.2s;
    }
    .stButton > button:hover { border-color: #58a6ff; color: #58a6ff; }
    .stTextArea textarea, .stTextInput input {
        background: #161b22 !important; border: 1px solid #30363d !important;
        color: #e6edf3 !important; border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Available Models ──────────────────────────────────────────────────────────
MODELS = {
    "qwen2:1.5b":    {"label": "Qwen2 1.5B",   "speed": "⚡⚡⚡", "quality": "⭐⭐",    "desc": "Fastest — best for quick Q&A"},
    "gemma2:2b":     {"label": "Gemma2 2B",     "speed": "⚡⚡",   "quality": "⭐⭐⭐",  "desc": "Balanced speed & quality"},
    "mistral:latest":{"label": "Mistral 7B",    "speed": "⚡",     "quality": "⭐⭐⭐⭐", "desc": "Best quality, slower"},
    "deepseek-r1:8b":{"label": "DeepSeek R1 8B","speed": "⚡",     "quality": "⭐⭐⭐⭐", "desc": "Best for reasoning tasks"},
}

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "confirm_clear_db" not in st.session_state:
    st.session_state.confirm_clear_db = False
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "qwen2:1.5b"
if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def get_stats():
    try:
        r = requests.get(f"{API_BASE}/api/v1/stats", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def switch_model(model_name: str):
    """Tell the backend to switch model"""
    try:
        r = requests.post(f"{API_BASE}/api/v1/switch-model", json={"model_name": model_name}, timeout=30)
        return r.status_code == 200
    except:
        return False

def export_txt():
    lines = ["Enterprise RAG System — Chat Export", "=" * 50, ""]
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        model = f" [{msg.get('model', '')}]" if msg.get("model") else ""
        lines.append(f"[{msg['timestamp']}] {role}{model}:")
        lines.append(msg["content"])
        if msg.get("timing"):
            lines.append(f"  ⏱ {msg['timing']['total']:.1f}s")
        lines.append("")
    return "\n".join(lines).encode("utf-8")

def export_json():
    return json.dumps(st.session_state.chat_history, indent=2, default=str).encode("utf-8")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG System")
    st.markdown("*100% FREE · Local AI*")
    st.divider()

    # Status
    api_ok = check_api()
    if api_ok:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")

    # Stats
    stats = get_stats()
    if stats:
        col1, col2 = st.columns(2)
        col1.metric("📄 Chunks", stats.get("total_documents", 0))
        col2.metric("💬 Chats", len(st.session_state.chat_history))

    st.divider()

    # ── Model Selector ────────────────────────────────────────────────────────
    st.markdown("### 🤖 Active Model")
    
    current_info = MODELS.get(st.session_state.selected_model, {})
    st.markdown(f"""
    <div style='background:#1f6feb22;border:1px solid #1f6feb;border-radius:8px;
    padding:10px;margin-bottom:8px;'>
        <b style='color:#58a6ff;'>✅ {current_info.get("label", st.session_state.selected_model)}</b><br>
        <span style='color:#8b949e;font-size:12px;'>{current_info.get("desc","")}</span>
    </div>""", unsafe_allow_html=True)
    
    new_model = st.selectbox(
        "Switch model",
        options=list(MODELS.keys()),
        format_func=lambda x: f"{MODELS[x]['label']} {MODELS[x]['speed']}",
        index=list(MODELS.keys()).index(st.session_state.selected_model),
        label_visibility="collapsed"
    )
    
    if st.button("🔄 Switch Model", use_container_width=True, key="switch_btn"):
        if new_model != st.session_state.selected_model:
            with st.spinner(f"Switching to {MODELS[new_model]['label']}..."):
                if switch_model(new_model):
                    st.session_state.selected_model = new_model
                    st.success(f"✅ Now using {MODELS[new_model]['label']}!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed! Run: ollama pull {new_model}")
        else:
            st.info("Already using this model!")

    st.divider()

    # ── Streaming Toggle ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Options")
    st.session_state.use_streaming = st.toggle(
        "🌊 Streaming responses",
        value=st.session_state.use_streaming,
        help="Words appear live as the model generates them"
    )

    st.divider()

    # ── Chat Controls ─────────────────────────────────────────────────────────
    st.markdown("### 💬 Chat")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        st.download_button("📥 Export TXT", data=export_txt(),
            file_name=f"rag_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True)
        st.download_button("📥 Export JSON", data=export_json(),
            file_name=f"rag_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json", use_container_width=True)

    st.divider()

    # ── DB Controls ───────────────────────────────────────────────────────────
    st.markdown("### 🗄️ Database")
    if not st.session_state.confirm_clear_db:
        if st.button("🗑️ Clear Database", use_container_width=True):
            st.session_state.confirm_clear_db = True
            st.rerun()
    else:
        st.warning("⚠️ Delete ALL documents?")
        c1, c2 = st.columns(2)
        if c1.button("✅ Yes", use_container_width=True):
            try:
                r = requests.delete(f"{API_BASE}/api/v1/clear", timeout=30)
                if r.status_code == 200:
                    st.session_state.ingested_docs = []
                    st.success("Cleared!")
            except Exception as e:
                st.error(str(e))
            st.session_state.confirm_clear_db = False
            st.rerun()
        if c2.button("❌ No", use_container_width=True):
            st.session_state.confirm_clear_db = False
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Enterprise RAG System")
st.markdown("*Ask questions about your documents — local AI, 100% free & private*")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "⚙️ Settings"])

# ════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════
with tab1:
    # Chat display
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#8b949e;'>
            <div style='font-size:48px;'>🧠</div>
            <div style='font-size:20px;margin-top:12px;'>Start a conversation</div>
            <div style='font-size:14px;margin-top:8px;'>Upload documents in the Documents tab, then ask questions here</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='user-msg'>
                    {msg['content']}
                </div>
                <div style='text-align:right;margin-right:20%;margin-top:-4px;'>
                    <span style='font-size:11px;color:#8b949e;'>🕐 {msg['timestamp']}</span>
                </div>""", unsafe_allow_html=True)
            else:
                # ✅ Use st.markdown for rendered markdown in answers
                with st.container():
                    st.markdown(f"""
                    <div style='background:#161b22;border:1px solid #30363d;
                    border-radius:18px 18px 18px 4px;padding:12px 16px;
                    margin:8px 0;margin-right:20%;'>""", unsafe_allow_html=True)
                    st.markdown(msg["content"])  # ✅ renders **bold**, bullets etc.
                    meta = f"🕐 {msg['timestamp']}"
                    if msg.get("timing"):
                        meta += f" &nbsp;·&nbsp; ⏱ {msg['timing']['total']:.1f}s"
                    if msg.get("confidence"):
                        meta += f" &nbsp;·&nbsp; 🎯 {int(msg['confidence']*100)}%"
                    if msg.get("model"):
                        meta += f" &nbsp;·&nbsp; 🤖 {msg['model']}"
                    st.markdown(f"<div class='msg-meta'>{meta}</div></div>", unsafe_allow_html=True)

                    if msg.get("sources"):
                        with st.expander(f"📚 {len(msg['sources'])} source(s) used"):
                            for src in msg["sources"]:
                                page = src['metadata'].get('page', '')
                                page_str = f" · Page {page}" if page != '' else ''
                                st.markdown(f"""
                                <div class='source-card'>
                                    <b>📄 {src['metadata'].get('source','Unknown')}{page_str}</b><br><br>
                                    {src['content']}
                                </div>""", unsafe_allow_html=True)

    st.divider()

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_area("Question", placeholder="Ask anything about your documents...",
                                height=80, label_visibility="collapsed", key="q_input")
    with col2:
        k_val = st.number_input("Chunks", 1, 10, 4, help="Docs to retrieve")
        ask = st.button("🔍 Ask", type="primary", use_container_width=True)

    if ask and question.strip():
        if not api_ok:
            st.error("❌ API offline.")
        else:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            if st.session_state.use_streaming:
                # ── STREAMING MODE ────────────────────────────────────────────
                active_model = st.session_state.selected_model
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "",
                    "model": active_model,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
            else:
                # ── NORMAL MODE ───────────────────────────────────────────────
                with st.spinner("🤔 Thinking..."):
                    try:
                        start = time.time()
                        r = requests.post(f"{API_BASE}/api/v1/query",
                            json={"question": question, "k": k_val}, timeout=120)
                        if r.status_code == 200:
                            data = r.json()
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data.get("sources", []),
                                "confidence": data.get("confidence_score", 0),
                                "timing": {"total": data.get("total_time", time.time()-start)},
                                "model": data.get("model_used", st.session_state.selected_model),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"❌ Error {r.status_code}: {r.text}",
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                    except requests.exceptions.Timeout:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "⏱️ Timeout. Try reducing chunk count or switching to a faster model.",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": f"❌ {str(e)}",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                st.rerun()

    # ── Streaming handler ─────────────────────────────────────────────────────
    last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
    if (last and last["role"] == "assistant" and last.get("content") == ""
            and st.session_state.use_streaming):
        question_msg = next(
            (m["content"] for m in reversed(st.session_state.chat_history[:-1])
             if m["role"] == "user"), None
        )
        if question_msg:
            with st.spinner("🌊 Streaming..."):
                try:
                    start = time.time()
                    full_text = ""
                    stream_placeholder = st.empty()
                    with requests.post(
                        f"{API_BASE}/api/v1/query-stream",
                        json={"question": question_msg, "k": k_val if 'k_val' in dir() else 4},
                        stream=True, timeout=120
                    ) as r:
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_text += chunk
                                stream_placeholder.markdown(full_text + "▌")
                    stream_placeholder.empty()
                    elapsed = time.time() - start
                    st.session_state.chat_history[-1]["content"] = full_text
                    st.session_state.chat_history[-1]["timing"] = {"total": elapsed}
                except Exception as e:
                    st.session_state.chat_history[-1]["content"] = f"❌ Streaming error: {e}"
            st.rerun()

# ════════════════════════════════════════════════
# TAB 2 — DOCUMENTS
# ════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### 📤 Add Documents")
        d1, d2, d3 = st.tabs(["📄 File", "📝 Text", "🌐 URL"])

        with d1:
            uploaded = st.file_uploader("PDF, TXT or DOCX", type=["pdf","txt","docx"])
            if st.button("⬆️ Upload & Ingest", use_container_width=True, key="up_btn"):
                if not uploaded:
                    st.warning("Select a file first.")
                else:
                    with st.spinner(f"Ingesting '{uploaded.name}'..."):
                        try:
                            r = requests.post(f"{API_BASE}/api/v1/upload",
                                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                                timeout=120)
                            if r.status_code == 200:
                                d = r.json()
                                st.success(f"✅ {d['num_chunks']} chunks in {d['ingestion_time']:.1f}s")
                                st.session_state.ingested_docs.append({
                                    "name": uploaded.name, "chunks": d['num_chunks'],
                                    "time": datetime.now().strftime("%H:%M:%S"), "type": "file"
                                })
                            else:
                                st.error(r.text)
                        except Exception as e:
                            st.error(str(e))

        with d2:
            src = st.text_input("Source name", "manual_input.txt", key="src")
            txt = st.text_area("Text content", height=150, key="txt_in")
            if st.button("➕ Ingest", use_container_width=True, key="txt_btn"):
                if not txt.strip():
                    st.warning("Enter text first.")
                else:
                    with st.spinner("Ingesting..."):
                        try:
                            r = requests.post(f"{API_BASE}/api/v1/ingest-text",
                                json={"text": txt, "source_name": src}, timeout=60)
                            if r.status_code == 200:
                                d = r.json()
                                st.success(f"✅ {d['num_chunks']} chunks")
                                st.session_state.ingested_docs.append({
                                    "name": src, "chunks": d['num_chunks'],
                                    "time": datetime.now().strftime("%H:%M:%S"), "type": "text"
                                })
                            else:
                                st.error(r.text)
                        except Exception as e:
                            st.error(str(e))

        with d3:
            url = st.text_input("URL", placeholder="https://example.com", key="url_in")
            if st.button("🌐 Fetch & Ingest", use_container_width=True, key="url_btn"):
                if not url.strip():
                    st.warning("Enter URL first.")
                else:
                    with st.spinner("Fetching..."):
                        try:
                            r = requests.post(f"{API_BASE}/api/v1/ingest-url",
                                json={"url": url}, timeout=60)
                            if r.status_code == 200:
                                d = r.json()
                                st.success(f"✅ {d['num_chunks']} chunks")
                                st.session_state.ingested_docs.append({
                                    "name": url, "chunks": d['num_chunks'],
                                    "time": datetime.now().strftime("%H:%M:%S"), "type": "url"
                                })
                            else:
                                st.error(r.text)
                        except Exception as e:
                            st.error(str(e))

    with col_r:
        st.markdown("### 📋 Ingested Documents")
        if not st.session_state.ingested_docs:
            st.markdown("""
            <div style='text-align:center;padding:40px;color:#8b949e;
            border:1px dashed #30363d;border-radius:8px;'>
                No documents ingested yet
            </div>""", unsafe_allow_html=True)
        else:
            type_icons = {"file": "📄", "text": "📝", "url": "🌐"}
            for i, doc in enumerate(st.session_state.ingested_docs):
                icon = type_icons.get(doc.get("type", "file"), "📄")
                ca, cb = st.columns([5, 1])
                with ca:
                    st.markdown(f"""
                    <div style='background:#161b22;border:1px solid #30363d;
                    border-radius:8px;padding:10px;margin:4px 0;'>
                        <b style='color:#58a6ff;'>{icon} {doc['name']}</b><br>
                        <span style='color:#8b949e;font-size:12px;'>
                        {doc['chunks']} chunks · {doc['time']}</span>
                    </div>""", unsafe_allow_html=True)
                with cb:
                    if st.button("🗑️", key=f"d_{i}"):
                        st.session_state.ingested_docs.pop(i)
                        st.rerun()

        st.markdown("### 📊 Vector Store")
        if stats:
            total = stats.get('total_documents', 0)
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid #30363d;
            border-radius:8px;padding:16px;'>
                <div style='color:#58a6ff;font-size:28px;font-weight:bold;'>{total}</div>
                <div style='color:#8b949e;'>Total chunks stored</div>
                <br>
                <div style='color:#8b949e;font-size:12px;'>
                🧮 {stats.get('embedding_model','N/A')}<br>
                🤖 {stats.get('llm_model','N/A')}<br>
                💰 $0/month
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 — SETTINGS
# ════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Comparison")
    cols = st.columns(len(MODELS))
    for col, (mid, info) in zip(cols, MODELS.items()):
        is_active = st.session_state.selected_model == mid
        border = "#1f6feb" if is_active else "#30363d"
        col.markdown(f"""
        <div style='background:#161b22;border:2px solid {border};
        border-radius:12px;padding:16px;text-align:center;'>
            <div style='font-size:18px;font-weight:bold;color:#e6edf3;'>{info['label']}</div>
            <div style='color:#8b949e;font-size:13px;margin:8px 0;'>{info['desc']}</div>
            <div>Speed: {info['speed']}</div>
            <div>Quality: {info['quality']}</div>
            {'<div style="color:#1f6feb;margin-top:8px;font-weight:bold;">✅ Active</div>' if is_active else ''}
        </div>""", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📥 Export Chat")
        if st.session_state.chat_history:
            st.download_button("📄 Download TXT", data=export_txt(),
                file_name=f"rag_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain", use_container_width=True)
            st.download_button("📋 Download JSON", data=export_json(),
                file_name=f"rag_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json", use_container_width=True)
        else:
            st.info("No chat history yet.")

    with c2:
        st.markdown("### 🛠️ Quick Commands")
        st.code("""# Start API
cd backend
uvicorn app.main:app --reload

# Start UI
streamlit run streamlit_app.py

# Pull models
ollama pull qwen2:1.5b
ollama pull gemma2:2b
ollama pull mistral""")