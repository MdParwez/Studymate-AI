# app.py
import os
import math
import re
import uuid
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

# LangChain memory container (we'll use it as a convenience wrapper)
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

try:
    # recommended packages (new langchain wrappers)
    from langchain_huggingface import HuggingFaceEmbeddings as LCHF_Embeddings
    from langchain_chroma import Chroma as LC_Chroma
    EMBEDDINGS_BACKEND = "langchain_huggingface"
    _USE_NEW_LANGCHAIN = True
except Exception:
    # fallback to community packages (older)
    from langchain_community.embeddings import SentenceTransformerEmbeddings as LCH_Embeddings
    from langchain_community.vectorstores import Chroma as LCC_Chroma
    EMBEDDINGS_BACKEND = "langchain_community"
    _USE_NEW_LANGCHAIN = False

ChromaClass = LC_Chroma if _USE_NEW_LANGCHAIN else LCC_Chroma
EmbeddingsClass = LCHF_Embeddings if _USE_NEW_LANGCHAIN else LCH_Embeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Groq client for LLM calls
from groq import Groq

# Cross-encoder lazy import placeholder
_crossencoder = None

# sentence-transformers (for batch embeddings)
from sentence_transformers import SentenceTransformer

# -------------------------
# Basic config & constants
# -------------------------
load_dotenv()
st.set_page_config(page_title="StudyMate AI", layout="wide")
PRIMARY = "#a855f7"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing. Put GROQ_API_KEY=your_key in a .env file and restart.")
    st.stop()

GEMMA_MODEL = "gemma2-9b-it"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast and compact
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
TOP_K = 8
EMBED_BATCH_SIZE = 64

# UI images (stable permanent URLs)
BANNER_URL = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80"
SIDEBAR_BOOK_ICON = "https://cdn-icons-png.flaticon.com/512/29/29302.png"
SIDEBAR_LIBRARY_IMG = "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"

# Styling and SVGs
ASSISTANT_SVG = """
<img src="data:image/svg+xml;utf8,
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' width='22' height='22'>
  <defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'><stop offset='0' stop-color='%23a855f7'/><stop offset='1' stop-color='%236b21a8'/></linearGradient></defs>
  <rect x='8' y='12' width='48' height='36' rx='6' fill='url(%23g)'/>
  <circle cx='24' cy='28' r='3.2' fill='%23fff'/>
  <circle cx='40' cy='28' r='3.2' fill='%23fff'/>
  <rect x='22' y='38' width='20' height='3' rx='1.5' fill='%23fff' opacity='0.9'/>
  <rect x='28' y='8' width='8' height='6' rx='2' fill='%23a855f7'/>
</svg>"/>
"""

st.markdown(
    f"""
    <style>
    :root {{ --primary: {PRIMARY}; }}
    [data-testid="stSidebar"] {{ padding: 1rem; }}
    .header-card {{ background-color: {PRIMARY}; padding:16px; border-radius:10px; color: white; display:flex; gap:16px; align-items:center; }}
    .chat-user {{ background: linear-gradient(90deg, rgba(168,85,247,0.12), rgba(168,85,247,0.08)); border-left: 4px solid var(--primary); padding:10px; border-radius:10px; margin:8px 0; display:flex; gap:8px; align-items:flex-start; }}
    .chat-assistant {{ background:#f3f4f6; padding:10px; border-radius:10px; margin:8px 0; display:flex; gap:8px; align-items:flex-start; }}
    .icon-left {{ width:26px; height:26px; margin-top:2px; }}
    .quiz-card {{ border:1px solid #eee; padding:14px; border-radius:10px; margin-bottom:12px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
    .question-badge {{ display:inline-block; background:var(--primary); color:white; padding:6px 10px; border-radius:12px; font-weight:700; margin-right:10px; }}
    .correct {{ color: green; font-weight:700; }}
    .summary-card {{ border-left:6px solid var(--primary); padding:12px; border-radius:8px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.03); margin-bottom:12px; }}
    .small-muted {{ color:#666; font-size:13px; }}
    .recent-file-btn {{ text-align: left; width: 100%; padding:6px 8px; border-radius:6px; border: none; background: transparent; }}
    .recent-chat-btn {{ text-align: left; width: 100%; padding:6px 8px; border-radius:6px; border: none; background: transparent; }}
    .recent-chat-active {{ font-weight:700; color: var(--primary); }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Prompts 
# -------------------------
QNA_SYSTEM = """
You are StudyMate, a friendly and accurate study assistant. Use ONLY the provided context passages.
- If the topic or keyword appears in the context, return the relevant details from the context.
- Do NOT invent facts or use outside knowledge.
- If the answer is not present in the provided context, reply exactly: "I could not find the answer in the provided material."
- When you reference a fact, add a friendly source note like: (from filename.pdf)
Answer in a simple, customer-friendly tone.
"""

SUMMARY_SYSTEM = """
You are StudyMate, expert teaching assistant. Using ONLY the provided context, OUTPUT in Markdown:

### Summary
<3–6 sentences overview>

### Study Notes
- 5–7 concise bullets

### Key Definitions
- **Term**: short definition

If insufficient info, say: "The provided material does not contain enough information to summarize fully."
"""

QUIZ_SYSTEM = """
You are StudyMate, a quiz generator. Using ONLY the provided context, create EXACTLY 5 multiple-choice questions.
Output strictly in this format for each question:

Q1. Question text
A. option
B. option
C. option
D. option
Answer: X

Rules:
- Do not use outside knowledge.
- Make questions clear and relevant to the context.
"""

# -------------------------
# Utilities (PDF extraction, LLM call, rerank, rewrite)
# -------------------------
def extract_page_text(page):
    try:
        return page.get_text()
    except Exception:
        return ""

def extract_text_from_pdf_parallel(file_bytes: bytes, max_workers: int = 6) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        pages = list(pdf)
        results = [None] * len(pages)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pages))) as ex:
            futures = {ex.submit(extract_page_text, p): idx for idx, p in enumerate(pages)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = ""
        text = "\n\n".join(results)
    return text

def call_groq_chat(messages: List[dict], model: str = GEMMA_MODEL, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def rewrite_short_query_for_retrieval(query: str) -> str:
    if len(query.strip().split()) <= 2:
        return f"What do the provided documents say about {query.strip()}?"
    return query

def lazy_load_crossencoder():
    global _crossencoder
    if _crossencoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _crossencoder = None
    return _crossencoder

def rerank_with_crossencoder(query: str, docs: List[Document]) -> List[Document]:
    model = lazy_load_crossencoder()
    if not model:
        return docs
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]

# -------------------------
# Session state initialization
# -------------------------
if "embeddings" not in st.session_state:
    try:
        st.session_state.embeddings = EmbeddingsClass(model_name=EMBEDDING_MODEL_NAME)
    except Exception:
        try:
            st.session_state.embeddings = EmbeddingsClass()
        except Exception:
            st.session_state._sbert = SentenceTransformer(EMBEDDING_MODEL_NAME)
            st.session_state.embeddings = None

if "chroma_db" not in st.session_state:
    try:
        st.session_state.chroma_db = ChromaClass(embedding_function=st.session_state.embeddings, persist_directory=None)
        st.session_state._chroma_direct = False
    except Exception:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))
        try:
            collection = client.get_collection("studymate")
        except Exception:
            collection = client.create_collection("studymate")
        st.session_state.chroma_db = collection
        st.session_state._chroma_direct = True

# Document/indexing state
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Recent files & per-file histories (existing)
if "recent_files" not in st.session_state:
    st.session_state.recent_files = []
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "active_file" not in st.session_state:
    st.session_state.active_file = None

# -------------------------
# New: Conversation memory, recent chats & interactive quiz
# -------------------------
# Conversation memory (LangChain wrapper used only to manage message objects)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# recent chat session names (max 5) and chat logs mapping
if "recent_chats" not in st.session_state:
    st.session_state.recent_chats = []  # list of chat names, newest first
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = {}  # chat_name -> [ {role, content, timestamp} ]

# identifier for the currently active chat session (not file)
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# interactive quiz storage
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []  # list of {q, options, answer, raw}
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}  # keys quiz_{i} -> selected

# Helper: create a new chat session name and initialize structures
def start_new_chat_session(name: Optional[str] = None) -> str:
    if not name:
        name = f"Chat {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # push to recent_chats front
    st.session_state.recent_chats = [c for c in st.session_state.recent_chats if c != name]
    st.session_state.recent_chats.insert(0, name)
    st.session_state.recent_chats = st.session_state.recent_chats[:5]  # keep last 5
    st.session_state.chat_logs.setdefault(name, [])
    st.session_state.current_chat = name
    # clear memory and optionally reload with logs if any (fresh session begins empty)
    st.session_state.memory.clear()
    return name

# Helper: append message to current chat (and memory)
def append_message_to_current_chat(role: str, content: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    chat_name = st.session_state.current_chat or start_new_chat_session()
    st.session_state.chat_logs.setdefault(chat_name, []).append({"role": role, "content": content, "timestamp": ts})
    # Also keep a lightweight visible chat_history (render order)
    st.session_state.chat_history.append({"role": role, "content": content, "timestamp": ts})
    # add to LangChain memory wrapper (so we can later inject)
    try:
        if role.lower() == "user":
            st.session_state.memory.chat_memory.add_user_message(content)
        else:
            st.session_state.memory.chat_memory.add_ai_message(content)
    except Exception:
        # Fallback: memory wrapper may not have chat_memory attribute in some versions
        pass

# -------------------------
# Sidebar UI
# -------------------------
with st.sidebar:
    st.image(SIDEBAR_BOOK_ICON, width=80)
    st.markdown(f"<h2 style='color:{PRIMARY};margin:6px 0 4px 0'>StudyMate AI</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#444;margin-bottom:8px;'>Upload PDFs/TXT — ask questions, summarize chapters, and generate quizzes.</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload files (PDF / TXT)", type=["pdf", "txt"], accept_multiple_files=True)

    st.markdown("---")
    if st.button("📝 Summarize Uploaded Docs"):
        st.session_state._run_summarize = True

    if st.button("🎯 Generate Quiz (5 Qs)"):
        st.session_state._run_quiz = True

    st.markdown("---")
    st.markdown("### 📜 Recent Files (click to load history)")
    if st.session_state.get("recent_files"):
        for idx, entry in enumerate(st.session_state.recent_files[:5]):
            fname = entry.get("filename", "unknown.pdf")
            ts = entry.get("timestamp", "")
            label = f"{fname}  ({ts})"
            btn_key = f"recent_file_btn_{idx}_{fname}"
            if fname == st.session_state.active_file:
                display_label = f"**{fname}**  <span class='recent-chat-active'>({ts})</span>"
                if st.button(display_label, key=btn_key, help="Click to load this file's chat history"):
                    st.session_state.active_file = fname
                    st.session_state.chat_histories.setdefault(fname, [])
                    st.session_state.chat_history = list(st.session_state.chat_histories.get(fname, []))
                    st.session_state.last_sources = []
                    st.rerun()
            else:
                if st.button(label, key=btn_key):
                    st.session_state.active_file = fname
                    st.session_state.chat_histories.setdefault(fname, [])
                    st.session_state.chat_history = list(st.session_state.chat_histories.get(fname, []))
                    st.session_state.last_sources = []
                    st.rerun()
    else:
        st.markdown("<div class='small-muted'>No recent files.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 Recent Chats")
    # recent chat list (max 5) - clickable to load chat session into memory and main view
    if st.session_state.recent_chats:
        for i, chat_name in enumerate(st.session_state.recent_chats[:5]):
            btn_key = f"recent_chat_btn_{i}_{chat_name}"
            display = f"{chat_name}"
            if chat_name == st.session_state.current_chat:
                display = f"**{chat_name}** <span class='recent-chat-active'>(active)</span>"
                if st.button(display, key=btn_key):
                    # reload (no-op) -> but ensure memory is populated from logs
                    st.session_state.chat_history = list(st.session_state.chat_logs.get(chat_name, []))
                    # populate LangChain memory with messages from chat logs
                    st.session_state.memory.clear()
                    for msg in st.session_state.chat_logs.get(chat_name, []):
                        if msg["role"].lower() == "user":
                            try:
                                st.session_state.memory.chat_memory.add_user_message(msg["content"])
                            except Exception:
                                pass
                        else:
                            try:
                                st.session_state.memory.chat_memory.add_ai_message(msg["content"])
                            except Exception:
                                pass
                    st.rerun()
            else:
                if st.button(display, key=btn_key):
                    st.session_state.current_chat = chat_name
                    # load visible history
                    st.session_state.chat_history = list(st.session_state.chat_logs.get(chat_name, []))
                    # populate LangChain memory with messages from chat logs
                    st.session_state.memory.clear()
                    for msg in st.session_state.chat_logs.get(chat_name, []):
                        if msg["role"].lower() == "user":
                            try:
                                st.session_state.memory.chat_memory.add_user_message(msg["content"])
                            except Exception:
                                pass
                        else:
                            try:
                                st.session_state.memory.chat_memory.add_ai_message(msg["content"])
                            except Exception:
                                pass
                    st.rerun()
    else:
        st.markdown("<div class='small-muted'>No recent chats.</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("➕ New Conversation"):
        # create a new named chat session
        start_new_chat_session()
        st.success("Started a new conversation.")
        st.rerun()

# -------------------------
# Main header
# -------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="header-card">
        <div style="flex-shrink:0;">
            <img src="" width="240" style="border-radius:8px;">
        </div>
        <div>
            <h1 style="color:white;margin:0;">StudyMate AI</h1>
            <p style="color:#f0f0f0;margin:4px 0 0 0;">Your friendly study assistant — upload documents, ask questions, learn & quiz yourself.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

if st.session_state.active_file:
    st.markdown(f"**Active file:** {st.session_state.active_file}", unsafe_allow_html=True)

# -------------------------
# Handle uploads: chunk + embed + add
# -------------------------
if uploaded_files:
    new_docs = []
    added_any = False

    for file_idx, uploaded in enumerate(uploaded_files, start=1):
        if uploaded.name in st.session_state.indexed_files:
            st.info(f"File '{uploaded.name}' already indexed in this session — skipping.")
            continue

        file_bytes = uploaded.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        status = st.info(f"Processing file {file_idx}/{len(uploaded_files)}: {uploaded.name} ({file_size_mb:.2f} MB)")

        with st.spinner("Extracting text (parallel per-page)..."):
            try:
                text = extract_text_from_pdf_parallel(file_bytes, max_workers=6)
            except Exception:
                text = ""
                with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                    for page in pdf:
                        text += page.get_text() + "\n\n"

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        total_chunks = len(chunks)
        status.info(f"Processing file {file_idx}/{len(uploaded_files)}: {uploaded.name} — {total_chunks} chunks")

        batch_size = EMBED_BATCH_SIZE
        embeddings_list = []
        ids = []
        metadatas = []
        documents_texts = []

        progress = st.progress(0.0)
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch_chunks = chunks[start:end]
            if hasattr(st.session_state, "_sbert"):
                embs = st.session_state._sbert.encode(batch_chunks, show_progress_bar=False, convert_to_numpy=True)
                for i, chunk_text in enumerate(batch_chunks):
                    uid = str(uuid.uuid4())
                    ids.append(uid)
                    documents_texts.append(chunk_text)
                    metadatas.append({"source": uploaded.name, "chunk_index": start + i})
                    embeddings_list.append(embs[i].tolist())
            else:
                if st.session_state.embeddings is not None:
                    try:
                        embs = st.session_state.embeddings.embed_documents(batch_chunks)
                        for i, chunk_text in enumerate(batch_chunks):
                            uid = str(uuid.uuid4())
                            ids.append(uid)
                            documents_texts.append(chunk_text)
                            metadatas.append({"source": uploaded.name, "chunk_index": start + i})
                            embeddings_list.append(embs[i])
                    except Exception:
                        for i, chunk_text in enumerate(batch_chunks):
                            uid = str(uuid.uuid4())
                            ids.append(uid)
                            documents_texts.append(chunk_text)
                            metadatas.append({"source": uploaded.name, "chunk_index": start + i})
                            embeddings_list.append(None)
                else:
                    for i, chunk_text in enumerate(batch_chunks):
                        uid = str(uuid.uuid4())
                        ids.append(uid)
                        documents_texts.append(chunk_text)
                        metadatas.append({"source": uploaded.name, "chunk_index": start + i})
                        embeddings_list.append(None)

            progress.progress(min(end / max(total_chunks, 1), 1.0))

        progress.empty()

        try:
            if getattr(st.session_state, "_chroma_direct", False):
                valid_ids, valid_docs, valid_metas, valid_embs = [], [], [], []
                for i, emb in enumerate(embeddings_list):
                    if emb is not None:
                        valid_ids.append(ids[i])
                        valid_docs.append(documents_texts[i])
                        valid_metas.append(metadatas[i])
                        valid_embs.append(embeddings_list[i])
                if valid_ids:
                    st.session_state.chroma_db.add(
                        ids=valid_ids,
                        documents=valid_docs,
                        metadatas=valid_metas,
                        embeddings=valid_embs
                    )
                else:
                    st.session_state.chroma_db.add(documents=documents_texts, metadatas=metadatas, ids=ids)
            else:
                docs_to_add = [Document(page_content=t, metadata=m) for t, m in zip(documents_texts, metadatas)]
                try:
                    st.session_state.chroma_db.add_documents(docs_to_add)
                except Exception:
                    try:
                        st.session_state.chroma_db.add_documents(docs_to_add, embedding=st.session_state.embeddings)
                    except Exception:
                        pass
        except Exception:
            pass

        for t, m in zip(documents_texts, metadatas):
            st.session_state.all_docs.append(Document(page_content=t, metadata=m))

        st.session_state.indexed_files.add(uploaded.name)
        status.success(f"Indexed '{uploaded.name}' ✅")
        added_any = True

        try:
            entry = {"filename": uploaded.name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state.recent_files = [e for e in st.session_state.recent_files if e.get("filename") != uploaded.name]
            st.session_state.recent_files.insert(0, entry)
            st.session_state.recent_files = st.session_state.recent_files[:5]
            if uploaded.name not in st.session_state.chat_histories:
                st.session_state.chat_histories[uploaded.name] = []
        except Exception:
            pass

        st.session_state.chat_history = []
        st.session_state.last_sources = []

    if added_any:
        st.success("All set — your study materials are ready! Ask a question or generate a summary/quiz.")

st.markdown("---")

# -------------------------
# Retrieval function 
# -------------------------
def retrieve_for_query(query: str, k: int = TOP_K) -> List[Document]:
    if getattr(st.session_state, "_chroma_direct", False):
        try:
            if hasattr(st.session_state, "_sbert"):
                q_emb = st.session_state._sbert.encode([query], convert_to_numpy=True)[0].tolist()
                res = st.session_state.chroma_db.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas"])
                doc_texts = res.get("documents", [[]])[0]
                metadatas = res.get("metadatas", [[]])[0]
                docs = [Document(page_content=dt, metadata=md) for dt, md in zip(doc_texts, metadatas)]
            else:
                res = st.session_state.chroma_db.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
                doc_texts = res.get("documents", [[]])[0]
                metadatas = res.get("metadatas", [[]])[0]
                docs = [Document(page_content=dt, metadata=md) for dt, md in zip(doc_texts, metadatas)]
        except Exception:
            docs = []
    else:
        try:
            docs = st.session_state.chroma_db.similarity_search(query, k=k)
        except Exception:
            docs = []

    try:
        docs = rerank_with_crossencoder(query, docs)
    except Exception:
        pass

    st.session_state.last_sources = docs
    return docs

# -------------------------
# Summarization 
# -------------------------
SUMMARY_BATCH_CHUNKS = 8
SUMMARY_BATCH_LIMIT = 40

def summarize_batches(docs: List[Document], batch_size: int = SUMMARY_BATCH_CHUNKS) -> List[str]:
    partials = []
    docs_to_use = docs[:SUMMARY_BATCH_LIMIT]
    if not docs_to_use:
        return partials
    num_batches = math.ceil(len(docs_to_use) / batch_size)
    prog = st.progress(0)
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(docs_to_use))
        batch = docs_to_use[start:end]
        snippets = []
        chars_limit = 900
        for d in batch:
            txt = d.page_content.strip()
            if len(txt) > chars_limit:
                txt = txt[:chars_limit] + " ..."
            snippets.append(f"Source: {d.metadata.get('source','uploaded_doc')}\n{txt}")
        context_block = "\n\n".join(snippets)
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Context:\n{context_block}"}
        ]
        try:
            partial = call_groq_chat(messages, temperature=0.3, max_tokens=700)
        except Exception:
            partial = "Partial summary failed for this batch."
        partials.append(partial)
        prog.progress((i + 1) / num_batches)
    prog.empty()
    return partials

def synthesize_final_summary(partials: List[str]) -> str:
    if not partials:
        return "The provided material does not contain enough information to summarize fully."
    joined = "\n\n".join(partials[:20])
    prompt = (
        "You are StudyMate, an expert teaching assistant. Combine the following partial summaries into ONE final output in Markdown format with headings 'Summary', 'Study Notes', and 'Key Definitions'.\n\n"
        "Partial summaries:\n" + joined
    )
    messages = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}]
    try:
        final = call_groq_chat(messages, temperature=0.3, max_tokens=800)
    except Exception:
        final = "Failed to synthesize final summary."
    return final

# -------------------------
# Quiz parsing
# -------------------------
def parse_quiz_raw(raw: str) -> List[Dict[str, Any]]:
    """
    Parse the raw quiz text produced by the model into a list of questions:
    [{ 'q': str, 'options': [A,B,C,D], 'answer': 'A' }]
    """
    questions = []
    # Find Q blocks
    qblocks = re.findall(r"(Q\d+\..*?)(?=(?:Q\d+\.|$))", raw, flags=re.S)
    for qb in qblocks:
        # extract question header
        header = re.match(r"Q\d+\.\s*(.*?)\n", qb)
        qtext = header.group(1).strip() if header else qb.split("\n")[0].strip()
        opts = re.findall(r"^([A-D])\.\s*(.*)$", qb, flags=re.M)
        options = []
        for _, txt in opts:
            options.append(txt.strip())
        ans_m = re.search(r"Answer:\s*([A-D])", qb)
        answer = ans_m.group(1).strip() if ans_m else None
        # Ensure 4 options if possible (pad with blanks)
        while len(options) < 4:
            options.append("—")
        questions.append({"q": qtext, "options": options[:4], "answer": answer, "raw": qb.strip()})
    return questions

# -------------------------
# Summarize & Quiz triggers (with interactive quiz storage)
# -------------------------
if st.session_state.pop("_run_summarize", False):
    if not st.session_state.all_docs:
        st.warning("No documents uploaded yet. Please upload files to summarize.")
    else:
        st.info("Creating a structured summary for your materials...")
        partials = summarize_batches(st.session_state.all_docs, batch_size=SUMMARY_BATCH_CHUNKS)
        final_summary = synthesize_final_summary(partials)
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.markdown(final_summary, unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)
        msg_entry = {"role": "assistant", "content": final_summary, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        append_message_to_current_chat("assistant", final_summary)
        # Also attach to per-file history if a file is active
        if st.session_state.active_file:
            st.session_state.chat_histories.setdefault(st.session_state.active_file, []).append(msg_entry)

if st.session_state.pop("_run_quiz", False):
    if not st.session_state.all_docs:
        st.warning("No documents uploaded yet. Please upload files to generate a quiz.")
    else:
        results = retrieve_for_query("quiz", k=TOP_K)
        snippets = [d.page_content[:1200] for d in results]
        context_block = "\n\n".join(snippets)
        messages = [{"role": "system", "content": QUIZ_SYSTEM}, {"role": "user", "content": f"Context:\n{context_block}"}]
        try:
            raw_quiz = call_groq_chat(messages, temperature=0.5, max_tokens=700)
            # parse quiz into interactive structure
            parsed = parse_quiz_raw(raw_quiz)
            if not parsed:
                st.error("Quiz generator didn't return questions in expected format.")
            else:
                st.session_state.quiz_questions = parsed
                # store raw quiz in assistant message and append to chat logs
                msg_entry = {"role": "assistant", "content": raw_quiz, "meta": {"type": "quiz"}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                append_message_to_current_chat("assistant", raw_quiz)
                # Also store per-file if active file
                st.session_state.chat_histories.setdefault(st.session_state.active_file or "unassigned", []).append(msg_entry)
        except Exception as e:
            st.error(f"Quiz generation failed: {e}")

# -------------------------
# Chat input and handling (integrate memory)
# -------------------------
user_input = st.chat_input("Ask anything about your uploaded documents...")
if user_input:
    original_query = user_input.strip()
    # Append user message to current chat and LangChain memory
    append_message_to_current_chat("user", original_query)

    if not st.session_state.all_docs:
        st.warning("No documents uploaded. Please upload PDFs/TXT on the left.")
    else:
        with st.spinner("Looking up your materials..."):
            retrieval_query = rewrite_short_query_for_retrieval(original_query)
            results = retrieve_for_query(retrieval_query, k=TOP_K)

        # Build context with friendly anchors
        context_items = []
        for i, d in enumerate(results, start=1):
            snippet = d.page_content.strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + " ..."
            src = d.metadata.get("source", "uploaded_doc")
            context_items.append(f"[{i}](#source-{i}) Source: {src}\n{snippet}")

        context_block = "\n\n".join(context_items)

        # Build messages for Groq call: include system, memory messages, context, and user question.
        messages: List[dict] = [{"role": "system", "content": QNA_SYSTEM}]
        # Inject memory history if available (as user/assistant roles)
        try:
            mem_msgs = st.session_state.memory.chat_memory.messages
            for m in mem_msgs:
                if getattr(m, "type", "").lower() in ("human", "user"):
                    messages.append({"role": "user", "content": m.content})
                else:
                    messages.append({"role": "assistant", "content": m.content})
        except Exception:
            # fallback: if memory wrapper doesn't expose messages, use stored chat_history visible list
            for m in st.session_state.chat_history[-20:]:
                role = m.get("role", "user").lower()
                r = "user" if role == "user" else "assistant"
                messages.append({"role": r, "content": m.get("content", "")})

        # add retrieval context and the current question
        messages.append({"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {original_query}"})

        try:
            assistant_text = call_groq_chat(messages, temperature=0.0, max_tokens=512)
            append_message_to_current_chat("assistant", assistant_text)
            # also save per-file history if active_file present
            target_file = st.session_state.active_file or "unassigned"
            st.session_state.chat_histories.setdefault(target_file, []).append({"role": "user", "content": original_query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.session_state.chat_histories.setdefault(target_file, []).append({"role": "assistant", "content": assistant_text, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
        except Exception as e:
            st.error(f"LLM error: {e}")

# -------------------------
#  chat (friendly)
# -------------------------
for msg in st.session_state.chat_history:
    if msg["role"].lower() == "user":
        st.markdown(f"<div class='chat-user'><div class='icon-left'>🧑‍🎓</div><div><strong>You:</strong> {msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-assistant'><div class='icon-left'>{ASSISTANT_SVG}</div><div><strong>StudyMate:</strong></div></div>", unsafe_allow_html=True)
        if msg.get("meta", {}).get("type") == "quiz":
            raw = msg["content"]
            qs = re.findall(r"(Q\d+\..*?)(?=(?:Q\d+\.|$))", raw, flags=re.S)
            if not qs:
                st.markdown(raw, unsafe_allow_html=False)
            else:
                for q_i, qblock in enumerate(qs, start=1):
                    header_match = re.match(r"Q\d+\.\s*(.*?)\n", qblock)
                    qtext = header_match.group(1).strip() if header_match else qblock.strip().split("\n")[0]
                    opts = re.findall(r"^([A-D])\.\s*(.*)$", qblock, flags=re.M)
                    ans_match = re.search(r"Answer:\s*([A-D])", qblock)
                    correct = ans_match.group(1).strip() if ans_match else None

                    st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
                    st.markdown(f"<span class='question-badge'>Q{q_i}</span> **{qtext}**", unsafe_allow_html=True)
                    if opts:
                        for label, text in opts:
                            if label == correct:
                                st.markdown(f"- **{label}. {text}**  <span class='correct'>✅</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {label}. {text}", unsafe_allow_html=True)
                    else:
                        st.markdown(qblock, unsafe_allow_html=False)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"], unsafe_allow_html=False)

# -------------------------
# interactive quiz UI (if parsed quiz questions exist)
# -------------------------
if st.session_state.quiz_questions:
    st.markdown("---")
    st.subheader("📝 Interactive Quiz")
    with st.form("interactive_quiz_form"):
        for i, q in enumerate(st.session_state.quiz_questions):
            key = f"quiz_{i}"
            # pre-populate previously selected answer if present
            default = st.session_state.quiz_answers.get(key, None)
            # use radio with options
            selected = st.radio(f"Q{i+1}. {q['q']}", q["options"], index=0 if default is None else q["options"].index(default) if default in q["options"] else 0, key=key)
            st.session_state.quiz_answers[key] = selected
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        score = 0
        results_ui = []
        for i, q in enumerate(st.session_state.quiz_questions):
            key = f"quiz_{i}"
            selected = st.session_state.quiz_answers.get(key)
            correct_label = q.get("answer")
            correct_option_text = None
            # convert letter to option text if possible
            if correct_label and correct_label in ("A", "B", "C", "D"):
                idx = ord(correct_label) - ord("A")
                if 0 <= idx < len(q["options"]):
                    correct_option_text = q["options"][idx]
            # Compare selected with correct text
            if correct_option_text and selected == correct_option_text:
                score += 1
                results_ui.append((q["q"], selected, True, correct_option_text))
            else:
                results_ui.append((q["q"], selected, False, correct_option_text))
        st.success(f"Your score: {score}/{len(st.session_state.quiz_questions)}")
        for idx, (qtext, chosen, ok, correct_text) in enumerate(results_ui, start=1):
            status = "✅ Correct" if ok else "❌ Incorrect"
            st.markdown(f"**Q{idx}. {qtext}** — {status}")
            st.markdown(f"- Your answer: {chosen}")
            if not ok:
                st.markdown(f"- Correct answer: **{correct_text}**")

# -------------------------
# Sources expander with anchors
# -------------------------
if st.session_state.last_sources:
    with st.expander("View Sources used (click a citation to jump)"):
        for idx, doc in enumerate(st.session_state.last_sources, start=1):
            st.markdown(f"<a name='source-{idx}'></a>", unsafe_allow_html=True)
            src = doc.metadata.get("source", "uploaded_doc")
            snippet = doc.page_content[:1000].replace("\n", " ")
            st.markdown(f"**[{idx}] {src}**")
            st.write(snippet)

st.markdown("</div>", unsafe_allow_html=True)





