# app.py
"""
StudyMate AI — Customer-ready single-file Streamlit RAG app (IN-MEMORY)
Features:
 - PyMuPDF parallel extraction
 - SentenceTransformers (all-MiniLM-L6-v2) batched embeddings
 - In-memory vector store (numpy) + cosine similarity retrieval
 - Optional cross-encoder reranking (lazy)
 - Groq/Gemma chat completions for Q&A, summaries, quizzes
 - Styled UI: chat bubbles, assistant SVG, quiz cards, smart actions, pinned sources
 - Session-only recent chats, automatic save when starting new conversation
"""
import os
import math
import re
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Config (defaults) ---
load_dotenv()
st.set_page_config(page_title="StudyMate AI", layout="wide")
PRIMARY = "#a855f7"
BANNER_URL = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?q=80&w=1400&auto=format&fit=crop"
SIDEBAR_BOOK_ICON = "https://cdn-icons-png.flaticon.com/512/29/29302.png"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing — add to .env (GROQ_API_KEY=...) and restart.")
    st.stop()
GEMMA_MODEL = "gemma2-9b-it"

# default chunk settings (you asked chunk size 50)
DEFAULT_CHUNK_SIZE = 50
DEFAULT_CHUNK_OVERLAP = 20

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast & compact
EMBED_BATCH_SIZE = 64
PAGE_EXTRACT_WORKERS = 6
TOP_K_DEFAULT = 8

# --- Styling & SVG ---
ASSISTANT_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' width='26' height='26'>
<defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'><stop offset='0' stop-color='#a855f7'/><stop offset='1' stop-color='#6b21a8'/></linearGradient></defs>
<rect x='8' y='12' width='48' height='36' rx='6' fill='url(#g)'/>
<circle cx='24' cy='28' r='3.2' fill='#fff'/>
<circle cx='40' cy='28' r='3.2' fill='#fff'/>
<rect x='22' y='38' width='20' height='3' rx='1.5' fill='#fff' opacity='0.9'/>
</svg>"""

st.markdown(f"""
<style>
:root {{ --p1: {PRIMARY}; --p2: #6b21a8; }}
body {{ font-family: Inter, Roboto, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial; }}
.header-card {{ background: linear-gradient(90deg,var(--p1),var(--p2)); padding:16px; border-radius:10px; color:white; display:flex; gap:16px; align-items:center; margin-bottom:10px; }}
.sidebar-book {{ width:64px; height:64px; }}
.chat-row {{ display:flex; gap:12px; margin:8px 0; align-items:flex-start; }}
.chat-user {{ justify-content:flex-end; }}
.bubble {{ padding:12px 14px; border-radius:14px; max-width:85%; line-height:1.35; box-shadow: 0 6px 18px rgba(0,0,0,0.04); animation: fadeIn 180ms ease-in; }}
.bubble.user {{ background: linear-gradient(90deg,var(--p1),var(--p2)); color:white; border-bottom-right-radius:6px; }}
.bubble.assistant {{ background:#ffffff; color:#111827; border:1px solid #eee; border-bottom-left-radius:6px; }}
.icon-left {{ width:36px; height:36px; display:flex; align-items:center; justify-content:center; }}
.summary-card {{ border-left:6px solid var(--p1); padding:12px; border-radius:8px; background:#fff; margin-bottom:12px; }}
.quiz-card {{ border:1px solid #eee; padding:14px; border-radius:10px; margin-bottom:12px; background:#fff; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
.action-btn {{ background: linear-gradient(90deg,var(--p1),var(--p2)); color: white; padding:6px 10px; border-radius:8px; border: none; cursor: pointer; }}
.source-row {{ display:flex; justify-content:space-between; align-items:center; gap:8px; padding:6px 0; border-bottom:1px dashed #f6f6f6; }}
.pinned {{ background:#fff8e6; padding:6px; border-radius:6px; border:1px solid #ffe6a7; margin-bottom:6px; }}
@keyframes fadeIn {{ from {{ opacity:0; transform: translateY(6px) }} to {{ opacity:1; transform: translateY(0) }} }}
.stChatInput {{ margin-top: 8px !important; }}
</style>
""", unsafe_allow_html=True)

# --- Prompts ---
QNA_SYSTEM = """
You are StudyMate, a friendly and accurate study assistant. Use ONLY the provided context passages.
- If the topic appears in the context, return relevant details.
- If the answer is not present, reply exactly: "I could not find the answer in the provided material."
- When referencing facts, include citation tokens like [1], [2].
Answer concisely and in a friendly tone.
"""
SUMMARY_SYSTEM = """
You are StudyMate, an expert teaching assistant. Using ONLY the provided context produce EXACT Markdown:

### Summary
<3-6 sentences>

### Study Notes
- bullet points

### Key Definitions
- **Term**: definition
"""
QUIZ_SYSTEM = """
You are StudyMate, a quiz generator. Using ONLY the provided context produce EXACTLY 5 MCQs in this format:

Q1. question?
A. opt
B. opt
C. opt
D. opt
Answer: X
"""

# -------------------------
# Helpers: PDF extraction & chunking
# -------------------------
def extract_page_text(page):
    try:
        return page.get_text()
    except Exception:
        return ""

def extract_pages_parallel(file_bytes: bytes, max_workers: int = PAGE_EXTRACT_WORKERS) -> List[str]:
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
    return results  # list of page texts in order

def chunk_texts(pages_texts: List[str], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Return list of {"text":..., "metadata": {"source":filename, "page":pno, "chunk_index":i}}
    We call this after we already mapped which pages belong to which file.
    """
    splitter = []
    # We will do a simple sliding splitter per page to respect chunk_size & overlap
    chunks = []
    for pno, pt in enumerate(pages_texts):
        if not pt or not pt.strip():
            continue
        tokens = pt.split()
        if len(tokens) <= chunk_size:
            chunks.append(" ".join(tokens))
        else:
            i = 0
            while i < len(tokens):
                chunk_tokens = tokens[i:i+chunk_size]
                chunks.append(" ".join(chunk_tokens))
                if i+chunk_size >= len(tokens):
                    break
                i = max(i + chunk_size - chunk_overlap, i+1)
    return chunks

# -------------------------
# In-memory vector store
# -------------------------
class InMemoryVectorStore:
    def __init__(self):
        self.embeddings = []       # list of lists
        self.texts = []            # chunk texts
        self.metadatas = []        # metadata dicts
        self.ids = []              # ids
        self.matrix = None         # numpy matrix of embeddings for fast search

    def add(self, texts: List[str], metadatas: List[dict], ids: Optional[List[str]] = None, embs: Optional[List[List[float]]] = None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if embs is None:
            # caller must ensure embeddings computed else throw
            raise ValueError("Embeddings not provided to add()")
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.embeddings.extend(embs)
        self.matrix = np.vstack(self.embeddings) if self.embeddings else None

    def similarity_search(self, query_emb: List[float], k: int = 8):
        if self.matrix is None or len(self.texts) == 0:
            return []
        q = np.array(query_emb, dtype=float).reshape(1, -1)
        # cosine similarity
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(q)
        dots = (self.matrix @ q.T).squeeze()
        # avoid div by zero
        denom = norms
        denom[denom == 0] = 1e-8
        sims = dots / denom
        idxs = np.argsort(-sims)[:k]
        results = []
        for idx in idxs:
            results.append({
                "id": self.ids[int(idx)],
                "text": self.texts[int(idx)],
                "metadata": self.metadatas[int(idx)],
                "score": float(sims[int(idx)])
            })
        return results

# -------------------------
# LLM / Embedding init
# -------------------------
# load SBERT model for batch encoding
@st.cache_resource(show_spinner=False)
def get_sbert(model_name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(model_name)

sbert = get_sbert()

def embed_texts(texts: List[str], batch_size=EMBED_BATCH_SIZE):
    embs = []
    # sentence-transformers encode in batches efficiently
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors = sbert.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs.extend([v.tolist() for v in vectors])
    return embs

@st.cache_resource(show_spinner=False)
def get_inmem_store():
    return InMemoryVectorStore()

store = get_inmem_store()

# Cross-encoder reranker (lazy)
_crossencoder = None
def lazy_crossencoder():
    global _crossencoder
    if _crossencoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _crossencoder = None
    return _crossencoder

def rerank(query: str, hits: List[dict]) -> List[dict]:
    model = lazy_crossencoder()
    if not model or not hits:
        return hits
    pairs = [[query, h["text"]] for h in hits]
    scores = model.predict(pairs)
    scored = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [h for h, s in scored]

# Groq LLM call
def call_groq_chat(messages: List[dict], model=GEMMA_MODEL, max_tokens=512, temperature=0.0):
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content

# -------------------------
# Session state init
# -------------------------
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []      # store texts + metadata for summarization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of dicts {role, content, meta?}
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []  # last retrieval hits
if "pinned_sources" not in st.session_state:
    st.session_state.pinned_sources = []  # list of metadata dicts
if "recent_chats" not in st.session_state:
    st.session_state.recent_chats = []    # session-only saved chats

# -------------------------
# Sidebar UI (upload + settings)
# -------------------------
with st.sidebar:
    st.image(SIDEBAR_BOOK_ICON, width=80)
    st.markdown(f"<h2 style='color:{PRIMARY};margin:6px 0 4px 0'>StudyMate AI</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#444;margin-bottom:6px;'>Upload PDFs/TXT — ask questions, summarize chapters, and generate quizzes.</div>", unsafe_allow_html=True)

    # Settings
    st.markdown("## Settings")
    chunk_size = st.slider("Chunk size (words)", 20, 400, value=DEFAULT_CHUNK_SIZE, step=5, help="How many words per chunk")
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 200, value=DEFAULT_CHUNK_OVERLAP, step=5, help="Overlap between chunks for coherence")
    top_k = st.slider("Top-k retrieval", 1, 12, value=TOP_K_DEFAULT)
    rerank_enabled = st.checkbox("Enable reranking (slower)", value=True, help="Use cross-encoder to rerank retrieved chunks")
    st.markdown("---")

    uploaded_files = st.file_uploader("Upload PDFs / TXT (multiple)", type=["pdf", "txt"], accept_multiple_files=True)

    st.markdown("---")
    if st.button("📝 Summarize Uploaded Docs"):
        st.session_state._run_summarize = True
    if st.button("🎯 Generate Quiz (5 Qs)"):
        st.session_state._run_quiz = True
    st.markdown("---")
    if st.button("➕ New Conversation"):
        # auto-save conversation to recent_chats (session-only)
        if st.session_state.chat_history:
            title = st.session_state.chat_history[0]["content"][:48]
            st.session_state.recent_chats.insert(0, {"title": title, "messages": st.session_state.chat_history.copy(), "ts": datetime.now().isoformat()})
            st.session_state.chat_history = []
            st.session_state.last_retrieved = []
            st.success("Started a new conversation — previous saved in recent chats (session).")

    st.markdown("## Recent Chats (session)")
    for i, rc in enumerate(st.session_state.recent_chats):
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            if st.button(rc["title"], key=f"load_{i}"):
                st.session_state.chat_history = rc["messages"].copy()
        with cols[1]:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.recent_chats.pop(i)
                st.experimental_rerun()

# -------------------------
# Main header
# -------------------------
st.markdown(f"""<div class="header-card">
    <div style="flex-shrink:0;"><img src="{BANNER_URL}" width="220" style="border-radius:8px;"></div>
    <div><h1 style="margin:0">StudyMate AI</h1><div style="opacity:.9">Your friendly study assistant — upload documents, ask questions, quiz yourself.</div></div>
</div>""", unsafe_allow_html=True)

# -------------------------
# Handle uploads (sequential per-file; parallel per-page; batched embeddings)
# -------------------------
if uploaded_files:
    added_any = False
    for uploaded in uploaded_files:
        if uploaded.name in st.session_state.indexed_files:
            st.info(f"'{uploaded.name}' already indexed this session; skipping.")
            continue
        file_bytes = uploaded.read()
        file_mb = len(file_bytes) / (1024*1024)
        status = st.info(f"Processing {uploaded.name} ({file_mb:.2f} MB)")

        # extract pages in parallel
        with st.spinner("Extracting pages..."):
            try:
                pages_texts = extract_pages_parallel(file_bytes, max_workers=PAGE_EXTRACT_WORKERS)
            except Exception:
                # fallback serial
                pages_texts = []
                with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                    for p in pdf:
                        pages_texts.append(p.get_text())

        # chunk per page & preserve ordering
        chunks = []
        chunk_metadatas = []
        for pno, ptext in enumerate(pages_texts, start=1):
            page_chunks = chunk_texts([ptext], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for ci, ch in enumerate(page_chunks):
                chunks.append(ch)
                chunk_metadatas.append({"source": uploaded.name, "page": pno, "chunk_index": len(chunks)-1})

        total_chunks = len(chunks)
        if total_chunks == 0:
            status.error(f"No text found in {uploaded.name}; skipping.")
            continue
        status.info(f"Embedding {total_chunks} chunks (this can take a moment)...")

        # compute embeddings in batches (use sbert.encode)
        embs = embed_texts(chunks, batch_size=EMBED_BATCH_SIZE)

        # add to in-memory store
        try:
            store.add(texts=chunks, metadatas=chunk_metadatas, ids=[str(uuid.uuid4()) for _ in chunks], embs=embs)
        except Exception as e:
            # if something goes wrong, append to session-level fallback
            st.warning("Failed to add to in-memory store: " + str(e))

        # append to session all_chunks for summarization
        for t, m in zip(chunks, chunk_metadatas):
            st.session_state.all_chunks.append({"text": t, "metadata": m})

        st.session_state.indexed_files.add(uploaded.name)
        status.success(f"Indexed '{uploaded.name}' ({total_chunks} chunks) ✅")
        added_any = True

    if added_any:
        # clear chat for fresh start (per your earlier preference)
        st.session_state.chat_history = []
        st.session_state.last_retrieved = []
        st.success("All uploads processed in memory — ready to ask questions.")

st.markdown("---")

# -------------------------
# Retrieval helpers (embedding query + similarity)
# -------------------------
def retrieve(query: str, k: int = TOP_K_DEFAULT, rerank_flag: bool = True):
    q_emb = sbert.encode([query], convert_to_numpy=True)[0].tolist()
    hits = store.similarity_search(q_emb, k=k)
    if rerank_flag:
        try:
            hits = rerank(query, hits)
        except Exception:
            pass
    st.session_state.last_retrieved = hits
    return hits

# -------------------------
# Summarize / Quiz actions -> put into main chat
# -------------------------
if st.session_state.pop("_run_summarize", False):
    if not st.session_state.all_chunks:
        st.warning("No uploaded content to summarize.")
    else:
        # sample top N chunks for summarization
        sample_texts = [c["text"] for c in st.session_state.all_chunks[:min(len(st.session_state.all_chunks), 200)]]
        context_block = "\n\n".join(sample_texts[:120])
        messages = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": f"Context:\n{context_block}"}]
        with st.spinner("Generating structured summary..."):
            try:
                out = call_groq_chat(messages, temperature=0.3, max_tokens=800)
            except Exception as e:
                out = "Summarization failed: " + str(e)
        st.session_state.chat_history.append({"role": "assistant", "content": out, "meta": {"type": "summary"}})

if st.session_state.pop("_run_quiz", False):
    if not st.session_state.all_chunks:
        st.warning("No uploaded content to generate quiz.")
    else:
        # retrieve some chunks as context
        hits = retrieve("quiz", k=top_k, rerank_flag=rerank_enabled)
        snippets = [h["text"][:1200] for h in hits]
        context_block = "\n\n".join(snippets)
        messages = [{"role": "system", "content": QUIZ_SYSTEM}, {"role": "user", "content": f"Context:\n{context_block}"}]
        with st.spinner("Generating quiz..."):
            try:
                out = call_groq_chat(messages, temperature=0.5, max_tokens=700)
            except Exception as e:
                out = "Quiz generation failed: " + str(e)
        st.session_state.chat_history.append({"role": "assistant", "content": out, "meta": {"type": "quiz"}})

# -------------------------
# Chat input (Q&A)
# -------------------------
user_input = st.chat_input("Ask anything about the uploaded documents...")
if user_input:
    original = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": original})
    if not store.texts:
        st.warning("No documents indexed — upload PDFs on the left.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            hits = retrieve(original, k=top_k, rerank_flag=rerank_enabled)
        if not hits:
            st.session_state.chat_history.append({"role": "assistant", "content": "I could not find the answer in the provided material."})
        else:
            # build context with numbered citations
            context_items = []
            for i, h in enumerate(hits, start=1):
                src = h["metadata"].get("source", "uploaded_doc")
                snippet = h["text"][:900] + (" ..." if len(h["text"])>900 else "")
                context_items.append(f"[{i}] Source: {src}\n{snippet}")
            context_block = "\n\n".join(context_items)
            messages = [{"role":"system","content":QNA_SYSTEM}, {"role":"user","content":f"Context:\n{context_block}\n\nQuestion: {original}"}]
            try:
                answer = call_groq_chat(messages, temperature=0.0, max_tokens=512)
            except Exception as e:
                answer = "LLM error: " + str(e)
            st.session_state.chat_history.append({"role":"assistant","content":answer, "meta": {"hits": hits}})

# -------------------------
# Render chat (bubbles, smart actions, pinned sources)
# -------------------------
for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-row chat-user'><div class='bubble user'><strong>👤 You:</strong> {st.escape(msg['content'])}</div></div>", unsafe_allow_html=True)
    else:
        # assistant header with SVG
        st.markdown(f"<div class='chat-row'><div class='icon-left'>{ASSISTANT_SVG}</div><div class='bubble assistant'><strong>🤖 StudyMate:</strong></div></div>", unsafe_allow_html=True)
        # content below
        content = msg["content"]
        st.markdown(f"<div style='margin-left:46px;margin-bottom:6px'>{content}</div>", unsafe_allow_html=True)

        # smart actions (Summarize this answer / Quiz from this answer / Find more context)
        action_cols = st.columns([0.2,0.2,0.2,0.4])
        with action_cols[0]:
            if st.button("Summarize this answer", key=f"summarize_snip_{idx}"):
                # run summarization on assistant content
                messages = [{"role":"system","content":SUMMARY_SYSTEM}, {"role":"user","content":f"Context:\n{content}"}]
                try:
                    out = call_groq_chat(messages, temperature=0.25, max_tokens=400)
                except Exception as e:
                    out = "Summarize failed: " + str(e)
                st.session_state.chat_history.append({"role":"assistant","content":out,"meta":{"type":"summary_from_answer"}})
                st.experimental_rerun()
        with action_cols[1]:
            if st.button("Generate quiz from this", key=f"quiz_snip_{idx}"):
                messages = [{"role":"system","content":QUIZ_SYSTEM}, {"role":"user","content":f"Context:\n{content}"}]
                try:
                    out = call_groq_chat(messages, temperature=0.5, max_tokens=600)
                except Exception as e:
                    out = "Quiz failed: " + str(e)
                st.session_state.chat_history.append({"role":"assistant","content":out,"meta":{"type":"quiz_from_answer"}})
                st.experimental_rerun()
        with action_cols[2]:
            if st.button("Find more context", key=f"morectx_{idx}"):
                # retrieve by taking top phrases/keywords from answer (simple: take short keywords)
                # We'll just perform retrieval with the whole assistant answer
                hits = retrieve(content, k=top_k, rerank_flag=rerank_enabled)
                st.session_state.chat_history.append({"role":"assistant","content":"I found these related sources.", "meta":{"hits":hits}})
                st.experimental_rerun()

# -------------------------
# Show pinned sources (if any)
# -------------------------
if st.session_state.pinned_sources:
    st.markdown("<div class='pinned'><strong>Pinned sources</strong></div>", unsafe_allow_html=True)
    for p in st.session_state.pinned_sources:
        st.markdown(f"**{p.get('source')} (page {p.get('page', '?')})**")
        st.write(p.get("text","")[:400] + "...")

# -------------------------
# Show last retrieval (sources) with pin option
# -------------------------
if st.session_state.last_retrieved:
    with st.expander("View Sources used (click pin to keep)"):
        for i, h in enumerate(st.session_state.last_retrieved, start=1):
            meta = h["metadata"]
            src = meta.get("source", "uploaded_doc")
            cols = st.columns([0.8, 0.08, 0.12])
            with cols[0]:
                st.markdown(f"**[{i}] {src}**")
                st.write(h["text"][:800] + (" ..." if len(h["text"])>800 else ""))
            with cols[1]:
                if st.button("📌", key=f"pin_{i}"):
                    st.session_state.pinned_sources.append({"source":src, "page":meta.get("page"), "text":h["text"]})
            with cols[2]:
                st.markdown(f"<div style='font-size:12px;color:#666'>score: {h.get('score',0):.3f}</div>", unsafe_allow_html=True)

# -------------------------
# Render quiz blocks inside chat (parse and beautify)
# -------------------------
# we render at the end to keep structure stable
for msg in st.session_state.chat_history:
    if msg.get("meta",{}).get("type") == "quiz" or "quiz" in (msg.get("meta") or {}):
        # parse MCQ blocks Q1.. format
        raw = msg["content"]
        qs = re.findall(r"(Q\d+\..*?)(?=(?:Q\d+\.|$))", raw, flags=re.S)
        if qs:
            for q_i, qblock in enumerate(qs, start=1):
                header_match = re.match(r"Q\d+\.\s*(.*?)\n", qblock)
                qtext = header_match.group(1).strip() if header_match else qblock.strip().split("\n")[0]
                opts = re.findall(r"^([A-D])\.\s*(.*)$", qblock, flags=re.M)
                ans_match = re.search(r"Answer:\s*([A-D])", qblock)
                correct = ans_match.group(1).strip() if ans_match else None
                st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
                st.markdown(f"**Q{q_i}. {qtext}**", unsafe_allow_html=True)
                if opts:
                    for label, text in opts:
                        if label == correct:
                            st.markdown(f"- **{label}. {text}** ✅", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- {label}. {text}", unsafe_allow_html=True)
                else:
                    st.markdown(qblock, unsafe_allow_html=False)
                st.markdown("</div>", unsafe_allow_html=True)

# end of app


