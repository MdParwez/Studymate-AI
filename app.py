# app.py
import os
import math
import re
import uuid
import html
from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Try modern LangChain imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    EMBEDDINGS_BACKEND = "langchain_huggingface"
    _USE_NEW_LANGCHAIN = True
except Exception:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma
    EMBEDDINGS_BACKEND = "langchain_community"
    _USE_NEW_LANGCHAIN = False

ChromaClass = Chroma
EmbeddingsClass = HuggingFaceEmbeddings if _USE_NEW_LANGCHAIN else SentenceTransformerEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------------
# Basic config & constants
# -------------------------
load_dotenv()
st.set_page_config(page_title="StudyMate AI", layout="wide")
PRIMARY = "#a855f7"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing. Add it to .env")
    st.stop()

GEMMA_MODEL = "gemma2-9b-it"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Developer settings in session_state defaults
if "dev_settings" not in st.session_state:
    st.session_state.dev_settings = {
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 50,
        "TOP_K": 8,
        "EMBED_BATCH_SIZE": 64
    }

# UI images
BANNER_URL = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?q=80&w=1400&auto=format&fit=crop"
SIDEBAR_BOOK_ICON = "https://cdn-icons-png.flaticon.com/512/29/29302.png"
SIDEBAR_LIBRARY_IMG = "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?q=80&w=800&auto=format&fit=crop"

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

# -------------------------
# Styling
# -------------------------
st.markdown(
    f"""
    <style>
    :root {{ --primary: {PRIMARY}; }}
    .header-card {{ background-color: {PRIMARY}; padding:16px; border-radius:10px; color: white; display:flex; gap:16px; align-items:center; }}
    .chat-user {{ background: linear-gradient(90deg, rgba(168,85,247,0.12), rgba(168,85,247,0.08)); border-left: 4px solid var(--primary); padding:10px; border-radius:10px; margin:8px 0; }}
    .chat-assistant {{ background:#f3f4f6; padding:10px; border-radius:10px; margin:8px 0; }}
    .icon-left {{ width:26px; height:26px; margin-top:2px; }}
    .quiz-card {{ border:1px solid #eee; padding:14px; border-radius:10px; background: #fff; }}
    .summary-card {{ border-left:6px solid var(--primary); padding:12px; border-radius:8px; background:#fff; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Prompts
# -------------------------
QNA_SYSTEM = """You are StudyMate..."""
SUMMARY_SYSTEM = """You are StudyMate..."""
QUIZ_SYSTEM = """You are StudyMate..."""

# -------------------------
# Utility functions
# -------------------------
def extract_page_text(page):
    try:
        return page.get_text()
    except:
        return ""

def extract_text_from_pdf_parallel(file_bytes: bytes, max_workers=6) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        pages = list(pdf)
        results = [None] * len(pages)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pages))) as ex:
            futures = {ex.submit(extract_page_text, p): i for i, p in enumerate(pages)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    return "\n\n".join(results)

def call_groq_chat(messages, model=GEMMA_MODEL, max_tokens=1024, temperature=0.0):
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# -------------------------
# Session state init
# -------------------------
if "embeddings" not in st.session_state:
    try:
        st.session_state.embeddings = EmbeddingsClass(model_name=EMBEDDING_MODEL_NAME)
    except:
        st.session_state._sbert = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st.session_state.embeddings = None

if "chroma_db" not in st.session_state:
    from chromadb import Client
    from chromadb.config import Settings
    st.session_state.chroma_db = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=None)).create_collection("studymate")

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.image(SIDEBAR_BOOK_ICON, width=80)
    st.image(SIDEBAR_LIBRARY_IMG, use_container_width=True)
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt"], accept_multiple_files=True)

    st.markdown("---")
    if st.button("➕ New Conversation"):
        st.session_state.chat_history = []
        st.session_state.last_sources = []

    st.markdown("---")
    st.subheader("⚙ Developer Settings")
    st.session_state.dev_settings["CHUNK_SIZE"] = st.number_input("Chunk Size", 100, 2000, st.session_state.dev_settings["CHUNK_SIZE"])
    st.session_state.dev_settings["CHUNK_OVERLAP"] = st.number_input("Chunk Overlap", 0, 500, st.session_state.dev_settings["CHUNK_OVERLAP"])
    st.session_state.dev_settings["TOP_K"] = st.number_input("Top K Retrieval", 1, 20, st.session_state.dev_settings["TOP_K"])
    st.session_state.dev_settings["EMBED_BATCH_SIZE"] = st.number_input("Embed Batch Size", 1, 128, st.session_state.dev_settings["EMBED_BATCH_SIZE"])

# -------------------------
# File processing
# -------------------------
if uploaded_files:
    for uploaded in uploaded_files:
        if uploaded.name in st.session_state.indexed_files:
            continue
        text = extract_text_from_pdf_parallel(uploaded.read())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.dev_settings["CHUNK_SIZE"],
            chunk_overlap=st.session_state.dev_settings["CHUNK_OVERLAP"]
        )
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            st.session_state.chroma_db.add(
                ids=[str(uuid.uuid4())],
                documents=[chunk],
                metadatas=[{"source": uploaded.name, "chunk": i}]
            )
            st.session_state.all_docs.append(Document(page_content=chunk, metadata={"source": uploaded.name}))
        st.session_state.indexed_files.add(uploaded.name)

# -------------------------
# Chat input
# -------------------------
user_input = st.chat_input("Ask anything...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    results = st.session_state.chroma_db.query(query_texts=[user_input], n_results=st.session_state.dev_settings["TOP_K"])
    context_block = "\n\n".join(results["documents"][0])
    assistant_reply = call_groq_chat([
        {"role": "system", "content": QNA_SYSTEM},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {user_input}"}
    ])
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

# -------------------------
# Render chat
# -------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'><b>You:</b> {html.escape(msg['content'])}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-assistant'><div class='icon-left'>{ASSISTANT_SVG}</div>{msg['content']}</div>", unsafe_allow_html=True)



