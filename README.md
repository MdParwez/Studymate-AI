# 📚 StudyMate AI — RAG-based PDF Study Assistant

StudyMate AI is an interactive **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**.  
It allows students and researchers to **upload PDFs**, ask questions, get **summaries**, generate **quizzes**, and explore content interactively using **Groq’s Gemma2-9B-IT LLM** and **ChromaDB**.

---

## 🚀 Features
- 📂 Upload **one or multiple PDFs**
- 🔍 **Semantic search** using embeddings
- 💡 Summarization & **quiz generation** from PDFs
- 🧠 RAG-based conversational Q&A
- ⚡ **Fast** response using Groq API
- 🎯 Supports **sentence-transformers** for embeddings
- 🖥 Clean **Streamlit** frontend

---

## 📌 Workflow

### **1️⃣ Uploading the File**
The user uploads one or more documents (PDF/TXT).
<img width="960" height="540" alt="Step 1 - Uploading Files" src="https://github.com/user-attachments/assets/ffcfbfbf-dcb5-48d5-9d94-e1195c00b723" />

---

### **2️⃣ Successful Upload & Home Screen**
After upload, a confirmation is displayed, and the user sees the home interface.
<img width="960" height="540" alt="Step 2 - Successful Upload" src="https://github.com/user-attachments/assets/10b264e7-ec55-4290-98a8-567283309b82" />

---

### **3️⃣ Q&A from the Document**
Ask any question from the uploaded document, and get AI-generated answers.
<img width="960" height="540" alt="Step 3 - Q&A" src="https://github.com/user-attachments/assets/65de7bc5-f276-4309-9f9c-93aa615d52f0" />

---

### **4️⃣ Summarization with Key Notes**
Get a concise summary along with important bullet points.
<img width="960" height="540" alt="Step 4 - Summarization" src="https://github.com/user-attachments/assets/7450fa03-e493-4e72-8213-286041095fd5" />

---

### **5️⃣ Quiz Generation**
Generate quizzes from the uploaded content for self-assessment.
<img width="960" height="540" alt="Step 5 - Quiz" src="https://github.com/user-attachments/assets/ec1f6464-8f58-438b-a07c-1d32c16949cf" />

---

## 🖼 System Flowchart
Below is the complete workflow architecture:

![Flowchart](https://github.com/user-attachments/assets/fd12e08c-fb88-4367-9652-2a442c705ec9)


---


## 🛠 Tech Stack
| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Language Model** | Groq - Gemma2-9B-IT |
| **Vector Database** | ChromaDB |
| **Embeddings** | Sentence-Transformers (HuggingFace) |
| **PDF Parsing** | PyMuPDF |
| **Environment Config** | python-dotenv |

---

## 🧩 RAG Concept
RAG (**Retrieval-Augmented Generation**) is a technique where the LLM retrieves relevant information from a vector store before generating a response.

**Why use RAG here?**
- Keeps answers **accurate** to the uploaded PDFs
- Reduces hallucinations from the LLM
- Allows working with **large documents** without exceeding token limits

---

## 📊 Project Flow

```

---

## 📥 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/studymate-ai.git
cd studymate-ai
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set environment variables**
   Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```



---

```
