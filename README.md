
# 🚀 AI Resume Portfolio — RAG Powered

> An intelligent multi-page AI portfolio built with Streamlit, RAG (Retrieval Augmented Generation), Vector Search, and Groq LLM.

Live AI assistant that answers questions about my:

* Experience
* Projects
* Skills
* Architecture
* Achievements
* Resume

---

## 🧠 Architecture Overview

This project follows a **Retrieval-Augmented Generation (RAG)** architecture:

1. Portfolio data stored in `/data`
2. Text converted into embeddings
3. Stored in a local vector store
4. User query → semantic search
5. Top matches passed to Groq LLM
6. Context-aware response generated

---

## 🛠 Tech Stack

* 🐍 Python
* 🎨 Streamlit (UI Framework)
* 🧠 RAG Architecture
* 🔎 Vector Search (FAISS / Local Vector Store)
* ⚡ Groq API (LLM Inference)
* 🤗 HuggingFace (Embeddings)
* 🔐 Environment Variable Security

---

## 📂 Project Structure

```
AI-Resume-Portfolio/
│
├── data/                    # Portfolio content files
├── Sections/                # Multi-page sections
│   ├── about_me.py
│   ├── experience.py
│   ├── skills.py
│   ├── projects.py
│   ├── education.py
│   ├── achievements.py
│   ├── resume.py
│   ├── contact.py
│   └── architecture.py
│
├── rag_pipeline.py          # RAG Logic
├── app.py                   # Main Streamlit app
├── .env                     # Environment variables
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ai-resume-portfolio.git
cd ai-resume-portfolio
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Get:

* Groq API key → [https://console.groq.com](https://console.groq.com)
* HuggingFace token → [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 💡 Features

✅ Multi-page portfolio UI
✅ AI Chat Assistant (RAG Powered)
✅ Cached Vector Store
✅ Context-aware responses
✅ First-time auto index building
✅ Clean sidebar navigation
✅ Recruiter-friendly design

---

## 🎯 Why This Project Matters

This isn’t just a portfolio.

It demonstrates:

* Real-world RAG implementation
* LLM integration
* Embedding pipelines
* Vector indexing
* Caching strategies
* Secure key management
* Modular architecture
* Production-grade Streamlit design

---

## 📸 AI Assistant Preview

The AI assistant can answer:

* “Tell me about your latest role”
* “Explain your system architecture”
* “What are your main skills?”
* “Why should we hire you?”

---

## 🚀 Future Improvements

* Deploy on Streamlit Cloud
* Add authentication
* Add PDF Resume download
* Docker support
* Cloud vector database
* Analytics dashboard

---

## 👨‍💻 Author

**Chandrashekhar Robbi**
AI Engineer | Backend Developer | Automation Specialist

* LinkedIn
* GitHub

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub!

