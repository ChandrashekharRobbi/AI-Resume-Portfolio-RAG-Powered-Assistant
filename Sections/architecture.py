import streamlit as st
from pathlib import Path


def page_architecture():
    """System Architecture Page — RAG Portfolio"""

    # ---------- CONFIG ----------
    IMAGE_PATH = Path("img/Architecture.png")

    # ---------- PAGE HEADER ----------
    st.title("🏗️ System Architecture")
    st.caption("Understanding how the AI Portfolio + RAG System operates")

    st.divider()

    # ---------- OVERVIEW ----------
    st.markdown(
        """
### 🚀 Overview

This portfolio uses **Retrieval Augmented Generation (RAG)** to deliver intelligent,
context-aware responses about experience, projects, and skills.

The system combines:

- ⚡ **Streamlit** → Interactive UI
- 🔎 **Vector Similarity Search** → Context retrieval
- 🧠 **LLM Inference (Groq + Llama)** → Response generation
- 📚 **Knowledge Base Indexing** → Structured data storage
- 🤖 **AI Pipeline** → End-to-end query processing
"""
    )

    st.divider()

    # ---------- ARCHITECTURE DIAGRAM ----------
    st.subheader("📊 Architecture Diagram")

    if IMAGE_PATH.exists():
        st.image(
            IMAGE_PATH,
            caption="Retrieval Augmented Generation Pipeline",
            width='stretch',
        )
    else:
        st.warning("⚠️ Architecture diagram not found")
        st.info("Place your architecture image inside the **img/** folder.")

    st.divider()

    # ---------- PIPELINE FLOW ----------
    st.subheader("⚙️ System Workflow")

    steps = [
        (
            "1️⃣ User Interaction",
            """
- User submits query via Streamlit chat interface
- Request forwarded to RAG pipeline
""",
        ),
        (
            "2️⃣ Query Processing",
            """
- Query classification
- Cache lookup for fast response
- Preprocessing and normalization
""",
        ),
        (
            "3️⃣ Context Retrieval",
            """
- Query converted into embeddings
- FAISS performs similarity search
- Relevant documents retrieved
""",
        ),
        (
            "4️⃣ AI Response Generation",
            """
- Context passed to Groq LLM
- Llama model generates response
- Answer validated and formatted
""",
        ),
        (
            "5️⃣ Response Delivery",
            """
- Response streamed back to UI
- Optimized for low latency experience
""",
        ),
    ]

    for title, description in steps:
        with st.expander(title, expanded=True):
            st.markdown(description)

    st.divider()

    # ---------- TECH STACK ----------
    st.subheader("🧰 Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
### 🎨 Frontend
- Streamlit
- Python
- Interactive UI Components
"""
        )

    with col2:
        st.markdown(
            """
### 🔎 Vector Search
- FAISS
- Sentence Transformers
- all-MiniLM-L6-v2 embeddings
"""
        )

    with col3:
        st.markdown(
            """
### 🧠 AI / Backend
- Groq API
- Llama 3.1
- Retrieval Augmented Generation
"""
        )

    st.divider()

    # ---------- OPTIMIZATION ----------
    with st.expander("🚀 Performance Optimizations"):
        st.markdown(
            """
- Vector index caching
- Embedding caching
- Response caching
- Top-K similarity retrieval
- Streaming responses
- Low-latency inference pipeline
"""
        )

    # ---------- FOOTER ----------
    st.caption("Built with scalable RAG architecture principles.")