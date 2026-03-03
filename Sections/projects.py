import streamlit as st

# ---------- PAGE: PROJECTS ----------
def page_projects():

    st.markdown("""
    <style>

    /* Page background */
    .stApp {
        background: linear-gradient(120deg,#f6f9ff,#eef1ff,#f8f5ff);
    }

    /* Header */
    .proj-main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg,#667eea,#764ba2,#6B8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .proj-subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Timeline */
    .proj-timeline {
        position: relative;
        margin-top: 2rem;
        padding-left: 30px;
    }

    .proj-timeline:before {
        content: "";
        position: absolute;
        left: 8px;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(#667eea,#764ba2);
        border-radius: 5px;
    }

    /* Card */
    .proj-item {
        position: relative;
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }

    .proj-item:before {
        content: "";
        position: absolute;
        left: -27px;
        top: 30px;
        height: 16px;
        width: 16px;
        border-radius: 50%;
        background: linear-gradient(135deg,#667eea,#764ba2);
        box-shadow: 0 0 0 5px rgba(102,126,234,0.15);
    }

    .proj-item:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.15);
    }

    .proj-top-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .proj-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .proj-date-badge {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        padding: 10px 18px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .proj-description {
        color: #555;
        line-height: 1.8;
        margin-top: 1rem;
    }

    .proj-tech-section {
        margin-top: 1.5rem;
    }

    .proj-tech-item {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        background: linear-gradient(white,white) padding-box,
                    linear-gradient(135deg,#667eea,#764ba2) border-box;
        border: 1.5px solid transparent;
        color: #555;
        transition: 0.3s;
    }

    .proj-tech-item:hover {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        cursor: default;
    }

    .proj-github-btn {
        display: inline-block;
        margin-top: 1.2rem;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none !important;
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white !important;
        transition: 0.3s;
    }

    .proj-github-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102,126,234,0.3);
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="proj-main-title">🚀 Projects</h1>', unsafe_allow_html=True)
    st.markdown('<p class="proj-subtitle">Selected work demonstrating impact and technical expertise</p>', unsafe_allow_html=True)

    projects_data = [
        {
            "name": "AI Resume Portfolio — RAG Powered Assistant",
            "date": "2025",
            "description": "Multi-page AI portfolio application with a context-aware assistant built using Retrieval Augmented Generation (RAG). Implemented document ingestion pipeline, vector indexing, semantic search, and LLM-powered responses using Groq inference and HuggingFace embeddings. Designed modular AI architecture with caching and session management for high-performance query handling.",
            "tech": [
                "Python",
                "Streamlit",
                "Retrieval Augmented Generation (RAG)",
                "Large Language Models",
                "Vector Search",
                "Embeddings",
                "HuggingFace",
                "Groq",
                "NLP",
                "AI System Architecture"
            ],
            "github": "https://github.com/ChandrashekharRobbi/AI-Resume-Portfolio-RAG-Powered-Assistant"
        },

        {
            "name": "GenMed — Medicine Information Chatbot",
            "date": "2024",
            "description": "AI-powered chatbot providing generic medicine prices and nearest Jan Aushadi Kendra search by pincode. Implemented NLP-based interaction, web scraping, and LLM integration for healthcare assistance.",
            "tech": ["Python", "Streamlit", "NLP", "Web Scraping", "LLM Integration"],
            "github": "https://github.com/ChandrashekharRobbi/GenMed"
        },
        {
            "name": "Paddy Doctor — Crop Disease Classification System",
            "date": "2024",
            "description": "AI-powered crop disease classification system using deep learning and transfer learning (VGG16). Built image classification pipeline achieving 98% accuracy for automated paddy disease detection.",
            "tech": ["Python", "TensorFlow", "VGG16", "Deep Learning", "Computer Vision"],
            "github": "https://github.com/ChandrashekharRobbi/Convolutional-Neural-Network-with-Tensorflow/tree/main/Classification%20Neural%20Networks/Paddy%20Doctor"
        },


        {
            "name": "Movie & Series Watchlist Automation Pipeline",
            "date": "2024",
            "description": "End-to-end automation pipeline that fetches movie data using OMDB API, validates in Google Sheets, and stores structured metadata in Notion database automatically.",
            "tech": ["Python", "OMDB API", "Google Sheets", "Apps Script", "Notion API"],
            "github": "https://github.com/ChandrashekharRobbi/Movie-Series-Watchlist-Automation-Pipeline/tree/main"
        },

        {
            "name": "LipNet — Visual Speech Recognition System",
            "date": "2023",
            "description": "Deep learning system that predicts spoken text from lip movement without audio input. Built custom model training pipeline and Streamlit interface for real-time prediction.",
            "tech": ["Python", "Deep Learning", "Computer Vision", "Video Processing", "Streamlit"],
            "github": "https://github.com/ChandrashekharRobbi/LipNet"
        },

        {
            "name": "GitHub Automation Watchdog",
            "date": "2023",
            "description": "Automation script that detects new functions using git diff and regex, automatically generates commit messages, and pushes changes to GitHub to improve development workflow efficiency.",
            "tech": ["Python", "Git", "Regex", "Automation"],
            "github": "https://github.com"
        },

        {
            "name": "LeetCode Automation Tool",
            "date": "2023",
            "description": "Automation tool that retrieves LeetCode problem descriptions and examples using problem name input, enabling faster and structured coding practice workflow.",
            "tech": ["Python", "Automation"],
            "github": "https://github.com/ChandrashekharRobbi/Web-Scraping/tree/main/LeetCode%20Type%20OF%20Questions"
        },

        {
            "name": "MNIST Digit Classifier",
            "date": "2023",
            "description": "Neural network model for handwritten digit recognition (0–9) demonstrating foundational deep learning and pattern recognition capability.",
            "tech": ["Python", "Neural Networks", "Deep Learning"],
            "github": "https://github.com/ChandrashekharRobbi/Convolutional-Neural-Network-with-Tensorflow/tree/main/Classification%20Neural%20Networks/MNIST%20Digit%20Classifier"
        }
    ]

    st.markdown('<div class="proj-timeline">', unsafe_allow_html=True)

    for project in projects_data:
        tech_badges = "".join(
            [f'<span class="proj-tech-item">{t}</span>' for t in project["tech"]]
        )

        st.markdown(f"""
        <div class="proj-item">
            <div class="proj-top-row">
                <div class="proj-title">{project['name']}</div>
                <div class="proj-date-badge">{project['date']}</div>
            </div>
            <div class="proj-description">
                {project['description']}
            </div>
            <div class="proj-tech-section">
                {tech_badges}
            </div>
            <a href="{project['github']}" target="_blank" class="proj-github-btn">
                🔗 View on GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
