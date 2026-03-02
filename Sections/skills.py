import streamlit as st


def page_skills():
    """Technical Skills page with modern interactive UI."""

    st.header("🛠️ Technical Skills")

    # ---------- Initialize Session State ----------
    if "selected_skill" not in st.session_state:
        st.session_state.selected_skill = "Python"

    # ---------- Skill Data ----------
    skill_data = {
    "Python": "Primary programming language for backend development, automation engineering, data processing pipelines, and AI/ML systems. Strong experience building scalable production-ready solutions.",

    "Streamlit": "Interactive AI application development, multi-page app architecture, chat interfaces, and rapid ML product prototyping.",

    "Retrieval Augmented Generation (RAG)": "End-to-end RAG pipeline development including document indexing, embedding generation, vector search, and context-aware LLM responses.",

    "Large Language Models (LLMs)": "LLM integration, prompt engineering, response optimization, and context-aware AI system design.",

    "Generative AI": "Design and implementation of AI-powered assistants, conversational systems, and intelligent automation workflows.",

    "Vector Databases & Embeddings": "Semantic search, document embedding pipelines, similarity retrieval, and vector indexing for knowledge-based systems.",

    "NLP": "Natural language processing for conversational AI, semantic search, and intelligent query understanding.",

    "HuggingFace Ecosystem": "Model integration, embedding models, inference optimization, and transformer-based NLP workflows.",

    "Groq LLM Inference": "Ultra-fast LLM inference integration and performance optimization for real-time AI applications.",

    "AI System Architecture": "Design of scalable AI pipelines including retrieval layers, caching, inference workflows, and modular architecture.",

    "Machine Learning": "Model training, evaluation, and optimization including classification and predictive systems.",

    "Deep Learning": "Computer vision models, transfer learning (VGG16), and neural network pipelines.",

    "FastAPI": "High-performance REST API development and scalable backend service implementation.",

    "Flask": "Lightweight backend development and rapid API prototyping.",

    "Automation Engineering": "Workflow automation, task orchestration, and productivity optimization systems.",

    "Data Processing Pipelines": "Batch processing, document ingestion pipelines, caching strategies, and performance optimization.",

    "Session Management & Caching": "Resource caching, session state handling, and performance optimization for AI applications.",

    "SQL": "Database querying, data manipulation, and backend data handling with multi-environment execution support.",

    "Git & Version Control": "Collaborative development, version management, and CI-friendly workflows.",

    "Azure (Data Environment)": "Backend workflows and automation deployment in cloud environments.",

    "Web Scraping": "Automated data extraction and processing pipelines.",

    "Regex Processing": "Pattern-based extraction, text processing, and validation logic.",
    "LangChain": "LLM orchestration, prompt management, and AI workflow design using the LangChain framework.",
}

    # ---------- Layout ----------
    col1, col2 = st.columns([2, 1])

    # ---------- Left Side : Skill Selection ----------
    with col1:
        st.subheader("Select a Skill")

        # Search box
        search = st.text_input("🔍 Search skill")

        filtered_skills = [
            skill for skill in skill_data
            if search.lower() in skill.lower()
        ] if search else list(skill_data.keys())

        # Skill grid
        grid_cols = st.columns(4)

        for i, skill in enumerate(filtered_skills):
            with grid_cols[i % 4]:
                if st.button(
                    skill,
                    width='stretch',
                    key=f"skill_{skill}"
                ):
                    st.session_state.selected_skill = skill

    # ---------- Right Side : Description ----------
    with col2:
        st.subheader("Description")

        selected = st.session_state.selected_skill

        if selected in skill_data:
            st.markdown(
                f"""
                <div style="
                    padding:1.2rem;
                    border-radius:12px;
                    border-left:5px solid #6366f1;
                    box-shadow:0 4px 10px rgba(0,0,0,0.2);
                ">
                <h4 style="margin-bottom:0.5rem;">{selected}</h4>
                <p style="line-height:1.6;">
                {skill_data[selected]}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------- Skills Overview ----------
    # ---------- Skills Overview (Card UI) ----------
    st.divider()
    st.subheader("Complete Skills Overview")

    # Skill categories
    skill_categories = {
    "💻 Programming": [
        "Python (Primary)",
        "SQL",
    ],

    "⚡ Backend & APIs": [
        "FastAPI",
        "Flask",
        "REST API Development",
        "API Integration",
        "Backend Workflow Design",
        "Session Management & Caching",
    ],

    "🤖 AI & Generative AI": [
        "Large Language Models (LLMs)",
        "Generative AI Applications",
        "Prompt Engineering",
        "Retrieval Augmented Generation (RAG)",
        "Context-Aware AI Systems",
        "AI Assistant Development",
    ],

    "🧠 NLP & Intelligent Systems": [
        "Natural Language Processing",
        "Semantic Search",
        "Conversational AI",
        "Query Understanding & Context Retrieval",
    ],

    "🔎 Vector Search & Knowledge Systems": [
        "Vector Databases",
        "Document Embeddings",
        "Similarity Search",
        "Document Indexing Pipelines",
        "Knowledge Retrieval Systems",
    ],

    "🤗 AI Frameworks & Model Ecosystem": [
        "HuggingFace Transformers",
        "Embedding Models",
        "Groq LLM Inference",
        "Model Integration & Inference Optimization",
    ],

    "⚙️ Automation Engineering": [
        "Workflow Automation",
        "Process Optimization",
        "Task Automation Pipelines",
        "Automation Tools Development",
    ],

    "📊 Data Engineering & Pipelines": [
        "Data Processing Pipelines",
        "Document Ingestion Pipelines",
        "Batch Processing",
        "Performance Optimization",
        "Caching Strategies",
    ],

    "🏗️ AI System Design": [
        "AI Pipeline Architecture",
        "Modular System Design",
        "Scalable AI Application Design",
        "End-to-End ML System Development",
    ],

    "☁️ Cloud & Developer Tools": [
        "Azure (Data Environment)",
        "Git & Version Control",
        "Streamlit Application Development",
        "Web Scraping",
        "Regex Processing",
    ],

    "🧠 Computer Science Foundations": [
        "Data Structures & Algorithms",
        "Software Design Principles",
        "Problem Solving",
    ],
}

    # Create grid layout
    cols = st.columns(3)

    for idx, (category, skills) in enumerate(skill_categories.items()):
        with cols[idx % 3]:

            skill_list = "".join(
                f"<li style='margin-bottom:4px'>{skill}</li>" for skill in skills
            )

            st.markdown(
                f"""
                <div style="
                    padding:1.3rem;
                    border-radius:14px;
                    margin-bottom:1rem;
                    border:1px solid rgba(255,255,255,0.08);
                    box-shadow:0 6px 18px rgba(0,0,0,0.2);
                    min-height:220px;
                ">
                    <h4 style="margin-bottom:10px;">{category}</h4>
                    <ul style="padding-left:18px;">
                        {skill_list}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )