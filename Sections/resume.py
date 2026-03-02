import streamlit as st

# ---------- PAGE: RESUME ----------
def page_resume():

    st.markdown("""
    <style>

    /* Background */
    .stApp {
        background: linear-gradient(120deg,#f6f9ff,#eef1ff,#f8f5ff);
    }

    /* Header */
    .resume-main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg,#667eea,#764ba2,#6B8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .resume-subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 2rem;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 15px 40px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }

    /* Hero Section */
    .resume-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a1a;
    }

    .resume-role {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .resume-contact {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #333;
    }

    /* Skill Tags */
    .skill-tag {
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

    .skill-tag:hover {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
    }

    /* Download Button */
    .resume-download-btn {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none !important;
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white !important;
        transition: 0.3s;
    }

    .resume-download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102,126,234,0.3);
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="resume-main-title">📄 Resume</h1>', unsafe_allow_html=True)
    st.markdown('<p class="resume-subtitle">Professional profile and technical expertise</p>', unsafe_allow_html=True)

    # HERO PROFILE CARD
    st.markdown("""
    <div class="glass-card">
        <div class="resume-name">Chandrashekhar Robbi</div>
        <div class="resume-role">Analyst – Python Developer | Automation Engineer | Generative AI Specialist</div>
        <div class="resume-contact">
            📍 Thane, Maharashtra, India <br>
            📧 chandrashekarrobbi789@gmail.com <br>
            🔗 linkedin.com/in/chandrashekharrobbi <br>
            💻 github.com/ChandrashekharRobbi <br>
            ⏳ Notice Period: 30 Days
        </div>
        <br>
        <a href="data:text/plain;charset=utf-8,Your%20Resume%20Content"
           download="Chandrashekhar_Robbi_Resume.txt"
           class="resume-download-btn">
           📥 Download Resume
        </a>
    </div>
    """, unsafe_allow_html=True)

    # TWO COLUMN LAYOUT
    col1, col2 = st.columns([1.2, 1])

    # LEFT COLUMN (Experience + Summary)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">📝 Professional Summary</div>
            Analyst – Python Developer with 1 year 9 months of professional experience specializing in 
            Automation Engineering, Backend Development, and Generative AI applications.
            <br><br>
            Proven track record of reducing automation processing time by 50–90%, 
            building scalable data pipelines, and delivering production-grade backend systems 
            in data-driven environments.
            <br><br>
            B.E in Computer Science Engineering (AI & ML) with CGPI 9.63.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">💼 Experience</div>
            <b>eClerx — Analyst (Python Developer)</b><br>
            <small>June 2024 – Present | Domain: Data</small>
            <ul>
                <li>Reduced automation execution time by 50–90% through workflow redesign</li>
                <li>Built production-to-environment data sync pipeline reducing manual effort by 80–90%</li>
                <li>Optimized document processing using caching strategy (50–70% faster)</li>
                <li>Developed dynamic Chrome driver handling automation</li>
                <li>Built SQL multi-environment execution tool</li>
                <li>Designed deployment validation script to prevent configuration errors</li>
            </ul>
            <b>Qodeit — Python AI Intern</b><br>
            <small>3 Months</small>
            <ul>
                <li>Worked on AI-based backend systems</li>
                <li>Implemented FastAPI services</li>
                <li>Contributed to AI solution development and deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT COLUMN (Skills + Education + Achievements)
    with col2:

        skills = [
            "Python",
            "SQL",

            # AI / LLM / Agentic AI
            "Large Language Models (LLMs)",
            "Generative AI",
            "Retrieval Augmented Generation (RAG)",
            "Natural Language Processing",
            "Vector Search & Embeddings",
            "AI Assistant Development",
            "AI System Architecture",

            # ML / DL
            "Machine Learning",
            "Deep Learning",
            "Computer Vision",

            # Backend
            "FastAPI",
            "Flask",
            "API Integration",

            # AI Frameworks / Tools
            "HuggingFace Transformers",
            "Groq LLM Inference",

            # Data / Pipelines
            "Data Processing Pipelines",
            "Automation Engineering",

            # Cloud / Dev Tools
            "Azure (Data Environment)",
            "Streamlit",
            "Git"
        ]
        skill_html = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])

        st.markdown(f"""
        <div class="glass-card">
            <div class="section-title">🛠 Skills</div>
            {skill_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">🎓 Education</div>
            Bachelor of Engineering (B.E) — Computer Science Engineering (AI & ML)<br>
            Saraswati College of Engineering, Mumbai University<br>
            Graduation: 2024 | CGPI: 9.63
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">🏆 Achievements</div>
            • Super Achiever Award — 2025<br>
            • Appreciation Award — 2024<br>
            • Avishkar Competition — 3rd Rank<br>
            • GenMed Project Publication<br>
            • 510+ Students Taught (Since 2020)
        </div>
        """, unsafe_allow_html=True)