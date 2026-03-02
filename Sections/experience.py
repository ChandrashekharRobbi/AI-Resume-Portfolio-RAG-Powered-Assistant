import streamlit as st

# ---------- PAGE: EXPERIENCE ----------
def page_experience():

    st.markdown("""
    <style>

    /* Page background */
    .stApp {
        background: linear-gradient(120deg,#f6f9ff,#eef1ff,#f8f5ff);
    }

    /* Header */
    .exp-main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg,#667eea,#764ba2,#6B8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .exp-subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Timeline container */
    .timeline-container {
        position: relative;
        margin-top: 2rem;
        padding-left: 30px;
    }

    /* vertical line */
    .timeline-container:before {
        content: "";
        position: absolute;
        left: 8px;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(#667eea,#764ba2);
        border-radius: 5px;
    }

    /* Experience Card */
    .exp-item {
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

    /* timeline dot */
    .exp-item:before {
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

    .exp-item:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.15);
    }

    .exp-top-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .exp-job-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .exp-company-name {
        color: #667eea;
        font-weight: 600;
        margin-top: 3px;
    }

    .exp-date-badge {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        padding: 19px 14px;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .exp-description {
        color: #555;
        line-height: 1.8;
        margin-top: 1rem;
    }

    .exp-description li {
        margin-bottom: 0.6rem;
    }

    /* Tech section */
    .exp-tech-section {
        margin-top: 1.5rem;
    }

    .exp-tech-label {
        font-size: 0.9rem;
        color: #777;
        margin-bottom: 0.8rem;
    }

    /* Gradient tech badges */
    .exp-tech-item {
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

    .exp-tech-item:hover {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        cursor: default;
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="exp-main-title">💼 Professional Experience</h1>', unsafe_allow_html=True)
    st.markdown('<p class="exp-subtitle">My professional journey, impact, and growth</p>', unsafe_allow_html=True)

    experiences = [
    {
        "role": "Analyst – Python Developer",
        "company": "eClerx",
        "duration": "June 2024 – Present",
        "responsibilities": [
            "Redesigned data automation workflows, reducing processing time by 50–90% and significantly improving operational efficiency",
            "Built production-to-environment data synchronization pipelines to eliminate manual data handling",
            "Developed automated SQL execution tools across multiple environments, reducing manual configuration effort",
            "Implemented dynamic Chrome driver handling to resolve version mismatch issues and improve system reliability",
            "Designed automated error reporting systems including Production Disabled List and Production Error Reports",
            "Optimized bulk data processing using scheduled document downloads and database caching, reducing redundant operations",
            "Collaborated with cross-functional teams to resolve technical issues and improve development workflows"
        ],
        "tech": ["Python", "SQL", "Azure", "Automation", "Data Pipelines", "Backend Systems"]
    },

    {
        "role": "Python AI Intern",
        "company": "Qodeit",
        "duration": "3 Months",
        "responsibilities": [
            "Worked on AI-based solution development and backend implementation tasks",
            "Built backend services using Python and FastAPI for AI-driven applications",
            "Implemented machine learning features and optimized model performance using data preprocessing techniques",
            "Contributed to project delivery within deadlines while gaining practical AI development experience",
            "Gained hands-on experience in building production-ready backend and AI solutions"
        ],
        "tech": ["Python", "FastAPI", "Machine Learning", "AI Development", "Backend"]
    }
]

    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)

    for exp in experiences:
        resp_list = "".join([f"<li>{r}</li>" for r in exp["responsibilities"]])
        tech_badges = "".join([f'<span class="exp-tech-item">{t}</span>' for t in exp["tech"]])

        st.markdown(f"""
        <div class="exp-item">
            <div class="exp-top-row">
                <div>
                    <div class="exp-job-title">{exp['role']}</div>
                    <div class="exp-company-name">{exp['company']}</div>
                </div>
                <div class="exp-date-badge">{exp['duration']}</div>
            </div>
            <ul class="exp-description">{resp_list}</ul>
            <div class="exp-tech-section">
                <div class="exp-tech-label">🛠 Tech Stack</div>
                {tech_badges}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)