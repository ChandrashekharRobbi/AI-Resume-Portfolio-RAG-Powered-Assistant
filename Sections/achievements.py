import streamlit as st

# ---------- PAGE: ACHIEVEMENTS ----------
def page_achievements():

    st.markdown("""
    <style>

    /* Background */
    .stApp {
        background: linear-gradient(120deg,#f6f9ff,#eef1ff,#f8f5ff);
    }

    /* Header */
    .ach-main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg,#667eea,#764ba2,#6B8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .ach-subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    .ach-metric-card {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: 0.3s;
        margin-bottom: 1.5rem;
    }

    .ach-metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.15);
    }

    .ach-metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg,#667eea,#764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .ach-metric-label {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.5rem;
    }

    /* Achievement List Card */
    .ach-list-card {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }

    .ach-list-card li {
        margin-bottom: 0.8rem;
        color: #555;
        line-height: 1.7;
    }

    /* Certification Badges */
    .cert-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 6px;
        background: linear-gradient(white,white) padding-box,
                    linear-gradient(135deg,#667eea,#764ba2) border-box;
        border: 1.5px solid transparent;
        color: #555;
        transition: 0.3s;
    }

    .cert-badge:hover {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        transform: translateY(-3px);
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="ach-main-title">⭐ Achievements & Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="ach-subtitle">Impact-driven results and continuous professional growth</p>', unsafe_allow_html=True)

    # ---------- Highlight Metrics ----------
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("9.63", "Engineering CGPI"),
        ("50–90%", "Automation Efficiency Improvement"),
        ("510+", "Students Taught"),
        ("2", "Professional Awards")
    ]

    for col, (value, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="ach-metric-card">
                <div class="ach-metric-value">{value}</div>
                <div class="ach-metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---------- Detailed Achievements ----------
    st.markdown("""
    <div class="ach-list-card">
        <ul>
            <li>🏆 Received <b>Super Achiever Award (2025)</b> for automation improvements and technical contribution</li>
            <li>⭐ Received <b>Appreciation Award (2024)</b> for strong performance and project delivery</li>
            <li>🎓 Achieved <b>CGPI 9.63</b> in B.E Computer Science (AI & ML) from Mumbai University</li>
            <li>🥉 Secured <b>3rd Rank in Avishkar</b> technical competition demonstrating innovation and problem-solving</li>
            <li>📘 Published AI-based project <b>GenMed</b> for medicine information and healthcare assistance</li>
            <li>👨‍🏫 Teaching experience since 2020 with <b>510+ students</b>, strengthening communication and leadership skills</li>
            <li>🚀 Delivered automation solutions reducing processing time by up to 90%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Certifications ----------
    st.subheader("📜 Certificate / Certifications")

    # Certification data (name + url)
    certifications = [
        ("Python Certification — IIT Bombay",
        "https://drive.google.com/file/d/134twDe74tBvKGxOsBcVCq4d_6l6Vac0Z/view"),

        ("Generative AI Certification — University of Munich",
        "https://drive.google.com/file/d/1lfAG-FzguP-dnWDBC5GSf0oXj6wmOs4H/view"),

        ("Django Web Framework — Coursera",
        "https://www.coursera.org/account/accomplishments/verify/AWL8VWXVW3NL"),

        ("SQL for Data Science — Coursera",
        "https://www.coursera.org/account/accomplishments/verify/DFBJ7Q72EA85"),

        ("Programming in Python — Coursera",
        "https://www.coursera.org/account/accomplishments/verify/8JHY4RP82TVA"),

        ("Machine Learning & Data Science — GeeksforGeeks",
        "https://media.geeksforgeeks.org/courses/certificates/f094d8c5d96e524e893895067a0cea02.pdf"),

        ("Introduction to MCP Protocol — Anthropic",
        "https://verify.skilljar.com/c/eepxfbvedypg"),

        ("AI Engineer Agentic Track — Udemy",
        "https://drive.google.com/file/d/1XDaC1Q2pf3P4dLTa8IFpZ3snFfrFvAco/view"),
    ]

    # Generate clickable badges
    cert_html = "".join([
        f"""
        <a href="{url}" target="_blank" style="text-decoration:none;">
            <span class="cert-badge">🎓 {name}</span>
        </a>
        """
        for name, url in certifications
    ])

    st.markdown(cert_html, unsafe_allow_html=True)