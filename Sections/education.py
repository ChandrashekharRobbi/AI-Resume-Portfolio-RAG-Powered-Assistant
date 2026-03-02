import streamlit as st

# ---------- PAGE: EDUCATION ----------
def page_education():

    st.markdown("""
    <style>

    /* Page background */
    .stApp {
        background: linear-gradient(120deg,#f6f9ff,#eef1ff,#f8f5ff);
    }

    /* Header */
    .edu-main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg,#667eea,#764ba2,#6B8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .edu-subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Timeline container */
    .edu-timeline {
        position: relative;
        margin-top: 2rem;
        padding-left: 30px;
    }

    /* Vertical line */
    .edu-timeline:before {
        content: "";
        position: absolute;
        left: 8px;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(#667eea,#764ba2);
        border-radius: 5px;
    }

    /* Education Card */
    .edu-item {
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

    /* Timeline dot */
    .edu-item:before {
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

    .edu-item:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.15);
    }

    .edu-top-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .edu-degree {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .edu-institute {
        color: #667eea;
        font-weight: 600;
        margin-top: 3px;
    }

    .edu-date-badge {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        padding: 16px 18px;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .edu-field {
        color: #555;
        line-height: 1.8;
        margin-top: 1rem;
        font-size: 1rem;
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="edu-main-title">🎓 Education</h1>', unsafe_allow_html=True)
    st.markdown('<p class="edu-subtitle">My academic journey and foundational growth</p>', unsafe_allow_html=True)

    education_data = [
        {
            "institute": "Saraswati College of Engineering, Mumbai University",
            "degree": "Bachelor of Engineering (B.E)",
            "field": "Computer Science Engineering (Artificial Intelligence & Machine Learning) • CGPI: 9.63",
            "duration": "2020 – 2024"
        },
        {
            "institute": "K.J Somaiya College of Science and Commerce",
            "degree": "Higher Secondary Certificate (HSC)",
            "field": "Science Stream",
            "duration": "2018 – 2020"
        },
        {
            "institute": "Mithila English High School",
            "degree": "Secondary School Certificate (SSC)",
            "field": "General Education",
            "duration": "2005 – 2018"
        }
    ]

    st.markdown('<div class="edu-timeline">', unsafe_allow_html=True)

    for edu in education_data:
        st.markdown(f"""
        <div class="edu-item">
            <div class="edu-top-row">
                <div>
                    <div class="edu-degree">{edu['degree']}</div>
                    <div class="edu-institute">{edu['institute']}</div>
                </div>
                <div class="edu-date-badge">{edu['duration']}</div>
            </div>
            <div class="edu-field">
                📘 <strong>Field:</strong> {edu['field']}
                {"<br>⭐ Consistent high academic performance with strong foundation in AI/ML, Data Structures, and Software Engineering" if "Engineering" in edu['degree'] else ""}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)