import streamlit as st
import time


def page_home():
    """Animated Hero Landing Page"""

    # ---------- HERO STYLING ----------
    st.markdown("""
    <style>

    .hero-container {
        text-align: center;
        padding-top: 80px;
        padding-bottom: 60px;
    }

    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(90deg,#6366f1,#9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        color: #94a3b8;
        margin-bottom: 40px;
    }

    .hero-tagline {
        font-size: 1.1rem;
        color: #64748b;
        max-width: 700px;
        margin: auto;
        margin-bottom: 50px;
        line-height: 1.7;
    }

    .feature-box {
        padding: 20px;
        border-radius: 14px;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }

    </style>
    """, unsafe_allow_html=True)

    # ---------- HERO ----------
    st.markdown("<div class='hero-container'>", unsafe_allow_html=True)

    # Animated typing effect
    title_placeholder = st.empty()

    title = "Chandrashekhar Robbi"

    typed = ""
    for char in title:
        typed += char
        title_placeholder.markdown(
            f"<div class='hero-title'>{typed}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.03)

    st.markdown(
        "<div class='hero-subtitle'>AI Engineer • Python Developer • Automation Specialist</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='hero-tagline'>Building intelligent systems, scalable automation pipelines, and production-ready AI solutions.
    Explore my work, projects, and talk with my AI assistant to learn more.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- CTA BUTTONS ----------
    st.markdown("###")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🤖 Talk to AI Assistant", width='stretch'):
            st.session_state.navigate = "AI Assistant"
            st.rerun()

    with col2:
        if st.button("📁 Explore Projects", width='stretch'):
            st.session_state.navigate = "Projects"
            st.rerun()

    with col3:
        if st.button("📄 View Resume", width='stretch'):
            st.session_state.navigate = "Resume"
            st.rerun()

    st.markdown("---")

    # ---------- FEATURE HIGHLIGHTS ----------
    st.subheader("What I Build")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='feature-box'>
        🤖 <b>AI Systems</b><br>
        RAG pipelines, ML models, intelligent automation
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='feature-box'>
        ⚡ <b>Automation</b><br>
        Scalable workflows & performance optimization
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='feature-box'>
        🧠 <b>Backend Engineering</b><br>
        Production-ready Python systems
        </div>
        """, unsafe_allow_html=True)