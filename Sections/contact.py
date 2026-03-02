import streamlit as st

# ---------- PAGE: CONTACT ----------
def page_contact():

    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(120deg,#f6f9ff,#eef1ff,#f8f5ff);
    }

    .contact-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg,#667eea,#764ba2,#6B8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .contact-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    .contact-btn {
        display: inline-block;
        padding: 12px 26px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none !important;
        margin: 8px;
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white !important;
        transition: 0.3s;
    }

    .contact-btn:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(102,126,234,0.3);
    }

    .form-container {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 2.5rem;
        max-width: 750px;
        margin: auto;
        box-shadow: 0 15px 40px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    # HERO SECTION
    st.markdown('<div class="contact-title">Let’s Build Scalable Systems Together</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="contact-subtitle">
    Analyst – Python Developer | Open to Backend, Automation & Generative AI Opportunities <br>
    📍 Thane, Maharashtra, India | ⏳ Notice Period: 30 Days
    </div>
    ''',
            unsafe_allow_html=True
        )

    # CONTACT BUTTONS (Centered)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align:center;">
            <a href="mailto:chandrashekarrobbi789@gmail.com" class="contact-btn">
                📧 Email Me
            </a>
            <a href="https://linkedin.com/in/chandrashekharrobbi" target="_blank" class="contact-btn">
                🔗 LinkedIn
            </a>
            <a href="https://github.com/ChandrashekharRobbi" target="_blank" class="contact-btn">
                💻 GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <style>

    /* Style Streamlit form directly */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 2.5rem;
        max-width: 750px;
        margin: auto;
        box-shadow: 0 15px 40px rgba(0,0,0,0.08);
        border: none;
    }

    </style>
    """, unsafe_allow_html=True)
# ---------- CONTACT FORM (FORM SUBMIT) ----------
    st.markdown("""
    <div class="form-container">
    <form action="https://formsubmit.co/081368f628eda88ba6f25f09e1ac02e7" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="hidden" name="_subject" value="New Portfolio Contact Message">
        <input 
            type="text" 
            name="name" 
            placeholder="Your name" 
            required
            style="width:100%;padding:12px;margin-bottom:15px;border-radius:10px;border:1px solid #ddd;"
        >
        <input 
            type="email" 
            name="email" 
            placeholder="Your email" 
            required
            style="width:100%;padding:12px;margin-bottom:15px;border-radius:10px;border:1px solid #ddd;"
        >
        <textarea 
            name="message" 
            placeholder="Your message here"
            required
            style="width:100%;padding:12px;height:150px;margin-bottom:15px;border-radius:10px;border:1px solid #ddd;"
        ></textarea>
        <button 
            type="submit"
            style="
                width:100%;
                padding:14px;
                border:none;
                border-radius:30px;
                font-weight:600;
                background:linear-gradient(135deg,#667eea,#764ba2);
                color:white;
                cursor:pointer;
            "
        >
            Send Message
        </button>

    </form>

    </div>
    """, unsafe_allow_html=True)