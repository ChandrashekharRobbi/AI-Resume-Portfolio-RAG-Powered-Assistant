"""
UI Components Module - Reusable Streamlit components.

Provides:
- Sidebar navigation
- Chat interface components
- Profile sections
- Loading indicators
- Styled containers
"""

import streamlit as st
from typing import List, Dict, Callable


def init_page_config():
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Resume | Portfolio",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling for dark theme."""
    st.markdown("""
    <style>
    /* Dark theme styling */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #4caf50;
        --error: #f44336;
        --warning: #ff9800;
        --dark-bg: #0e1419;
        --light-text: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        padding: 20px 15px;
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0 0 10px 0;
        font-size: 2.5em;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        word-wrap: break-word;
    }
    
    .user-message {
        background-color: #1e3a5f;
        border-left: 4px solid #2196F3;
        margin-left: 20px;
    }
    
    .assistant-message {
        background-color: #2d1b4e;
        border-left: 4px solid #9c27b0;
        margin-right: 20px;
    }
    
    .user-message strong {
        color: #64B5F6;
    }
    
    .assistant-message strong {
        color: #CE93D8;
    }
    
    /* Containers */
    .profile-section {
        background-color: #1a1f2e;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #667eea;
    }
    
    .profile-section h2 {
        margin-top: 0;
        color: #64B5F6;
    }
    
    .info-card {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #667eea;
    }
    
    /* Messages */
    .error-message {
        background-color: #3d1a1a;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #ffcdd2;
    }
    
    .success-message {
        background-color: #1a3d2a;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #c8e6c9;
    }
    
    .info-message {
        background-color: #1a2d3d;
        border-left: 4px solid #2196F3;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #b3e5fc;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Links */
    a {
        color: #64B5F6;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Divider */
    hr {
        border-color: #667eea;
        opacity: 0.3;
    }
    
    /* Section title */
    .section-title {
        color: #64B5F6;
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #999;
        font-size: 0.9em;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #667eea;
        opacity: 0.7;
    }
    
    </style>
    """, unsafe_allow_html=True)


def display_header(name: str = "Alex", title: str = "AI Resume Assistant"):
    """Display main header with gradient background."""
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 {title}</h1>
        <p>Chat with my AI • Explore my profile • Learn about my skills</p>
    </div>
    """, unsafe_allow_html=True)


def display_section_title(title: str, icon: str = ""):
    """Display styled section title."""
    st.markdown(f"""
    <div class="section-title">{icon} {title}</div>
    """, unsafe_allow_html=True)


def display_info_card(title: str, content: str, icon: str = "ℹ️"):
    """Display styled info card."""
    st.markdown(f"""
    <div class="info-card">
        <strong>{icon} {title}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)


def display_error_message(message: str):
    """Display styled error message."""
    st.markdown(
        f'<div class="error-message"><strong>❌ Error:</strong> {message}</div>',
        unsafe_allow_html=True
    )


def display_success_message(message: str):
    """Display styled success message."""
    st.markdown(
        f'<div class="success-message"><strong>✅ Success:</strong> {message}</div>',
        unsafe_allow_html=True
    )


def display_info_message(message: str):
    """Display styled info message."""
    st.markdown(
        f'<div class="info-message"><strong>ℹ️ Info:</strong> {message}</div>',
        unsafe_allow_html=True
    )


def display_chat_message(role: str, content: str):
    """Display styled chat message."""
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong> {content}</div>',
            unsafe_allow_html=True
        )


def display_chat_history(messages: List[Dict]):
    """Display entire chat history."""
    for message in messages:
        display_chat_message(message["role"], message["content"])


def create_sidebar_menu(sections: Dict[str, Dict]) -> str:
    """
    Create sidebar navigation menu.
    
    Args:
        sections: Dictionary of {section_id: {title, icon, description}}
        
    Returns:
        Selected section ID
    """
    with st.sidebar:
        st.markdown("### 📍 Navigation")
        st.markdown("---")
        
        # Create buttons for each section
        selected_section = None
        for section_id, section_info in sections.items():
            if st.button(
                f"{section_info['icon']} {section_info['title']}",
                width='stretch',
                key=f"nav_{section_id}"
            ):
                selected_section = section_id
                st.session_state.current_page = section_id
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("LinkedIn", "https://linkedin.com", width='stretch')
        with col2:
            st.link_button("GitHub", "https://github.com", width='stretch')
        with col3:
            st.link_button("Email", "mailto:contact@example.com", width='stretch')
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        show_sources = st.checkbox("Show context sources", value=False)
        theme = st.radio("Theme", ["Dark", "Light"], index=0)
        
        return show_sources, theme
    
    return False, "Dark"


def display_suggested_questions(questions: List[str], on_click: Callable):
    """
    Display suggested question buttons.
    
    Args:
        questions: List of suggested questions
        on_click: Callback function when question is clicked
    """
    st.markdown("### 💡 Suggested Questions")
    
    cols = st.columns(2)
    for i, question in enumerate(questions):
        col = cols[i % 2]
        with col:
            if st.button(
                f"❓ {question}",
                key=f"suggestion_{i}",
                width='stretch'
            ):
                on_click(question)


def display_profile_card(name: str, title: str, bio: str):
    """Display profile card in sidebar."""
    st.markdown(f"""
    <div class="profile-section">
        <h3>👤 Profile</h3>
        <strong>{name}</strong><br>
        <em>{title}</em><br><br>
        {bio}
    </div>
    """, unsafe_allow_html=True)


def display_skills_overview(skills: Dict[str, List[str]]):
    """Display skills in columns."""
    cols = st.columns(len(skills))
    
    for col, (category, skill_list) in zip(cols, skills.items()):
        with col:
            st.markdown(f"**{category}**")
            for skill in skill_list:
                st.markdown(f"- {skill}")


def display_footer():
    """Display footer section."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔗 Connect**")
        st.markdown("[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)")
    
    with col2:
        st.markdown("**📧 Contact**")
        st.markdown("Email: contact@example.com")
    
    with col3:
        st.markdown("**🛠️ Stack**")
        st.markdown("LangChain • Groq • FAISS • Streamlit")


def display_chat_input_area() -> str:
    """
    Display chat input area.
    
    Returns:
        User input text
    """
    col_input, col_button = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Your question:",
            placeholder="Ask about my skills, projects, experience...",
            label_visibility="collapsed"
        )
    
    with col_button:
        send_button = st.button("Send", width='stretch', key="send_btn")
    
    return user_input, send_button


def display_loading_spinner(message: str = "Processing..."):
    """Display loading spinner."""
    with st.spinner(message):
        pass


def display_statistics(stats: Dict[str, any]):
    """Display chat statistics."""
    st.markdown("### 📊 Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Messages", stats.get("total_messages", 0))
    
    with col2:
        st.metric("Avg Response", f"{stats.get('avg_tokens', 0):.0f}ms")
    
    with col3:
        st.metric("Tokens Used", stats.get("total_tokens", 0))


def display_empty_state():
    """Display empty chat state."""
    st.markdown("""
    <div class="info-card">
        <h3>👋 Welcome!</h3>
        <p>Start a conversation by asking me about my:</p>
        <ul>
            <li>Professional experience and skills</li>
            <li>Notable projects and achievements</li>
            <li>Education and certifications</li>
            <li>Technical expertise and technologies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
