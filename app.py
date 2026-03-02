"""
AI Resume Portfolio - Multi-Page Application
Inspired by Nikita's Portfolio + RAG Pipeline
"""

import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from rag_pipeline import RAGPipeline
from Sections.experience import page_experience
from Sections.about_me import page_about
from Sections.skills import page_skills
from Sections.education import page_education
from Sections.projects import page_projects
from Sections.achievements import page_achievements
from Sections.resume import page_resume
from Sections.contact import page_contact
from Sections.architecture import page_architecture
from huggingface_hub import login
# from Sections.home import page_home

# Load environment variables
load_dotenv()

# ---------- HUGGINGFACE LOGIN (RUN ONLY ONCE) ----------
@st.cache_resource
def hf_login():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token)
    else:
        st.warning("HF_TOKEN not found in environment variables")

hf_login()
# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Chandrashekhar Robbi's Portfolio",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------- INITIALIZE RAG PIPELINE ----------
@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline with caching."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        st.info("""
        **Setup Instructions:**
        1. Create a `.env` file in the project root
        2. Add: `GROQ_API_KEY=your_api_key_here`
        3. Get free API key from: https://console.groq.com
        """)
        return None
    
    try:
        pipeline = RAGPipeline(groq_api_key)
        
        # Check if index exists locally
        index_path = ".vector_store"
        if os.path.exists(index_path):
            pipeline.load_index(index_path)
        else:
            with st.spinner("Building vector store (first time only)..."):
                pipeline.build_index("data")
                pipeline.save_index(index_path)
        
        return pipeline
    
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None



# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    choose = option_menu(
        "Chandrashekhar Robbi",
        [
            "AI Assistant",
            "About Me",
            "Experience",
            "Technical Skills",
            "Education",
            "Projects",
            "Achievements",
            "Resume",
            "Contact",
            "Architecture",
        ],
        icons=[
            "robot",
            "person-fill",
            "briefcase-fill",
            "tools",
            "book-fill",
            "kanban-fill",
            "trophy-fill",
            "file-earmark-person",
            "envelope-fill",
            "diagram-3-fill",
        ],
        menu_icon="mortarboard",
        default_index=0,
    )

    # ---------- SOCIAL ICONS ----------
    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <a href='https://linkedin.com/in/chandrashekharrobbi' target='_blank'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' width='35'>
            </a>
            &nbsp;&nbsp;
            <a href='https://github.com' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/128/25/25657.png' width='35'>
            </a>
            &nbsp;&nbsp;
            <a href='mailto:chandrashekarrobbi789@gmail.com'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png' width='35'>
            </a>
        </div>
    """, unsafe_allow_html=True)

    # ---------- FOOTER ----------
    st.markdown("""
        <hr style="margin-top: 25px; margin-bottom: 10px;">
        <div style='text-align: center; font-size: 12px; color: #888; line-height: 1.4;'>
            Made with ❤️<br>by <b style="color: #667eea;">Chandrashekhar Robbi</b><br>
            © 2026 All rights reserved.
        </div>
    """, unsafe_allow_html=True)


# ---------- LOAD PAGE CONTENT ----------
def load_content(section: str) -> str:
    """Load content from data files."""
    file_mapping = {
        "about": "about.txt",
        "experience": "experience.txt",
        "skills": "skills.txt",
        "education": "education.txt",
        "projects": "projects.txt",
        "achievements": "achievements.txt",
        "resume": "resume.txt",
        "contact": "contact.txt",
    }
    
    file_path = f"data/{file_mapping.get(section, 'about.txt')}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Content file not found: {file_path}"

# ---------- PAGE: AI ASSISTANT ----------
def page_ai_assistant():
    """Modern AI Chat page using native Streamlit chat components"""

    if pipeline is None:
        st.error("RAG Pipeline not initialized")
        return

    st.title("🤖 Chandu Intelligence")
    st.caption("Context-aware AI powered by RAG + Vector Search")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # --------- Helper Function ----------
    def process_user_query(query):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = pipeline.query(query, top_k=3)

                if response["status"] == "success":
                    answer = response["answer"]
                else:
                    answer = f"⚠️ Error: {response['answer']}"

                st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    # --------- Display Chat History ----------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------- Suggested Prompts ----------
    if not st.session_state.messages:
        st.markdown("### Try asking:")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👤 Who is Chandrashekhar?", width='stretch'):
                process_user_query("Tell me about Chandrashekhar Robbi")
                st.rerun()

            if st.button("🛠️ What are the main skills?", width='stretch'):
                process_user_query("What are your main skills?")
                st.rerun()

            if st.button("💼 Latest Experience", width='stretch'):
                process_user_query("Tell me about your latest role and experience")
                st.rerun()

        with col2:
            if st.button("🚀 Show Projects", width='stretch'):
                process_user_query("Tell me about your key projects")
                st.rerun()

            if st.button("📈 Biggest Achievement", width='stretch'):
                process_user_query("What are your biggest achievements?")
                st.rerun()

            if st.button("🤖 AI & ML Work", width='stretch'):
                process_user_query("Explain your AI and machine learning work")
                st.rerun()

        with col3:
            if st.button("⚡ Automation Work", width='stretch'):
                process_user_query("Tell me about your automation projects")
                st.rerun()

            if st.button("🏗️ System Architecture", width='stretch'):
                process_user_query("Explain your system architecture and technical design approach")
                st.rerun()

            if st.button("✨ Why Hire Chandrashekhar?", width='stretch'):
                process_user_query("Why should a company hire Chandrashekhar Robbi?")
                st.rerun()
    else:
        if st.button("Clear Chat History", width='stretch'):
            st.session_state.messages = []
            st.rerun()
    # --------- Chat Input ----------
    user_input = st.chat_input("Ask me anything about Chandrashekhar...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        process_user_query(user_input)
        st.rerun()


# ---------- HANDLE PAGE NAVIGATION ----------
if "navigate" in st.session_state:
    choose = st.session_state.navigate
    print(f"Navigating to: {choose}")
    del st.session_state.navigate

# Initialize pipeline
pipeline = initialize_rag_pipeline()


# ---------- PAGE ROUTING ----------
page_functions = {
    "AI Assistant": page_ai_assistant,
    "About Me": page_about,
    "Experience": page_experience,
    "Technical Skills": page_skills,
    "Education": page_education,
    "Projects": page_projects,
    "Achievements": page_achievements,
    "Resume": page_resume,
    "Contact": page_contact,
    "Architecture": page_architecture,
}

page_function = page_functions.get(choose, page_ai_assistant)
page_function()

