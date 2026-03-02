import streamlit as st

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def page_about():
    """About Me page with modern design and improved layout."""
    img_base64 = get_base64_image("img/About_Me.jpg")
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        .about-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem 0;
        }
        
        .profile-section {
            display: flex;
            gap: 3rem;
            align-items: flex-start;
            margin-bottom: 3rem;
        }
        
        .profile-image-wrapper {
            flex-shrink: 0;
            display: flex;
            justify-content: center;
        }
        
        .profile-image {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 200px;
            height: 200px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 100px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }
        
        .profile-image:hover {
            transform: translateY(-5px);
        }
        
        .profile-content {
            flex: 1;
            padding-top: 1rem;
        }
        
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .description-text {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #444;
            margin-bottom: 1.5rem;
        }
        
        .achievements-card {
            background: linear-gradient(135deg, #f5f7ff 0%, #f0f2ff 100%);
            border-left: 4px solid #667eea;
            padding: 2rem;
            border-radius: 12px;
            margin: 2rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        .achievements-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 1.5rem;
        }
        
        .achievements-list {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        
        .achievement-item {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .achievement-icon {
            font-size: 1.6rem;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        
        .achievement-text {
            font-size: 1rem;
            color: #555;
            line-height: 1.5;
        }
        
        .divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 2rem 0;
        }
        
        .interests-text {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #444;
            margin: 2rem 0;
            padding: 1.5rem;
            background: linear-gradient(135deg, #f5f7ff 0%, #f0f2ff 100%);
            border-radius: 8px;
            border-left: 4px solid #764ba2;
        }
        
        .social-links {
            display: flex;
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .social-link {
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        
        .social-link:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.35);
        }
        
        .social-link img {
            width: 24px;
            height: 24px;
            filter: brightness(0) invert(1);
        }
        
        @media (max-width: 768px) {
            .profile-section {
                flex-direction: column;
                gap: 2rem;
            }
            
            .profile-image {
                width: 150px;
                height: 150px;
                font-size: 70px;
            }
            
            .achievements-list {
                grid-template-columns: 1fr;
            }
            
            .section-title {
                font-size: 2rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="about-container">', unsafe_allow_html=True)
    
    # Profile section with image and intro
    st.markdown(f"""
    <div class="profile-section">
        <div class="profile-image-wrapper">
            <div class="profile-image">
                <img src="data:image/jpeg;base64,{img_base64}" 
                    style="width:100%; height:100%; object-fit:cover; border-radius:16px;">
            </div>
        </div>
        <div class="profile-content">
            <div class="section-title">About Me</div>
                <p class="description-text">
                    Hello! I'm <strong>Chandrashekhar Robbi</strong>, a Backend and Automation-focused Python Developer specializing in 
                    <strong>Python, Generative AI, and scalable backend systems</strong>. I design data-driven solutions that reduce manual effort, 
                    optimize workflows, and deliver measurable operational impact.
                </p>
                <p class="description-text">
                    Currently working as an <strong>Analyst – Python Developer at eClerx</strong>, I build automation pipelines, 
                    production-ready backend solutions, and AI-powered systems that improve efficiency and reliability. 
                    I hold a <strong>Bachelor's degree in Computer Science Engineering (AI & ML)</strong> from Mumbai University 
                    with a CGPI of <strong>9.63</strong>, and I focus on solving real-world problems through structured engineering and continuous learning.
                </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Achievements section
    st.markdown("""
    <div class="achievements-card">
        <div class="achievements-title">🎯 Key Achievements</div>
        <div class="achievements-list">
            <div class="achievement-item">
                <span class="achievement-icon">⚡</span>
                <div class="achievement-text">
                    <strong>50–90% Automation Optimization</strong><br>
                    Reduced data processing time by redesigning production workflows at eClerx
                </div>
            </div>
            <div class="achievement-item">
                <span class="achievement-icon">🔄</span>
                <div class="achievement-text">
                    <strong>Production Data Synchronization Systems</strong><br>
                    Built bulk data pipelines and automated environment sync solutions
                </div>
            </div>
            <div class="achievement-item">
                <span class="achievement-icon">🤖</span>
                <div class="achievement-text">
                    <strong>AI Solution Development</strong><br>
                    Built AI applications including disease classification and automation tools
                </div>
            </div>
            <div class="achievement-item">
                <span class="achievement-icon">🏆</span>
                <div class="achievement-text">
                    <strong>Industry Recognition</strong><br>
                    Received Appreciation Award (2024) and Super Achiever Award (2025)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Interests section
    st.markdown("""
    <div class="interests-text">
        <strong>Beyond Code:</strong> I enjoy exploring emerging AI technologies, building automation tools, 
        and continuously learning new advancements in backend engineering and Generative AI. 
        I am also passionate about teaching and knowledge sharing, which has strengthened my communication 
        skills and ability to explain complex technical concepts clearly.
    </div>
    """, unsafe_allow_html=True)
    
    # Closing statement
    st.markdown("""
    <p style="font-size: 1.1rem; color: #666; margin-top: 2rem; font-style: italic;">
        Thank you for visiting my portfolio. Let's connect and build something impactful together! 🚀
    </p>
    """, unsafe_allow_html=True)
    
    link_img_base64 = get_base64_image("img/linkedin.png")
    email_img_base64 = get_base64_image("img/email.png")
    # Social links
    st.markdown(f"""
    <div style="margin-top: 3rem;">
        <p style="font-size: 1rem; color: #888; margin-bottom: 1.5rem;">Connect with me:</p>
        <div class="social-links">
            <a href='https://linkedin.com/in/chandrashekharrobbi' target='_blank' class='social-link' title='LinkedIn'>
                <img src="data:image/jpeg;base64,{link_img_base64}">
            </a>
            <a href='https://github.com' target='_blank' class='social-link' title='GitHub'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/2/24/Github_logo_svg.svg' alt='GitHub'>
            </a>
            <a href='mailto:chandrashekarrobbi789@gmail.com' class='social-link' title='Email'>
                <img src="data:image/jpeg;base64,{email_img_base64}">
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)