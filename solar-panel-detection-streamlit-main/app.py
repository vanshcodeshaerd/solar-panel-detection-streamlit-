# -*- coding: utf-8 -*-
"""
Main Entry Point for Solar Panel Detection App
Redirects to login page or main app based on authentication status
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth import is_authenticated, get_current_user

# Page configuration
st.set_page_config(
    page_title="Solar Panel Detection - Welcome",
    page_icon="☀️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for welcome page
st.markdown("""
<style>
    .welcome-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .welcome-subtitle {
        font-size: 1.4rem;
        color: #64748b;
        margin-bottom: 3rem;
        font-weight: 500;
        line-height: 1.5;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 35px -5px rgba(0, 0, 0, 0.15), 0 15px 15px -5px rgba(0, 0, 0, 0.08);
        border-left-color: #764ba2;
    }
    
    .feature-card h4 {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .feature-card p {
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
        font-weight: 400;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .btn-secondary {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .btn-secondary:hover {
        background: #e5e7eb;
        border-color: #9ca3af;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stHeader {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main entry point logic"""
    
    # Check if user is already authenticated
    if is_authenticated():
        user = get_current_user()
        
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">☀️ Welcome Back!</div>
            <div class="welcome-subtitle">Solar Panel Detection System</div>
        </div>
        """, unsafe_allow_html=True)
        
        # User info
        st.success(f"👤 Logged in as **{user['name']}** ({user['role'].title()})")
        st.info(f"📧 Email: {user['email']}")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Go to Dashboard", type="primary", use_container_width=True):
                st.switch_page("pages/streamlit_roboflow_7.py")
        
        with col2:
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                from auth import logout_user
                logout_user()
        
        # Features section
        st.markdown("---")
        st.markdown("### 🌟 System Features")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>📸 Image Detection</h4>
                <p>Upload images for advanced solar panel detection using cutting-edge AI models with high accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🗺️ Satellite Analysis</h4>
                <p>Analyze satellite imagery for comprehensive solar panel identification and mapping</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        # Show welcome page for non-authenticated users
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">☀️ Solar Panel Detection</div>
            <div class="welcome-subtitle">Advanced AI-Powered Solar Panel Identification System</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("### 🌟 Key Features")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🤖 AI Detection</h4>
                <p>State-of-the-art machine learning models for accurate solar panel detection with high precision and reliability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Real-time Analysis</h4>
                <p>Instant detection results with confidence scores and detailed analytics reports for comprehensive insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🗺️ Satellite Integration</h4>
                <p>Google Maps integration for location-based solar panel analysis with aerial imagery processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📱 Mobile Responsive</h4>
                <p>Access the system from any device with our fully responsive design and optimized user experience</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("---")
        st.markdown("### 🚀 Get Started")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Login to Access", type="primary", use_container_width=True):
                st.switch_page("pages/login.py")
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #bae6fd;">
                <h5 style="color: #0369a1; margin-bottom: 1rem; font-weight: 600;">🔑 Demo Credentials</h5>
                <div style="background: white; padding: 1rem; border-radius: 8px; font-family: 'Courier New', monospace;">
                    <div style="color: #0f172a; margin-bottom: 0.5rem;"><strong>Email:</strong> hemal@visioninfosoft.in</div>
                    <div style="color: #0f172a;"><strong>Password:</strong> Hemal@vision</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
        <p>© 2024 Solar Panel Detection System | Powered by AI & Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
