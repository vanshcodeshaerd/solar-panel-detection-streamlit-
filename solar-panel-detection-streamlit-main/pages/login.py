# -*- coding: utf-8 -*-
"""
Login Page for Solar Panel Detection App
Modern, professional UI with authentication functionality
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import auth module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth import login_user, is_authenticated, get_current_user

# Page configuration
st.set_page_config(
    page_title="Login - Solar Panel Detection",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Redirect if already authenticated
if is_authenticated():
    user = get_current_user()
    st.success(f"✅ Already logged in as {user['name']}")
    if st.button("Go to Dashboard", type="primary"):
        st.switch_page("pages/streamlit_roboflow_7.py")
    st.stop()

# Custom CSS for modern login design
st.markdown("""
<style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        color: #6b7280;
        font-size: 1rem;
    }
    
    .login-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
    }
    
    .form-group {
        margin-bottom: 1.5rem;
    }
    
    .form-label {
        display: block;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .input-field {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 0.875rem;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    
    .input-field:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .login-btn {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .login-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .error-message {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
        padding: 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #16a34a;
        padding: 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    
    .demo-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    
    .demo-title {
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    
    .demo-credentials {
        font-size: 0.875rem;
        color: #64748b;
        line-height: 1.5;
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

# Login form
def show_login_form():
    """Display the login form"""
    
    # Header
    st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <div class="login-title">🔐 Welcome Back</div>
            <div class="login-subtitle">Solar Panel Detection System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login card
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        # Email field
        email = st.text_input(
            "📧 Email Address",
            placeholder="Enter your email",
            key="login_email",
            help="Enter your registered email address"
        )
        
        # Password field with show/hide toggle
        col1, col2 = st.columns([4, 1])
        with col1:
            password = st.text_input(
                "🔒 Password",
                placeholder="Enter your password",
                type="password",
                key="login_password",
                help="Enter your password"
            )
        with col2:
            show_password = st.checkbox("Show", key="show_password")
            if show_password:
                st.session_state.login_password = st.session_state.get("login_password", "")
        
        # Login button with loading state
        if st.button("🚀 Sign In", type="primary", use_container_width=True):
            if not email or not password:
                st.error("⚠️ Please enter both email and password")
                return
            
            # Show loading state
            with st.spinner("🔐 Authenticating..."):
                # Attempt login
                if login_user(email, password):
                    st.success("✅ Login successful! Redirecting...")
                    # Check if there's a redirect target
                    redirect_target = st.session_state.get("redirect_after_login", "pages/streamlit_roboflow_7.py")
                    if "redirect_after_login" in st.session_state:
                        del st.session_state["redirect_after_login"]
                    st.switch_page(redirect_target)
                else:
                    st.error("❌ Invalid email or password. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo credentials section
        st.markdown("""
        <div class="demo-info">
            <div class="demo-title">📋 Demo Credentials</div>
            <div class="demo-credentials">
                <strong>Admin User:</strong><br>
                Email: hemal@visioninfosoft.in<br>
                Password: Hemal@vision<br><br>
                
                <strong>Test User:</strong><br>
                Email: user@example.com<br>
                Password: User@123
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main login page
def main():
    """Main login page logic"""
    
    # Check session timeout
    if is_authenticated():
        from auth import check_session_timeout
        check_session_timeout()
    
    # Show login form
    show_login_form()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #6b7280; font-size: 0.875rem;">
        <p>© 2024 Solar Panel Detection System | Secure Authentication</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
