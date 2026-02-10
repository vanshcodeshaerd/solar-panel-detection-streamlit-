# -*- coding: utf-8 -*-
"""
Authentication module for Solar Panel Detection App
Handles user login, session management, and route protection
"""

import streamlit as st
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Predefined user dataset
USERS_DATA = [
    {
        "email": "hemal@visioninfosoft.in",
        "password": "Hemal@vision",
        "role": "admin",
        "name": "Hemal Administrator"
    },
    {
        "email": "admin@example.com",
        "password": "Admin@123",
        "role": "admin",
        "name": "System Administrator"
    },
    {
        "email": "user@example.com", 
        "password": "User@123",
        "role": "user",
        "name": "Demo User"
    },
    {
        "email": "student@example.com",
        "password": "Student@123", 
        "role": "user",
        "name": "Student User"
    },
    {
        "email": "researcher@example.com",
        "password": "Research@123",
        "role": "user", 
        "name": "Research User"
    }
]

def hash_password(password: str) -> str:
    """Hash password using SHA-256 for basic security"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email: str, password: str) -> Optional[Dict]:
    """Verify user credentials against predefined dataset"""
    for user in USERS_DATA:
        if user["email"].lower() == email.lower():
            # In production, use: hash_password(password) == user["hashed_password"]
            if user["password"] == password:
                return {
                    "email": user["email"],
                    "role": user["role"], 
                    "name": user["name"],
                    "login_time": datetime.now().isoformat()
                }
    return None

def is_authenticated() -> bool:
    """Check if user is currently authenticated"""
    return st.session_state.get("authenticated", False)

def get_current_user() -> Optional[Dict]:
    """Get current authenticated user info"""
    if is_authenticated():
        return st.session_state.get("user")
    return None

def login_user(email: str, password: str) -> bool:
    """Authenticate user and create session"""
    user = verify_user(email, password)
    if user:
        st.session_state.update({
            "authenticated": True,
            "user": user,
            "login_time": datetime.now()
        })
        return True
    return False

def logout_user():
    """Clear user session and logout"""
    keys_to_clear = ["authenticated", "user", "login_time"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def require_auth():
    """Route protection - redirect to login if not authenticated"""
    if not is_authenticated():
        st.session_state["redirect_after_login"] = "main_app"
        st.switch_page("pages/login.py")

def get_all_users() -> List[Dict]:
    """Get all users (for admin purposes)"""
    return [{"email": user["email"], "role": user["role"], "name": user["name"]} 
            for user in USERS_DATA]

def check_session_timeout() -> bool:
    """Check if session has timed out (24 hours)"""
    if "login_time" in st.session_state:
        login_time = st.session_state["login_time"]
        if isinstance(login_time, str):
            login_time = datetime.fromisoformat(login_time)
        if datetime.now() - login_time > timedelta(hours=24):
            logout_user()
            return False
    return True
