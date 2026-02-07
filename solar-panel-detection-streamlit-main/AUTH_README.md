# 🔐 Authentication System Documentation

## Overview
The Solar Panel Detection System now includes a secure, modern authentication system with user login, session management, and role-based access control.

## 🚀 Quick Start

### 1. Run the Application
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the main application
python -m streamlit run app.py
```

### 2. Access Points
- **Main Entry**: http://localhost:8502 (or your assigned port)
- **Login Page**: Available from main entry or directly at `/pages/login.py`
- **Main App**: Protected - requires authentication

## 👥 User Accounts

### Predefined Users
| Email | Password | Role | Name |
|-------|----------|------|------|
| `hemal@visioninfosoft.in` | `Hemal@vision` | Admin | Hemal Administrator |
| `admin@example.com` | `Admin@123` | Admin | System Administrator |
| `user@example.com` | `User@123` | User | Demo User |
| `student@example.com` | `Student@123` | User | Student User |
| `researcher@example.com` | `Research@123` | User | Research User |

### Role Types
- **Admin**: Full access to all features
- **User**: Standard access to detection features

## 🔧 Features

### ✅ Authentication Features
- **Secure Login**: Email/password authentication with validation
- **Session Management**: 24-hour session timeout
- **Route Protection**: Main app requires authentication
- **User Info Display**: Shows current user details in sidebar
- **Logout Functionality**: Secure session termination

### 🎨 UI/UX Features
- **Modern Design**: Professional, card-based layout
- **Responsive Design**: Mobile-friendly interface
- **Loading States**: Visual feedback during authentication
- **Error Handling**: Clear error messages for invalid credentials
- **Password Toggle**: Show/hide password option
- **Smooth Transitions**: Hover effects and animations

### 🛡️ Security Features
- **Session Timeout**: Automatic logout after 24 hours
- **Route Protection**: Prevents unauthorized access
- **Input Validation**: Email and password validation
- **Session State Management**: Secure session handling

## 📁 File Structure

```
solar-panel-detection-streamlit-main/
├── app.py                    # Main entry point with authentication
├── auth.py                   # Authentication module
├── pages/
│   └── login.py             # Login page UI
├── streamlit_roboflow_7.py  # Main app (now protected)
└── AUTH_README.md           # This documentation
```

## 🔌 Integration Details

### Authentication Flow
1. **Entry Point**: `app.py` checks authentication status
2. **Not Authenticated**: Redirects to login page
3. **Login Process**: Validates credentials against user dataset
4. **Session Creation**: Stores user info in session state
5. **Access Granted**: Redirects to main application
6. **Session Management**: Maintains login state across pages

### Key Functions

#### `auth.py` Module
- `verify_user(email, password)`: Validate credentials
- `login_user(email, password)`: Create user session
- `logout_user()`: Clear session data
- `is_authenticated()`: Check login status
- `get_current_user()`: Get current user info
- `require_auth()`: Route protection
- `check_session_timeout()`: Session timeout management

#### Session State Variables
- `authenticated`: Boolean login status
- `user`: User information dictionary
- `login_time`: Session timestamp
- `redirect_after_login`: Post-login redirect target

## 🎯 How to Use

### For Users
1. Open the application URL
2. Click "Login to Access" or "Go to Login"
3. Enter your email and password
4. Click "Sign In"
5. Access the main dashboard
6. Use "Logout" when finished

### For Developers
1. **Add New Users**: Update `USERS_DATA` in `auth.py`
2. **Modify Roles**: Extend role system in authentication logic
3. **Customize UI**: Modify CSS in login page and main app
4. **Add Routes**: Use `require_auth()` for new protected pages

## 🔧 Configuration

### Environment Variables
No additional environment variables required for authentication.

### Customization Options
- **Session Timeout**: Modify `timedelta(hours=24)` in `check_session_timeout()`
- **User Dataset**: Update `USERS_DATA` list in `auth.py`
- **Styling**: Modify CSS in individual files
- **Redirect Logic**: Update `redirect_after_login` handling

## 🚨 Important Notes

### Security Considerations
- This is a **demo authentication system** for educational purposes
- Passwords are stored in plain text (for demo purposes only)
- In production, use proper password hashing and secure storage
- Consider using OAuth/Firebase for production applications

### Session Management
- Sessions are stored in Streamlit's session state
- Sessions expire after 24 hours of inactivity
- Browser refresh maintains login state
- Multiple tabs share the same session

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- JavaScript required for optimal functionality
- Mobile responsive design

## 🆘 Troubleshooting

### Common Issues
1. **Login Not Working**: Check email/password spelling
2. **Session Lost**: Clear browser cache and retry
3. **Page Not Loading**: Restart the Streamlit server
4. **Access Denied**: Ensure you're logged in before accessing main app

### Debug Mode
Add debugging by checking session state:
```python
st.write(st.session_state)
```

## 📞 Support

For issues with the authentication system:
1. Check this documentation
2. Verify user credentials
3. Restart the application
4. Check browser console for errors

---

**© 2024 Solar Panel Detection System | Authentication Module**
