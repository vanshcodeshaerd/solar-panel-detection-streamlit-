import os
import toml

# Create .streamlit directory if it doesn't exist
streamlit_dir = ".streamlit"
os.makedirs(streamlit_dir, exist_ok=True)

# Create secrets.toml file
secrets_file = os.path.join(streamlit_dir, "secrets.toml")

secrets_content = """
# Streamlit Secrets Configuration
# This file will be automatically loaded by Streamlit

# Google Maps API Key
GOOGLE_API_KEY = "AIzaSyDkIwC8qL9a7f3kP5mN2rX6tY8vZ9wB1qJ4"

# Roboflow API Key (replace with your actual key)
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

# Project Configuration
ROBOFLOW_PROJECT = "solar-panel-detection-2"
ROBOFLOW_WORKSPACE = "roboflow-ai-hackathon"
"""

with open(secrets_file, 'w') as f:
    f.write(secrets_content)

print(f"✅ Created secrets file: {secrets_file}")
print("📝 Please edit this file and add your Roboflow API key")
print("🔄 Restart the Streamlit app after editing")
