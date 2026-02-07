import streamlit as st
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tempfile
import os
from datetime import datetime
import numpy as np

# --- Initialize Roboflow ---
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace().project(os.getenv("ROBOFLOW_PROJECT", ""))
model = project.version(4).model

# --- Helper function: Google Maps image download ---
def download_google_satellite(lat, lon, gapi_key, zoom=20, size="640x640"):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={gapi_key}"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        # Convert to RGB mode to avoid JPEG saving issues
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    else:
        st.error("Failed to fetch Google Maps image.")
        return None

# --- Helper function: Run Roboflow model ---
def infer_with_roboflow(image):
    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name, format="JPEG")
        tmp_path = tmp.name
    
    try:
        # Run inference using file path
        preds = model.predict(tmp_path, confidence=40).json()
        return preds
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# --- Helper function: Create Roboflow-style visualization ---
def create_roboflow_style_visualization(image, predictions):
    # Create a copy of the image
    img = image.copy()
    
    # Create overlay for segmentation masks
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Count detections
    detection_count = len(predictions.get("predictions", []))
    
    # Draw each prediction
    for pred in predictions.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        confidence = pred.get("confidence", 0)
        
        # Create bounding box coordinates
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2
        
        # Draw green semi-transparent rectangle (Roboflow style)
        overlay_draw.rectangle([left, top, right, bottom], 
                              fill=(0, 255, 0, 80),  # Green with transparency
                              outline=(0, 255, 0, 255))  # Solid green outline
        
        # Add confidence text
        conf_text = f"{confidence:.2f}"
        overlay_draw.text((left, top - 15), conf_text, fill=(0, 255, 0, 255))
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    
    return result, detection_count

# --- Helper function: Add UI elements to image ---
def add_ui_elements(image, detection_count, timestamp=None):
    """Add Roboflow-style UI elements to the image"""
    # Create a copy to work with
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    
    # Add detection counter (bottom-right corner)
    counter_text = f"Solar panels detected: {detection_count}"
    # Estimate text size (approximate)
    text_width = len(counter_text) * 8
    text_height = 20
    
    # Create semi-transparent background for counter
    counter_bg = Image.new('RGBA', (text_width + 20, text_height + 10), (0, 0, 0, 180))
    img.paste(counter_bg, (width - text_width - 25, height - text_height - 15), counter_bg)
    
    # Add counter text
    draw.text((width - text_width - 15, height - text_height - 10), 
              counter_text, fill=(255, 255, 255, 255))
    
    # Add timestamp if provided
    if timestamp:
        timestamp_text = f"Image © {timestamp}"
        ts_width = len(timestamp_text) * 6
        ts_bg = Image.new('RGBA', (ts_width + 10, 15), (0, 0, 0, 180))
        img.paste(ts_bg, (width - ts_width - 15, 10), ts_bg)
        draw.text((width - ts_width - 10, 12), timestamp_text, fill=(255, 255, 255, 255))
    
    return img

# --- Streamlit UI ---
st.set_page_config(page_title="Roboflow-Style Solar Panel Detector", page_icon="☀️", layout="wide")

# Custom CSS for Roboflow-style appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-counter {
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        display: inline-block;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🔆 Roboflow-Style Solar Panel Detection</h1></div>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    
    # Visualization options
    st.subheader("🎨 Visualization")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_counter = st.checkbox("Show Detection Counter", value=True)
    show_timestamp = st.checkbox("Show Timestamp", value=True)
    
    # Timeline slider (simulating Roboflow's timeline feature)
    st.subheader("📅 Timeline")
    timeline_date = st.slider("Date", 2020, 2025, 2023)

# Main content
mode = st.radio("Choose input method:", ["Upload Image", "Latitude/Longitude + Google Maps API"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img)
        
        with col2:
            st.subheader("🔍 Detection Results")
            
            # Run inference
            preds = infer_with_roboflow(img)
            
            # Create Roboflow-style visualization
            result_img, detection_count = create_roboflow_style_visualization(img, preds)
            
            # Add UI elements
            if show_counter or show_timestamp:
                timestamp = f"{timeline_date} Airbus" if show_timestamp else None
                result_img = add_ui_elements(result_img, detection_count, timestamp)
            
            # Display result
            st.image(result_img)
            
            # Show detection statistics
            if detection_count > 0:
                st.markdown(f'<div class="detection-counter">✅ Solar panels detected: {detection_count}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="detection-counter">❌ No solar panels detected</div>', unsafe_allow_html=True)

elif mode == "Latitude/Longitude + Google Maps API":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location Input")
        lat = st.text_input("Latitude", "")
        lon = st.text_input("Longitude", "")
        gapi_key = st.text_input("Google Maps API Key", type="password")
        
        if st.button("🔍 Fetch & Detect", type="primary"):
            if lat and lon and gapi_key:
                with st.spinner("Fetching satellite image..."):
                    img = download_google_satellite(lat, lon, gapi_key)
                
                if img:
                    # Display original satellite image
                    st.subheader("🛰️ Satellite Image")
                    st.image(img)
                    
                    # Run inference
                    with st.spinner("Detecting solar panels..."):
                        preds = infer_with_roboflow(img)
                    
                    # Create Roboflow-style visualization
                    result_img, detection_count = create_roboflow_style_visualization(img, preds)
                    
                    # Add UI elements
                    if show_counter or show_timestamp:
                        timestamp = f"{timeline_date} Airbus" if show_timestamp else None
                        result_img = add_ui_elements(result_img, detection_count, timestamp)
                    
                    # Display result in second column
                    with col2:
                        st.subheader("🔍 Detection Results")
                        st.image(result_img)
                        
                        # Show detection statistics
                        if detection_count > 0:
                            st.markdown(f'<div class="detection-counter">✅ Solar panels detected: {detection_count}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="detection-counter">❌ No solar panels detected</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter latitude, longitude, and API key.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔆 Advanced Solar Panel Detection with Roboflow-Style Visualization</p>
</div>
""", unsafe_allow_html=True) 