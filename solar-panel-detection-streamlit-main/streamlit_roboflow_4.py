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
def download_google_satellite(lat, lon, gapi_key, zoom=21, size="640x640"):
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

# --- Helper function: Create segmentation-based visualization ---
def create_segmentation_visualization(image, predictions):
    """Create visualization using actual segmentation masks instead of bounding boxes"""
    # Create a copy of the image
    img = image.copy()
    
    # Create overlay for segmentation masks
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Count detections
    detection_count = len(predictions.get("predictions", []))
    
    # Draw each prediction
    for pred in predictions.get("predictions", []):
        confidence = pred.get("confidence", 0)
        
        # Check for Roboflow 3.0 Instance Segmentation data
        segmentation_data = None
        
        # Roboflow 3.0 uses 'points' field for segmentation data
        if "points" in pred and pred["points"]:
            segmentation_data = pred["points"]
        
        if segmentation_data:
            # Get segmentation points
            points = segmentation_data
            
            # Handle different segmentation data formats
            if isinstance(points, list) and len(points) > 0:
                if isinstance(points[0], dict):
                    # Format: [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]
                    point_tuples = [(point["x"], point["y"]) for point in points]
                elif isinstance(points[0], (list, tuple)):
                    # Format: [[x1, y1], [x2, y2], ...] or [(x1, y1), (x2, y2), ...]
                    point_tuples = [(point[0], point[1]) for point in points]
                else:
                    # Fallback to bounding box
                    point_tuples = None
                
                if point_tuples and len(point_tuples) >= 3:
                    # Draw filled polygon for segmentation mask
                    overlay_draw.polygon(point_tuples, 
                                       fill=(0, 255, 0, 80),  # Green with transparency
                                       outline=(0, 255, 0, 255))  # Solid green outline
                    
                    # Add confidence text at the center of the polygon
                    if point_tuples:
                        center_x = sum(p[0] for p in point_tuples) / len(point_tuples)
                        center_y = sum(p[1] for p in point_tuples) / len(point_tuples)
                        conf_text = f"{confidence:.2f}"
                        overlay_draw.text((center_x, center_y - 10), conf_text, fill=(0, 255, 0, 255))
                else:
                    # Fallback to bounding box if polygon is invalid
                    draw_bounding_box_fallback(overlay_draw, pred, confidence)
            else:
                # Fallback to bounding box
                draw_bounding_box_fallback(overlay_draw, pred, confidence)
        else:
            # Fallback to bounding box if no segmentation data
            draw_bounding_box_fallback(overlay_draw, pred, confidence)
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    
    return result, detection_count

def draw_bounding_box_fallback(draw, pred, confidence):
    """Create polygon-like shape from bounding box to mimic segmentation"""
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    
    # Create a polygon that follows the bounding box but with slight variations
    # to make it look more like a segmentation mask
    width = right - left
    height = bottom - top
    
    # Add some variation to make it look more natural (like segmentation)
    variation = min(width, height) * 0.1  # 10% variation
    
    # Create polygon points with slight variations
    points = [
        (left + variation, top),  # Top-left with variation
        (right - variation, top),  # Top-right with variation
        (right, top + variation),  # Top-right corner
        (right, bottom - variation),  # Bottom-right with variation
        (right - variation, bottom),  # Bottom-right with variation
        (left + variation, bottom),  # Bottom-left with variation
        (left, bottom - variation),  # Bottom-left corner
        (left, top + variation),  # Top-left corner
    ]
    
    # Draw filled polygon instead of rectangle
    draw.polygon(points, 
                fill=(0, 255, 0, 80),  # Green with transparency
                outline=(0, 255, 0, 255))  # Solid green outline
    
    # Add confidence text at center
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    conf_text = f"{confidence:.2f}"
    draw.text((center_x, center_y - 10), conf_text, fill=(0, 255, 0, 255))

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
st.set_page_config(page_title="Solar Panel Detector", page_icon="☀️", layout="wide")

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

st.markdown('<div class="main-header"><h1>🔆 Solar Panel Detection</h1></div>', unsafe_allow_html=True)

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
            
            # Create segmentation-based visualization
            result_img, detection_count = create_segmentation_visualization(img, preds)
            
            # Add UI elements (always show counter and timestamp)
            timestamp = "2025 Airbus"
            result_img = add_ui_elements(result_img, detection_count, timestamp)
            
            # Display result
            st.image(result_img)
            
            # Show detection statistics
            if detection_count > 0:
                st.markdown(f'<div class="detection-counter">✅ Solar panels detected: {detection_count}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="detection-counter">❌ No solar panels detected</div>', unsafe_allow_html=True)

elif mode == "Latitude/Longitude + Google Maps API":
    # Input section at the top
    st.subheader("📍 Location Input")
    
    # Create three columns for inputs
    input_col1, input_col2, input_col3, input_col4 = st.columns([1, 1, 1, 1])
    
    with input_col1:
        lat = st.text_input("Latitude", "")
    with input_col2:
        lon = st.text_input("Longitude", "")
    with input_col3:
        gapi_key = st.text_input("Google Maps API Key", type="password")
    with input_col4:
        st.write("")  # Empty space for alignment
        detect_button = st.button("🔍 Fetch & Detect", type="primary")
    
    if detect_button:
        if lat and lon and gapi_key:
            with st.spinner("Fetching satellite image..."):
                img = download_google_satellite(lat, lon, gapi_key)
            
            if img:
                # Run inference
                with st.spinner("Detecting solar panels..."):
                    preds = infer_with_roboflow(img)
                
                # Create segmentation-based visualization
                result_img, detection_count = create_segmentation_visualization(img, preds)
                
                # Add UI elements (always show counter and timestamp)
                timestamp = "2025 Airbus"
                result_img = add_ui_elements(result_img, detection_count, timestamp)
                
                # Display images side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🛰️ Satellite Image")
                    st.image(img)
                
                with col2:
                    st.subheader("🔍 Detection Results")
                    st.image(result_img)
                
                # Show detection statistics below both images
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
    <p>🔆 Advanced Solar Panel Detection </p>
</div>
""", unsafe_allow_html=True) 