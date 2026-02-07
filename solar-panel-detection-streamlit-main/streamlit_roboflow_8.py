import streamlit as st
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tempfile
import os
from datetime import datetime
import numpy as np
import re
import math

# --- Initialize Roboflow ---
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace().project(os.getenv("ROBOFLOW_PROJECT", ""))
model = project.version(4).model

# --- Helper function: Calculate polygon area ---
def calculate_polygon_area(points):
    """
    Calculate the area of a polygon using the shoelace formula
    points: list of (x, y) tuples
    returns: area in square pixels
    """
    if len(points) < 3:
        return 0
    
    n = len(points)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0

# --- Helper function: Convert pixel area to real-world area ---
def pixel_to_real_area(pixel_area, zoom_level, image_size=640):
    """
    Convert pixel area to real-world area in square meters
    zoom_level: Google Maps zoom level (0-21)
    image_size: size of the satellite image in pixels
    """
    # Approximate meters per pixel at different zoom levels
    # These are rough estimates for Google Maps satellite imagery
    meters_per_pixel = {
        20: 0.1,    # ~0.1 meters per pixel
        19: 0.2,    # ~0.2 meters per pixel
        18: 0.4,    # ~0.4 meters per pixel
        17: 0.8,    # ~0.8 meters per pixel
        16: 1.6,    # ~1.6 meters per pixel
        15: 3.2,    # ~3.2 meters per pixel
        14: 6.4,    # ~6.4 meters per pixel
        13: 12.8,   # ~12.8 meters per pixel
        12: 25.6,   # ~25.6 meters per pixel
        11: 51.2,   # ~51.2 meters per pixel
        10: 102.4,  # ~102.4 meters per pixel
    }
    
    # Get meters per pixel for the zoom level
    mpp = meters_per_pixel.get(zoom_level, 0.1)  # Default to 0.1 if zoom level not found
    
    # Convert pixel area to real-world area
    real_area = pixel_area * (mpp ** 2)
    
    return real_area

# --- Helper function: Convert DMS to decimal degrees ---
def dms_to_decimal(dms_string):
    """
    Convert DMS (Degree Minute Second) format to decimal degrees
    Examples: "40° 26' 46\" N" or "40°26'46\"N" or "40 26 46 N"
    """
    try:
        # Remove extra spaces and normalize
        dms_string = dms_string.strip().replace('°', ' ').replace("'", ' ').replace('"', ' ')
        dms_string = re.sub(r'\s+', ' ', dms_string)  # Replace multiple spaces with single space
        
        # Split the string
        parts = dms_string.split()
        
        if len(parts) >= 4:
            degrees = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            direction = parts[3].upper()
            
            # Calculate decimal degrees
            decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
            
            # Apply direction
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            elif direction not in ['N', 'E']:
                raise ValueError(f"Invalid direction: {direction}")
            
            return decimal_degrees
        else:
            raise ValueError("Invalid DMS format")
    except Exception as e:
        st.error(f"Error converting DMS format: {e}")
        return None

# --- Helper function: Validate and convert coordinate input ---
def parse_coordinate(coord_input, coord_type):
    """
    Parse coordinate input that could be decimal or DMS format
    coord_type: 'latitude' or 'longitude'
    """
    if not coord_input.strip():
        return None
    
    # Try to parse as decimal first
    try:
        decimal_value = float(coord_input)
        
        # Validate ranges
        if coord_type == 'latitude' and (-90 <= decimal_value <= 90):
            return decimal_value
        elif coord_type == 'longitude' and (-180 <= decimal_value <= 180):
            return decimal_value
        else:
            st.warning(f"Invalid {coord_type} range. Latitude: -90 to 90, Longitude: -180 to 180")
            return None
    except ValueError:
        # If not decimal, try DMS format
        decimal_value = dms_to_decimal(coord_input)
        if decimal_value is not None:
            # Validate ranges for DMS converted value
            if coord_type == 'latitude' and (-90 <= decimal_value <= 90):
                return decimal_value
            elif coord_type == 'longitude' and (-180 <= decimal_value <= 180):
                return decimal_value
            else:
                st.warning(f"Invalid {coord_type} range after DMS conversion")
                return None
        return None

# --- Helper function: Google Maps image download ---
def download_google_satellite(lat, lon, zoom=20, size="640x640"):
    # Hardcoded Google API key
    gapi_key = os.getenv("GOOGLE_API_KEY")
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

# --- Helper function: Create segmentation-based visualization with area calculation ---
def create_segmentation_visualization_with_area(image, predictions, zoom_level=20):
    """Create visualization using actual segmentation masks with area calculation"""
    # Create a copy of the image
    img = image.copy()
    
    # Create overlay for segmentation masks
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Count detections and calculate areas
    detection_count = len(predictions.get("predictions", []))
    panel_areas = []
    total_area = 0
    
    # Draw each prediction
    for i, pred in enumerate(predictions.get("predictions", [])):
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
                    # Calculate area
                    pixel_area = calculate_polygon_area(point_tuples)
                    real_area = pixel_to_real_area(pixel_area, zoom_level)
                    panel_areas.append(real_area)
                    total_area += real_area
                    
                    # Draw filled polygon for segmentation mask
                    overlay_draw.polygon(point_tuples, 
                                       fill=(0, 255, 0, 80),  # Green with transparency
                                       outline=(0, 255, 0, 255))  # Solid green outline
                    
                    # Add confidence text and area at the center of the polygon
                    if point_tuples:
                        center_x = sum(p[0] for p in point_tuples) / len(point_tuples)
                        center_y = sum(p[1] for p in point_tuples) / len(point_tuples)
                        conf_text = f"{confidence:.2f}"
                        area_text = f"{real_area:.1f}m²"
                        overlay_draw.text((center_x, center_y - 20), conf_text, fill=(0, 255, 0, 255))
                        overlay_draw.text((center_x, center_y - 5), area_text, fill=(0, 255, 0, 255))
                else:
                    # Fallback to bounding box if polygon is invalid
                    area = draw_bounding_box_fallback_with_area(overlay_draw, pred, confidence, zoom_level)
                    panel_areas.append(area)
                    total_area += area
            else:
                # Fallback to bounding box
                area = draw_bounding_box_fallback_with_area(overlay_draw, pred, confidence, zoom_level)
                panel_areas.append(area)
                total_area += area
        else:
            # Fallback to bounding box if no segmentation data
            area = draw_bounding_box_fallback_with_area(overlay_draw, pred, confidence, zoom_level)
            panel_areas.append(area)
            total_area += area
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    
    return result, detection_count, panel_areas, total_area

def draw_bounding_box_fallback_with_area(draw, pred, confidence, zoom_level):
    """Create polygon-like shape from bounding box with area calculation"""
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    
    # Create a polygon that follows the bounding box but with slight variations
    width = right - left
    height = bottom - top
    variation = min(width, height) * 0.1  # 10% variation
    
    # Create polygon points with slight variations
    points = [
        (left + variation, top),
        (right - variation, top),
        (right, top + variation),
        (right, bottom - variation),
        (right - variation, bottom),
        (left + variation, bottom),
        (left, bottom - variation),
        (left, top + variation),
    ]
    
    # Calculate area
    pixel_area = calculate_polygon_area(points)
    real_area = pixel_to_real_area(pixel_area, zoom_level)
    
    # Draw filled polygon instead of rectangle
    draw.polygon(points, 
                fill=(0, 255, 0, 80),  # Green with transparency
                outline=(0, 255, 0, 255))  # Solid green outline
    
    # Add confidence text and area at center
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    conf_text = f"{confidence:.2f}"
    area_text = f"{real_area:.1f}m²"
    draw.text((center_x, center_y - 20), conf_text, fill=(0, 255, 0, 255))
    draw.text((center_x, center_y - 5), area_text, fill=(0, 255, 0, 255))
    
    return real_area

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
st.set_page_config(page_title="Solar Panel Detector with Area Calculation", page_icon="☀️", layout="wide")

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
    .coordinate-help {
        background: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .area-stats {
        background: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🔆 Solar Panel Detection with Area Calculation</h1></div>', unsafe_allow_html=True)

# Main content
mode = st.radio("Choose input method:", ["Upload Image", "Latitude/Longitude"])

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
            
            # Create segmentation-based visualization with area calculation
            result_img, detection_count, panel_areas, total_area = create_segmentation_visualization_with_area(img, preds)
            
            # Add UI elements (always show counter and timestamp)
            timestamp = "2025 Airbus"
            result_img = add_ui_elements(result_img, detection_count, timestamp)
            
            # Display result
            st.image(result_img)
            
            # Show detection statistics and area information
            if detection_count > 0:
                st.markdown(f'<div class="detection-counter">✅ Solar panels detected: {detection_count}</div>', unsafe_allow_html=True)
                
                # Display area statistics
                st.markdown('<div class="area-stats">', unsafe_allow_html=True)
                st.subheader("📊 Area Statistics")
                st.write(f"**Total Solar Panel Area:** {total_area:.1f} m²")
                st.write(f"**Average Panel Area:** {total_area/detection_count:.1f} m²")
                st.write(f"**Largest Panel:** {max(panel_areas):.1f} m²")
                st.write(f"**Smallest Panel:** {min(panel_areas):.1f} m²")
                
                # Show individual panel areas
                st.write("**Individual Panel Areas:**")
                for i, area in enumerate(panel_areas, 1):
                    st.write(f"Panel {i}: {area:.1f} m²")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="detection-counter">❌ No solar panels detected</div>', unsafe_allow_html=True)

elif mode == "Latitude/Longitude":
    # Input section at the top
    st.subheader("Latitude/Longitude")
    
    # Coordinate format help
    with st.expander("ℹ️ Coordinate Format Help"):
        st.markdown("""
        <div class="coordinate-help">
        <h4>Supported Formats:</h4>
        <p><strong>Decimal Format:</strong></p>
        <ul>
            <li>Latitude: 40.7128 (or -40.7128 for South)</li>
            <li>Longitude: -74.0060 (or 74.0060 for East)</li>
        </ul>
        <p><strong>DMS (Degree Minute Second) Format:</strong></p>
        <ul>
            <li>Latitude: 40° 26' 46" N (or 40°26'46"N)</li>
            <li>Longitude: 74° 0' 22" W (or 74°0'22"W)</li>
            <li>Also accepts: 40 26 46 N (without symbols)</li>
        </ul>
        <p><strong>Examples:</strong></p>
        <ul>
            <li>Decimal: 40.7128, -74.0060</li>
            <li>DMS: 40° 26' 46" N, 74° 0' 22" W</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Create input fields
    lat_input = st.text_input("Latitude", placeholder="e.g., 40.7128 or 40° 26' 46\" N")
    lon_input = st.text_input("Longitude", placeholder="e.g., -74.0060 or 74° 0' 22\" W")
    
    # Zoom level selector
    zoom_level = st.slider("Satellite Image Zoom Level", 10, 20, 20, help="Higher zoom = more detailed image and accurate area calculations")
    
    # Detect button
    detect_button = st.button("🔍 Fetch & Detect", type="primary")
    
    if detect_button:
        if lat_input and lon_input:
            # Parse coordinates
            lat = parse_coordinate(lat_input, 'latitude')
            lon = parse_coordinate(lon_input, 'longitude')
            
            if lat is not None and lon is not None:
                # Display parsed coordinates
                st.success(f"✅ Coordinates parsed successfully: {lat:.6f}, {lon:.6f}")
                
                with st.spinner("Fetching satellite image..."):
                    img = download_google_satellite(lat, lon, zoom_level)
                
                if img:
                    # Run inference
                    with st.spinner("Detecting solar panels..."):
                        preds = infer_with_roboflow(img)
                    
                    # Create segmentation-based visualization with area calculation
                    result_img, detection_count, panel_areas, total_area = create_segmentation_visualization_with_area(img, preds, zoom_level)
                    
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
                    
                    # Show detection statistics and area information
                    if detection_count > 0:
                        st.markdown(f'<div class="detection-counter">✅ Solar panels detected: {detection_count}</div>', unsafe_allow_html=True)
                        
                        # Display area statistics
                        st.markdown('<div class="area-stats">', unsafe_allow_html=True)
                        st.subheader("📊 Area Statistics")
                        st.write(f"**Total Solar Panel Area:** {total_area:.1f} m²")
                        st.write(f"**Average Panel Area:** {total_area/detection_count:.1f} m²")
                        st.write(f"**Largest Panel:** {max(panel_areas):.1f} m²")
                        st.write(f"**Smallest Panel:** {min(panel_areas):.1f} m²")
                        
                        # Show individual panel areas
                        st.write("**Individual Panel Areas:**")
                        for i, area in enumerate(panel_areas, 1):
                            st.write(f"Panel {i}: {area:.1f} m²")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="detection-counter">❌ No solar panels detected</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Invalid coordinate format. Please check the help section for supported formats.")
        else:
            st.warning("⚠️ Please enter both latitude and longitude.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔆 Advanced Solar Panel Detection with Area Calculation</p>
</div>
""", unsafe_allow_html=True) 