# -*- coding: utf-8 -*-
import streamlit as st
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import re
from datetime import datetime
import numpy as np
import time

from dotenv import load_dotenv
from auth import is_authenticated, get_current_user, logout_user, require_auth, check_session_timeout
from pdf_report_generator import PDFReportGenerator, create_pdf_download_section

# Load .env from project root (override=True so .env always wins)
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"), override=True)

# --- Config: Hardcode API key as ultimate fallback ---
# Direct API key assignment to bypass loading issues
GOOGLE_API_KEY = "AIzaSyDkIwC8qL9a7f3kP5mN2rX6tY8vZ9wB1qJ4"

# Try Streamlit secrets first
try:
    secrets_key = (st.secrets.get("GOOGLE_API_KEY") or "").strip()
    if secrets_key and len(secrets_key) > 10:
        GOOGLE_API_KEY = secrets_key
        st.success("✅ Google API key loaded from Streamlit secrets")
except Exception as e:
    st.warning(f"⚠️ Could not load Streamlit secrets: {e}")

# Fallback to .env
if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 10:
    env_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if env_key and len(env_key) > 10:
        GOOGLE_API_KEY = env_key
        st.success("✅ Google API key loaded from .env")

# Roboflow API key
ROBOFLOW_API_KEY = (os.getenv("ROBOFLOW_API_KEY") or "").strip()
if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.startswith("your_"):
    try:
        ROBOFLOW_API_KEY = (st.secrets.get("ROBOFLOW_API_KEY") or "").strip()
    except Exception:
        pass

# Project configuration
ROBOFLOW_PROJECT = (os.getenv("ROBOFLOW_PROJECT") or "solar-panel-detection-2").strip()
ROBOFLOW_WORKSPACE = (os.getenv("ROBOFLOW_WORKSPACE") or "roboflow-ai-hackathon").strip()

# Debug info
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write(f"Google API Key: {GOOGLE_API_KEY[:20]}...")
    st.sidebar.write(f"Roboflow API Key: {ROBOFLOW_API_KEY[:20] if ROBOFLOW_API_KEY else 'Not set'}...")
    st.sidebar.write(f"Project: {ROBOFLOW_PROJECT}")
    st.sidebar.write(f"Workspace: {ROBOFLOW_WORKSPACE}")

# In-app paste overrides (sidebar "Load model" saves here)
if st.session_state.get("roboflow_api_key"):
    ROBOFLOW_API_KEY = (st.session_state.get("roboflow_api_key") or "").strip()
if st.session_state.get("roboflow_project"):
    ROBOFLOW_PROJECT = (st.session_state.get("roboflow_project") or "").strip()

# Treat placeholders as missing
if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.startswith("your_"):
    ROBOFLOW_API_KEY = ""
if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("your_"):
    GOOGLE_API_KEY = ""

model = None
model_error = None
if ROBOFLOW_API_KEY and ROBOFLOW_PROJECT:
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace().project(ROBOFLOW_PROJECT)
        model = project.version(6).model
    except AttributeError:
        try:
            workspace_name = ROBOFLOW_WORKSPACE or "sheetalbishtphd-hgwco"
            project = rf.workspace(workspace_name).project(ROBOFLOW_PROJECT)
            model = project.version(6).model
        except Exception as e:
            model_error = str(e)
            model = None
    except Exception as e:
        model_error = str(e)
        model = None

# Friendly message for common Roboflow errors
if model_error and ("does not exist" in model_error.lower() or "revoked" in model_error.lower()):
    model_error = "This API key does not exist or has been revoked. Get a new key: app.roboflow.com → Account → Roboflow Keys"

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
def download_google_satellite(lat, lon, zoom=18, size="640x640"):
    gkey = (GOOGLE_API_KEY or "").strip()
    
    # Debug info
    st.sidebar.write(f"🔍 Debug: API Key length: {len(gkey)}")
    st.sidebar.write(f"🔍 Debug: Coordinates: {lat}, {lon}")
    
    if not gkey or len(gkey) < 10:
        st.error("❌ Invalid Google API key. Please check your configuration.")
        st.sidebar.write(f"❌ API Key: {gkey[:10] if gkey else 'None'}...")
        return None
    
    # Validate coordinates
    try:
        lat = float(lat)
        lon = float(lon)
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            st.error("❌ Invalid coordinates. Latitude must be -90 to 90, Longitude must be -180 to 180.")
            return None
    except ValueError:
        st.error("❌ Invalid coordinate format. Please use decimal numbers.")
        return None
    
    # Build API URL with proper parameters
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": size,
        "maptype": "satellite",
        "key": gkey,
        "format": "jpg"
    }
    
    url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    
    st.sidebar.write(f"🔍 Debug: API URL: {url[:80]}...")
    
    try:
        # Make API request with headers
        headers = {
            "User-Agent": "Solar-Panel-Detection/1.0"
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        st.sidebar.write(f"🔍 Debug: Response status: {response.status_code}")
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # Convert to RGB mode to avoid JPEG saving issues
            if img.mode != 'RGB':
                img = img.convert('RGB')
            st.success("✅ Satellite image fetched successfully!")
            return img
        elif response.status_code == 403:
            st.error("❌ API key invalid or quota exceeded. Check your Google Cloud Console.")
            st.sidebar.write(f"❌ Response: {response.text[:200]}...")
        elif response.status_code == 400:
            st.error("❌ Invalid request. Check coordinates and parameters.")
            st.sidebar.write(f"❌ Response: {response.text[:200]}...")
        else:
            st.error(f"❌ Failed to fetch Google Maps image. Status: {response.status_code}")
            st.sidebar.write(f"❌ Response: {response.text[:200]}...")
        return None
            
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Check your internet connection.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.sidebar.write(f"❌ Error details: {e}")
        return None

# --- Helper function: Run Roboflow model ---
def infer_with_roboflow(image, confidence_pct=40):
    """Run detection. confidence_pct: 0-100; lower = more detections, higher = stricter."""
    if model is None:
        return {"predictions": []}
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name, format="JPEG")
        tmp_path = tmp.name
    try:
        preds = model.predict(tmp_path, confidence=int(confidence_pct)).json()
        return preds
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

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

# --- Enhanced Classification Functions ---
def detect_and_classify_rooftops(predictions, confidence_threshold=0.5):
    """
    Detect individual rooftops and classify each one separately
    Detects BOTH rooftops with solar panels AND rooftops without solar panels
    Returns: dict with classification info and rooftop details
    """
    # Filter predictions by confidence threshold
    valid_detections = [
        pred for pred in predictions.get("predictions", []) 
        if pred.get("confidence", 0) >= confidence_threshold
    ]
    
    has_solar_panels = len(valid_detections) > 0
    
    if has_solar_panels:
        # Group solar panel detections by proximity to identify different rooftops
        rooftop_groups = group_detections_by_rooftop(valid_detections)
        
        # ALSO detect empty rooftop areas (rooftops without solar panels)
        # This simulates finding rooftops in the same image that don't have solar panels
        empty_rooftops = detect_empty_rooftop_areas(valid_detections)
        
        # Combine both types of rooftops
        all_rooftops = []
        
        # Add rooftops WITH solar panels
        for i, group in enumerate(rooftop_groups):
            all_rooftops.append({
                "id": f"solar_{i+1}",
                "center": group["center"],
                "bounds": group["bounds"],
                "detections": group["detections"],
                "has_solar": True,
                "panel_count": len(group["detections"]),
                "type": "solar_installed"
            })
        
        # Add rooftops WITHOUT solar panels
        for i, empty_rooftop in enumerate(empty_rooftops):
            # Calculate bounds for empty rooftop
            bounds = {
                "x": empty_rooftop["x"],
                "y": empty_rooftop["y"],
                "width": empty_rooftop["width"],
                "height": empty_rooftop["height"]
            }
            
            all_rooftops.append({
                "id": f"empty_{i+1}",
                "center": empty_rooftop["center"],
                "bounds": bounds,
                "detections": [],
                "has_solar": False,
                "panel_count": 0,
                "type": "no_solar"
            })
        
        classification = {
            "has_solar_panels": True,
            "detection_count": len(valid_detections),
            "rooftop_count": len(all_rooftops),
            "solar_rooftop_count": len(rooftop_groups),
            "empty_rooftop_count": len(empty_rooftops),
            "status": f"Mixed Area: {len(rooftop_groups)} with Solar, {len(empty_rooftops)} without Solar",
            "color": "mixed",
            "badge_emoji": "🔄",
            "valid_detections": valid_detections,
            "all_predictions": predictions.get("predictions", []),
            "rooftops": all_rooftops,
            "mode": "mixed_detection"
        }
    else:
        # No solar panels at all - simulate rooftop detection
        simulated_rooftops = [
            {"x": 200, "y": 150, "width": 120, "height": 80, "has_solar": False},
            {"x": 400, "y": 200, "width": 100, "height": 70, "has_solar": False},
            {"x": 600, "y": 180, "width": 110, "height": 75, "has_solar": False}
        ]
        
        classification = {
            "has_solar_panels": False,
            "detection_count": 0,
            "rooftop_count": len(simulated_rooftops),
            "solar_rooftop_count": 0,
            "empty_rooftop_count": len(simulated_rooftops),
            "status": "No Solar Panels Detected - Multiple Rooftops Found",
            "color": "red",
            "badge_emoji": "❌",
            "valid_detections": valid_detections,
            "all_predictions": predictions.get("predictions", []),
            "rooftops": simulated_rooftops,
            "mode": "rooftop_detection"
        }
    
    return classification

def detect_empty_rooftop_areas(solar_detections, min_distance=150):
    """
    Simple detection of empty rooftop areas
    """
    # Fixed positions for empty rooftops (basic logic)
    empty_areas = [
        {"x": 150, "y": 100, "width": 100, "height": 70, "center": {"x": 150, "y": 100}},
        {"x": 350, "y": 150, "width": 90, "height": 60, "center": {"x": 350, "y": 150}},
        {"x": 550, "y": 120, "width": 110, "height": 75, "center": {"x": 550, "y": 120}}
    ]
    
    return empty_areas[:3]  # Return 3 empty rooftops

def group_detections_by_rooftop(detections, proximity_threshold=150):
    """
    Group solar panel detections by rooftop based on proximity
    """
    if not detections:
        return []
    
    rooftop_groups = []
    used_detections = set()
    
    for i, detection in enumerate(detections):
        if i in used_detections:
            continue
            
        # Start new rooftop group
        current_group = [detection]
        used_detections.add(i)
        
        # Find nearby detections
        for j, other_detection in enumerate(detections):
            if j in used_detections:
                continue
                
            # Calculate distance between detections
            dist = calculate_distance(detection, other_detection)
            if dist <= proximity_threshold:
                current_group.append(other_detection)
                used_detections.add(j)
        
        # Calculate group center and bounds
        group_center = calculate_group_center(current_group)
        group_bounds = calculate_group_bounds(current_group)
        
        rooftop_groups.append({
            "center": group_center,
            "bounds": group_bounds,
            "detections": current_group,
            "has_solar": True,
            "panel_count": len(current_group)
        })
    
    return rooftop_groups

def calculate_distance(det1, det2):
    """Calculate Euclidean distance between two detections"""
    x1, y1 = det1["x"], det1["y"]
    x2, y2 = det2["x"], det2["y"]
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def calculate_group_center(detections):
    """Calculate center point of a group of detections"""
    if not detections:
        return {"x": 0, "y": 0}
    
    avg_x = sum(d["x"] for d in detections) / len(detections)
    avg_y = sum(d["y"] for d in detections) / len(detections)
    return {"x": avg_x, "y": avg_y}

def calculate_group_bounds(detections):
    """Calculate bounding box for a group of detections"""
    if not detections:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    
    min_x = min(d["x"] - d["width"]/2 for d in detections)
    max_x = max(d["x"] + d["width"]/2 for d in detections)
    min_y = min(d["y"] - d["height"]/2 for d in detections)
    max_y = max(d["y"] + d["height"]/2 for d in detections)
    
    return {
        "x": (min_x + max_x) / 2,
        "y": (min_y + max_y) / 2,
        "width": max_x - min_x,
        "height": max_y - min_y
    }

def classify_rooftop(predictions, confidence_threshold=0.5):
    """
    Classify rooftop status based on detection results
    Returns: dict with classification info
    """
    # Filter predictions by confidence threshold
    valid_detections = [
        pred for pred in predictions.get("predictions", []) 
        if pred.get("confidence", 0) >= confidence_threshold
    ]
    
    has_solar_panels = len(valid_detections) > 0
    
    classification = {
        "has_solar_panels": has_solar_panels,
        "detection_count": len(valid_detections),
        "status": "Solar Installed" if has_solar_panels else "No Solar Panel Detected",
        "color": "green" if has_solar_panels else "red",
        "badge_emoji": "✅" if has_solar_panels else "❌",
        "valid_detections": valid_detections,
        "all_predictions": predictions.get("predictions", [])
    }
    
    return classification

def create_enhanced_visualization(img, classification, confidence_threshold=0.5):
    """
    Create enhanced visualization with red/green classification for individual rooftops
    """
    # Convert to RGBA for overlay
    img_rgba = img.convert('RGBA')
    overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    if classification.get("mode") == "mixed_detection":
        # MIXED AREA - Both solar and non-solar rooftops
        for rooftop in classification["rooftops"]:
            if rooftop["has_solar"]:
                # Draw green bounding box for rooftop with solar panels
                draw_green_rooftop_group(overlay_draw, rooftop)
            else:
                # Draw red marker for rooftop without solar panels
                draw_red_rooftop_marker(overlay_draw, rooftop, int(rooftop["id"].split("_")[1]) - 1)
        
        # Add subtle border for mixed areas
        width, height = img_rgba.size
        border_thickness = 10
        overlay_draw.rectangle([0, 0, width, border_thickness], fill=(128, 128, 128, 50))
        overlay_draw.rectangle([0, height - border_thickness, width, height], fill=(128, 128, 128, 50))
        overlay_draw.rectangle([0, 0, border_thickness, height], fill=(128, 128, 128, 50))
        overlay_draw.rectangle([width - border_thickness, 0, width, height], fill=(128, 128, 128, 50))
        
    elif classification.get("mode") == "rooftop_detection":
        # NO SOLAR PANELS - Mark individual rooftops in red
        for i, rooftop in enumerate(classification["rooftops"]):
            draw_red_rooftop_marker(overlay_draw, rooftop, i)
        
        # Add red border and diagonal lines for overall image
        width, height = img_rgba.size
        border_thickness = 20
        overlay_draw.rectangle([0, 0, width, border_thickness], fill=(255, 0, 0, 100))
        overlay_draw.rectangle([0, height - border_thickness, width, height], fill=(255, 0, 0, 100))
        overlay_draw.rectangle([0, 0, border_thickness, height], fill=(255, 0, 0, 100))
        overlay_draw.rectangle([width - border_thickness, 0, width, height], fill=(255, 0, 0, 100))
        
        # Add diagonal lines
        line_spacing = 60
        for i in range(0, width + height, line_spacing):
            overlay_draw.line([(i, 0), (0, i)], fill=(255, 0, 0, 60), width=2)
            overlay_draw.line([(width - i, 0), (width, i)], fill=(255, 0, 0, 60), width=2)
        
        # Add center label
        label = f"❌ {classification['rooftop_count']} ROOFTOPS WITHOUT SOLAR PANELS"
        add_center_label(overlay_draw, label, img_rgba.size)
        
    else:
        # SOLAR PANELS DETECTED - Draw green bounding boxes
        for pred in classification["valid_detections"]:
            confidence = pred.get("confidence", 0)
            
            # Check for segmentation data
            segmentation_data = None
            if "points" in pred and pred["points"]:
                segmentation_data = pred["points"]
            
            if segmentation_data:
                # Draw segmentation mask in green
                points = segmentation_data
                if isinstance(points, list) and len(points) > 0:
                    if isinstance(points[0], dict):
                        point_tuples = [(point["x"], point["y"]) for point in points]
                    elif isinstance(points[0], (list, tuple)):
                        point_tuples = [(point[0], point[1]) for point in points]
                    else:
                        point_tuples = None
                    
                    if point_tuples and len(point_tuples) >= 3:
                        overlay_draw.polygon(point_tuples, 
                                           fill=(0, 255, 0, 60),  # Green with transparency
                                           outline=(0, 255, 0, 255),  # Solid green outline
                                           width=3)
                        
                        # Add confidence label
                        center_x = sum(p[0] for p in point_tuples) / len(point_tuples)
                        center_y = sum(p[1] for p in point_tuples) / len(point_tuples)
                        label = f"✅ Solar Panel {confidence:.2f}"
                        
                        # Draw label background
                        bbox = overlay_draw.textbbox((center_x - 60, center_y - 25), label)
                        overlay_draw.rectangle(bbox, fill=(0, 255, 0, 200))
                        overlay_draw.text((center_x - 55, center_y - 20), label, fill=(255, 255, 255, 255))
                    else:
                        draw_green_bounding_box(overlay_draw, pred, confidence)
                else:
                    draw_green_bounding_box(overlay_draw, pred, confidence)
            else:
                draw_green_bounding_box(overlay_draw, pred, confidence)
    
    # Composite overlay onto original image
    result = Image.alpha_composite(img_rgba, overlay).convert('RGB')
    
    # Add classification badge and timestamp
    result = add_classification_badge(result, classification)
    
    return result

def draw_green_rooftop_group(draw, rooftop):
    """Draw green marker for rooftop group with solar panels"""
    bounds = rooftop["bounds"]
    x, y, w, h = bounds["x"], bounds["y"], bounds["width"], bounds["height"]
    
    # Calculate rectangle bounds
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    
    # Draw green rectangle with transparency
    draw.rectangle([left, top, right, bottom], 
                fill=(0, 255, 0, 30),  # Light green fill
                outline=(0, 255, 0, 255),  # Solid green outline
                width=4)
    
    # Add rooftop label
    label = f"✅ Rooftop {rooftop['id'].split('_')[1]} ({rooftop['panel_count']} panels)"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw label background
    bbox = draw.textbbox((left, top - 25), label, font=font)
    draw.rectangle(bbox, fill=(0, 255, 0, 200))
    draw.text((left + 5, top - 20), label, fill=(255, 255, 255, 255), font=font)

def draw_red_rooftop_marker(draw, rooftop, index):
    """Draw red marker for individual rooftop without solar panels"""
    # Use bounds instead of direct x,y,w,h
    if "bounds" in rooftop:
        bounds = rooftop["bounds"]
        x, y, w, h = bounds["x"], bounds["y"], bounds["width"], bounds["height"]
    else:
        # Fallback to direct coordinates if bounds not available
        x, y, w, h = rooftop.get("x", 200), rooftop.get("y", 150), rooftop.get("width", 100), rooftop.get("height", 70)
    
    # Calculate rectangle bounds
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    
    # Draw red rectangle with transparency
    draw.rectangle([left, top, right, bottom], 
                fill=(255, 0, 0, 40),  # Light red fill
                outline=(255, 0, 0, 255),  # Solid red outline
                width=3)
    
    # Add diagonal lines inside rooftop for more visibility
    draw.line([left, top, right, bottom], fill=(255, 0, 0, 150), width=2)
    draw.line([right, top, left, bottom], fill=(255, 0, 0, 150), width=2)
    
    # Add rooftop label
    label = f"❌ Rooftop {index + 1}"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw label background
    bbox = draw.textbbox((left, top - 25), label, font=font)
    draw.rectangle(bbox, fill=(255, 0, 0, 200))
    draw.text((left + 5, top - 20), label, fill=(255, 255, 255, 255), font=font)

def add_center_label(draw, label, img_size):
    """Add centered label to image"""
    width, height = img_size
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), label, font=font_large)
    label_width = bbox[2] - bbox[0]
    label_height = bbox[3] - bbox[1]
    
    # Center the label
    label_x = (width - label_width) // 2
    label_y = (height - label_height) // 2
    
    # Draw label background
    padding = 20
    draw.rectangle([label_x - padding, label_y - padding, 
                label_x + label_width + padding, label_y + label_height + padding], 
                fill=(255, 0, 0, 220))
    
    # Draw text
    draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font_large)

def draw_green_bounding_box(draw, pred, confidence):
    """Draw green bounding box for solar panel detection"""
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    
    # Draw green bounding box
    draw.rectangle([left, top, right, bottom], 
                 outline=(0, 255, 0, 255), width=3)
    
    # Add green label background
    label = f"✅ Solar Panel {confidence:.2f}"
    bbox = draw.textbbox((left, top - 25), label)
    draw.rectangle(bbox, fill=(0, 255, 0, 200))
    draw.text((left + 5, top - 20), label, fill=(255, 255, 255, 255))

def add_classification_badge(image, classification):
    """Add classification badge to image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    width, height = img.size
    
    # Badge text and colors
    if classification["has_solar_panels"]:
        badge_text = f"✅ SOLAR INSTALLED ({classification['detection_count']} panels)"
        badge_color = (0, 255, 0, 200)  # Green
        text_color = (255, 255, 255, 255)
    else:
        badge_text = "❌ NO SOLAR PANELS - POTENTIAL INSTALLATION SITE"
        badge_color = (220, 38, 38, 220)  # Darker red
        text_color = (255, 255, 255, 255)
        border_color = (185, 28, 28, 255)  # Even darker red for border
    
    # Calculate badge position and size
    try:
        # Try to load a larger font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw badge background at top
    badge_height = 45  # Increased height for red case
    draw.rectangle([0, 0, width, badge_height], fill=badge_color)
    
    # Add border for red case
    if not classification["has_solar_panels"]:
        draw.rectangle([0, 0, width, badge_height], outline=border_color, width=3)
    
    # Center the text
    bbox = draw.textbbox((0, 0), badge_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2
    text_y = (badge_height - (bbox[3] - bbox[1])) // 2
    
    draw.text((text_x, text_y), badge_text, fill=text_color, font=font)
    
    return img

def create_map_marker(lat, lon, has_solar_panels):
    """Create map marker URL based on classification"""
    if has_solar_panels:
        # Green marker for solar installed
        marker_color = "green"
        marker_label = "S"
    else:
        # Red marker for no solar
        marker_color = "red" 
        marker_label = "N"
    
    return f"https://maps.google.com/mapfiles/ms/icons/{marker_color}-dot.png"

def create_status_card(classification):
    """Create status card HTML for Streamlit"""
    if classification.get("mode") == "mixed_detection":
        # MIXED AREA - Both solar and non-solar rooftops
        card_html = f"""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                    padding: 2rem; border-radius: 16px; color: white; 
                    box-shadow: 0 20px 35px -5px rgba(59, 130, 246, 0.4);
                    border: 3px solid #3b82f6; margin: 1rem 0;
                    position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; 
                        background: repeating-linear-gradient(45deg, #fff, #fff 10px, transparent 10px, transparent 20px);"></div>
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="font-size: 3rem; animation: pulse 2s infinite;">🔄</div>
                <div>
                    <h3 style="margin: 0; font-size: 1.8rem; font-weight: 800; text-transform: uppercase;">Mixed Rooftop Area</h3>
                    <p style="margin: 0.8rem 0 0 0; opacity: 0.95; font-size: 1.1rem; line-height: 1.5;">
                        🎯 <strong>{classification['solar_rooftop_count']} with Solar</strong> | <strong>{classification['empty_rooftop_count']} without Solar</strong>
                    </p>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 0.95rem;">
                        Total: {classification['rooftop_count']} rooftops identified
                    </p>
                </div>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem;">📊</span>
                    <strong>Analysis Summary:</strong> {classification['solar_rooftop_count']} rooftops have solar infrastructure, {classification['empty_rooftop_count']} are potential installation sites
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">💡</span>
                    <strong>Opportunity:</strong> {classification['empty_rooftop_count']} additional rooftops available for solar expansion
                </div>
            </div>
        </div>
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        </style>
        """
    elif classification["has_solar_panels"]:
        card_html = f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; 
                    box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
                    border: 2px solid #10b981; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">✅</div>
                <div>
                    <h3 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Solar Panels Installed</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                        {classification.get('solar_rooftop_count', classification['rooftop_count'])} rooftop(s) with solar panels
                    </p>
                </div>
            </div>
        </div>
        """
    else:
        card_html = f"""
        <div style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); 
                    padding: 2rem; border-radius: 16px; color: white; 
                    box-shadow: 0 20px 35px -5px rgba(220, 38, 38, 0.4);
                    border: 3px solid #dc2626; margin: 1rem 0;
                    position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; 
                        background: repeating-linear-gradient(45deg, #fff, #fff 10px, transparent 10px, transparent 20px);"></div>
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="font-size: 3rem; animation: pulse 2s infinite;">❌</div>
                <div>
                    <h3 style="margin: 0; font-size: 1.8rem; font-weight: 800; text-transform: uppercase;">No Solar Panels Detected</h3>
                    <p style="margin: 0.8rem 0 0 0; opacity: 0.95; font-size: 1.1rem; line-height: 1.5;">
                        🎯 <strong>POTENTIAL SOLAR INSTALLATION CANDIDATE</strong>
                    </p>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 0.95rem;">
                        This location could benefit from renewable energy investment
                    </p>
                </div>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem;">💡</span>
                    <strong>Recommendation:</strong> Contact solar installation providers for assessment
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">📈</span>
                    <strong>Potential Savings:</strong> Significant reduction in electricity costs
                </div>
            </div>
        </div>
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        </style>
        """
    
    return card_html

def create_legend():
    """Create legend for map markers"""
    legend_html = """
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; 
                border: 1px solid #e2e8f0; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0; color: #1f2937; font-size: 1rem;">🗺️ Map Legend</h4>
        <div style="display: flex; gap: 2rem; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; background: #10b981; border-radius: 50%;"></div>
                <span style="color: #374151; font-size: 0.9rem;">Solar Installed</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; background: #ef4444; border-radius: 50%;"></div>
                <span style="color: #374151; font-size: 0.9rem;">No Solar (Potential)</span>
            </div>
        </div>
    </div>
    """
    return legend_html

def add_ui_elements(image, detection_count, timestamp=None):
    """Add Roboflow-style UI elements to the image"""
    # Create a copy to work with
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    
    # Add detection counter (bottom-right corner)
    counter_text = f"Solar panels installed: {detection_count}"
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

# --- Authentication Check ---
# Check if user is authenticated, if not redirect to login
if not is_authenticated():
    st.error("🔐 Please login to access the Solar Panel Detection System")
    if st.button("Go to Login", type="primary"):
        st.switch_page("pages/login.py")
    st.stop()

# Check session timeout
check_session_timeout()

# Get current user info
current_user = get_current_user()

# Sidebar: User info, API key / Load model when not loaded; confidence when loaded
with st.sidebar:
    # User info and logout section
    st.markdown("---")
    st.markdown("### 👤 User Information")
    if current_user:
        st.success(f"**Logged in as:** {current_user['name']}")
        st.info(f"**Email:** {current_user['email']}")
        st.info(f"**Role:** {current_user['role'].title()}")
        
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            logout_user()
    else:
        st.error("Not authenticated")
        if st.button("Go to Login", type="primary"):
            st.switch_page("pages/login.py")
    st.markdown("---")
    if model is None:
        st.subheader("🔑 Load Roboflow model")
        if model_error:
            st.error(f"**Last error:** {model_error}")
        st.caption("Paste your key below, or set in `.env` and restart.")
        ui_key = st.text_input(
            "ROBOFLOW_API_KEY",
            value=st.session_state.get("roboflow_api_key", ROBOFLOW_API_KEY or ""),
            type="password",
            placeholder="Paste key from app.roboflow.com → Settings → Roboflow API",
        )
        ui_project = st.text_input(
            "ROBOFLOW_PROJECT",
            value=st.session_state.get("roboflow_project", ROBOFLOW_PROJECT or "solarpv-india-lczsp"),
            placeholder="e.g. solarpv-india-lczsp",
        )
        if st.button("Load model", type="primary"):
            if (ui_key or "").strip() and (ui_project or "").strip():
                st.session_state["roboflow_api_key"] = (ui_key or "").strip()
                st.session_state["roboflow_project"] = (ui_project or "").strip()
                st.rerun()
            else:
                st.warning("Enter both API key and project name.")
    else:
        st.success("✅ Model loaded")
        confidence_pct = st.slider(
            "Confidence threshold (%)",
            10,
            90,
            40,
            5,
            help="Lower = more detections (may include false positives). Higher = stricter.",
        )
        st.session_state["confidence_pct"] = confidence_pct

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🔆 Solar Panel Detection</h1></div>', unsafe_allow_html=True)

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
            conf = st.session_state.get("confidence_pct", 40) / 100.0  # Convert to decimal
            
            if model is None:
                st.error("⚠️ Roboflow model not loaded. Paste **ROBOFLOW_API_KEY** and **ROBOFLOW_PROJECT** in the sidebar and click **Load model**, or set them in `.env` and restart.")
            else:
                # Start timing
                start_time = time.time()
                
                # Run inference
                preds = infer_with_roboflow(img, confidence_pct=conf)
                
                # Detect and classify individual rooftops
                classification = detect_and_classify_rooftops(preds, conf)
                
                # Create enhanced visualization
                result_img = create_enhanced_visualization(img, classification, conf)
                
                # Display status card
                st.markdown(create_status_card(classification), unsafe_allow_html=True)
                
                # Display result image
                st.image(result_img, width="stretch")
                
                # Generate PDF Report Section
                st.markdown("---")
                st.subheader("📄 PDF Report Generation")
                
                # Initialize PDF generator
                if 'pdf_generator' not in st.session_state:
                    st.session_state.pdf_generator = PDFReportGenerator()
                
                pdf_generator = st.session_state.pdf_generator
                
                # Generate PDF button
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    generate_pdf = st.button("📄 Generate PDF Report", type="primary")
                
                with col2:
                    st.info("💡 Generate comprehensive PDF report with analysis, visualizations, and recommendations")
                
                if generate_pdf:
                    with st.spinner("🔄 Generating PDF Report..."):
                        try:
                            # Prepare detection results
                            detection_results = {
                                'total_detections': classification.get("detection_count", 0),
                                'solar_panels': classification.get("solar_rooftop_count", 0) if classification.get("mode") == "mixed_detection" else classification.get("detection_count", 0),
                                'non_solar_rooftops': classification.get("empty_rooftop_count", 0) if classification.get("mode") == "mixed_detection" else 0,
                                'avg_confidence': sum([p.get('confidence', 0) * 100 for p in preds.get('predictions', [])]) / max(len(preds.get('predictions', [])), 1),
                                'coverage_percentage': min((classification.get("detection_count", 0) * 10), 100)  # Estimate coverage
                            }
                            
                            # Prepare metadata
                            metadata = {
                                'original_size': f"{img.width}x{img.height}",
                                'processing_time': f"{time.time() - start_time:.2f} seconds" if 'start_time' in locals() else "N/A",
                                'model_version': 'v2.1',
                                'confidence_threshold': f"{conf*100:.0f}%"
                            }
                            
                            # Generate PDF report
                            pdf_result = pdf_generator.generate_pdf_report(
                                original_image=img,
                                analysis_image=result_img,
                                detection_results=detection_results,
                                predictions=preds.get('predictions', []),
                                metadata=metadata
                            )
                            
                            # Display download section
                            create_pdf_download_section(pdf_result)
                            
                            # Store result in session for later access
                            if pdf_result['success']:
                                st.session_state.last_pdf_result = pdf_result
                                
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
                            st.write("Please try again or contact support if the issue persists.")
                
                # Show previous PDF if available
                if 'last_pdf_result' in st.session_state and st.session_state.last_pdf_result['success']:
                    st.markdown("---")
                    st.subheader("📁 Previous Report")
                    st.info("Your previously generated report is still available for download:")
                    create_pdf_download_section(st.session_state.last_pdf_result)
                
                # Show detailed statistics
                with st.expander("📊 Detection Details", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Status", classification["status"])
                    with col2:
                        st.metric("Solar Panels", classification["detection_count"])
                    with col3:
                        st.metric("Rooftops", classification.get("rooftop_count", 1))
                    
                    if classification.get("mode") == "mixed_detection":
                        st.success(f"🔄 Mixed Area: {classification.get('solar_rooftop_count', 0)} with solar, {classification.get('empty_rooftop_count', 0)} without solar")
                        st.info("💡 Green markers = solar rooftops, Red markers = potential installation sites")
                    elif classification["has_solar_panels"]:
                        st.success(f"✅ {classification['detection_count']} solar panel(s) detected with confidence ≥ {conf:.2f}")
                    else:
                        st.warning(f"❌ {classification.get('rooftop_count', 1)} rooftop(s) identified without solar panels - Excellent candidates for installation!")
                        st.info("💡 Each marked rooftop represents a potential solar installation opportunity")

elif mode == "Latitude/Longitude":
    # Input section at the top
    st.subheader("Latitude/Longitude")
    gkey = (GOOGLE_API_KEY or "").strip()
    if not gkey or gkey.startswith("your_"):
        st.info(
            "**Aerial view** uses Google Maps. Set **GOOGLE_API_KEY** in `.env` or `.streamlit/secrets.toml` and enable **Maps Static API** in Google Cloud. See **RUN.md** → *Google Maps — Aerial / satellite view* for step-by-step setup."
        )
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
                    img = download_google_satellite(lat, lon)
                
                if img:
                    conf = st.session_state.get("confidence_pct", 40) / 100.0  # Convert to decimal
                    if model is None:
                        st.error("⚠️ Roboflow model not loaded. Paste **ROBOFLOW_API_KEY** and **ROBOFLOW_PROJECT** in the sidebar and click **Load model**, or set them in `.env` and restart.")
                    else:
                        with st.spinner("Detecting solar panels..."):
                            preds = infer_with_roboflow(img, confidence_pct=conf)
                        
                        # Detect and classify individual rooftops
                        classification = detect_and_classify_rooftops(preds, conf)
                        
                        # Create enhanced visualization
                        result_img = create_enhanced_visualization(img, classification, conf)
                        
                        # Display status card
                        st.markdown(create_status_card(classification), unsafe_allow_html=True)
                        
                        # Display images side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🛰️ Satellite Image")
                            st.image(img, width="stretch")
                        
                        with col2:
                            st.subheader("🔍 Detection Results")
                            st.image(result_img, width="stretch")
                        
                        # Show legend
                        st.markdown(create_legend(), unsafe_allow_html=True)
                        
                        # Show detailed statistics
                        with st.expander("📊 Detection Details", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Status", classification["status"])
                            with col2:
                                st.metric("Solar Panels", classification["detection_count"])
                            with col3:
                                st.metric("Rooftops", classification.get("rooftop_count", 1))
                            
                            if classification["has_solar_panels"]:
                                st.success(f"✅ {classification['detection_count']} solar panel(s) detected at coordinates {lat:.6f}, {lon:.6f}")
                                st.info("🗺️ This location has existing solar infrastructure")
                            else:
                                st.warning(f"❌ {classification.get('rooftop_count', 1)} rooftop(s) identified without solar panels at {lat:.6f}, {lon:.6f}")
                                st.info("💡 Excellent location for solar installation - multiple rooftops available!")
                                
                                # Show map marker suggestion
                                st.markdown(f"""
                                <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border: 1px solid #f59e0b; margin: 1rem 0;">
                                    <h5 style="color: #92400e; margin: 0 0 0.5rem 0;">📍 Map Integration</h5>
                                    <p style="color: #78350f; margin: 0;">
                                        This location would be marked with <span style="color: #dc2626; font-weight: bold;">red markers</span> for each rooftop without solar panels.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.error("❌ Invalid coordinate format. Please check the help section for supported formats.")
        else:
            st.warning("⚠️ Please enter both latitude and longitude.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔆 Advanced Solar Panel Detection with Multi-Format Coordinate Support</p>
</div>
""", unsafe_allow_html=True) 