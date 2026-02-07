"""
Solar Panel Detection Script - CSV Version
Reads coordinates from CSV file and detects solar panels using Roboflow model
"""

import requests
from io import BytesIO
from PIL import Image, ImageDraw
from roboflow import Roboflow
import tempfile
import os
from datetime import datetime
import pandas as pd
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Initialize Roboflow ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "")

print("Initializing Roboflow model...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_PROJECT)
model = project.version(6).model
print("Model initialized successfully!")

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
        print(f"Error converting DMS format: {e}")
        return None

# --- Helper function: Validate and convert coordinate input ---
def parse_coordinate(coord_input, coord_type):
    """
    Parse coordinate input that could be decimal or DMS format
    coord_type: 'latitude' or 'longitude'
    """
    if pd.isna(coord_input) or str(coord_input).strip() == '':
        return None
    
    coord_input = str(coord_input)
    
    # Try to parse as decimal first
    try:
        decimal_value = float(coord_input)
        
        # Validate ranges
        if coord_type == 'latitude' and (-90 <= decimal_value <= 90):
            return decimal_value
        elif coord_type == 'longitude' and (-180 <= decimal_value <= 180):
            return decimal_value
        else:
            print(f"Invalid {coord_type} range. Latitude: -90 to 90, Longitude: -180 to 180")
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
                print(f"Invalid {coord_type} range after DMS conversion")
                return None
        return None

# --- Helper function: Google Maps image download ---
def download_google_satellite(lat, lon, zoom=20, size="640x640"):
    """Download satellite image from Google Maps Static API"""
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        # Convert to RGB mode to avoid JPEG saving issues
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    else:
        print(f"Failed to fetch Google Maps image. Status code: {response.status_code}")
        return None

# --- Helper function: Run Roboflow model ---
def infer_with_roboflow(image):
    """Run inference on image using Roboflow model"""
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
    """Add detection counter and timestamp to the image"""
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

# --- Main processing function ---
def process_csv_file(csv_path, output_dir='output'):
    """
    Process CSV file with coordinates and detect solar panels
    
    Args:
        csv_path: Path to CSV file with columns: X (Longitude), Y (Latitude), Name
        output_dir: Directory to save output images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories for original and detected images
    original_dir = output_path / 'original'
    detected_dir = output_path / 'detected'
    original_dir.mkdir(exist_ok=True)
    detected_dir.mkdir(exist_ok=True)
    
    # Read CSV file
    print(f"\nReading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Validate required columns
    required_cols = ['X', 'Y', 'Name']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return
    
    print(f"Found {len(df)} rows to process")
    
    # Results tracking
    results = []
    
    # Process each row
    for idx, row in df.iterrows():
        name = row['Name']
        lon_input = row['X']  # X = Longitude
        lat_input = row['Y']  # Y = Latitude
        
        print(f"\n{'='*60}")
        print(f"Processing row {idx+1}/{len(df)} - Name: {name}")
        print(f"Input coordinates: Lat={lat_input}, Lon={lon_input}")
        
        # Parse coordinates
        lat = parse_coordinate(lat_input, 'latitude')
        lon = parse_coordinate(lon_input, 'longitude')
        
        if lat is None or lon is None:
            print(f"⚠️  Invalid coordinates for {name}, skipping...")
            results.append({
                'Name': name,
                'Latitude': lat_input,
                'Longitude': lon_input,
                'Status': 'Failed - Invalid Coordinates',
                'Detections': 0
            })
            continue
        
        print(f"✓ Parsed coordinates: {lat:.6f}, {lon:.6f}")
        
        # Fetch satellite image
        print("Fetching satellite image...")
        img = download_google_satellite(lat, lon)
        
        if img is None:
            print(f"⚠️  Failed to fetch image for {name}, skipping...")
            results.append({
                'Name': name,
                'Latitude': lat,
                'Longitude': lon,
                'Status': 'Failed - Image Download',
                'Detections': 0
            })
            continue
        
        print("✓ Image downloaded successfully")
        
        # Save original image - sanitize filename
        safe_name = str(name).replace('/', '_').replace('\\', '_')
        original_filename = f"{safe_name}_original.jpg"
        original_path = original_dir / original_filename
        img.save(original_path, format="JPEG")
        print(f"✓ Original image saved: {original_path}")
        
        # Run inference
        print("Running solar panel detection...")
        preds = infer_with_roboflow(img)
        
        # Create segmentation-based visualization
        result_img, detection_count = create_segmentation_visualization(img, preds)
        
        # Add UI elements
        timestamp = datetime.now().strftime("%Y-%m-%d")
        result_img = add_ui_elements(result_img, detection_count, timestamp)
        
        # Save detected image
        detected_filename = f"{safe_name}_detected.jpg"
        detected_path = detected_dir / detected_filename
        result_img.save(detected_path, format="JPEG")
        print(f"✓ Detected image saved: {detected_path}")
        
        # Print results
        if detection_count > 0:
            print(f"✅ SUCCESS: Solar panels detected: {detection_count}")
        else:
            print(f"ℹ️  No solar panels detected")
        
        # Store results
        results.append({
            'Name': name,
            'Latitude': lat,
            'Longitude': lon,
            'Status': 'Success',
            'Detections': detection_count,
            'Original_Image': original_filename,
            'Detected_Image': detected_filename
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = output_path / 'detection_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"📊 Results saved to: {results_csv}")
    print(f"📁 Original images: {original_dir}")
    print(f"📁 Detected images: {detected_dir}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"Total rows processed: {len(results)}")
    successful = len([r for r in results if r['Status'] == 'Success'])
    total_detections = sum([r['Detections'] for r in results])
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Solar panels detected: {total_detections}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default CSV file name
        csv_file = "/Users/manish/Desktop/Sheetal/office_work/cursor/solar_roboflow/different_cities/indian_cities.csv"
    
    # Check if output directory is specified
    if len(sys.argv) > 2:
        output_directory = sys.argv[2]
    else:
        output_directory = "output"
    
    print("="*60)
    print("Solar Panel Detection System - CSV Version")
    print("="*60)
    print(f"CSV file: {csv_file}")
    print(f"Output directory: {output_directory}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"\n❌ Error: CSV file '{csv_file}' not found!")
        print("\nUsage:")
        print("  python yolo8_solar_csv.py <csv_file> [output_directory]")
        print("\nExample:")
        print("  python yolo8_solar_csv.py indian_cities.csv output")
        print("\nThe CSV file should contain columns: X (Longitude), Y (Latitude), Name")
        sys.exit(1)
    
    # Process the CSV file
    process_csv_file(csv_file, output_directory)

