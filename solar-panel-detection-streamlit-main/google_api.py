import os
import streamlit as st
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from roboflow import Roboflow

# --- Initialize Roboflow ---
# Replace with your API key and project details
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
    import tempfile
    import os
    
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


# --- Helper function: Draw predictions based on mode ---
def draw_predictions(image, predictions, mode="Bounding Boxes"):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for pred in predictions["predictions"]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2
        
        if mode == "Bounding Boxes":
            # Draw bounding box
            draw.rectangle([left, top, right, bottom], outline="red", width=8)
            # Add label above the box
            draw.text((left, top - 20), f"{pred['class']} {pred['confidence']:.2f}", fill="red")
            
        elif mode == "Segmentation Masks":
            # Check if segmentation data is available
            if "segmentation" in pred and pred["segmentation"]:
                points = pred["segmentation"]
                if isinstance(points, list) and len(points) > 0:
                    # Convert points to tuples for drawing
                    if isinstance(points[0], dict):
                        point_tuples = [(point["x"], point["y"]) for point in points]
                    else:
                        point_tuples = points
                    # Draw filled polygon for segmentation mask
                    draw.polygon(point_tuples, fill=(255, 0, 0, 128), outline="red")
                    # Add label
                    draw.text((x, y - 10), f"{pred['class']} {pred['confidence']:.2f}", fill="red")
            else:
                # Fallback to bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=8)
                draw.text((left, top - 20), f"{pred['class']} {pred['confidence']:.2f}", fill="red")
                
        elif mode == "Labels Only":
            # Just add labels without boxes
            draw.text((x, y), f"{pred['class']} {pred['confidence']:.2f}", fill="red", font=None)

    return img


# --- Streamlit UI ---
st.title("🔆 Solar Panel Detection using Roboflow + Google Maps")

# Visualization mode selector
viz_mode = st.selectbox("Visualization Mode:", ["Bounding Boxes", "Segmentation Masks", "Labels Only"])

mode = st.radio("Choose input method:", ["Upload Image", "Latitude/Longitude + Google Maps API"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image")

        preds = infer_with_roboflow(img)
        out_img = draw_predictions(img, preds, viz_mode)
        st.image(out_img, caption="Detected Solar Panels ✅")

elif mode == "Latitude/Longitude + Google Maps API":
    lat = st.text_input("Latitude", "")
    lon = st.text_input("Longitude", "")
    gapi_key = st.text_input("Google Maps API Key", type="password")

    if st.button("Fetch & Detect"):
        if lat and lon and gapi_key:
            img = download_google_satellite(lat, lon, gapi_key)
            if img:
                st.image(img, caption="Satellite Image")

                preds = infer_with_roboflow(img)
                out_img = draw_predictions(img, preds, viz_mode)
                st.image(out_img, caption="Detected Solar Panels ✅")
        else:
            st.warning("⚠️ Please enter latitude, longitude, and API key.")
