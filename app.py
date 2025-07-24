import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("📸 Face Emotion Detector")
st.write("Upload a photo and I’ll try to tell the detected emotion.")

# Upload image
image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    try:
        # Save image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img = Image.open(image_file).convert("RGB")
            img.save(tmp.name)
            tmp_path = tmp.name

        # Run detection
        results = model(tmp_path)

        # Show uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Extract and display emotions
        st.subheader("🎯 Detected Emotion(s):")
        found = False
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            st.success(label)
            found = True

        if not found:
            st.warning("Oops! No emotion detected.")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
