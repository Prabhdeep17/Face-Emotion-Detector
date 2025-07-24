import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Load model
model = YOLO("runs/train/emotion_yolov8n6/weights/best.pt")

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("📸 Emotion Detector")
st.write("Upload a photo and I’ll tell you the detected emotion.")

# Upload image
image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Save image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img = Image.open(image_file)
        img.save(tmp.name)
        tmp_path = tmp.name

    # Run YOLOv8 detection
    results = model(tmp_path)

    # Display image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Show detected emotions
    st.subheader("🎯 Detected Emotion(s):")
    found = False
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        st.success(label)
        found = True

    if not found:
        st.warning("No emotion detected.")
