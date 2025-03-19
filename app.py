import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
MODEL_PATH = "your_trained_model.pt"  # Ganti dengan path model yang benar
model = YOLO(MODEL_PATH)

# Function to calculate defect area (adjust pixel_to_mm2 as needed)
def calculate_defect_area(mask, pixel_to_mm2=1.0):
    return np.sum(mask > 0.5) * pixel_to_mm2

# Layout Configuration
st.set_page_config(layout="wide")  # Ensure full width layout

# Sidebar for settings
with st.sidebar:
    with st.expander("⚙️ Advanced Settings", expanded=False):
        confidence_threshold = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    with st.expander("🖥 Display Settings", expanded=False):
        transparency = st.slider("Select Mask Transparency", 0.0, 1.0, 0.50, 0.05)

# Main content layout
col1, col2 = st.columns([3, 1])

with col1:
    # Streamlit UI
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Defect Detection in Leather by Instance Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>YOLOv11m-seg</h2>", unsafe_allow_html=True)
    st.write("Upload an image and let the model detect fabric defects and calculate defect areas.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded image temporarily
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the uploaded image
        image = cv2.imread(temp_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        # Run YOLO model for segmentation
        st.write("Detecting defects with instance segmentation...")
        results = model(temp_image_path, conf=confidence_threshold)  # Adjust confidence threshold dynamically

        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_rgb)
        
        total_defect_area = 0

        # Overlay segmentation masks and bounding boxes
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()  # Convert segmentation masks to numpy
                
                for i, mask in enumerate(masks):
                    # Resize mask to match image size
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    
                    # Get confidence score
                    conf = float(r.boxes.conf[i])
                    
                    # Filter out defects below confidence threshold
                    if conf < confidence_threshold:
                        continue
                    
                    # Calculate defect area (adjust pixel_to_mm2 as needed)
                    defect_area = calculate_defect_area(mask_resized, pixel_to_mm2=1.0)
                    total_defect_area += defect_area
                    
                    # Apply mask overlay
                    colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
                    colored_mask[mask_resized > 0.5] = (255, 0, 0)  # Red mask
                    image_rgb = cv2.addWeighted(image_rgb, 1, colored_mask, transparency, 0)
                    
                ax.imshow(image_rgb)

        # Draw bounding boxes and confidence scores
        for box, conf in zip(r.boxes, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(conf)
            
            # Skip if confidence is below threshold
            if conf < confidence_threshold:
                continue
            
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='yellow', linewidth=2))
            ax.text(x1, y1 - 5, f"Defect: {conf:.2f}", color="yellow", fontsize=12, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5))

        # Show the final image with segmentation
        ax.axis("off")
        st.pyplot(fig)
        
        # Display total defect area
        st.write(f"**Total Defect Area: {total_defect_area:.2f} square mm** (adjust scale as needed)")

        # Clean up temporary file
        os.remove(temp_image_path)

with col2:
    # Display Metrodata logo in the designated area
    metrodata_logo_url = "https://github.com/husnanali05/FP_Datmin/raw/main/9NPV97S5WSDBSEKB48SP63YNKWGAFFVTJ52QGYKZ-5eabd8d4%20(1).png"
    st.image(metrodata_logo_url, use_container_width=True)