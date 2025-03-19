import streamlit as st
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

os.system("apt-get update && apt-get install -y libgl1 libglib2.0-0")

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Ensures Streamlit doesn't break
    
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calculate_defect_area(mask, pixel_to_mm2=1.0):
    return np.sum(mask > 0.5) * pixel_to_mm2

def main():
    st.set_page_config(
        page_title="Leather Defect Detection",
        page_icon="https://github.com/husnanali05/FP_Datmin/raw/main/9NPV97S5WSDBSEKB48SP63YNKWGAFFVTJ52QGYKZ-5eabd8d4%20(1).png",
        layout="wide"
    )
    st.sidebar.markdown("""
        <div style='display: flex; justify-content: center;'>
            <img src='https://github.com/husnanali05/FP_Datmin/raw/main/9NPV97S5WSDBSEKB48SP63YNKWGAFFVTJ52QGYKZ-5eabd8d4%20(1).png' width='150'>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("⚙️ Settings")

    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    transparency = st.sidebar.slider("Mask Transparency", 0.0, 1.0, 0.50, 0.05)

    st.markdown(
        """
        <h1 style='text-align: center; color: #1E88E5;'>Leather Defect Detection🔎</h1>
        <h3 style='text-align: center;'>Powered by YOLO Instance Segmentation</h3>
        """, unsafe_allow_html=True
    )

    st.write("Upload an image of leather material to detect defects and estimate their area.")

    model_path = "best.pt"
    model = load_model(model_path)

    if model is not None:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                image = cv2.imread(temp_image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

                results = model(temp_image_path, conf=confidence_threshold)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(image_rgb)
                total_defect_area = 0

                for r in results:
                    if r.masks is not None:
                        masks = r.masks.data.cpu().numpy()
                        for i, mask in enumerate(masks):
                            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                            conf = float(r.boxes.conf[i])
                            if conf < confidence_threshold:
                                continue

                            defect_area = calculate_defect_area(mask_resized, pixel_to_mm2=1.0)
                            total_defect_area += defect_area

                            colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
                            colored_mask[mask_resized > 0.5] = (255, 0, 0)
                            image_rgb = cv2.addWeighted(image_rgb, 1, colored_mask, transparency, 0)

                        ax.imshow(image_rgb)

                for box, conf in zip(r.boxes, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if conf < confidence_threshold:
                        continue
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='yellow', linewidth=2))
                    ax.text(x1, y1 - 5, f"Defect: {conf:.2f}", color="yellow", fontsize=12, fontweight="bold",
                            bbox=dict(facecolor="black", alpha=0.5))

                ax.axis("off")
                st.pyplot(fig)

                st.success(f"Total Defect Area: **{total_defect_area:.2f} square mm** (adjust scale as needed)")

                os.remove(temp_image_path)

    st.markdown("""
        ### ℹ️ About This App
        - This tool detects and highlights defects in leather materials using AI-powered segmentation.
        - Adjust the confidence threshold and transparency settings in the sidebar.
        - The model used is **YOLOv11m-seg**, optimized for precise defect segmentation.
        - Future updates will include automatic defect classification and severity analysis.
    """)

    st.markdown("""
        <hr>
        <p style='text-align: center;'>Developed by AI Research Team | Powered by YOLO</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
