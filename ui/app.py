import streamlit as st
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.transform import resize
import tempfile
import os

IMG_SIZE = (64, 64)
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# Load model
@st.cache_resource
def load_prediction_model():
    model = None  # todo load model
    return model


# Preprocessing
def process_input(file_path):
    # Read 13-band image
    img = tiff.imread(file_path)

    # Resize to (64, 64, 13)
    img_resized = resize(
        img, (IMG_SIZE[0], IMG_SIZE[1], 13), preserve_range=True, anti_aliasing=True
    )

    # Normalizations
    img_resized = img_resized.astype(np.float32)
    if img_resized.max() > 0:
        img_resized /= img_resized.max()

    # Add batch dimension: (1, 64, 64, 13)
    return np.expand_dims(img_resized, axis=0), img_resized


# Visualization helper
def get_rgb_view(img_array):
    """
    Extracts RGB channels for display. Blue=1, Green=2, Red=3
    """
    rgb = img_array[:, :, [3, 2, 1]]

    # Normalize just for display brightness
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    return rgb


# UI
st.title("🌍 Land Type Classifier")
st.write("Upload a **.tif** satellite image (13 bands) to classify the terrain.")

model = load_prediction_model()
uploaded_file = st.file_uploader("Choose a satellite image", type=["tif", "tiff"])

if uploaded_file is not None:
    # Save to temp file (because tifffile needs a path on disk)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Process
        input_tensor, raw_img = process_input(tmp_path)

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Satellite View (RGB)")
            rgb_img = get_rgb_view(raw_img)
            st.image(rgb_img, clamp=True, use_container_width=True)
            st.caption(f"Original Shape: {raw_img.shape}")

        with col2:
            st.subheader("Prediction")

            if model == None:  # todo
                st.warning("Model not loaded. Generating dummy data.")
                # Dummy prediction for testing UI
                probs = np.random.random((1, 10))
                probs = probs / probs.sum()  # Normalize
            else:
                # Real Inference
                probs = model.predict(input_tensor)

            top_idx = np.argmax(probs[0])
            top_class = CLASSES[top_idx]
            confidence = probs[0][top_idx]

            st.success(f"**{top_class}**")
            st.metric("Confidence", f"{confidence:.2%}")

        # chart
        st.markdown("---")
        st.subheader("Confidence Distribution")
        df_chart = pd.DataFrame({"Class": CLASSES, "Probability": probs[0]})
        st.bar_chart(df_chart.set_index("Class"))

    except Exception as e:
        st.error(f"Error processing file: {e}")

    finally:
        os.remove(tmp_path)
