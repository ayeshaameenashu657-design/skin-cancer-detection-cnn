
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

REPO_ID = "Ayesha-Ameen/skin-cancer-detection-cnn"

CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="skin_cancer_model.keras"
    )
    return tf.keras.models.load_model(model_path)

st.title("Skin Cancer Detection Using CNN")
st.write("Upload a skin image to get a prediction.")

st.warning(
    "This project is for educational purposes only. "
    "It is not a medical diagnostic tool."
)

try:
    model = load_model()
    uploaded_file = st.file_uploader(
        "Upload Skin Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image",
                 use_container_width=True)

        if st.button("Predict"):
            resized_image = image.resize((224, 224))
            image_array = np.array(
                resized_image, dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(image_array, axis=0)
            predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index]) * 100

            st.subheader("Prediction Result")
            st.success(CLASS_NAMES[predicted_index])
            st.write(f"Confidence: {confidence:.2f}%")

except Exception as e:
    st.error("Model loading failed. Please check the error below.")
    st.exception(e)
