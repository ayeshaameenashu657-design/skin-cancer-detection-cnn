
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

REPO_ID = "Ayesha-Ameen/skin-cancer-detection-cnn"

@st.cache_resource
def load_model_and_classes():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="skin_cancer_model.keras"
    )

    model = tf.keras.models.load_model(model_path)

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names

st.title("Skin Cancer Detection Using CNN")

st.write("Upload a skin image to get a model prediction.")

st.warning(
    "Educational project only. This model is not a medical diagnostic tool."
)

try:
    model, class_names = load_model_and_classes()

    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Skin Image",
            use_container_width=True
        )

        if st.button("Predict"):
            resized_image = image.resize((224, 224))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.subheader("Prediction Result")
            st.write("Predicted Class:", predicted_class)
            st.write(f"Model Confidence: {confidence:.2f}%")

except Exception as e:
    st.error("Model loading failed.")
    st.exception(e)
