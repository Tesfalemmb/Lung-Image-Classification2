import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Sheep Lung Classification App",
    page_icon="🫁",
    layout="wide"
)

# -------------------------
# TensorFlow check
# -------------------------
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not available. Please check requirements.txt.")

MODEL_PATH = "lung2026_classification_model_efficientnetb0.h5"

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():

    if not TENSORFLOW_AVAILABLE:
        return None

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


model = load_model()

# -------------------------
# Class definitions
# -------------------------
class_names = ["Healthy", "Inflammatory", "Neoplastic"]
class_colors = ["green", "orange", "red"]

# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(img):

    img = img.resize((224,224))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    return img_array


# -------------------------
# Grad-CAM
# -------------------------
def get_gradcam(img_array, model, class_index):

    last_conv_layer = None

    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions,(list,tuple)):
            predictions = predictions[0]

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap,0)

    heatmap /= (np.max(heatmap)+1e-8)

    return heatmap.numpy()


# -------------------------
# Heatmap overlay
# -------------------------
def overlay_heatmap(img, heatmap):

    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.65,heatmap,0.35,0)

    return overlay


# -------------------------
# Pathology explanation
# -------------------------
def pathology_explanation(label):

    explanations = {

        "Healthy":
        """
        Lung tissue appears **normal**.

        No visible pathological structures were detected by the model.
        """,

        "Inflammatory":
        """
        The model detected **patterns consistent with inflammation**.

        Possible causes include:

        • pneumonia  
        • inflammatory infiltration  
        • infectious lesions
        """,

        "Neoplastic":
        """
        Suspicious **tumor-like tissue patterns** detected.

        Possible neoplastic lesions may include abnormal cell growth.
        Veterinary histopathology is recommended.
        """
    }

    return explanations.get(label,"No explanation available")


# -------------------------
# Probability chart
# -------------------------
def probability_chart(preds):

    fig, ax = plt.subplots(figsize=(5,4))

    ax.barh(class_names, preds*100, color=class_colors)

    ax.set_xlim([0,100])
    ax.set_xlabel("Probability (%)")

    for i,v in enumerate(preds*100):
        ax.text(v+1, i, f"{v:.2f}%", va="center")

    return fig


# -------------------------
# Main App
# -------------------------
def main():

    st.title("🫁 Sheep Lung Pathology Classification")

    st.markdown(
    """
    AI system for classifying sheep lung pathology into:

    • Healthy  
    • Inflammatory  
    • Neoplastic  

    Includes **Grad-CAM explainability** for visual interpretation.
    """
    )

    uploaded_files = st.file_uploader(
        "Upload lung images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if uploaded_files is None or len(uploaded_files)==0:
        st.info("👆 Upload one or more lung images to start")
        return

    for file in uploaded_files:

        img = Image.open(file).convert("RGB")

        img_array = preprocess_image(img)

        preds = model.predict(img_array, verbose=0)[0]

        pred_index = np.argmax(preds)

        pred_class = class_names[pred_index]

        confidence = np.max(preds)*100

        color = class_colors[pred_index]

        st.markdown("---")

        col1,col2,col3 = st.columns([1.3,1,1])

        # ---------------- Image ----------------
        with col1:

            st.subheader("🖼️ Lung Image")

            st.image(img, use_column_width=True)

        # ---------------- Prediction ----------------
        with col2:

            st.subheader("📊 Prediction")

            st.pyplot(probability_chart(preds))

            st.markdown(
                f"<h3 style='color:{color}'>Prediction: {pred_class}</h3>",
                unsafe_allow_html=True
            )

            st.write(f"Confidence: **{confidence:.2f}%**")

        # ---------------- Explanation ----------------
        with col3:

            st.subheader("🧠 Pathology Explanation")

            st.info(pathology_explanation(pred_class))

        # ---------------- GradCAM ----------------
        st.subheader("🔥 Grad-CAM Visualization")

        heatmap = get_gradcam(img_array, model, pred_index)

        if heatmap is not None:

            overlay = overlay_heatmap(np.array(img), heatmap)

            st.image(overlay, use_column_width=True)

        else:

            st.warning("Grad-CAM could not be generated")

    st.markdown("---")

    st.caption(
        "Research tool for sheep lung disease classification using EfficientNet."
    )


if __name__ == "__main__":
    main()
