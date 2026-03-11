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

# TensorFlow check
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

    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            return None

        try:
            # Load full trained model
            model = tf.keras.models.load_model(MODEL_PATH)
            return model

        except Exception:
            # Fallback architecture
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                input_shape=(224,224,3),
                weights=None
            )

            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)

            predictions = tf.keras.layers.Dense(3, activation="softmax")(x)

            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            model.load_weights(MODEL_PATH)

            return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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
def preprocess_image(img: Image.Image):

    img_resized = img.resize((224,224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    return img_array


# -------------------------
# Grad-CAM
# -------------------------
def get_gradcam(img_array, model, pred_class_index):
    """Generate Grad-CAM heatmap with debugging"""
    try:
        # Debug: Print model summary info
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Get the last convolutional layer
        last_conv_layer = None
        for i, layer in enumerate(reversed(model.layers)):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                print(f"Found Conv2D layer: {layer.name}")
                break
        
        if last_conv_layer is None:
            st.warning("No convolutional layer found for Grad-CAM")
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        print(f"Grad model outputs: {grad_model.outputs}")
        
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions value: {predictions.numpy()}")
            
            # Handle predictions based on shape
            if len(predictions.shape) == 1:
                # Single output
                pred_class = predictions[pred_class_index]
            elif predictions.shape[-1] > 1:
                # Multiple outputs (classification)
                pred_class = predictions[:, pred_class_index]
            else:
                # Single output but with batch dimension
                pred_class = predictions[:, 0]
            
            print(f"Selected class: {pred_class}")
        
        # Calculate gradients
        grads = tape.gradient(pred_class, conv_outputs)
        print(f"Gradients shape: {grads.shape if grads is not None else 'None'}")
        
        if grads is None:
            st.warning("Gradients are None - cannot generate Grad-CAM")
            return None
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        )
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
        
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {str(e)}")
        print(f"Grad-CAM error details: {str(e)}")  # This will show in logs
        import traceback
        traceback.print_exc()  # Print full traceback
        return None


# -------------------------
# Heatmap explanation
# -------------------------
def heatmap_explanation():
    return [
        ("red","Very high activation: strongest influence on prediction"),
        ("yellow","Strong activation"),
        ("green","Moderate activation"),
        ("blue","Low activation"),
    ]


# -------------------------
# Main App
# -------------------------
def main():

    st.title("🫁 Sheep Lung Image Classification App")

    st.write(
        "Upload a lung image to classify it into **Healthy, Inflammatory, or Neoplastic** "
        "and visualize the regions influencing the model prediction."
    )

    uploaded_file = st.file_uploader(
        "Choose a lung image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is None:
        st.info("👆 Upload a lung image to start")
        return

    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Cannot open image: {str(e)}")
        return

    if model is None:
        st.error("Model failed to load.")
        return

    img_array = preprocess_image(img)

    preds = model.predict(img_array, verbose=0)[0]

    pred_class_index = np.argmax(preds)

    pred_class = class_names[pred_class_index]

    confidence = np.max(preds) * 100

    prediction_color = class_colors[pred_class_index]

    # -------------------------
    # Top row
    # -------------------------

    col_img, col_pred = st.columns([1.3,1])

    with col_img:

        st.subheader("🖼️ Uploaded Image")

        st.image(np.array(img), use_column_width=True)

    with col_pred:

        st.subheader("📊 Prediction Confidence")

        fig, ax = plt.subplots(figsize=(5,4))

        ax.barh(class_names, preds*100, color=class_colors)

        ax.set_xlim([0,100])

        ax.set_xlabel("Probability (%)")

        for i,v in enumerate(preds*100):
            ax.text(v+1, i, f"{v:.2f}%", va="center")

        st.pyplot(fig)

        st.markdown(
            f"<h2 style='color:{prediction_color}'>Prediction: {pred_class}</h2>",
            unsafe_allow_html=True
        )

        st.markdown(f"**Confidence:** {confidence:.2f}%")

    # -------------------------
    # GradCAM
    # -------------------------

    col_heatmap, col_interpret = st.columns([1.3,1])

    with col_heatmap:

        st.subheader("🔥 Grad-CAM Visualization")

        heatmap = get_gradcam(img_array, model, pred_class_index)

        if heatmap is not None:

            heatmap_resized = cv2.resize(heatmap, (img.width, img.height))

            heatmap_resized = np.uint8(255 * heatmap_resized)

            heatmap_resized = cv2.applyColorMap(
                heatmap_resized,
                cv2.COLORMAP_JET
            )

            img_np = np.array(img)

            superimposed = cv2.addWeighted(
                img_np,
                0.6,
                heatmap_resized,
                0.4,
                0
            )

            st.image(superimposed, use_column_width=True)

        else:
            st.warning("Grad-CAM could not be generated")

    with col_interpret:

        st.subheader("📝 Heatmap Interpretation")

        for color, text in heatmap_explanation():
            st.markdown(f"- <span style='color:{color}'>●</span> {text}", unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "🔬 Research tool for sheep lung pathology classification."
    )


if __name__ == "__main__":
    main()
