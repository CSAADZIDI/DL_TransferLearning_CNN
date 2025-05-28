import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# Text-to-speech
import pyttsx3
from gtts import gTTS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# Load your model (update this path)
model = tf.keras.models.load_model("../models/model_densenet.keras")

# Set last conv layer name for DenseNet121
last_conv_layer_name = "conv5_block16_concat"  # DenseNet121 example

# Grad-CAM function
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([[img_array]]) 
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlayed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlayed_img

# Local TTS (offline)
def speak_local(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# Web TTS with gTTS (streaming in browser)
def speak_web(message):
    tts = gTTS(text=message, lang='en')
    tts.save("message.mp3")
    audio_file = open("message.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")
    audio_file.close()
    os.remove("message.mp3")

# Streamlit UI
st.title("🩺 Détection de Pneumonie sur Radiographies Thoraciques (avec Grad-CAM & Rétroaction vocale)")

uploaded_file = st.file_uploader("Choisir une image radiographie thoracique", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", width=200)

    # Preprocess for model
    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Prediction
    prediction_proba = model.predict(img_input)[0][0]
    label = "PNEUMONIA" if prediction_proba > 0.5 else "NORMAL"
    confidence = float(np.max(prediction_proba))

    #st.write(f"### Prediction: `{label}`")
    #st.write(f"Confidence: `{confidence:.2f}`")

    message = f"Le diagnostic est {label} avec {confidence:.0%} confiance."

    # Diagnostic Report Section
    st.markdown("## 📋 Diagnostic")
    st.markdown(f"**Diagnosis:** `{label}`")
    st.markdown(f"**Confiance:** `{confidence:.2%}`")
    # Vocal feedback (local + web)
    speak_local(message)  # Works offline on your machine
    speak_web(message)    # Plays audio in browser (works in cloud)

    # Explanation text to help verify correctness
    if label == "PNEUMONIA":
        attention_text = (
            "⚠️ Le modèle s'est fortement activé dans des zones pouvant correspondre à des «infiltrats ou à des opacités pulmonaires»," 
            "généralement associées à une pneumonie."
        )
  
    st.markdown(f"** {attention_text} **")

    # Grad-CAM heatmap visualization
    heatmap = get_gradcam_heatmap(model, img_input, last_conv_layer_name)
    overlayed = overlay_heatmap(img_resized, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Originale X-ray", use_container_width=True)
    with col2:
        st.image(overlayed, caption="Zones Grad-CAM Heatmap", use_container_width=True)



