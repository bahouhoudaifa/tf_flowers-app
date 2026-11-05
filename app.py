import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
IMG_SIZE = (180, 180)
# Même ordre que metadata.features["label"].names dans TFDS
CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]
CLASS_NAMES_FR = {
    "dandelion": "Pissenlit(dandelion) 🌼",
    "daisy": "Marguerite(daisy) 🌼",
    "tulips": "Tulipe(tulips) 🌷",
    "sunflowers": "Tournesol(sunflowers) 🌻",
    "roses": "Rose(roses) 🌹",
}

st.set_page_config(page_title="Classification de fleurs", page_icon="🌸", layout="centered")


# ------------------------------------------------------------------
# Chargement du modèle (mis en cache pour ne le charger qu'une fois)
# ------------------------------------------------------------------
def build_model():
    """Reconstruit exactement l'architecture du notebook."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ], name="data_augmentation")

    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(180, 180, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(255.0)(x)
    x = layers.Lambda(preprocess_input)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    return models.Model(inputs, outputs)


@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("model_transfer.weights.h5")
    return model




model = load_model()


# ------------------------------------------------------------------
# Prétraitement (identique au notebook)
# ------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    arr = np.array(img)
    arr = tf.image.resize(arr, IMG_SIZE)          # redimensionnement 180x180
    arr = tf.cast(arr, tf.float32) / 255.0        # [0,255] -> [0,1]
    return tf.expand_dims(arr, axis=0).numpy()    # (1, 180, 180, 3)


def predict(pil_image: Image.Image):
    batch = preprocess_image(pil_image)
    probs = model.predict(batch, verbose=0)[0]
    return probs


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
st.title("🌸 Classification de fleurs")
st.write(
    "Charge une photo (ou prends-en une) et le modèle **MobileNetV2** "
    "(transfer learning) devine l'espèce parmi 5 classes."
)

with st.sidebar:
    st.header("À propos")
    st.write("Dataset : **TF Flowers** (3670 images)")
    st.write("Classes : " + ", ".join(CLASS_NAMES_FR.values()))
    threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)

source = st.radio("Source de l'image", ["📁 Fichier", "📷 Caméra"], horizontal=True)

if source == "📁 Fichier":
    uploaded = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])
else:
    uploaded = st.camera_input("Prends une photo")

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image analysée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        probs = predict(image)

    best_idx = int(np.argmax(probs))
    best_class = CLASS_NAMES[best_idx]
    confidence = float(probs[best_idx])

    with col2:
        if confidence >= threshold:
            st.success(f"**{CLASS_NAMES_FR[best_class]}**")
        else:
            st.warning(f"Incertain : **{CLASS_NAMES_FR[best_class]}**")
        st.metric("Confiance", f"{confidence * 100:.1f} %")

    st.subheader("Probabilités par classe")
    df = pd.DataFrame({
        "Classe": [CLASS_NAMES_FR[c] for c in CLASS_NAMES],
        "Probabilité": probs,
    }).set_index("Classe")
    st.bar_chart(df)
else:
    st.info("👆 Ajoute une image pour lancer la prédiction.")