import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.title("🐾 Cats vs Dogs Classifier")

@st.cache_resource
def get_model():
    return load_model("model.h5", compile=False)

model = get_model()

with open("labels.txt") as f:
    class_names = [line.strip() for line in f]

uploaded = st.file_uploader("Upload a cat or dog photo", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300, caption="Uploaded Image")

    img = ImageOps.fit(img, (224,224), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32")
    arr = (arr / 127.5) - 1  # normalize
    batch = np.expand_dims(arr, 0)

    preds = model.predict(batch)[0]
    idx = np.argmax(preds)
    label = class_names[idx]
    conf = preds[idx]

    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")
