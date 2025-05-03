import streamlit as st
from PIL import Image
from src import Detect
from dotenv import load_dotenv
import os

load_dotenv()  

artifact_url = os.getenv("ARTIFACT_URL_MODEL") # Define variables with .env or add path here
model_path = os.getenv('MODELS_FOLDER') # Define variables with .env or add path here

st.title("Orange Detection")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])


if uploaded_file is not None:
    img_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded image", use_column_width=True)

    with st.spinner("Loading the model..."):
        orange = Detect(model_path, img_path)
        label = orange.pred()
        st.success(f"Prediction: {label}")
