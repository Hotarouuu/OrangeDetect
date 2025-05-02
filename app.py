import streamlit as st
from PIL import Image
from src import Detect
from dotenv import load_dotenv
import os

load_dotenv()  

artifact_url = os.getenv("ARTIFACT_URL_MODEL")
model_path = os.getenv('MODELS_FOLDER')

st.title("Orange Detection")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "png"])


if uploaded_file is not None:
    img_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Imagem carregada", use_column_width=True)

    with st.spinner("Rodando o modelo..."):
        Detect()
        orange = Detect(model_path, artifact_url, img_path)
        label = orange.pred()
        st.success(f"Predição: {label}")
