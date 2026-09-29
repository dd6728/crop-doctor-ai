import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# Page settings
st.set_page_config(
    page_title="Crop Doctor AI",
    page_icon="🌱",
    layout="centered"
)

# Title
st.title("🌱 Crop Doctor AI")
st.write("Upload a crop image for preliminary plant-health screening.")

# Model
MODEL_ID = "ayerr/plant-disease-classification"

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID,
        subfolder="ayerr/plant-disease-classification"
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        subfolder="ayerr/plant-disease-classification"
    )

    model.eval()
    return processor, model

processor, model = load_model()

# Language selection
language = st.radio(
    "Choose Language / மொழியை தேர்வு செய்யவும்",
    ["English", "Tamil"]
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload Crop Image / பயிர் படத்தை Upload செய்யவும்",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Crop Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Crop"):

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

       
