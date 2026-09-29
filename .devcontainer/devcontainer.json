import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Crop Doctor AI",
    page_icon="🌱",
    layout="centered"
)


# ==================================================
# TITLE
# ==================================================

st.title("🌱 Crop Doctor AI")

st.write(
    "AI-based preliminary crop health screening"
)


# ==================================================
# MODEL SETTINGS
# ==================================================

MODEL_ID = "ayerr/plant-disease-classification"

MODEL_SUBFOLDER = "ayerr/plant-disease-classification"


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID,
        subfolder=MODEL_SUBFOLDER
    )

    disease_model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        subfolder=MODEL_SUBFOLDER
    )

    disease_model.eval()

    return processor, disease_model


# Load AI model

try:

    processor, disease_model = load_model()

    st.success("✅ AI model loaded successfully!")

except Exception as e:

    st.error("❌ AI model could not be loaded.")

    st.exception(e)

    st.stop()


# ==================================================
# LANGUAGE SELECTION
# ==================================================

language = st.radio(
    "Choose Language / மொழியை தேர்வு செய்யவும்",
    ["English", "Tamil"]
)


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📷 Upload Crop Image / பயிர் படத்தை Upload செய்யவும்",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# IMAGE PROCESSING
# ==================================================

if uploaded_file is not None:

    # Open image

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image

    st.image(
        image,
        caption="Uploaded Crop Image",
        use_container_width=True
    )


    # ==================================================
    # ANALYZE BUTTON
    # ==================================================

    if st.button(
        "🔍 Analyze Crop",
        type="primary"
    ):

        try:

            with st.spinner(
                "🔬 Analyzing crop..."
            ):

                # ------------------------------------------
                # PROCESS IMAGE
                # ------------------------------------------

                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )


                # ------------------------------------------
                # AI PREDICTION
                # ------------------------------------------

                with torch.no_grad():

                    outputs = disease_model(
                        **inputs
                    )


                # ------------------------------------------
                # PROBABILITIES
                # ------------------------------------------

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )


                # ------------------------------------------
                # BEST PREDICTION
                # ------------------------------------------

                predicted_class = torch.argmax(
                    probabilities,
                    dim=-1
                ).item()


                # ------------------------------------------
                # LABEL
                # ------------------------------------------

                label = disease_model.config.id2label[
                    predicted_class
                ]


                # ------------------------------------------
                # CONFIDENCE
                # ------------------------------------------

                confidence = probabilities[
                    0,
                    predicted_class
                ].item()


            # ==================================================
            # RESULT
            # ==================================================

            st.divider()

            st.subheader(
                "🌱 Analysis Result"
            )


            # ==================================================
            # ENGLISH RESULT
            # ==================================================

            if language == "English":

                if label.lower() == "healthy":

                    st.success(
                        "✅ The plant appears healthy."
                    )

                    st.write(
                        "🌱 Continue regular monitoring."
                    )

                else:

                    st.warning(
                        "⚠️ Possible disease symptoms detected."
                    )

                    st.write(
                        "📋 Please consult an agricultural "
                        "expert for confirmation."
                    )


                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


            # ==================================================
            # TAMIL RESULT
            # ==================================================

            else:

                if label.lower() == "healthy":

                    st.success(
                        "✅ செடி ஆரோக்கியமாக இருப்பது போல் தெரிகிறது."
                    )

                    st.write(
                        "🌱 தொடர்ந்து செடியை கண்காணிக்கவும்."
                    )

                else:

                    st.warning(
                        "⚠️ செடியில் நோய் அறிகுறிகள் இருக்கலாம்."
                    )

                    st.write(
                        "📋 வேளாண்மை நிபுணரிடம் பரிசோதனை "
                        "செய்து உறுதி செய்யவும்."
                    )


                st.metric(
                    "நம்பகத்தன்மை",
                    f"{confidence * 100:.2f}%"
                )


            # ==================================================
            # TOP PREDICTIONS
            # ==================================================

            st.divider()

            st.subheader(
                "🔎 Top Predictions"
            )


            top_probabilities, top_indices = torch.topk(
                probabilities[0],
                k=min(
                    2,
                    probabilities.shape[-1]
                )
            )


            for probability, index in zip(
                top_probabilities,
                top_indices
            ):

                prediction_name = (
                    disease_model.config.id2label[
                        index.item()
                    ]
                )

                percentage = (
                    probability.item() * 100
                )


                st.write(
                    f"**{prediction_name}** — "
                    f"{percentage:.2f}%"
                )


                st.progress(
                    float(
                        probability.item()
                    )
                )


            # ==================================================
            # DISCLAIMER
            # ==================================================

            st.info(
                "📋 This AI provides preliminary screening only. "
                "It is not a guaranteed diagnosis. "
                "For confirmation, consult an agricultural expert."
            )


        except Exception as e:

            st.error(
                "❌ Analysis failed."
            )

            st.exception(e)


else:

    st.info(
        "📷 Please upload a crop image to begin."
    )
