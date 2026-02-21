import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

# --------------------------------
# Add fusion model path
# --------------------------------
sys.path.append("../deep_learning/phase5_fusion")
from fusion_model import FusionModel

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="AlzClassNet - Alzheimer Detection",
    layout="centered",
    page_icon="🧠"
)

# --------------------------------
# Constants
# --------------------------------
IMG_SIZE = 224
CLASS_NAMES = [
    "Non Demented",
    "Very Mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

CNN_WEIGHTS = "../deep_learning/phase5_fusion/cnn_mri_4class_baseline.pth"
VIT_WEIGHTS = "../deep_learning/phase5_fusion/vit_mri_4class_baseline.pth"
FUSION_WEIGHTS = "../deep_learning/phase5_fusion/fusion_model.pth"

# --------------------------------
# Device
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Load model (cached)
# --------------------------------
@st.cache_resource
def load_model():
    model = FusionModel(CNN_WEIGHTS, VIT_WEIGHTS)
    model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --------------------------------
# Image transform
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------
# UI HEADER
# --------------------------------
st.title("🧠 AlzClassNet")
st.subheader("CNN–Vision Transformer Fusion for Alzheimer’s Disease Detection")

st.markdown(
    """
    Upload a **brain MRI image** to predict the **Alzheimer’s disease stage**.
    This system uses a **CNN–ViT fusion deep learning model** trained on MRI data.
    """
)

st.divider()

# --------------------------------
# IMAGE UPLOAD
# --------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_column_width=True
    )

    st.markdown("### 🔍 Prediction")

    if st.button("Predict Alzheimer’s Stage"):
        with st.spinner("Analyzing MRI image..."):
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]

        st.success("Prediction completed successfully ✅")

        # --------------------------------
        # RESULT DISPLAY
        # --------------------------------
        pred_idx = torch.argmax(probs).item()
        pred_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item() * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Stage",
                value=pred_label
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        # --------------------------------
        # PROBABILITY BAR CHART
        # --------------------------------
        st.markdown("### 📊 Class-wise Confidence")

        prob_dict = {
            CLASS_NAMES[i]: float(probs[i] * 100)
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(prob_dict)

        st.info(
            "⚠️ This tool is intended for research and educational purposes only "
            "and should not be used as a sole clinical diagnostic system."
        )

else:
    st.info("Please upload an MRI image to start prediction.")

st.divider()

# --------------------------------
# FOOTER
# --------------------------------
st.caption(
    "© AlzClassNet | CNN–ViT Fusion Framework for Alzheimer’s Detection"
)
