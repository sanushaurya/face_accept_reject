import streamlit as st
import numpy as np
from PIL import Image
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.60  # Accept/Reject threshold

st.title("Face Verification - Accept / Reject")
st.write("Capture with your camera. The system will check if the person is known.")

# =========================
# LOAD MODELS (PICKLE ONLY)
# =========================
@st.cache_resource
def load_models():
    # Load SVM
    with open("models/svm_model.joblib", "rb") as f:
        clf = pickle.load(f)

    # Load Label Encoder
    with open("models/label_encoder.joblib", "rb") as f:
        le = pickle.load(f)

    # Load MTCNN + FaceNet
    mtcnn = MTCNN(image_size=160, margin=14, device=DEVICE)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    return clf, le, mtcnn, resnet


clf, le, mtcnn, resnet = load_models()

# =========================
# EMBEDDING FUNCTION
# =========================
def get_embedding(img):
    img_np = np.array(img)
    face = mtcnn(img_np)

    if face is None:
        return None

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(DEVICE))

    return emb.cpu().numpy()


# =========================
# STREAMLIT CAMERA INPUT
# =========================
camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")
    st.image(img, caption="Captured Image", use_column_width=True)

    emb = get_embedding(img)

    if emb is None:
        st.error("❌ No face detected. Try again with clearer lighting.")
    else:
        probs = clf.predict_proba(emb)[0]
        best_prob = np.max(probs)

        st.write(f"**Similarity Score:** {best_prob:.3f}")

        if best_prob >= THRESHOLD:
            st.success("✅ ACCEPT — Face matches trained person.")
        else:
            st.error("❌ REJECT — Unknown person.")
