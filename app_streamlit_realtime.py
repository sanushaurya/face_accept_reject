# app_streamlit_realtime.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --------------------
# Config
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.60  # probability threshold for Accept/Reject

st.set_page_config(page_title="Real-time Face Verify", layout="centered")
st.title("Real-time Face Verification — ACCEPT / REJECT")
st.write("Allow camera access when prompted. Live video will show a bounding box and ACCEPT/REJECT label.")

# --------------------
# Load pickled models (cloud-safe)
# --------------------
@st.cache_resource
def load_models():
    # Load classifier + label encoder (saved with pickle)
    with open("models/svm_model.joblib", "rb") as f:
        clf = pickle.load(f)
    with open("models/label_encoder.joblib", "rb") as f:
        le = pickle.load(f)

    # face detector and embedding model
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=True, device=DEVICE)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    return clf, le, mtcnn, resnet

clf, le, mtcnn, resnet = load_models()

# --------------------
# Helper functions
# --------------------
def embedding_from_face_tensor(face_tensor):
    """face_tensor: torch tensor (N,3,160,160) or (3,160,160) -> returns numpy embeddings (N,512)"""
    if face_tensor is None:
        return None
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(DEVICE)
    with torch.no_grad():
        emb = resnet(face_tensor)
    return emb.cpu().numpy()

def classify_embedding(emb):
    """emb: (512,) or (1,512) -> returns best_prob, best_label"""
    emb2 = emb.reshape(1, -1)
    probs = clf.predict_proba(emb2)[0]
    idx = int(np.argmax(probs))
    return probs[idx], le.inverse_transform([idx])[0], probs

# --------------------
# Video transformer
# --------------------
class FaceVerifierTransformer(VideoTransformerBase):
    def __init__(self):
        # We'll reuse the loaded resources from the module-level variables
        # (they are already created in main thread via load_models)
        self.mtcnn = mtcnn
        self.resnet = resnet
        self.clf = clf
        self.le = le
        self.threshold = THRESHOLD
        # optional small font for overlay
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            self.font = None

    def transform(self, frame):
        # frame -> numpy bgr
        img_bgr = frame.to_ndarray(format="bgr24")
        # convert to RGB for face processing & PIL draw
        img_rgb = cv2_to_rgb(img_bgr)

        # run detection (boxes) and aligned face tensors
        try:
            boxes, probs = self.mtcnn.detect(img_rgb)
        except Exception as e:
            # if detection fails, just return original frame
            return img_bgr

        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        if boxes is not None and len(boxes) > 0:
            # get face tensors (aligned crops) for each detected face
            # mtcnn returns face tensors when called directly, but in transform we call extract separately
            # We'll call self.mtcnn.extract to create aligned faces for each box if possible
            try:
                faces_tensor = self.mtcnn.extract(img_rgb, boxes)
            except Exception:
                # fallback: call mtcnn on whole image to get tensors; but mtcnn(img) returns stacked faces if keep_all True
                faces_tensor = self.mtcnn(img_rgb)

            if faces_tensor is not None:
                # faces_tensor shape: (N,3,160,160)
                embs = embedding_from_face_tensor(faces_tensor)
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                    label_text = "NO FACE"
                    best_prob = 0.0

                    if embs is not None and i < embs.shape[0]:
                        emb = embs[i]
                        best_prob, best_label, probs_all = classify_embedding(emb)
                        if best_prob >= self.threshold:
                            label_text = f"ACCEPT {best_label} {best_prob:.2f}"
                            color = (0, 255, 0)  # green
                        else:
                            label_text = f"REJECT {best_prob:.2f}"
                            color = (255, 0, 0)  # red
                    else:
                        color = (255, 0, 0)

                    # draw rectangle and label
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                    text_w = 200
                    draw.rectangle([(x1, y1 - 24), (x1 + text_w, y1)], fill=(0,0,0))
                    if self.font:
                        draw.text((x1 + 4, y1 - 22), label_text, fill=color, font=self.font)
                    else:
                        draw.text((x1 + 4, y1 - 22), label_text, fill=color)

        # convert back to BGR for returning
        out_bgr = rgb_to_cv2(np.array(pil_img))
        return out_bgr

# small helpers to convert color orders
def cv2_to_rgb(img_bgr):
    import cv2
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_cv2(img_rgb):
    import cv2
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# --------------------
# Run streamer
# --------------------
webrtc_streamer(
    key="face-verifier",
    video_transformer_factory=FaceVerifierTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
