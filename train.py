# train.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import torch

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

# ================================
# CONFIG
# ================================
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUGMENT_TIMES = 20  # Increase for better accuracy
print("Using device:", DEVICE)

# ================================
# LOAD MODELS
# ================================
mtcnn = MTCNN(image_size=160, margin=14, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ================================
# AUGMENTATION PIPELINE
# ================================
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# ================================
# EMBEDDING FUNCTION
# ================================
def get_embedding(img):
    img_np = np.array(img)

    try:
        face = mtcnn(img_np)
    except Exception as e:
        print("MTCNN Error:", e)
        return None

    if face is None:
        return None

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(DEVICE))

    return emb.cpu().numpy()[0]


# ================================
# TRAINING LOOP
# ================================
X, y = [], []

print("\n=== Extracting & Augmenting Images ===\n")

for person in os.listdir(DATA_DIR):
    person_folder = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_folder):
        continue

    for img_name in tqdm(os.listdir(person_folder), desc=f"Processing {person}"):
        img_path = os.path.join(person_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        # Original embedding
        emb = get_embedding(img)
        if emb is not None:
            X.append(emb)
            y.append(person)

        # Augmented embeddings
        for _ in range(AUGMENT_TIMES):
            aug_img = augment(img)
            emb_aug = get_embedding(aug_img)
            if emb_aug is not None:
                X.append(emb_aug)
                y.append(person)

X = np.array(X)
y = np.array(y)
print("\nTotal embeddings:", len(X))

# ================================
# LABEL ENCODER
# ================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ================================
# TRAIN SVM CLASSIFIER
# ================================
clf = SVC(kernel="rbf", probability=True)

print("\n=== Training SVM Classifier ===")
clf.fit(X, y_encoded)

# ================================
# CROSS-VALIDATION ACCURACY
# ================================
print("\n=== Evaluating Accuracy (Cross-Validation) ===")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y_encoded, cv=cv)

print(f"\nCross-validation accuracy: {scores.mean():.4f}")
print(f"Std deviation: {scores.std():.4f}")

# ================================
# SAVE MODELS (CLOUD-SAFE PICKLE)
# ================================
os.makedirs("models", exist_ok=True)

with open("models/svm_model.joblib", "wb") as f:
    pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("models/label_encoder.joblib", "wb") as f:
    pickle.dump(le, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n=== TRAINING COMPLETE ===")
print("Saved: models/svm_model.joblib")
print("Saved: models/label_encoder.joblib")
