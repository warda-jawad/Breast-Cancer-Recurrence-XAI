import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.models as models

# ============================================================
# CONFIG
# ============================================================

PATCH_DIR = "/Users/wardajawad/Downloads/Patches"
OUTPUT_FEATURES_DIR = "/Users/wardajawad/Downloads/Features"

os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)

IMAGE_SIZE = 224  # for CNN input

# quality thresholds
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 220

# ============================================================
# CNN MODEL (ResNet18 pretrained)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
model = model.to(device)
model.eval()

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ============================================================
# QUALITY FILTER
# ============================================================

def is_good_patch(img):
    img_np = np.array(img)

    brightness = np.mean(img_np)

    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        return False

    # remove mostly empty patches
    if np.std(img_np) < 10:
        return False

    return True

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(img):
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img)

    feat = feat.squeeze().cpu().numpy()
    return feat

# ============================================================
# MAIN PROCESS
# ============================================================

all_features = []
metadata = []

patch_folders = [
    f for f in os.listdir(PATCH_DIR)
    if os.path.isdir(os.path.join(PATCH_DIR, f))
]

print("="*70)
print("PATCH FILTER + NORMALIZATION + FEATURE EXTRACTION")
print("="*70)
print("Patients found:", len(patch_folders))

for patient_id in tqdm(patch_folders):

    patient_path = os.path.join(PATCH_DIR, patient_id)

    patch_files = [
        f for f in os.listdir(patient_path)
        if f.endswith(".jpg")
    ]

    if len(patch_files) == 0:
        continue

    patient_features = []

    for patch_file in patch_files:

        patch_path = os.path.join(patient_path, patch_file)

        try:
            img = Image.open(patch_path).convert("RGB")

            # quality filtering
            if not is_good_patch(img):
                continue

            # feature extraction
            feat = extract_feature(img)

            patient_features.append(feat)

        except Exception as e:
            continue

    if len(patient_features) == 0:
        continue

    patient_features = np.array(patient_features)

    save_path = os.path.join(
        OUTPUT_FEATURES_DIR,
        f"{patient_id}.npy"
    )

    np.save(save_path, patient_features)

    metadata.append({
        "patient_id": patient_id,
        "num_patches": len(patient_features),
        "feature_shape": patient_features.shape
    })

    print(f"{patient_id} → {patient_features.shape}")

# ============================================================
# SAVE METADATA
# ============================================================

df = pd.DataFrame(metadata)

df.to_csv(
    os.path.join(OUTPUT_FEATURES_DIR, "features_metadata.csv"),
    index=False
)

print("\nDONE")
print("Saved features to:", OUTPUT_FEATURES_DIR)
print("Total patients processed:", len(df))
