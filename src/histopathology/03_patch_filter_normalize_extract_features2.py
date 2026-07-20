import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import traceback

import torch
import torchvision.transforms as transforms
import torchvision.models as models

# ============================================================
# CONFIG
# ============================================================

PATCH_DIR = "/Users/wardajawad/Downloads/Patches"
OUTPUT_FEATURES_DIR = "/Users/wardajawad/Downloads/Features"

os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)

IMAGE_SIZE = 224

# ============================================================
# IMPORTANT CHANGE (PATHOLOGY SAFE SETTINGS)
# ============================================================

# We REMOVE brightness filtering (too sensitive for histology)
# Keep ONLY very mild texture filter
STD_THRESHOLD = 7   # relaxed compared to your 10

# ============================================================
# MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ============================================================
# LOGGING
# ============================================================

processed_patients = 0
skipped_patients = 0
failed_patients = 0

total_patches_all = 0
accepted_patches_all = 0
rejected_patches_all = 0
error_logs = []

# ============================================================
# FUNCTIONS
# ============================================================

def is_good_patch(img):
    """
    SAFE pathology filter:
    - NO brightness rejection (important improvement)
    - ONLY removes completely empty / uniform patches
    """

    img_np = np.array(img)
    rightness = np.mean(img_np)
    std = np.std(img_np)
    
    # only remove extreme blank/background patches
    if std < 3:      # almost flat image
        return False

    # optional: remove fully black/white artifacts
    if brightness < 5 or brightness > 250:
        return False

    return True


def extract_feature(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img)
    return feat.squeeze().cpu().numpy()


def log_error(patient_id, patch_file, err):
    if len(error_logs) < 20:
        error_logs.append((patient_id, patch_file, str(err)))

# ============================================================
# MAIN
# ============================================================

metadata = []

patch_folders = [
    f for f in os.listdir(PATCH_DIR)
    if os.path.isdir(os.path.join(PATCH_DIR, f))
]

print("=" * 70)
print("PATCH FILTER + NORMALIZATION + FEATURE EXTRACTION (PATHOLOGY SAFE)")
print("=" * 70)
print("Patients found:", len(patch_folders))

for patient_id in tqdm(patch_folders):

    save_path = os.path.join(OUTPUT_FEATURES_DIR, f"{patient_id}.npy")

    # resume
    if os.path.exists(save_path):
        print(f"[SKIP] {patient_id}")
        skipped_patients += 1
        continue

    patient_path = os.path.join(PATCH_DIR, patient_id)

    if not os.path.exists(patient_path):
        print(f"[ERROR] Missing folder: {patient_id}")
        failed_patients += 1
        continue

    patch_files = [f for f in os.listdir(patient_path) if f.endswith(".jpg")]

    total_patches = len(patch_files)
    total_patches_all += total_patches

    if total_patches == 0:
        print(f"[WARNING] No patches: {patient_id}")
        continue

    patient_features = []
    accepted = 0
    rejected = 0
    errors = 0

    print(f"\nProcessing {patient_id} | patches: {total_patches}")

    for patch_file in patch_files:

        patch_path = os.path.join(patient_path, patch_file)

        try:
            img = Image.open(patch_path).convert("RGB")
            # DEBUG: inspect patch statistics (temporary)
            img_np = np.array(img)

            brightness = np.mean(img_np)
            std = np.std(img_np)

            print(f"[DEBUG] {patient_id}/{patch_file} | brightness={brightness:.2f} | std={std:.2f}")

            if not is_good_patch(img):
                rejected += 1
                continue

            try:
                feat = extract_feature(img)
                patient_features.append(feat)
                accepted += 1

            except Exception as e:
                errors += 1
                if len(error_logs) < 10:
                    print(f"[FEATURE ERROR] {patch_file}: {str(e)}")
                continue

        except Exception as e:
            errors += 1
            log_error(patient_id, patch_file, e)

    rejected_patches_all += rejected

    if len(patient_features) == 0:
        print(f"[FAILED] No valid patches: {patient_id}")
        failed_patients += 1
        continue

    patient_features = np.array(patient_features)

    np.save(save_path, patient_features)

    metadata.append({
        "patient_id": patient_id,
        "num_patches_total": total_patches,
        "num_patches_accepted": accepted,
        "num_patches_rejected": rejected,
        "num_errors": errors,
        "feature_shape": patient_features.shape
    })

    processed_patients += 1

    print(f"[DONE] {patient_id} → {patient_features.shape} "
          f"(accepted={accepted}, rejected={rejected}, errors={errors})")

# ============================================================
# SAVE METADATA (SAFE APPEND)
# ============================================================

metadata_file = os.path.join(OUTPUT_FEATURES_DIR, "features_metadata.csv")

new_df = pd.DataFrame(metadata)

if os.path.exists(metadata_file):
    old_df = pd.read_csv(metadata_file)
    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset="patient_id", keep="first")
else:
    df = new_df

df.to_csv(metadata_file, index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Processed patients : {processed_patients}")
print(f"Skipped patients   : {skipped_patients}")
print(f"Failed patients    : {failed_patients}")

print(f"Total patches      : {total_patches_all}")
print(f"Accepted patches   : {accepted_patches_all}")
print(f"Rejected patches   : {rejected_patches_all}")

print(f"Total errors       : {len(error_logs)}")

if error_logs:
    print("\nFirst errors:")
    for e in error_logs[:5]:
        print(e)

print("\nSaved features to:", OUTPUT_FEATURES_DIR)
print("Metadata file:", metadata_file)
print("Accepted patches:", len(patient_features))