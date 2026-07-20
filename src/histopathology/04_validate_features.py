import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = "/Users/wardajawad/Downloads/Features"

EXPECTED_FEATURE_DIM = 512

# ============================================================
# START
# ============================================================

print("=" * 70)
print("TCGA BRCA FEATURE DATASET VALIDATION")
print("=" * 70)

feature_files = sorted([
    f for f in os.listdir(FEATURE_DIR)
    if f.endswith(".npy")
])

print(f"\nFeature files found : {len(feature_files)}")

patient_ids = []

records = []

corrupted_files = []

wrong_dimension = []

empty_files = []

duplicate_ids = set()

seen = set()

# ============================================================
# CHECK EVERY FILE
# ============================================================

for file in feature_files:

    patient_id = file.replace(".npy", "")

    if patient_id in seen:
        duplicate_ids.add(patient_id)

    seen.add(patient_id)

    path = os.path.join(
        FEATURE_DIR,
        file
    )

    try:

        features = np.load(path)
        if not np.isfinite(features).all():
            print(f"[WARNING] NaN/Inf detected: {file}")
            corrupted_files.append(file)
            continue

    except Exception:

        corrupted_files.append(file)
        continue

    if len(features.shape) != 2:

        wrong_dimension.append(file)
        continue

    num_patches = features.shape[0]
    feature_dim = features.shape[1]

    if feature_dim != EXPECTED_FEATURE_DIM:

        wrong_dimension.append(file)

    if num_patches == 0:

        empty_files.append(file)

    patient_ids.append(patient_id)

    records.append({

        "patient_id": patient_id,
        "patches": num_patches,
        "feature_dimension": feature_dim

    })

# ============================================================
# BUILD REPORT
# ============================================================

df = pd.DataFrame(records)
total_patches = df["patches"].sum()
print(f"Total patches           : {total_patches}")

csv_path = os.path.join(
    FEATURE_DIR,
    "feature_validation_report.csv"
)

df.to_csv(
    csv_path,
    index=False
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Patients processed      : {len(df)}")
print(f"Feature files           : {len(feature_files)}")

print(f"Corrupted files         : {len(corrupted_files)}")
print(f"Wrong dimensions        : {len(wrong_dimension)}")
print(f"Empty feature files     : {len(empty_files)}")
print(f"Duplicate patient IDs   : {len(duplicate_ids)}")

if len(df) > 0:

    print()
    print(f"Average patches         : {int(df['patches'].mean())}")
    print(f"Median patches          : {int(df['patches'].median())}")
    print(f"Minimum patches         : {int(df['patches'].min())}")
    print(f"Maximum patches         : {int(df['patches'].max())}")

print()

# ============================================================
# DATASET STATUS
# ============================================================

if (
    len(corrupted_files) == 0 and
    len(wrong_dimension) == 0 and
    len(empty_files) == 0 and
    len(duplicate_ids) == 0
):

    print("Dataset Status          : READY FOR NEXT BATCH")
    print("All feature files passed validation.")

else:

    print("Dataset Status          : CHECK REQUIRED")

    if corrupted_files:

        print("\nCorrupted files:")

        for f in corrupted_files:

            print("   ", f)

    if wrong_dimension:

        print("\nWrong dimensions:")

        for f in wrong_dimension:

            print("   ", f)

    if empty_files:

        print("\nEmpty files:")

        for f in empty_files:

            print("   ", f)

    if duplicate_ids:

        print("\nDuplicate IDs:")

        for f in duplicate_ids:

            print("   ", f)

print("\nValidation report saved to:")
print(csv_path)

print("\nDONE")
