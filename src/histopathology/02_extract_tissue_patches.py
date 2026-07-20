import os
import openslide
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/Users/wardajawad/Downloads/Images"

OUTPUT_DIR = "/Users/wardajawad/Downloads/Patches"


# Patch size
PATCH_SIZE = 512


# Tissue percentage required inside patch
MIN_TISSUE_PERCENT = 70


# Slide level
# Level 0 = highest resolution
# Level 1/2 = faster but lower resolution
LEVEL = 0



# Create output folder

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



patch_records = []



# ============================================================
# FUNCTIONS
# ============================================================


def calculate_tissue_percentage(mask_patch):

    tissue_pixels = np.sum(mask_patch > 0)

    total_pixels = mask_patch.size

    percentage = (
        tissue_pixels / total_pixels
    ) * 100

    return percentage




def extract_patches(slide_path, mask_path, patient_id):


    print("\n======================================")
    print(patient_id)
    print("======================================")


    slide = openslide.OpenSlide(slide_path)


    width, height = slide.level_dimensions[LEVEL]


    print("Slide size:", width, height)



    mask = Image.open(mask_path)

    mask = mask.resize(
        (width, height)
    )

    mask_array = np.array(mask)



    patient_output = os.path.join(
        OUTPUT_DIR,
        patient_id
    )


    os.makedirs(
        patient_output,
        exist_ok=True
    )



    patch_number = 0



    for y in tqdm(
        range(0, height-PATCH_SIZE, PATCH_SIZE),
        desc="Extracting"
    ):


        for x in range(
            0,
            width-PATCH_SIZE,
            PATCH_SIZE
        ):


            mask_patch = mask_array[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ]



            tissue_percent = calculate_tissue_percentage(
                mask_patch
            )


            # Remove background

            if tissue_percent < MIN_TISSUE_PERCENT:

                continue



            # Read slide patch

            patch = slide.read_region(
                (x,y),
                LEVEL,
                (
                    PATCH_SIZE,
                    PATCH_SIZE
                )
            )


            patch = patch.convert("RGB")



            patch_array = np.array(
                patch
            )


            # Remove almost white patches

            brightness = np.mean(
                patch_array
            )


            if brightness > 220:

                continue



            filename = (

                f"{patient_id}_"
                f"patch_{patch_number}_"
                f"x{x}_y{y}.jpg"

            )


            save_path = os.path.join(
                 patient_output,
                 filename
            )

            # Skip patch if it already exists
            if os.path.exists(save_path):
                continue

            patch.save(
                save_path,
                "JPEG",
                quality=90,
                optimize=True
            )



            patch_records.append(
            {
            "patient_id": patient_id,
            "patch_file": save_path,
            "x": x,
            "y": y,
            "tissue_percent": tissue_percent
            }
            )



            patch_number += 1



    slide.close()

    # Mark extraction as completed
    with open(
         os.path.join(patient_output, "finished.txt"),
         "w"
    ) as f:
        f.write("done")

    print(
        "Saved patches:",
         patch_number
    )




# ============================================================
# MAIN
# ============================================================


print("="*70)
print("TCGA BRCA TISSUE PATCH EXTRACTION")
print("="*70)



patients = os.listdir(BASE_DIR)



svs_files = []



for patient in patients:


    patient_folder = os.path.join(
        BASE_DIR,
        patient
    )


    if not os.path.isdir(patient_folder):

        continue



    for file in os.listdir(patient_folder):


        if file.endswith(".svs"):

            svs_files.append(
                (
                patient,
                os.path.join(
                    patient_folder,
                    file
                )
                )
            )



print(
    "Slides found:",
    len(svs_files)
)



for patient_id, slide_path in svs_files:



    slide_name = os.path.basename(
        slide_path
    )


    mask_path = slide_path.replace(
        ".svs",
        "_tissue_mask.png"
    )


    if not os.path.exists(mask_path):

        print(
            "Mask missing:",
            patient_id
        )

        continue

    patient_output = os.path.join(
        OUTPUT_DIR,
        patient_id
    )

    finished_file = os.path.join(
           patient_output,
           "finished.txt"
    )

    # Skip only patients that were completely processed
    if os.path.exists(finished_file):

        print(
            f"Skipping {patient_id} (already finished)"
        )

        continue

    extract_patches(
        slide_path,
        mask_path,
        patient_id
    )




# Save metadata


csv_path = os.path.join(
    OUTPUT_DIR,
    "patch_metadata.csv"
)


df = pd.DataFrame(
    patch_records
)


df.to_csv(
    csv_path,
    index=False
)



print("\n======================================")
print("DONE")
print("======================================")

 
print(
    "Total patches:",
    len(df)
)


print(
    "Saved metadata:",
    csv_path
)
