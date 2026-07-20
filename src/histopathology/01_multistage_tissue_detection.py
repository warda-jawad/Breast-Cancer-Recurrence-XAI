import os
import cv2
import numpy as np
from PIL import Image
import openslide

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------

INPUT_FOLDER = "/Users/wardajawad/Downloads/Images"

THUMBNAIL_SIZE = 1024

MIN_COMPONENT_AREA = 500

# ----------------------------------------------------


def create_thumbnail(slide):
    thumbnail = slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    return np.array(thumbnail)


def tissue_mask(thumbnail):

    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:

            cleaned[labels == i] = 255

    flood = cleaned.copy()

    h, w = flood.shape

    flood_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0,0), 255)

    flood_inv = cv2.bitwise_not(flood)

    final_mask = cleaned | flood_inv

    return final_mask


def process_slide(svs_path):

    print("="*70)
    print(os.path.basename(svs_path))

    slide = openslide.OpenSlide(svs_path)

    thumbnail = create_thumbnail(slide)

    mask = tissue_mask(thumbnail)

    output_mask = svs_path.replace(".svs", "_tissue_mask.png")

    Image.fromarray(mask).save(output_mask)

    tissue_pixels = np.sum(mask > 0)

    total_pixels = mask.size

    tissue_percent = tissue_pixels / total_pixels * 100

    print(f"Slide size      : {slide.dimensions}")
    print(f"Tissue percent  : {tissue_percent:.2f}%")
    print(f"Saved mask      : {output_mask}")


def main():

    total = 0

    for patient in os.listdir(INPUT_FOLDER):

        patient_folder = os.path.join(INPUT_FOLDER, patient)

        if not os.path.isdir(patient_folder):
            continue

        for file in os.listdir(patient_folder):

            if file.endswith(".svs"):

                process_slide(
                    os.path.join(patient_folder, file)
                )

                total += 1

    print("\n===================================")
    print(f"Processed slides : {total}")
    print("DONE")
    print("===================================")


if __name__ == "__main__":
    main()
