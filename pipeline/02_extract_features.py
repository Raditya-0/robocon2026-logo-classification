"""
Tahap 2: ekstraksi fitur dari hasil crop data/processed
Fitur: color histogram HSV + HOG
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

INPUT_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join("results", "features.csv")
SPLITS = ["train", "valid", "test"]
RESIZE_DIM = (128, 128)


# hitung histogram warna HSV
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


# hitung fitur HOG dari citra grayscale
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


# gabung histogram warna dan hog jadi satu vektor
def extract_features(image):
    resized = cv2.resize(image, RESIZE_DIM)
    color_feat = extract_color_histogram(resized)
    hog_feat = extract_hog_features(resized)
    return np.concatenate([color_feat, hog_feat])


# kumpulkan fitur dari satu split
def process_split(split):
    split_dir = os.path.join(INPUT_DIR, split)
    rows = []

    if not os.path.isdir(split_dir):
        return rows

    class_names = sorted(os.listdir(split_dir))
    for class_name in class_names:
        if class_name == "rejected":
            continue
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            feat = extract_features(image)
            rows.append((feat, class_name, split))

    print(f"split {split}: {len(rows)} citra diekstrak")
    return rows


def main():
    all_rows = []
    for split in SPLITS:
        all_rows.extend(process_split(split))

    feat_matrix = np.array([r[0] for r in all_rows])
    labels = [r[1] for r in all_rows]
    splits = [r[2] for r in all_rows]

    feat_cols = [f"f{i}" for i in range(feat_matrix.shape[1])]
    df = pd.DataFrame(feat_matrix, columns=feat_cols)
    df["label"] = labels
    df["split"] = splits

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"tahap 2 selesai, disimpan ke {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
