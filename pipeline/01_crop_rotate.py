"""
Tahap 1: crop dan rotate objek OBB dari dataset Roboflow
Kelas BLUE dan RED sengaja dikecualikan dari output.
"""
import os
from collections import defaultdict

import yaml
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "processed")
SPLITS = ["train", "valid", "test"]
EXCLUDED_CLASSES = {"BLUE", "RED"}
REJECTED_LOG_PATH = os.path.join("results", "logs", "rejected_log.csv")

BLUR_THRESHOLD = 50
MIN_SIZE = 25


# baca mapping kelas
def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


# parse 4 titik obb dari baris label
def parse_obb_line(line):
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:9]))
    points = np.array(coords, dtype=np.float32).reshape(4, 2)
    return class_id, points


# denormalisasi titik ke koordinat pixel
def denormalize_points(points, img_w, img_h):
    px = points.copy()
    px[:, 0] *= img_w
    px[:, 1] *= img_h
    return px


# luruskan dan crop objek pakai minAreaRect
def warp_crop_obb(image, points_px):
    rect = cv2.minAreaRect(points_px)
    (cx, cy), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90

    rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    x1, y1 = max(x1, 0), max(y1, 0)
    x2 = min(x2, rotated.shape[1])
    y2 = min(y2, rotated.shape[0])

    crop = rotated[y1:y2, x1:x2]
    return crop


# cek blur dan ukuran crop, kembalikan status lolos/reject
def check_quality(crop):
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if min(h, w) < MIN_SIZE:
        return False, "terlalu_kecil", lap_var
    if lap_var < BLUR_THRESHOLD:
        return False, "blur", lap_var
    return True, None, lap_var


# proses satu split dataset
def process_split(split, class_names, reject_rows, stats):
    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    saved_count = 0

    for label_file in label_files:
        stem = os.path.splitext(label_file)[0]
        image_path = find_image_path(images_dir, stem)
        if image_path is None:
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]

        for idx, line in enumerate(lines):
            class_id, points = parse_obb_line(line)
            class_name = class_names[class_id]
            if class_name in EXCLUDED_CLASSES:
                continue

            points_px = denormalize_points(points, img_w, img_h)
            crop = warp_crop_obb(image, points_px)
            if crop.size == 0:
                continue

            out_filename = f"{stem}_{idx}.jpg"
            passed, reason, lap_var = check_quality(crop)

            if passed:
                class_out_dir = os.path.join(OUTPUT_DIR, split, class_name)
                os.makedirs(class_out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(class_out_dir, out_filename), crop)
                saved_count += 1
                stats[(split, class_name, "lolos")] += 1
            else:
                reject_dir = os.path.join(OUTPUT_DIR, split, "rejected", class_name)
                os.makedirs(reject_dir, exist_ok=True)
                cv2.imwrite(os.path.join(reject_dir, out_filename), crop)
                stats[(split, class_name, reason)] += 1
                reject_rows.append(
                    {
                        "filename": out_filename,
                        "split": split,
                        "kelas": class_name,
                        "alasan": reason,
                        "laplacian_var": lap_var,
                        "width": crop.shape[1],
                        "height": crop.shape[0],
                    }
                )

    print(f"split {split}: {saved_count} crop tersimpan")


# print ringkasan lolos vs reject per kelas dan alasan
def print_summary(stats):
    print("ringkasan filtering:")
    keys = sorted(stats.keys())
    for split, class_name, status in keys:
        print(f"  {split}/{class_name}/{status}: {stats[(split, class_name, status)]}")


# cari file gambar dengan berbagai ekstensi
def find_image_path(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    class_names = load_class_names(data_yaml_path)

    reject_rows = []
    stats = defaultdict(int)

    for split in SPLITS:
        process_split(split, class_names, reject_rows, stats)

    print_summary(stats)

    pd.DataFrame(reject_rows).to_csv(REJECTED_LOG_PATH, index=False)
    print(f"rejected log disimpan ke {REJECTED_LOG_PATH}")
    print("tahap 1 selesai")


if __name__ == "__main__":
    main()
