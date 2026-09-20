"""
Tahap 0: import foto manual kamera pribadi ke train set
Foto full/close-up per simbol, bukan hasil crop OBB dari video.
"""
import os
import re

import cv2
import pandas as pd

PHOTOS_DIR = os.path.join("data", "raw", "Foto")
TRAIN_DIR = os.path.join("data", "processed", "train")
LOG_PATH = os.path.join("results", "logs", "manual_photo_log.csv")
RESIZE_DIM = (128, 128)

BLUR_THRESHOLD = 50
MIN_SIZE = 25

FILENAME_PREFIX = "manual_"


# cek blur dan ukuran, sama seperti tahap 1
def check_quality(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if min(h, w) < MIN_SIZE:
        return False, "terlalu_kecil", lap_var
    if lap_var < BLUR_THRESHOLD:
        return False, "blur", lap_var
    return True, None, lap_var


# ambil nama kelas dari daftar folder train yang sudah ada
def load_existing_classes(train_dir):
    class_names = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and d != "rejected"
    ]
    lookup = {}
    for name in class_names:
        key = re.sub(r"[^A-Za-z0-9]", "", name).upper()
        lookup[key] = name
    return lookup


# petakan nama file ke nama kelas yang sudah ada
def map_filename_to_class(filename, class_lookup):
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    key = re.sub(r"[^A-Za-z0-9]", "", stem).upper()
    return class_lookup.get(key)


# proses satu foto manual
def process_photo(filename, class_lookup, log_rows):
    class_name = map_filename_to_class(filename, class_lookup)
    if class_name is None:
        log_rows.append(
            {"filename": filename, "kelas": None, "status": "reject", "alasan": "kelas_tidak_dikenali"}
        )
        return None

    image_path = os.path.join(PHOTOS_DIR, filename)
    image = cv2.imread(image_path)
    if image is None:
        log_rows.append(
            {"filename": filename, "kelas": class_name, "status": "reject", "alasan": "gagal_dibaca"}
        )
        return None

    passed, reason, lap_var = check_quality(image)
    if not passed:
        log_rows.append(
            {
                "filename": filename,
                "kelas": class_name,
                "status": "reject",
                "alasan": reason,
                "laplacian_var": lap_var,
            }
        )
        return None

    resized = cv2.resize(image, RESIZE_DIM)
    out_dir = os.path.join(TRAIN_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    out_stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(filename)[0])
    out_path = os.path.join(out_dir, f"{FILENAME_PREFIX}{out_stem}.jpg")
    cv2.imwrite(out_path, resized)

    log_rows.append(
        {
            "filename": filename,
            "kelas": class_name,
            "status": "masuk",
            "alasan": None,
            "laplacian_var": lap_var,
        }
    )
    return class_name


def main():
    class_lookup = load_existing_classes(TRAIN_DIR)
    photo_files = [
        f for f in os.listdir(PHOTOS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    log_rows = []
    unmapped = []

    for filename in photo_files:
        class_name = map_filename_to_class(filename, class_lookup)
        if class_name is None:
            unmapped.append(filename)
        process_photo(filename, class_lookup, log_rows)

    if unmapped:
        print("gagal dimapping ke kelas manapun:")
        for name in unmapped:
            print(f"  {name}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False)

    masuk = (log_df["status"] == "masuk").sum()
    reject = (log_df["status"] == "reject").sum()
    print(f"foto manual masuk: {masuk}, reject: {reject}")
    print(f"log disimpan ke {LOG_PATH}")


if __name__ == "__main__":
    main()
