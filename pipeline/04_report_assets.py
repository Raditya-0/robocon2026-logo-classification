"""
Tahap 4: generate aset visual dan statistik untuk laporan
"""
import os
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from skimage.feature import hog

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

DATASET_DIR = os.path.join("data", "raw")
CLASSIFIED_DIR = os.path.join("data", "processed")
FEATURES_PATH = os.path.join("results", "features.csv")
REJECTED_LOG_PATH = os.path.join("results", "logs", "rejected_log.csv")
MANUAL_LOG_PATH = os.path.join("results", "logs", "manual_photo_log.csv")
ASSETS_DIR = os.path.join("docs", "laporan", "assets")
STATS_PATH = os.path.join("results", "logs", "laporan_stats.json")

BLUR_THRESHOLD = 50
MIN_SIZE = 25
SAMPLES_PER_CLASS = 2
THUMB_SIZE = (110, 110)

MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf"),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200),
}


# baca mapping kelas dari yaml
def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


# parse baris label obb
def parse_obb_line(line):
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:9]))
    points = np.array(coords, dtype=np.float32).reshape(4, 2)
    return class_id, points


# denormalisasi titik ke pixel
def denormalize_points(points, img_w, img_h):
    px = points.copy()
    px[:, 0] *= img_w
    px[:, 1] *= img_h
    return px


# crop lurus tanpa rotasi, buat pembanding before
def axis_aligned_crop(image, points_px):
    x1 = max(int(points_px[:, 0].min()), 0)
    y1 = max(int(points_px[:, 1].min()), 0)
    x2 = min(int(points_px[:, 0].max()), image.shape[1])
    y2 = min(int(points_px[:, 1].max()), image.shape[0])
    return image[y1:y2, x1:x2]


# luruskan dan crop pakai minAreaRect
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

    return rotated[y1:y2, x1:x2]


# cek blur dan ukuran, sama seperti tahap 1
def check_quality(crop):
    h, w = crop.shape[:2]
    if min(h, w) < MIN_SIZE:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var >= BLUR_THRESHOLD


# cari path gambar dengan ekstensi umum
def find_image_path(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# kumpulkan sample before/after per kelas dari train split
def collect_grid_samples(class_names, excluded_classes):
    images_dir = os.path.join(DATASET_DIR, "train", "images")
    labels_dir = os.path.join(DATASET_DIR, "train", "labels")

    target_classes = [c for c in class_names.values() if c not in excluded_classes]
    samples = {c: [] for c in target_classes}
    needed = {c: SAMPLES_PER_CLASS for c in target_classes}

    label_files = sorted(os.listdir(labels_dir))
    for label_file in label_files:
        if all(v == 0 for v in needed.values()):
            break

        stem = os.path.splitext(label_file)[0]
        image_path = find_image_path(images_dir, stem)
        if image_path is None:
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        with open(os.path.join(labels_dir, label_file), "r") as f:
            lines = [l for l in f.readlines() if l.strip()]

        for line in lines:
            class_id, points = parse_obb_line(line)
            class_name = class_names[class_id]
            if class_name not in needed or needed[class_name] == 0:
                continue

            points_px = denormalize_points(points, img_w, img_h)
            after_crop = warp_crop_obb(image, points_px)
            if after_crop.size == 0 or not check_quality(after_crop):
                continue

            before_crop = axis_aligned_crop(image, points_px)
            if before_crop.size == 0:
                continue

            samples[class_name].append((before_crop, after_crop))
            needed[class_name] -= 1

    return samples


# tempel satu crop ke thumbnail seragam
def to_thumbnail(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize(THUMB_SIZE)
    return img


# susun grid contoh kelas jadi satu gambar
def build_class_grid(samples, out_path):
    label_w = 130
    cell_w, cell_h = THUMB_SIZE
    n_cols = SAMPLES_PER_CLASS * 2
    class_names = sorted(samples.keys())
    n_rows = len(class_names)

    canvas_w = label_w + n_cols * cell_w
    canvas_h = n_rows * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for row, class_name in enumerate(class_names):
        y = row * cell_h
        draw.text((5, y + cell_h // 2 - 6), class_name, fill="black")

        pairs = samples[class_name]
        col = 0
        for before_crop, after_crop in pairs:
            x_before = label_w + col * cell_w
            canvas.paste(to_thumbnail(before_crop), (x_before, y))
            col += 1
            x_after = label_w + col * cell_w
            canvas.paste(to_thumbnail(after_crop), (x_after, y))
            col += 1

    canvas.save(out_path)
    print(f"grid contoh kelas disimpan ke {out_path}")


# hitung metrik kualitas semua crop yang lolos
def compute_lolos_metrics():
    rows = []
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(CLASSIFIED_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            if class_name == "rejected":
                continue
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                img = cv2.imread(os.path.join(class_dir, filename))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                rows.append({"laplacian_var": lap_var, "width": w, "height": h})
    return pd.DataFrame(rows)


# plot distribusi laplacian dan ukuran, reject vs lolos
def plot_reject_vs_lolos(lolos_df, reject_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(lolos_df["laplacian_var"], bins=40, alpha=0.6, label="lolos", range=(0, 500))
    axes[0].hist(reject_df["laplacian_var"], bins=40, alpha=0.6, label="reject", range=(0, 500))
    axes[0].axvline(BLUR_THRESHOLD, color="red", linestyle="--", label="threshold blur")
    axes[0].set_title("distribusi laplacian variance")
    axes[0].set_xlabel("laplacian variance")
    axes[0].legend()

    lolos_minside = lolos_df[["width", "height"]].min(axis=1)
    reject_minside = reject_df[["width", "height"]].min(axis=1)
    axes[1].hist(lolos_minside, bins=40, alpha=0.6, label="lolos", range=(0, 150))
    axes[1].hist(reject_minside, bins=40, alpha=0.6, label="reject", range=(0, 150))
    axes[1].axvline(MIN_SIZE, color="red", linestyle="--", label="threshold ukuran")
    axes[1].set_title("distribusi ukuran sisi terpendek crop")
    axes[1].set_xlabel("pixel")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"plot distribusi reject disimpan ke {out_path}")


# klasifikasi keluarga kelas real/fake/lainnya
def get_family(class_name):
    if class_name.startswith("REAL_"):
        return "REAL"
    if class_name.startswith("FAKE_"):
        return "FAKE"
    return "OTHER"


# load fitur dan pisah berdasar kolom split
def load_features():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c.startswith("f")]

    encoder = LabelEncoder()
    df["label_enc"] = encoder.fit_transform(df["label"])

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])

    y_train = train_df["label_enc"].values
    y_test = test_df["label_enc"].values

    return x_train, y_train, x_test, y_test, encoder


# latih semua model, kembalikan prediksi test set
def train_all_models(x_train, y_train, x_test):
    predictions = {}
    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_test)
    return predictions


# gambar dan simpan confusion matrix satu model
def save_confusion_plot(name, cm, class_labels, out_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("prediksi")
    ax.set_ylabel("aktual")
    ax.set_title(f"confusion matrix - {name}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# hitung persentase error silang real-fake per model
def compute_real_fake_error_stats(cm, class_labels):
    families = [get_family(c) for c in class_labels]
    total_errors = 0
    cross_errors = 0

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            if i == j:
                continue
            count = int(cm[i, j])
            total_errors += count
            if {families[i], families[j]} == {"REAL", "FAKE"}:
                cross_errors += count

    pct = (cross_errors / total_errors * 100) if total_errors > 0 else 0.0
    return {
        "total_error": total_errors,
        "real_fake_cross_error": cross_errors,
        "real_fake_cross_pct": round(pct, 2),
    }


# cari pasangan real-fake paling sering tertukar, agregat semua model
def find_most_confused_pair(all_cm, class_labels):
    agg = np.zeros_like(all_cm[0])
    for cm in all_cm:
        agg += cm

    best_pair = None
    best_score = -1
    for i, ci in enumerate(class_labels):
        if get_family(ci) != "REAL":
            continue
        for j, cj in enumerate(class_labels):
            if get_family(cj) != "FAKE":
                continue
            score = agg[i, j] + agg[j, i]
            if score > best_score:
                best_score = score
                best_pair = (ci, cj)

    return best_pair, int(best_score)


# ambil satu contoh citra kelas dari data/processed
def load_sample_for_class(class_name):
    for split in ("train", "valid", "test"):
        class_dir = os.path.join(CLASSIFIED_DIR, split, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(os.listdir(class_dir))
        if files:
            return cv2.imread(os.path.join(class_dir, files[0]))
    return None


# render hog descriptor untuk satu citra
def render_hog(image):
    resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), hog_image


# simpan visualisasi hog real vs fake berdampingan
def save_hog_comparison(real_class, fake_class, out_path):
    real_img = load_sample_for_class(real_class)
    fake_img = load_sample_for_class(fake_class)

    real_orig, real_hog = render_hog(real_img)
    fake_orig, fake_hog = render_hog(fake_img)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(real_orig)
    axes[0, 0].set_title(f"asli - {real_class}")
    axes[0, 1].imshow(real_hog, cmap="gray")
    axes[0, 1].set_title(f"hog - {real_class}")
    axes[1, 0].imshow(fake_orig)
    axes[1, 0].set_title(f"asli - {fake_class}")
    axes[1, 1].imshow(fake_hog, cmap="gray")
    axes[1, 1].set_title(f"hog - {fake_class}")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"visualisasi hog disimpan ke {out_path}")


# hitung statistik sumber data manual vs video
def compute_source_stats(class_names, excluded_classes):
    target_classes = [c for c in class_names.values() if c not in excluded_classes]

    video_final = defaultdict(int)
    manual_final = defaultdict(int)

    for split in ("train", "valid", "test"):
        split_dir = os.path.join(CLASSIFIED_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in target_classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.startswith("manual_"):
                    manual_final[class_name] += 1
                else:
                    video_final[class_name] += 1

    reject_df = pd.read_csv(REJECTED_LOG_PATH)
    manual_log_df = pd.read_csv(MANUAL_LOG_PATH)

    video_reject_total = len(reject_df)
    video_reject_by_reason = reject_df["alasan"].value_counts().to_dict()

    manual_reject_df = manual_log_df[manual_log_df["status"] == "reject"]
    manual_reject_total = len(manual_reject_df)
    manual_reject_by_reason = manual_reject_df["alasan"].value_counts().to_dict()

    video_final_total = sum(video_final.values())
    manual_final_total = sum(manual_final.values())

    video_reject_pct = (
        video_reject_total / (video_reject_total + video_final_total) * 100
        if (video_reject_total + video_final_total) > 0 else 0.0
    )
    manual_reject_pct = (
        manual_reject_total / (manual_reject_total + manual_final_total) * 100
        if (manual_reject_total + manual_final_total) > 0 else 0.0
    )

    return {
        "video_final_total": video_final_total,
        "manual_final_total": manual_final_total,
        "video_final_per_class": dict(video_final),
        "manual_final_per_class": dict(manual_final),
        "video_reject_total": video_reject_total,
        "video_reject_by_reason": video_reject_by_reason,
        "video_reject_pct": round(video_reject_pct, 2),
        "manual_reject_total": manual_reject_total,
        "manual_reject_by_reason": manual_reject_by_reason,
        "manual_reject_pct": round(manual_reject_pct, 2),
    }


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    class_names = load_class_names(data_yaml_path)
    excluded_classes = {"BLUE", "RED"}

    # item 1: grid contoh kelas
    samples = collect_grid_samples(class_names, excluded_classes)
    build_class_grid(samples, os.path.join(ASSETS_DIR, "grid_contoh_kelas.png"))

    # item 2: distribusi reject vs lolos
    lolos_df = compute_lolos_metrics()
    reject_df = pd.read_csv(REJECTED_LOG_PATH)
    plot_reject_vs_lolos(lolos_df, reject_df, os.path.join(ASSETS_DIR, "distribusi_reject.png"))

    # item 4: training dan confusion matrix semua model
    x_train, y_train, x_test, y_test, encoder = load_features()
    predictions = train_all_models(x_train, y_train, x_test)

    class_labels = list(encoder.classes_)
    all_cm = []
    model_stats = {}

    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred, labels=range(len(class_labels)))
        all_cm.append(cm)
        save_confusion_plot(name, cm, class_labels, os.path.join(ASSETS_DIR, f"confmat_{name}.png"))
        model_stats[name] = compute_real_fake_error_stats(cm, class_labels)

    best_pair, best_pair_score = find_most_confused_pair(all_cm, class_labels)

    # item 3: visualisasi hog untuk pasangan real-fake paling mirip
    if best_pair is not None:
        save_hog_comparison(
            best_pair[0], best_pair[1], os.path.join(ASSETS_DIR, "hog_real_vs_fake.png")
        )

    # item 5: rekap sumber data manual vs video
    source_stats = compute_source_stats(class_names, excluded_classes)

    split_counts = pd.read_csv(FEATURES_PATH)["split"].value_counts().to_dict()

    stats = {
        "total_kelas_final": len(class_labels),
        "kelas_final": class_labels,
        "total_citra_per_split": split_counts,
        "confusion_real_fake_per_model": model_stats,
        "pasangan_real_fake_paling_tertukar": {
            "real_class": best_pair[0] if best_pair else None,
            "fake_class": best_pair[1] if best_pair else None,
            "skor_confusion_gabungan": best_pair_score,
        },
        "sumber_data": source_stats,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))

    print(f"statistik laporan disimpan ke {STATS_PATH}")

    print("ringkasan akhir:")
    print(f"  total kelas final: {len(class_labels)}")
    print(f"  total citra per split: {split_counts}")
    print(f"  total citra final: video={source_stats['video_final_total']} manual={source_stats['manual_final_total']}")
    print(f"  pasangan paling mirip: {best_pair}")
    for name, s in model_stats.items():
        print(f"  {name}: error real<->fake = {s['real_fake_cross_error']}/{s['total_error']} ({s['real_fake_cross_pct']}%)")


if __name__ == "__main__":
    main()
