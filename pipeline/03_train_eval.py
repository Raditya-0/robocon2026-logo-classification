"""
Tahap 3: training dan evaluasi lima algoritma classical ML
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

FEATURES_PATH = "results/features.csv"
RESULTS_SUMMARY_PATH = "results/reports/results_summary.csv"
CONFUSION_DIR = "results/confusion_matrices"

MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf"),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200),
}


# load fitur dan pisah berdasar kolom split
def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith("f")]

    encoder = LabelEncoder()
    df["label_enc"] = encoder.fit_transform(df["label"])

    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_valid = scaler.transform(valid_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])

    y_train = train_df["label_enc"].values
    y_valid = valid_df["label_enc"].values
    y_test = test_df["label_enc"].values

    return x_train, y_train, x_valid, y_valid, x_test, y_test, encoder


# latih satu model, evaluasi di valid dan test set
def train_and_evaluate(name, model, x_train, y_train, x_valid, y_valid, x_test, y_test, encoder):
    model.fit(x_train, y_train)

    y_valid_pred = model.predict(x_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": name,
        "valid_accuracy": valid_accuracy,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "report": report,
        "confusion_matrix": cm,
    }


# simpan confusion matrix tiap model ke csv
def save_confusion_matrix(name, cm, class_names):
    import os

    os.makedirs(CONFUSION_DIR, exist_ok=True)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(os.path.join(CONFUSION_DIR, f"{name}.csv"))


def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test, encoder = load_data(
        FEATURES_PATH
    )

    summary_rows = []
    for name, model in MODELS.items():
        result = train_and_evaluate(
            name, model, x_train, y_train, x_valid, y_valid, x_test, y_test, encoder
        )
        save_confusion_matrix(name, result["confusion_matrix"], encoder.classes_)

        print(f"{name}: accuracy={result['accuracy']:.4f} f1_macro={result['f1_macro']:.4f}")

        summary_rows.append(
            {
                "model": result["model"],
                "valid_accuracy": result["valid_accuracy"],
                "accuracy": result["accuracy"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
                "f1_macro": result["f1_macro"],
            }
        )

        report_path = f"results/reports/classification_report_{name}.txt"
        with open(report_path, "w") as f:
            f.write(result["report"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_SUMMARY_PATH, index=False)
    print(f"tahap 3 selesai, ringkasan disimpan ke {RESULTS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
