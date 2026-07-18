"""
Visualisasi & Perbandingan Model - Fish Classification
Membuat grafik training curve (accuracy & loss) dan confusion matrix
untuk membandingkan MobileNetV2 vs EfficientNetB0.

Output disimpan di folder models/plots/

Jalankan dari root project (setelah kedua model selesai di-training):
    python src/visualize.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
PLOTS_DIR = MODELS_DIR / "plots"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_history(model_name: str):
    with open(MODELS_DIR / f"{model_name}_history.json") as f:
        return json.load(f)


def plot_training_curves(histories: dict):
    """Bikin grafik perbandingan accuracy & loss (train vs val) untuk kedua model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"mobilenet": "#2563eb", "efficientnet": "#dc2626"}
    labels = {"mobilenet": "MobileNetV2", "efficientnet": "EfficientNetB0"}

    for model_name, history in histories.items():
        epochs = range(1, len(history["accuracy"]) + 1)
        color = colors[model_name]
        label = labels[model_name]

        axes[0, 0].plot(epochs, history["accuracy"], color=color, label=label, marker="o", markersize=3)
        axes[0, 1].plot(epochs, history["val_accuracy"], color=color, label=label, marker="o", markersize=3)
        axes[1, 0].plot(epochs, history["loss"], color=color, label=label, marker="o", markersize=3)
        axes[1, 1].plot(epochs, history["val_loss"], color=color, label=label, marker="o", markersize=3)

    titles = [
        "Training Accuracy", "Validation Accuracy",
        "Training Loss", "Validation Loss",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy" if "Accuracy" in title else "Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = PLOTS_DIR / "training_curves_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafik training curves tersimpan di: {output_path}")


def plot_final_comparison_bar(histories: dict):
    """Bar chart perbandingan test accuracy & test loss final."""
    model_names = list(histories.keys())
    labels = {"mobilenet": "MobileNetV2", "efficientnet": "EfficientNetB0"}
    test_acc = [histories[m]["test_accuracy"] for m in model_names]
    test_loss = [histories[m]["test_loss"] for m in model_names]
    display_labels = [labels[m] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = axes[0].bar(display_labels, test_acc, color=["#2563eb", "#dc2626"])
    axes[0].set_title("Test Accuracy", fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    for bar, val in zip(bars1, test_acc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                     ha="center", fontweight="bold")

    bars2 = axes[1].bar(display_labels, test_loss, color=["#2563eb", "#dc2626"])
    axes[1].set_title("Test Loss", fontweight="bold")
    for bar, val in zip(bars2, test_loss):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(test_loss) * 0.02, f"{val:.4f}",
                     ha="center", fontweight="bold")

    plt.tight_layout()
    output_path = PLOTS_DIR / "test_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafik perbandingan test metrics tersimpan di: {output_path}")


def get_test_generator():
    datagen_raw = ImageDataGenerator()  # untuk efficientnet (tanpa rescale manual)
    return datagen_raw.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )


def plot_confusion_matrix(model_name: str, class_names: list):
    """Bikin confusion matrix untuk satu model di test set."""
    model_path = MODELS_DIR / f"{model_name}_best.keras"
    model = load_model(model_path)

    # MobileNetV2 butuh rescale manual, EfficientNetB0 tidak (lihat catatan di train.py)
    rescale_value = (1.0 / 255) if model_name == "mobilenet" else None
    datagen = ImageDataGenerator(rescale=rescale_value)
    test_gen = datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name.upper()}", fontweight="bold")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix {model_name} tersimpan di: {output_path}")

    # Simpan juga classification report (precision, recall, f1 per kelas)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = PLOTS_DIR / f"classification_report_{model_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Classification report {model_name} tersimpan di: {report_path}")


def main():
    print("Memuat history training kedua model...")
    histories = {
        "mobilenet": load_history("mobilenet"),
        "efficientnet": load_history("efficientnet"),
    }

    print("\nMembuat grafik training curves...")
    plot_training_curves(histories)

    print("\nMembuat grafik perbandingan test metrics...")
    plot_final_comparison_bar(histories)

    with open(MODELS_DIR / "class_indices.json") as f:
        class_indices = json.load(f)
    class_names = sorted(class_indices, key=lambda k: class_indices[k])

    print("\nMembuat confusion matrix MobileNetV2...")
    plot_confusion_matrix("mobilenet", class_names)

    print("\nMembuat confusion matrix EfficientNetB0...")
    plot_confusion_matrix("efficientnet", class_names)

    print(f"\nSemua visualisasi selesai. Cek folder: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()