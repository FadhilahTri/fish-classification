"""
Preprocessing Script - Fish Classification
Membagi dataset mentah menjadi folder train/val/test dengan struktur
yang siap dipakai oleh ImageDataGenerator / image_dataset_from_directory.

Struktur dataset "A Large Scale Fish Dataset" dari Kaggle biasanya:
data/raw/Fish_Dataset/Fish_Dataset/<NamaSpesies>/<NamaSpesies>/*.png
(ada 2 subfolder per kelas: gambar asli & ground truth/mask -- kita cuma pakai yang asli)

Jalankan dari root project:
    python src/preprocess.py
"""

import os
import shutil
import random
from pathlib import Path

# ====== KONFIGURASI ======
RAW_DATA_DIR = Path("data/raw/Fish_Dataset/Fish_Dataset")
OUTPUT_DIR = Path("data/processed")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)


def get_class_folders(raw_dir: Path):
    """Ambil semua folder kelas (nama spesies ikan), skip folder GT/ground truth."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Folder {raw_dir} tidak ditemukan. "
            "Pastikan dataset sudah diextract ke data/raw/. "
            "Cek juga apakah nama foldernya sama persis, sesuaikan RAW_DATA_DIR jika beda."
        )
    classes = [f for f in raw_dir.iterdir() if f.is_dir()]
    return sorted(classes)


def collect_images(class_folder: Path):
    """
    Ambil gambar asli dari dalam folder kelas.
    Dataset ini punya struktur: <Kelas>/<Kelas>/*.png (gambar asli)
    dan <Kelas>/<Kelas> GT/*.png (ground truth/mask -- di-skip).
    """
    image_subfolder = class_folder / class_folder.name
    if image_subfolder.exists():
        images = list(image_subfolder.glob("*.png")) + list(image_subfolder.glob("*.jpg"))
    else:
        # fallback kalau strukturnya flat
        images = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg"))
    return images


def split_and_copy(images, class_name, output_dir):
    random.shuffle(images)
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        split_class_dir = output_dir / split_name / class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            shutil.copy2(img_path, split_class_dir / img_path.name)

    return {k: len(v) for k, v in splits.items()}


def main():
    print("Membaca folder kelas dari dataset mentah...")
    class_folders = get_class_folders(RAW_DATA_DIR)
    print(f"Ditemukan {len(class_folders)} kelas: {[c.name for c in class_folders]}\n")

    summary = []
    for class_folder in class_folders:
        images = collect_images(class_folder)
        if len(images) == 0:
            print(f"[SKIP] {class_folder.name}: tidak ada gambar ditemukan")
            continue
        counts = split_and_copy(images, class_folder.name, OUTPUT_DIR)
        summary.append((class_folder.name, len(images), counts))
        print(f"[OK] {class_folder.name}: total={len(images)} -> "
              f"train={counts['train']}, val={counts['val']}, test={counts['test']}")

    print("\n=== Ringkasan ===")
    total = sum(s[1] for s in summary)
    print(f"Total kelas: {len(summary)}")
    print(f"Total gambar: {total}")
    print(f"Data hasil split tersimpan di: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()