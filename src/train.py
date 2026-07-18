"""
Training Script - Fish Classification
Melatih dan membandingkan dua arsitektur Transfer Learning:
1. MobileNetV2
2. EfficientNetB0

Jalankan dari root project:
    python src/train.py --model mobilenet
    python src/train.py --model efficientnet
"""

import os
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ====== KONFIGURASI ======
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

MODELS_DIR.mkdir(exist_ok=True)


def get_data_generators(model_name: str):
    """Siapkan data generator dengan augmentasi untuk training,
    dan tanpa augmentasi untuk validasi/testing.

    PENTING: EfficientNetB0 di tf.keras.applications sudah punya layer
    Rescaling/Normalization bawaan di dalam arsitekturnya sendiri (expect
    input piksel mentah 0-255). Kalau kita rescale manual (bagi 255) DI LUAR
    lagi, terjadi double-normalisasi -> nilai piksel jadi mendekati nol ->
    model gagal belajar sama sekali (stuck di accuracy 1/jumlah_kelas).
    MobileNetV2 sebaliknya TIDAK punya rescaling internal, jadi tetap perlu
    rescale=1./255 manual seperti biasa.
    """
    use_manual_rescale = model_name == "mobilenet"
    rescale_value = (1.0 / 255) if use_manual_rescale else None

    train_datagen = ImageDataGenerator(
        rescale=rescale_value,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=rescale_value)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )
    val_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model(base_model_name: str, num_classes: int):
    """Bangun model transfer learning berdasarkan nama base model."""
    if base_model_name == "mobilenet":
        base_model = MobileNetV2(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
        )
    elif base_model_name == "efficientnet":
        base_model = EfficientNetB0(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError("model harus 'mobilenet' atau 'efficientnet'")

    # Freeze base model dulu (feature extraction)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model_name: str):
    print(f"\n{'=' * 60}")
    print(f"TRAINING MODEL: {model_name.upper()}")
    print(f"{'=' * 60}\n")

    train_gen, val_gen, test_gen = get_data_generators(model_name)
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices

    model = build_model(model_name, num_classes)
    model.summary()

    checkpoint_path = MODELS_DIR / f"{model_name}_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Evaluasi di test set
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy ({model_name}): {test_acc:.4f}")
    print(f"Test Loss ({model_name}): {test_loss:.4f}")

    # Simpan history training buat bikin grafik nanti
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict["test_accuracy"] = float(test_acc)
    history_dict["test_loss"] = float(test_loss)

    with open(MODELS_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # Simpan mapping class_indices (dibutuhkan Flask app nanti)
    with open(MODELS_DIR / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)

    # Simpan model final juga (bukan cuma checkpoint terbaik)
    model.save(MODELS_DIR / f"{model_name}_final.keras")

    print(f"\nModel tersimpan di: {checkpoint_path}")
    print(f"History tersimpan di: {MODELS_DIR / f'{model_name}_history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mobilenet", "efficientnet"],
        help="Pilih arsitektur: mobilenet atau efficientnet",
    )
    args = parser.parse_args()
    train(args.model)