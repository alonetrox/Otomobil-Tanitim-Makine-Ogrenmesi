# ================================================================
# train_model.py — Otomobil Tanıma CNN Eğitim Scripti
# ResNet50V2 Transfer Learning (2 Aşamalı Eğitim)
# ================================================================

import os, json, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_P1 = 10
EPOCHS_P2 = 20
UNFREEZE = 30
BASE = os.path.dirname(__file__)
DATASET = os.path.join(BASE, "dataset")
MODELS = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")

def main():
    print("🚗 Otomobil Tanıma — Model Eğitimi")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🎮 GPU: {[g.name for g in gpus]}")
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    else:
        print("⚠️  GPU yok, CPU kullanılacak.")

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(OUTPUTS, exist_ok=True)

    # Veri yükleme
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")
    val_ds = keras.utils.image_dataset_from_directory(
        DATASET, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")

    names = train_ds.class_names
    n_cls = len(names)
    print(f"✅ {n_cls} sınıf: {names}")

    # Sınıf indeksleri kaydet
    with open(os.path.join(MODELS, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): n for i, n in enumerate(names)}, f, ensure_ascii=False, indent=2)

    AT = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AT)
    val_ds = val_ds.cache().prefetch(AT)

    # Sınıf ağırlıkları
    labels = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(2048)])
    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=labels)
    cw_dict = {i: w for i, w in enumerate(cw)}
    print(f"⚖️  Ağırlıklar: {cw_dict}")

    # Model
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.15),
        layers.RandomZoom(0.15), layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1)], name="augmentation")

    base = ResNet50V2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x = aug(inp)
    x = keras.applications.resnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_cls, activation="softmax")(x)
    model = Model(inp, out, name="OtomobilTanima")
    model.summary()

    # Aşama 1 — Feature Extraction
    print("\n🚀 AŞAMA 1 — Feature Extraction")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1, class_weight=cw_dict)

    # Aşama 2 — Fine-Tuning
    print("\n🔧 AŞAMA 2 — Fine-Tuning")
    base.trainable = True
    for layer in base.layers[:-UNFREEZE]: layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        ModelCheckpoint(os.path.join(MODELS, "best_model.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P2,
                   class_weight=cw_dict, callbacks=cbs)

    # Grafik
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    lo = h1.history["loss"] + h2.history["loss"]
    vlo = h1.history["val_loss"] + h2.history["val_loss"]
    ep = range(1, len(acc)+1)
    p1e = len(h1.history["accuracy"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eğitim Sonuçları", fontsize=14, fontweight="bold")
    ax[0].plot(ep, acc, "b-", label="Eğitim"); ax[0].plot(ep, vacc, "r-", label="Doğrulama")
    ax[0].axvline(p1e, color="gray", ls="--", alpha=.6); ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, lo, "b-", label="Eğitim"); ax[1].plot(ep, vlo, "r-", label="Doğrulama")
    ax[1].axvline(p1e, color="gray", ls="--", alpha=.6); ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "training_history.png"), dpi=150)
    plt.close()

    vl, va = model.evaluate(val_ds, verbose=0)
    print(f"\n🏆 TAMAMLANDI! Val Accuracy: {va:.4f} ({va*100:.1f}%)")

if __name__ == "__main__":
    main()
