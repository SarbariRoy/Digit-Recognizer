# 2‑Model Ensemble for Digit Recognizer (Keras)
# ---------------------------------------------
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

SEED1, SEED2 = 42, 99
np.random.seed(SEED1); tf.random.set_seed(SEED1)

import sys, tensorflow as tf, keras, numpy as np
print("exe   :", sys.executable)  # should be /opt/anaconda3/envs/kaggle_py39/bin/python
print("tf    :", tf.__version__)  # 2.15.1
print("keras :", keras.__version__)
print("numpy :", np.__version__)
print("GPUs  :", tf.config.list_physical_devices("GPU"))

# 2‑Model Ensemble for Digit Recognizer (Keras)
# ---------------------------------------------
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

SEED1, SEED2 = 42, 99
np.random.seed(SEED1); tf.random.set_seed(SEED1)

# --- Load CSVs ---
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
y = train["label"].values
X = train.drop(columns=["label"]).values.astype("float32") / 255.0
X_test = test.values.astype("float32") / 255.0
X = X.reshape(-1, 28,28,1)
X_test = X_test.reshape(-1, 28,28,1)

# Train/val split (same for both models for fair comparison)
X_tr, X_val, y_tr_int, y_val_int = train_test_split(
    X, y, test_size=0.1, random_state=SEED1, stratify=y
)
num_classes = 10
y_tr = keras.utils.to_categorical(y_tr_int, num_classes)
y_val = keras.utils.to_categorical(y_val_int, num_classes)

# --- Light augmentation ---
aug = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
)
aug_tr = aug.flow(X_tr, y_tr, batch_size=128, shuffle=True)

def build_model(variant=1, seed=42):
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(28,28,1))
    x = inputs
    # Shared stem
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    # Variant branch for diversity
    if variant == 1:
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    else:  # slightly different kernel sizes/filters
        x = layers.Conv2D(48, 5, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128 if variant==1 else 160, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

cbs = [
    callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2,
                                min_lr=1e-5, verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max",
                            restore_best_weights=True, verbose=1),
]

steps_per_epoch = int(np.ceil(len(X_tr) / 128))

# --- Train Model A ---
model_a = build_model(variant=1, seed=SEED1)
hist_a = model_a.fit(aug_tr, validation_data=(X_val, y_val),
                     epochs=30, steps_per_epoch=steps_per_epoch,
                     callbacks=cbs, verbose=2)

# --- Train Model B ---
model_b = build_model(variant=2, seed=SEED2)
# re-init generator (optional)
aug_tr_b = aug.flow(X_tr, y_tr, batch_size=128, shuffle=True)
hist_b = model_b.fit(aug_tr_b, validation_data=(X_val, y_val),
                     epochs=30, steps_per_epoch=steps_per_epoch,
                     callbacks=cbs, verbose=2)

# --- Ensemble: average softmax probabilities ---
probs_a = model_a.predict(X_test, batch_size=256, verbose=0)
probs_b = model_b.predict(X_test, batch_size=256, verbose=0)
probs_ens = (probs_a + probs_b) / 2.0
labels = probs_ens.argmax(axis=1)

# Optional: check val performance of ensemble (on X_val)
val_pa = model_a.predict(X_val, verbose=0)
val_pb = model_b.predict(X_val, verbose=0)
val_pens = (val_pa + val_pb) / 2.0
val_acc_ens = (val_pens.argmax(1) == y_val_int).mean()
print(f"Ensemble validation accuracy: {val_acc_ens:.4f}")

# --- Submission ---
sub = pd.DataFrame({"ImageId": np.arange(1, len(labels)+1), "Label": labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
