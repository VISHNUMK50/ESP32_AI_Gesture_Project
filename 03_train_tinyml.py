import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV = "gesture_data.csv"
MODEL = "gesture_model.tflite"

df = pd.read_csv(CSV)
y = df["label"].values.astype(np.int32)
X = df.drop(columns=["label"]).values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# The dataset is already normalized geometrically.
# Keep the same input range for the ESP32.
X_train = np.clip(X_train, -2.0, 2.0)
X_test = np.clip(X_test, -2.0, 2.0)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=35,
    batch_size=16,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")

# Float32 TFLite model: simplest and most reliable for a 3-hour demo.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(MODEL, "wb") as f:
    f.write(tflite_model)

print("Created:", MODEL)
print("Copy gesture_model.tflite into the Arduino sketch folder.")
