import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LATENT_DIM = 64
NUM_CLASSES = 10

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return (x_train, y_train), (x_test, y_test)


def build_model():

    encoder = keras.models.load_model("models/encoder.keras")
    encoder.trainable = False

    inputs = keras.Input(shape=(28, 28, 1))
    latent = encoder(inputs)

    x = layers.Dense(128, activation="relu")(latent)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model


def main():

    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=256,
        validation_data=(x_test, y_test)
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/final_classifier.keras")


if __name__ == "__main__":
    main()
