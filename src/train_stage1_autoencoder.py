import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LATENT_DIM = 64

def load_data():
    (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return x_train, x_test


def build_autoencoder():

    # Encoder
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    latent = layers.Dense(LATENT_DIM, name="latent_vector")(x)

    encoder = keras.Model(encoder_inputs, latent, name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    decoder_outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Autoencoder completo
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.Model(encoder_inputs, autoencoder_outputs, name="autoencoder")

    return encoder, decoder, autoencoder


def main():
    x_train, x_test = load_data()
    encoder, decoder, autoencoder = build_autoencoder()

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    autoencoder.summary()

    autoencoder.fit(
        x_train, x_train,
        epochs=20,
        batch_size=256,
        validation_data=(x_test, x_test)
    )

    os.makedirs("models", exist_ok=True)
    encoder.save("models/encoder.keras")
    autoencoder.save("models/autoencoder.keras")


if __name__ == "__main__":
    main()

