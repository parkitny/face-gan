import tensorflow as tf
from tensorflow import keras
import tempfile
import os
from src.conf import get_params


def create_model(ctx):
    # For this notebook, accuracy will be used to evaluate performance.
    METRICS = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]

    # The model consists of:
    # 1. An input layer that represents the 28x28x3 image flatten.
    # 2. A fully connected layer with 64 units activated by a ReLU function.
    # 3. A single-unit readout layer to output real-scores instead of probabilities.
    model = keras.Sequential(
        [
            keras.layers.Flatten(
                input_shape=(ctx.data.image_size, ctx.data.image_size, 3), name="image"
            ),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation=None),
        ]
    )

    # TFCO by default uses hinge loss — and that will also be used in the model.

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={"dense_1": "categorical_crossentropy", "dense_1": "hinge"},
        metrics=METRICS,
    )
    return model


def create_generator_model(ctx):
    model = keras.Sequential()

    model.add(keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model


def create_discriminator_model(ctx):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
        )
    )
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def save_model(model, subdir):
    base_dir = tempfile.mkdtemp(prefix="saved_models")
    model_location = os.path.join(base_dir, subdir)
    model.save(model_location, save_format="tf")
    return model_location


if __name__ == "__main__":
    ctx = get_params()
    create_model(ctx)
