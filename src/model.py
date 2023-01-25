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
        optimizer=tf.keras.optimizers.Adam(0.001), loss={'dense_1': 'categorical_crossentropy', 
                    'dense_1': 'hinge'}, metrics=METRICS
    )
    return model


def save_model(model, subdir):
    base_dir = tempfile.mkdtemp(prefix="saved_models")
    model_location = os.path.join(base_dir, subdir)
    model.save(model_location, save_format="tf")
    return model_location


if __name__ == "__main__":
    ctx = get_params()
    create_model(ctx)
