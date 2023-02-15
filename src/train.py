import tensorflow as tf
from src.conf import get_params, set_seeds
from src.model import (
    create_generator_model,
    create_discriminator_model,
    generator_loss,
    discriminator_loss,
)
from src.data import get_dataloader
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import time


# @tf.function
def train_step(
    images,
    generator,
    discriminator,
    noise_dim,
    generator_optimizer,
    discriminator_optimizer,
    batch_size,
):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        _images = tf.expand_dims(
            tf.reduce_mean(images, axis=-1), axis=-1
        )  # TODO: Fix the generator and discriminator so it works with RGB.
        generated_images = generator(noise, training=True)

        real_output = discriminator(_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


import matplotlib.pyplot as plt


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.close()
    # plt.show()


def train(ctx):

    # Set seeds to get reproducible results
    set_seeds(ctx.seed)
    dl = get_dataloader(ctx)
    generator = create_generator_model(ctx)
    discriminator = create_discriminator_model(ctx)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    BATCH_SIZE = 8

    for epoch in tqdm(range(EPOCHS), desc=" outer", position=0):
        start = time.time()

        for imgs, labels in tqdm(
            dl, desc=" inner loop", position=1, leave=False, total=len(dl)
        ):
            train_step(
                imgs,
                generator,
                discriminator,
                noise_dim,
                generator_optimizer,
                discriminator_optimizer,
                BATCH_SIZE,
            )
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        generate_and_save_images(generator, epoch + 1, seed)
    # decision = discriminator(generated_image)
    # model_unconstrained = create_model(ctx)
    # model_unconstrained.fit(
    #    dl,
    #    epochs=25,
    #    steps_per_epoch=1000,
    #    verbose=0,
    #    callbacks=[TqdmCallback(verbose=2)],
    # )


if __name__ == "__main__":
    ctx = get_params()
    train(ctx)
