from src.conf import get_params, set_seeds
from src.model import create_model
from src.data import get_dataloader
from tqdm.keras import TqdmCallback


def train(ctx):

    # Set seeds to get reproducible results
    set_seeds(ctx.seed)
    dl = get_dataloader(ctx)
    model_unconstrained = create_model(ctx)
    model_unconstrained.fit(
        dl,
        epochs=25,
        steps_per_epoch=1000,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2)],
    )


if __name__ == "__main__":
    ctx = get_params()
    train(ctx)
