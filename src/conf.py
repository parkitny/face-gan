from pathlib import Path
import munch
import yaml
import numpy as np
import tensorflow as tf

PARAMS = "params.yaml"


def get_params(params_file=PARAMS):
    with Path(params_file).open("rt") as fh:
        params = munch.munchify(yaml.safe_load(fh.read()))
    return params


def set_seeds(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
