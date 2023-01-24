# https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NO_GCE_CHECK"] = "true"
import tensorflow as tf
import tensorflow_datasets as tfds
import urllib
from functools import partial
from pathlib import Path
import src.const as C
from src.conf import get_params
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def local_test_filename_base(ctx):
  p = Path(ctx.paths.test)
  p.mkdir(parents=True, exist_ok=True)
  return p


def local_test_file_full_prefix(ctx):
  return local_test_filename_base(ctx) / "celeb_a-test.tfrecord"


def _download(url: str, output_path: Path) -> None:
  fn = output_path / Path(url).name
  _ = urllib.request.urlretrieve(url, fn)


def copy_test_files_to_local(ctx):
  num_test_shards = ctx.data.celeb_a.shards
  urls = []
  for shard in range(num_test_shards):
    urls.append(
        "https://storage.googleapis.com/"+
        "celeb_a_dataset/celeb_a/"+
        f"{ctx.data.celeb_a.version}/"+
        "celeb_a-test."+
        f"tfrecord-0000{shard}-of-0000{num_test_shards}"
    )
  Path(ctx.paths.test).mkdir(parents=True, exist_ok=True)
  with ProcessPoolExecutor(max_workers=min(len(urls), os.cpu_count())) as executor:
    results = [executor.submit(_download, url, Path(ctx.paths.test)) for (url) in urls]
    for future in tqdm(as_completed(results), total=len(urls)):
      _ = future.result()



def preprocess_input_dict(feat_dict, size):
  # Separate out the image and target variable from the feature dictionary.
  image = feat_dict[C.IMAGE_KEY]
  label = feat_dict[C.ATTR_KEY][C.LABEL_KEY]
  group = feat_dict[C.ATTR_KEY][C.GROUP_KEY]

  # Resize and normalize image.
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [size, size])
  image /= 255.0

  # Cast label and group to float32.
  label = tf.cast(label, tf.float32)
  group = tf.cast(group, tf.float32)

  feat_dict[C.IMAGE_KEY] = image
  feat_dict[C.ATTR_KEY][C.LABEL_KEY] = label
  feat_dict[C.ATTR_KEY][C.GROUP_KEY] = group

  return feat_dict


# Train data returning either 2 or 3 elements (the third element being the group)
def celeb_a_train_data_wo_group(ctx, batch_size, celeb_a_builder, get_image_and_label):
  celeb_a_train_data = (
    celeb_a_builder.as_dataset(split="train")
    .shuffle(1024)
    .repeat()
    .batch(batch_size)
    .map(partial(preprocess_input_dict, size=ctx.image_size))
  )
  return celeb_a_train_data.map(get_image_and_label)


def celeb_a_train_data_w_group(batch_size, celeb_a_builder, get_image_label_and_group):
  celeb_a_train_data = (
    celeb_a_builder.as_dataset(split="train")
    .shuffle(1024)
    .repeat()
    .batch(batch_size)
    .map(partial(preprocess_input_dict, size=ctx.image_size))
  )
  return celeb_a_train_data.map(get_image_label_and_group)


def get_data(ctx, batch_size=64):

  get_image_and_label = lambda feat_dict: (
    feat_dict[C.IMAGE_KEY],
    feat_dict[C.ATTR_KEY][C.LABEL_KEY],
  )
  get_image_label_and_group = lambda feat_dict: (
    feat_dict[C.IMAGE_KEY],
    feat_dict[C.ATTR_KEY][C.LABEL_KEY],
    feat_dict[C.ATTR_KEY][C.GROUP_KEY],
  )
  celeb_a_builder = tfds.builder(
    ctx.data.celeb_a.name,
    data_dir=ctx.data.celeb_a.url,
    version=ctx.data.celeb_a.version,
  )

  celeb_a_builder.download_and_prepare()
  # Test data for the overall evaluation
  celeb_a_test_data = (
      celeb_a_builder.as_dataset(split="test")
      .batch(1)
      .map(partial(preprocess_input_dict, size=ctx.data.image_size))
      .map(get_image_label_and_group)
  )
  # Copy test data locally to be able to read it into tfma
  copy_test_files_to_local(ctx)


if __name__ == "__main__":
    ctx = get_params()
    get_data(ctx)
