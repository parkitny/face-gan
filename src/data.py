# https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study
import os
os.environ['NO_GCE_CHECK'] = 'true'
import tensorflow as tf
import tensorflow_datasets as tfds
import urllib
from pathlib import Path
ATTR_KEY = "attributes"
IMAGE_KEY = "image"
LABEL_KEY = "Smiling"
GROUP_KEY = "Young"
IMAGE_SIZE = 28
DATA_DIR = "data/raw"
if tf.__version__ < "2.0.0":
  tf.compat.v1.enable_eager_execution()
  print("Eager execution enabled.")
else:
  print("Eager execution enabled by default.")

import tempfile

gcs_base_dir = "gs://celeb_a_dataset/"
celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')

celeb_a_builder.download_and_prepare()

num_test_shards_dict = {'0.3.0': 4, '2.0.0': 2} # Used because we download the test dataset separately
version = str(celeb_a_builder.info.version)
print('Celeb_A dataset version: %s' % version)
local_root = tempfile.mkdtemp(prefix='test-data')
def local_test_filename_base():
  p = Path(DATA_DIR) / "test-data"
  p.mkdir(parents=True, exist_ok=True)
  return p

def local_test_file_full_prefix():
  return local_test_filename_base() / "celeb_a-test.tfrecord"

def copy_test_files_to_local():
  filename_base = local_test_file_full_prefix().as_posix()
  num_test_shards = num_test_shards_dict[version]
  for shard in range(num_test_shards):
    url = "https://storage.googleapis.com/celeb_a_dataset/celeb_a/%s/celeb_a-test.tfrecord-0000%s-of-0000%s" % (version, shard, num_test_shards)
    filename = "%s-0000%s-of-0000%s" % (filename_base, shard, num_test_shards)
    res = urllib.request.urlretrieve(url, filename)

def preprocess_input_dict(feat_dict):
  # Separate out the image and target variable from the feature dictionary.
  image = feat_dict[IMAGE_KEY]
  label = feat_dict[ATTR_KEY][LABEL_KEY]
  group = feat_dict[ATTR_KEY][GROUP_KEY]

  # Resize and normalize image.
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
  image /= 255.0

  # Cast label and group to float32.
  label = tf.cast(label, tf.float32)
  group = tf.cast(group, tf.float32)

  feat_dict[IMAGE_KEY] = image
  feat_dict[ATTR_KEY][LABEL_KEY] = label
  feat_dict[ATTR_KEY][GROUP_KEY] = group

  return feat_dict


# Train data returning either 2 or 3 elements (the third element being the group)
def celeb_a_train_data_wo_group(batch_size, celeb_a_builder, get_image_and_label):
    celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(preprocess_input_dict)
    return celeb_a_train_data.map(get_image_and_label)
def celeb_a_train_data_w_group(batch_size, celeb_a_builder, get_image_label_and_group):
    celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(preprocess_input_dict)
    return celeb_a_train_data.map(get_image_label_and_group)


def get_data(batch_size=64):
    
    get_image_and_label = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY])
    get_image_label_and_group = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY], feat_dict[ATTR_KEY][GROUP_KEY])
    gcs_base_dir = "gs://celeb_a_dataset/"
    celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')

    celeb_a_builder.download_and_prepare()
    # Test data for the overall evaluation
    celeb_a_test_data = celeb_a_builder.as_dataset(split='test').batch(1).map(preprocess_input_dict).map(get_image_label_and_group)
    # Copy test data locally to be able to read it into tfma
    copy_test_files_to_local()

    # Test data for the overall evaluation
    celeb_a_test_data = celeb_a_builder.as_dataset(split='test').batch(1).map(preprocess_input_dict).map(get_image_label_and_group)
    # Copy test data locally to be able to read it into tfma
    copy_test_files_to_local()

if __name__ == "__main__":
    get_data()