from data_loader.data_loader import RecordDataLoader
import tensorflow as tf
slim = tf.contrib.slim


class COCODataLoader(RecordDataLoader):
    def _define_features(self):
        size = 512
        self.keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, ''),
            "image/format": tf.FixedLenFeature((), tf.string, 'jpeg'),
            "image/height": tf.FixedLenFeature([], tf.int64, tf.zeros([], tf.int64)),
            "image/width": tf.FixedLenFeature([], tf.int64, tf.zeros([], tf.int64)),
        }

        self.items_to_handlers = {
            "image": slim.tfexample_decoder.Image(shape=[size, size, 3]),
        }