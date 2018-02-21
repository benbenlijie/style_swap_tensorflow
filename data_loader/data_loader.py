import tensorflow as tf
from bunch import Bunch

slim = tf.contrib.slim


class BaseDataLoader:
    def __init__(self, config: Bunch, shuffle=True):
        self.config = config
        self.shuffle = shuffle

    def get_data(self):
        raise NotImplementedError

    def batch_data(self, input_tensor=None, num_threads=4, capacity=4):
        input_tensor = input_tensor if input_tensor is not None else self.get_data()
        if self.shuffle:
            images = tf.train.shuffle_batch(
                [input_tensor],
                batch_size=self.config.batch_size,
                capacity=5*self.config.batch_size,
                num_threads=num_threads,
                min_after_dequeue=2*self.config.batch_size,
            )
        else:
            images = tf.train.batch(
                [input_tensor],
                batch_size=self.config.batch_size,
                num_threads=num_threads,
                capacity=5 * self.config.batch_size
            )
        image_queue = slim.prefetch_queue.prefetch_queue([images], capacity=capacity)
        return image_queue.dequeue()


class RecordDataLoader(BaseDataLoader):
    def __init__(self, config: Bunch, shuffle=True):
        super(RecordDataLoader, self).__init__(config, shuffle)
        self.keys_to_features = None
        self.items_to_handlers = None
        self._define_features()

    def _define_features(self):
        raise NotImplementedError

    def get_data(self):
        reader = tf.TFRecordReader
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            self.keys_to_features, self.items_to_handlers
        )
        data_set = slim.dataset.Dataset(
            data_sources=self.config.record_file_name,
            reader=reader,
            decoder=decoder,
            items_to_descriptions={},
            num_samples=self.config.num_sampels
        )
        provider = slim.dataset_data_provider.DatasetDataProvider(
            data_set, shuffle=self.shuffle, num_epochs=self.config.num_epochs)

        item_keys = list(self.items_to_handlers.keys())
        return provider.get(item_keys)


