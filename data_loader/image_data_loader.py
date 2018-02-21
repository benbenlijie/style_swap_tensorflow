from utils.utils import get_data_files
from data_loader.data_loader import BaseDataLoader
import tensorflow as tf


class ImageDataLoader(BaseDataLoader):
    def __init__(self, config, shuffle=True):
        super(ImageDataLoader, self).__init__(config, shuffle)

    def get_data(self):
        data_files = get_data_files(self.config.Image_files)
        # print("data files", data_files)
        filename_queue = tf.train.string_input_producer(
            data_files, num_epochs=self.config.num_epochs, shuffle=self.shuffle,
            name='filenames')
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image = tf.image.decode_image(value)

        return image

