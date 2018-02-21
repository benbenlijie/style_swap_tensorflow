import argparse
import tensorflow as tf
import scipy.misc as sc


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument('--content', default="images/content/*")
    argparser.add_argument('--style', default="images/style/*")
    argparser.add_argument('-s', '--stylize', default=False, action='store_true')
    args = argparser.parse_args()
    return args


def get_data_files(data_sources):
    """Get data_files from data_sources.

    Args:
      data_sources: a list/tuple of files or the location of the data, i.e.
        /path/to/train@128, /path/to/train* or /tmp/.../train*

    Returns:
      a list of data_files.

    Raises:
      ValueError: if not data files are not found

    """
    if isinstance(data_sources, (list, tuple)):
        data_files = []
        for source in data_sources:
            data_files += get_data_files(source)
    else:
        if '*' in data_sources or '?' in data_sources or '[' in data_sources:
            data_files = tf.gfile.Glob(data_sources)
        else:
            data_files = [data_sources]
    if not data_files:
        raise ValueError('No data files found in %s' % (data_sources,))
    return data_files


def save_image(image, file_path):
    print("Save image at {}".format(file_path))
    sc.imsave(file_path, image)


def load_image(file_path, min_side=None):
    img = sc.imread(file_path)
    height, width = img.shape[:2]
    if min_side is not None:
        if min(height, width) < min_side:
            rate = min_side * 1.0 / min(height, width)
            img = sc.imresize(img, [int(height * rate), int(width * rate)])
    return img
