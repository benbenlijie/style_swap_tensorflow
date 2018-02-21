import argparse
import tensorflow as tf
import scipy.misc as sc
import cv2
import os


VIDEO_EXTS = [".mp4", ".avi", ".mkv", ".flv"]
IMG_EXTS = [".jpg", ".png"]


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


def is_video(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    return ext in VIDEO_EXTS


def is_image(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    return ext in IMG_EXTS


def get_video_capture(file_path):
    capture = cv2.VideoCapture(file_path)
    fps = capture.get(cv2.cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)))
    return capture, fps, size


def video_stylize(file_path, style_name, capture, style_fn):
    folder, file_name = os.path.split(file_path)
    file_name = os.path.join(folder, os.path.splitext(file_name)[0] + "_" + style_name + ".avi")
    print("save video at", file_name)
    fps = capture.get(cv2.cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    count = 1
    while capture.isOpened():
        ret, frame = capture.read()
        print(count)
        count += 1
        if ret:
            s = style_fn(frame)
            s = cv2.resize(s, size)
            writer.write(cv2.cvtColor(s, cv2.COLOR_RGB2BGR))
        else:
            break
    writer.release()

