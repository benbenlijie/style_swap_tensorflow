import tensorflow as tf

from data_loader import *
from models.style_swap_model import StyleSwapModel
from trainers.style_swap_trainer import StyleSwapTrainer
from utils.config import process_config
from utils.utils import *
import numpy as np
import os


def evaluate(config, content_pattern, style_pattern):

    contents = get_data_files(content_pattern)
    styles = get_data_files(style_pattern)

    model = StyleSwapModel(config, [None, None])

    for c_file_path in contents:
        if is_image(c_file_path):
            ci = load_image(c_file_path, 512)
            save_folder = os.path.split(c_file_path)[0]
            c_name = os.path.splitext(os.path.split(c_file_path)[-1])[0]
            evaluate_size = ci.shape[:2]

            def save(stylize_fn, s_name):
                save_image(stylize_fn(ci), "{}/{}_{}.jpg".format(save_folder, c_name, s_name))
        if is_video(c_file_path):
            cap, fps, size = get_video_capture(c_file_path)
            evaluate_size = size[::-1]

            def save(stylize_fn, s_name):
                video_stylize(c_file_path, s_name, cap, stylize_fn)
        with tf.Graph().as_default():
            model.evaluate_height, model.evaluate_width = evaluate_size
            model.init_evaluate_model()
            with tf.Session() as sess:
                if model.init_op is not None:
                    model.init_op(sess)
                model.load(sess)

                for s_file_path in styles:
                    if is_image(s_file_path):
                        si = load_image(s_file_path, 512)

                        def stylize(origin):
                            inversed = sess.run(model.evaluate_op, feed_dict={
                                model.input_image: origin, model.style_image: si,
                            })
                            return np.array(inversed, dtype=np.uint8)

                        s_name = os.path.splitext(os.path.split(s_file_path)[-1])[0]
                        save(stylize, s_name)
        if is_video(c_file_path):
            cap.release()


def train(config):
    record_loader = COCODataLoader(config, False)
    config.num_epochs = None
    image_loader = ImageDataLoader(config, True)

    model = StyleSwapModel(config, [record_loader, image_loader])
    model.init_train_model()
    with tf.Session() as sess:
        trainer = StyleSwapTrainer(sess, model, config)
        trainer.train()


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments", e)
        exit(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    if args.stylize:
        evaluate(config, args.content, args.style)
    else:
        train(config)


if __name__ == '__main__':
    main()
