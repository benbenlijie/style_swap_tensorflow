import tensorflow as tf

from data_loader import *
from models.style_swap_model import StyleSwapModel
from trainers.style_swap_trainer import StyleSwapTrainer
from utils.config import process_config
from utils.utils import get_args, save_image, load_image, get_data_files
import numpy as np
import os

TRAIN = False


def evaluate(config):

    contents = get_data_files(r"..\images\content\*")
    styles = get_data_files(r"..\images\style\*")

    model = StyleSwapModel(config, [None, None])

    for c in contents:
        print(c)
        ci = load_image(c, 512)

        with tf.Graph().as_default():
            model.evaluate_height, model.evaluate_width = ci.shape[:2]
            model.init_evaluate_model()
            with tf.Session() as sess:
                if model.init_op is not None:
                    model.init_op(sess)
                model.load(sess)

                for s in styles:
                    si = load_image(s, 512)
                    inversed = sess.run(model.evaluate_op, feed_dict={
                        model.input_image: ci, model.style_image: si,
                    })
                    inversed = np.array(inversed, dtype=np.uint8)
                    c_name = os.path.splitext(os.path.split(c)[-1])[0]
                    s_name = os.path.splitext(os.path.split(s)[-1])[0]
                    save_image(inversed, "../images/result/{}_{}.jpg".format(c_name, s_name))



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
        print("missing or invalid arguments")
        exit(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    if TRAIN:
        train(config)
    else:
        evaluate(config)


if __name__ == '__main__':
    main()
