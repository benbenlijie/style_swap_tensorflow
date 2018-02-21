import tensorflow as tf

from base.base_train import BaseTrain


class StyleSwapTrainer(BaseTrain):
    def __init__(self, sess: tf.Session, model, config):
        super(StyleSwapTrainer, self).__init__(sess, model, config)

    def train_step(self):
        self.sess.run(self.model.train_op)
