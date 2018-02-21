import tensorflow as tf
import os
from bunch import Bunch


class BaseModel:
    def __init__(self, config: Bunch, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.save_variables = None

    def init_train_model(self):
        self._init_global_step()
        self._build_train_model()
        self._init_saver()
        self._init_summary()

    def init_evaluate_model(self):
        self._build_evaluate_model()
        self._init_saver()

    def _init_global_step(self):
        self.global_step = tf.train.get_or_create_global_step()

    def _init_summary(self):
        if not tf.gfile.Exists(self.config.summary_dir):
            tf.gfile.MakeDirs(self.config.summary_dir)
        self.summary_op = tf.summary.merge_all()
        self.summary = tf.summary.FileWriter(logdir=self.config.summary_dir, graph=tf.get_default_graph())

    def _init_saver(self):

        if not tf.gfile.Exists(self.config.checkpoint_dir):
            tf.gfile.MakeDirs(self.config.checkpoint_dir)
        self.saver = tf.train.Saver(self.save_variables)

    def save(self, sess, global_step=None):
        tf.logging.info("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), global_step)
        tf.logging.info("Model saved")

    def load(self, sess):
        print(os.path.join(self.config.checkpoint_dir))
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.config.checkpoint_dir))
        if latest_checkpoint:
            tf.logging.info("Loading model checkpoint from {}".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            tf.logging.info("Model loaded")

    def _build_train_model(self):
        raise NotImplementedError

    def _build_evaluate_model(self):
        raise NotImplementedError

