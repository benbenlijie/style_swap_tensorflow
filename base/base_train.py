import tensorflow as tf
import time
import sys


class BaseTrain:
    def __init__(self, sess: tf.Session, model, config):
        self.sess = sess
        self.model = model
        self.config = config
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def train(self):
        tf.logging.info("Start to Train")
        if self.model.init_op is not None:
            self.model.init_op(self.sess)
        self.model.load(self.sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                start_time = time.time()
                self.train_step()
                elapsed_time = time.time() - start_time
                self.log_step(elapsed_time)

        except tf.errors.OutOfRangeError as e:
            tf.logging.info("Train finished")
        finally:
            self.model.save(self.sess)
            coord.request_stop()
        coord.join(threads)

    def train_step(self):
        raise NotImplementedError

    def log_step(self, elapsed_time=0):
        loss, step = self.sess.run([self.model.loss_op, self.model.global_step])
        if step % 10 == 0:
            sys.stdout.write("step {}, total loss {}, secs/step {}\r".format(step,
                                                                           loss,
                                                                           elapsed_time))
            sys.stdout.flush()
        if step % 25 == 0:
            summary_str = self.sess.run(self.model.summary_op)
            self.model.summary.add_summary(summary_str, step)
            self.model.summary.flush()
        if step % 500 == 0:
            self.model.save(self.sess, global_step=step)
