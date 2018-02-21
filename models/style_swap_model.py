import tensorflow as tf

from base.base_model import BaseModel
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


class StyleSwapModel(BaseModel):
    def __init__(self, config, data_loader):
        super(StyleSwapModel, self).__init__(config, data_loader)
        self.style_layer = "/conv3/conv3_3"
        self.records_loader = data_loader[0]
        self.style_loader = data_loader[1]
        self.PREPROCESS_SIZE = 256
        self.cell_size = 3

    def _build_train_model(self):
        preprocess_fn = preprocessing_factory.get_preprocessing(self.config.net_name, is_training=False)
        [image] = self.records_loader.get_data()
        preprocessed_image = preprocess_fn(image, self.PREPROCESS_SIZE, self.PREPROCESS_SIZE)
        images = self.records_loader.batch_data(preprocessed_image)

        style_image = self.style_loader.get_data()
        preprocessed_style_image = preprocess_fn(style_image, self.PREPROCESS_SIZE, self.PREPROCESS_SIZE)
        style_images = self.style_loader.batch_data(preprocessed_style_image)

        self.swaped_tensor = self._swap_net(images, style_images)
        self.generated = self._inverse_net(self.swaped_tensor)
        slim.summary.image("generated", self.generated)
        slim.summary.image("origin", images)
        slim.summary.image("style", style_images)
        self._train_inverse(self.generated, self.swaped_tensor)

        self.init_op = self._get_network_init_fn()

    def _build_evaluate_model(self):
        self.input_image = tf.placeholder(tf.float32, shape=[None, None, 3])
        self.style_image = tf.placeholder(tf.float32, shape=[None, None, 3])
        preprocess_fn = preprocessing_factory.get_preprocessing(self.config.net_name, is_training=False)

        height = self.evaluate_height if self.evaluate_height else self.PREPROCESS_SIZE
        width = self.evaluate_width if self.evaluate_width else self.PREPROCESS_SIZE

        preprocessed_image = preprocess_fn(self.input_image, height, width, resize_side_min=min(height, width))
        images = tf.expand_dims(preprocessed_image, axis=0)

        style_images = tf.expand_dims(preprocess_fn(self.style_image, self.PREPROCESS_SIZE, self.PREPROCESS_SIZE), axis=0)

        self.swaped_tensor = self._swap_net(images, style_images)

        #
        # network_fn = nets_factory.get_network_fn(self.config.net_name, num_classes=1, is_training=False)
        # _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        # self.swaped_tensor = endpoints_dict[self.config.net_name + self.style_layer]

        self.generated = self._inverse_net(self.swaped_tensor)

        self.evaluate_op = tf.squeeze(self.generated, axis=0)
        self.init_op = self._get_network_init_fn()
        self.save_variables = [var for var in tf.trainable_variables() if var.name.startswith("inverse_net")]

    def _swap_net(self, content, style):
        network_fn = nets_factory.get_network_fn(self.config.net_name, num_classes=1, is_training=False)
        # content_amount = content.get_shape()[0].value
        style_amount = style.get_shape()[0].value
        #
        # images = tf.concat([content, style], axis=0)
        _, endpoints_dict = network_fn(content, spatial_squeeze=False)
        content_feature = endpoints_dict[self.config.net_name + self.style_layer]

        with tf.variable_scope("", reuse=True):
            _, endpoints_dict = network_fn(style, spatial_squeeze=False)
            layer_names = list(endpoints_dict.keys())
            [layer_name] = [l_name for l_name in layer_names if self.style_layer in l_name]
            style_feature = endpoints_dict[layer_name]

        # content_feature, style_feature = tf.split(style_layer, num_or_size_splits=[content_amount, style_amount],
        #                                           axis=0)
        #
        # print(content_feature.get_shape())

        rows = tf.split(style_feature, num_or_size_splits=list(
            [self.cell_size] * (style_feature.get_shape()[1].value // self.cell_size) + [style_feature.get_shape()[1].value % self.cell_size]), axis=1)[:-1]
        cells = [tf.split(row, num_or_size_splits=list(
            [self.cell_size] * (style_feature.get_shape()[2].value // self.cell_size) + [style_feature.get_shape()[2].value % self.cell_size]), axis=2)[:-1]
                 for row in rows]

        stacked_cells = [tf.stack(row_cell, axis=4) for row_cell in cells]
        filters = tf.concat(stacked_cells, axis=-1)
        swaped_list = []
        for style_filter in tf.unstack(filters, axis=0, num=style_amount):
            swaped_list.append(self._swap_op(content_feature, style_filter))

        return tf.concat(swaped_list, axis=0)

    def _train_inverse(self, generated, swaped_tensor):
        preprocess_fn = preprocessing_factory.get_preprocessing(self.config.net_name, is_training=False)
        network_fn = nets_factory.get_network_fn(self.config.net_name, num_classes=1, is_training=False)
        with tf.variable_scope("", reuse=True):
            preprocessed_image = tf.stack([preprocess_fn(img, self.PREPROCESS_SIZE, self.PREPROCESS_SIZE)
                                           for img in tf.unstack(generated, axis=0)])
            _, inversed_endpoints_dict = network_fn(preprocessed_image, spatial_squeeze=False)
            layer_names = list(inversed_endpoints_dict.keys())
            [layer_name] = [l_name for l_name in layer_names if self.style_layer in l_name]
            inversed_style_layer = inversed_endpoints_dict[layer_name]
        # print(inversed_style_layer.get_shape())
        tf.losses.add_loss(tf.nn.l2_loss(swaped_tensor - inversed_style_layer))
        self.loss_op = tf.losses.get_total_loss()

        train_vars = [var for var in tf.trainable_variables() if var.name.startswith("inverse_net")]
        slim.summarize_tensor(self.loss_op, "loss")
        slim.summarize_tensors(train_vars)
        # print(train_vars)
        self.save_variables = train_vars

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, 1000, 0.66,
                                                   name="learning_rate")
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_op, self.global_step, train_vars)

    def _swap_op(self, content_feature, style_feature):
        height = tf.shape(content_feature)[1]
        width = tf.shape(content_feature)[2]
        print(style_feature)
        normalized_filters = tf.nn.l2_normalize(style_feature, dim=(0, 1, 2))

        """ change the strides to see difference"""
        similarity = tf.nn.conv2d(content_feature, normalized_filters, strides=[1, 1, 1, 1], padding="VALID")

        arg_max_filter = tf.argmax(similarity, axis=-1)
        one_hot_filter = tf.one_hot(arg_max_filter, depth=similarity.get_shape()[-1].value)

        swap = tf.nn.conv2d_transpose(one_hot_filter, style_feature, output_shape=tf.shape(content_feature),
                                      strides=[1, 1, 1, 1], padding="VALID")

        return swap / 9.0

    def _inverse_net(self, x):
        with tf.variable_scope("inverse_net"):
            with tf.variable_scope("conv1"):
                x = slim.conv2d(x, num_outputs=256, kernel_size=3, stride=1, padding="SAME",
                                weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
            with tf.variable_scope("residual1"):
                res = slim.repeat(x, 2, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,
                                  weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
                x = res + x
            with tf.variable_scope("residual2"):
                res = slim.repeat(x, 2, slim.conv2d, num_outputs=256, kernel_size=5, stride=1,
                                  weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
                x = res + x
            ## model 2 only use 2 resi module
            with tf.variable_scope("residual3"):
                res = slim.repeat(x, 2, slim.conv2d, num_outputs=256, kernel_size=7, stride=1,
                                  weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
                x = res + x
            with tf.variable_scope("deconv1"):
                x = self._deconv(x, num_outputs=128, kernel_size=5, stride=2, activation_fn=tf.nn.relu,
                                 weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
            with tf.variable_scope("deconv2"):
                x = self._deconv(x, num_outputs=64, kernel_size=5, stride=2, activation_fn=tf.nn.relu,
                                 weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
            with tf.variable_scope("conv2"):
                x = slim.conv2d(x, num_outputs=3, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.tanh,
                                weights_regularizer=slim.l2_regularizer(self.config.weight_regulation_scale))
            x = (x + 1) * (255.0 / 2)
        return x

    def _instance_norm(self, x):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    def _deconv(self, x, num_outputs, kernel_size, stride, activation_fn, weights_regularizer):
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        new_height = height * 2 * stride
        new_width = width * 2 * stride
        x = tf.image.resize_images(
            x, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR
        )
        x = slim.conv2d(x, num_outputs=num_outputs,
                        kernel_size=kernel_size, stride=stride, padding="SAME",
                        activation_fn=None, weights_regularizer=weights_regularizer)
        x = self._instance_norm(x)
        x = activation_fn(x)
        return x

    def _get_network_init_fn(self):
        tf.logging.info("Use pretrained model {}".format(self.config.net_name))
        exclusions = []
        if self.config.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in self.config.checkpoint_exclude_scopes.split(",")]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        return slim.assign_from_checkpoint_fn(
            self.config.loss_model_file,
            variables_to_restore,
            ignore_missing_vars=True
        )
