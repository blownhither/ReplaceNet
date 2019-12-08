import tensorflow as tf

from ReplaceNet import ReplaceNet


def guarded_log(tensor):
    return tf.log(tf.maximum(tensor, 1e-10))


class ReplaceGAN(ReplaceNet):
    def __init__(self, patch_size=512, skip_connection="add", input_img=None, truth_img=None,
                 input_mask=None, discriminator_real_input=None, ref_mask=None):
        super().__init__(patch_size=patch_size, skip_connection=skip_connection,
                         input_img=input_img, truth_img=truth_img, input_mask=input_mask,
                         ref_mask=ref_mask)
        # redefine these for easier tuning
        self.down_channels = [64, 64, 128, 128, 256, 256, 512]
        self.up_channels = [512, 256, 256, 128, 128, 64, 64]
        # new hyperparamters
        self.discriminator_channels = [64, 64, 128, 128, 256, 256, 512]
        self.lr = None  # DEPRECATED
        self.generator_lr = 1e-3
        self.discriminator_lr = 1e-3

        # i/o tensors
        self.discriminator_real_input = discriminator_real_input or tf.placeholder(dtype=tf.float32,
            shape=[None, patch_size, patch_size, 3])

        # tensor set after building for discriminator
        self.real_prediction = None
        self.fake_prediction = None
        self.loss = None  # DEPRECATE
        self.l2_loss = None
        self.elpips_distance = None
        self.generator_adversarial_loss = None
        self.generator_loss = None  # l2 + elpips + generator_adversarial_loss
        self.discriminator_loss = None
        self.generator_train_op = None
        self.discriminator_train_op = None

    def _build_discriminator(self, tensor, is_training=True):
        """build a discriminator that convolutes on input tensor until it's single pixel
        Output scalar value prediction in (0, 1) for confidence in the input being real/fake
        """
        layers = []
        for i, c in enumerate(self.discriminator_channels):
            tensor = tf.layers.conv2d(tensor, c, [4, 4], strides=(2, 2), activation=None,
                                      padding='SAME')
            tensor = tf.layers.batch_normalization(tensor, training=is_training)
            tensor = tf.nn.elu(tensor)
            layers.append(tensor)
        print('down', layers)
        size = tensor.get_shape().as_list()[1:3]
        tensor = tf.layers.conv2d(tensor, 1, kernel_size=size, strides=size, padding='VALID',
                                  activation=tf.nn.sigmoid)
        prediction = tf.squeeze(tensor, [1, 2], name='discriminator_out')
        print('Discriminator out', prediction)
        return prediction

    def build(self, is_training):
        with tf.variable_scope('generator'):
            self.down_layers, encoder_fc = self._build_down(self.input_img, self.input_mask,
                                                            is_training)
            self.output_img, up_layers = self._build_up(encoder_fc, self.down_layers, is_training)
        with tf.variable_scope('discriminator'):
            real_prediction = self._build_discriminator(self.discriminator_real_input, is_training)
        with tf.variable_scope('discriminator', reuse=True):
            fake_prediction = self._build_discriminator(self.output_img, is_training)

        # Generator loss
        self.l2_loss = tf.losses.mean_squared_error(self.truth_img, self.output_img)
        tf.summary.scalar('l2-loss', self.l2_loss)
        self.elpips_distance = self.metric.forward(self.truth_img, self.output_img)[0]
        tf.summary.scalar('elpips', self.elpips_distance)
        self.generator_adversarial_loss = tf.reduce_mean(-guarded_log(fake_prediction))
        self.generator_loss = self.l2_loss + self.elpips_distance + self.generator_adversarial_loss

        # Discriminator loss
        self.discriminator_loss = tf.reduce_mean(
            -guarded_log(real_prediction) - guarded_log(1 - fake_prediction))

        # train operations
        gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        print('Generator variables', gen_variables)
        gen_update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
        print('Generator update ops', gen_update_ops)
        op = tf.train.AdamOptimizer(self.generator_lr).minimize(self.generator_loss,
                var_list=gen_variables, global_step=tf.train.get_or_create_global_step())
        self.generator_train_op = tf.group([op, gen_update_ops])

        dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        dis_update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
        print('Discriminator update ops', dis_update_ops)
        op = tf.train.AdamOptimizer(self.discriminator_lr).minimize(self.discriminator_loss,
                var_list=dis_variables, global_step=tf.train.get_or_create_global_step())
        self.discriminator_train_op = tf.group([op, dis_update_ops])

        self.merged_summary = tf.summary.merge_all()
        self.global_step = tf.train.get_or_create_global_step()


def test():
    g = ReplaceGAN()
    g.build(True)


if __name__ == '__main__':
    test()

