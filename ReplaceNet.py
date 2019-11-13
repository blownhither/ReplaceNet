import os
import tensorflow as tf


class ReplaceNet:
    """
    This is a variation of UNet setting from "Deep Image Harmonization"
    The bottom branch (scene parsing decoder) is not currently included.
    """

    def __init__(self, patch_size=512, input_img=None, truth_img=None, input_mask=None,
                 ref_mask=None):
        # hyperparameters
        self.size = patch_size
        self.down_channels = [64, 64, 128, 128, 256, 256, 512]
        self.fc_size = 1024
        self.up_channels = [512, 256, 256, 128, 128, 64, 64]
        self.batch_size = None
        self.lr = 1e-3

        # i/o tensors
        # input img should be patches in size 512
        self.input_img = input_img or tf.placeholder(shape=[None, self.size, self.size, 3],
                                                     dtype=tf.float32)
        self.truth_img = truth_img or tf.placeholder(shape=[None, self.size, self.size, 3],
                                                     dtype=tf.float32)
        # `input_mask` is applied on `input_img` to locate foreground
        self.input_mask = input_mask or tf.placeholder(shape=[None, self.size, self.size],
                                                       dtype=tf.float32)
        # `ref_mask + input_mask` is the area to apply inpainting
        self.ref_mask = ref_mask

        self.output_img = None

        # internal tensors, set after building
        self.down_layers = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.merged_summary = None
        self.global_step = None
        self.saver = None

    def build(self, is_training):
        self.down_layers, encoder_fc = self._build_down(self.input_img, self.input_mask,
                                                        is_training)
        self.output_img, up_layers = self._build_up(encoder_fc, self.down_layers, is_training)

        self.loss = tf.losses.mean_squared_error(self.truth_img, self.output_img)
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # for Batch Norm update
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = self.optimizer.minimize(self.loss,
                                           global_step=tf.train.get_or_create_global_step())
        self.train_op = tf.group([train_op, update_ops])
        self.merged_summary = tf.summary.merge_all()
        self.global_step = tf.train.get_or_create_global_step()

    def _build_down(self, img, mask, is_training):
        # down sampling
        down_layers = []
        if len(mask.shape) == 3:  # add 4th dimension
            mask = tf.expand_dims(mask, 3)
        with tf.name_scope('encoder'):
            tensor = tf.concat([img, mask], axis=3)
            for i, c in enumerate(self.down_channels):
                tensor = tf.layers.conv2d(tensor, c, [4, 4], strides=(2, 2), activation=None,
                                          padding='SAME')
                tensor = tf.layers.batch_normalization(tensor, training=is_training)
                tensor = tf.nn.elu(tensor)
                down_layers.append(tensor)
            print('down', down_layers)
            tensor = tf.layers.dense(tf.layers.flatten(tensor), self.fc_size, name='out')
            return down_layers, tensor

    def _build_up(self, fc, down_layers, is_training):
        up_layers = []
        with tf.name_scope('h-decoder'):
            tensor = tf.reshape(fc, [-1, 1, 1, self.fc_size])

            for i, c in enumerate(self.up_channels):
                # TODO: check different stride setting
                tensor = tf.layers.conv2d_transpose(tensor, c, [4, 4],
                                                    strides=[2, 2] if i != 0 else [4, 4],
                                                    activation=None, padding='SAME')
                tensor = tf.layers.batch_normalization(tensor, training=is_training)
                tensor = tf.nn.elu(tensor)
                tensor = tensor + down_layers[~i]
                up_layers.append(tensor)

        print('up', up_layers)
        output_img = tf.layers.conv2d_transpose(tensor, 3, [4, 4], strides=[2, 2], activation=None,
                                                padding='SAME')
        return output_img, up_layers

    def save(self, sess, path="tmp/paired/", global_step=None):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.saver is None:
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
        self.saver.save(sess, path, global_step=global_step)

    def restore(self, sess, path="tmp/paired/"):
        self.saver.restore(sess, path)
