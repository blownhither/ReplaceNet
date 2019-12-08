import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../elpips') # clone repo  : https://github.com/mkettune/elpips
import elpips


class ReplaceNet:
    """
    This is a variation of UNet setting from "Deep Image Harmonization"
    The bottom branch (scene parsing decoder) is not currently included.
    """

    def __init__(self, patch_size=512, skip_connection="add", input_img=None, truth_img=None,
                 input_mask=None, ref_mask=None):
        """
        :param skip_connection: concat | add
        """
        # hyper parameters
        self.size = patch_size
        self.skip_connection = skip_connection
        self.down_channels = [64, 64, 128, 128, 256, 256, 512]
        # self.fc_size = 1024       # NOTE: static fc_size is deprecated, uses a 0.5x bottleneck
        self.up_channels = [512, 256, 256, 128, 128, 64, 64]
        self.lr = 1e-3

        # i/o tensors
        # input img should be patches in size 512
        self.input_img = input_img or tf.placeholder(shape=[None, self.size, self.size, 3],
                                                     dtype=tf.float32, name='input_img')
        self.truth_img = truth_img or tf.placeholder(shape=[None, self.size, self.size, 3],
                                                     dtype=tf.float32, name='truth_img')
        # `input_mask` is applied on `input_img` to locate foreground
        self.input_mask = input_mask or tf.placeholder(shape=[None, self.size, self.size, 3],
                                                       dtype=tf.float32, name='input_mask')

        self.output_img = None
        self.metric = elpips.Metric(elpips.elpips_vgg(batch_size=1, n=1), back_prop=False)

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
        self.elpips_distance = self.metric.forward(self.truth_img, self.output_img)[0]
        tf.summary.scalar('elpips', self.elpips_distance)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # for Batch Norm update
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = self.optimizer.minimize(self.loss + self.elpips_distance,
                                           global_step=tf.train.get_or_create_global_step())
        self.train_op = tf.group([train_op, update_ops])
        self.merged_summary = tf.summary.merge_all()
        self.global_step = tf.train.get_or_create_global_step()

    def _build_down(self, img, mask, is_training):
        # down sampling
        down_layers = []
        mask_shape = mask.get_shape().as_list()[1:3]
        with tf.name_scope('encoder'):
            tensor = img
            for i, c in enumerate(self.down_channels):
                current_shape = tensor.get_shape().as_list()[1:3]
                # concat mask to each layer, the up layers automatically gets them
                if current_shape != mask_shape:
                    mask_reshaped = tf.image.resize_images(mask, current_shape)
                else:
                    mask_reshaped = mask
                print(mask_reshaped, tensor)
                tensor = tf.concat([tensor, mask_reshaped], axis=3)

                tensor = tf.layers.conv2d(tensor, c, [4, 4], strides=(2, 2), activation=None,
                                          padding='SAME')
                tensor = tf.layers.batch_normalization(tensor, training=is_training)
                tensor = tf.nn.elu(tensor)
                down_layers.append(tensor)
            print('down', down_layers)

            # Add a 0.5x FC bottle neck, this enables flexible patch size.
            size = np.prod(tensor.get_shape().as_list()[1:])
            tensor = tf.layers.dense(tf.layers.flatten(tensor), size // 2, name='bottleneck')
            tensor = tf.layers.batch_normalization(tensor, training=is_training)
            tensor = tf.nn.elu(tensor)

            return down_layers, tensor

    def _build_up(self, fc, down_layers, is_training):
        up_layers = []
        with tf.name_scope('h-decoder'):
            # Recover from the 0.5x FC bottle neck with another FC
            last_block_shape = down_layers[-1].get_shape().as_list()[1:]
            size = np.prod(last_block_shape)
            tensor = tf.layers.dense(fc, size // 4, name='out')
            tensor = tf.layers.batch_normalization(tensor, training=is_training)
            tensor = tf.nn.elu(tensor)
            tensor = tf.reshape(tensor, [-1, last_block_shape[0] // 2, last_block_shape[1] // 2,
                                         last_block_shape[2]])

            for i, c in enumerate(self.up_channels):
                # TODO: check different stride setting
                tensor = tf.layers.conv2d_transpose(tensor, c, [4, 4], strides=[2, 2],
                                                    activation=None, padding='SAME')
                tensor = tf.layers.batch_normalization(tensor, training=is_training)
                tensor = tf.nn.elu(tensor)
                if self.skip_connection == "add":
                    tensor = tensor + down_layers[~i]
                elif self.skip_connection == "concat":
                    tensor = tf.concat([tensor, down_layers[~i]], axis=3)
                up_layers.append(tensor)

        print('up', up_layers)
        tensor = tf.layers.conv2d_transpose(tensor, 32, [4, 4], strides=[2, 2], activation=None,
                                            padding='SAME')

        # the two lines below are not in the original paper, they may fix checkerboard
        tensor = tf.nn.elu(tf.layers.batch_normalization(tensor))
        output_img = tf.layers.conv2d(tensor, 3, [4, 4], strides=[1, 1], padding='SAME',
                                      activation=tf.nn.sigmoid)

        return output_img, up_layers

    def save(self, sess, path="tmp/paired/", global_step=None):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.saver is None:
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
        self.saver.save(sess, path, global_step=global_step)

    def restore(self, sess, path="tmp/paired/"):
        self.saver.restore(sess, path)
