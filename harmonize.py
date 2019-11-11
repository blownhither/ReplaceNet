import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import skimage
from skimage.transform import resize

from load_data import load_parsed_sod



class ParameterizedUNet:
    def __init__(self, patch_size):
        # hyperparameters
        self.size = patch_size
        self.down_channels = [32, 64, 128, 256]
        self.up_channels = [128, 64, 32]
        # self.down_channels = [32, 64, 128, 256, 512]
        # self.up_channels = [256, 128, 64, 32]
        self.batch_size = None
        self.lr = 1e-3
        self.translator_hidden = 32

        # i/o tensor
        # input img should be patches in size 512
        self.input_img = tf.placeholder(shape=[None, self.size, self.size, 3], dtype=tf.float32)
        self.truth_img = tf.placeholder(shape=[None, self.size, self.size, 3], dtype=tf.float32)
        self.truth_mask = tf.placeholder(shape=[None, self.size, self.size], dtype=tf.float32)
        self.output_img = None
        self.output_mask = None
        self.is_training = tf.placeholder_with_default(True, shape=())

        # internal tensors, set after building
        self.down_layers = None
        self.up_layers = None
        self.l2_loss = None
        self.mask_loss = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.merged_summary = None
        self.global_step = None
        self.saver = None

    def build(self):
        self.down_layers, encoder_out = self._build_down(self.input_img)
        with tf.variable_scope('reconstruct'):
            self.output_img = self._build_up(encoder_out, self.down_layers, 3)
        with tf.variable_scope('mask'):
            output_mask = self._build_up(encoder_out, self.down_layers, 1)
        self.output_mask = tf.squeeze(output_mask, axis=3)

        self.l2_loss = tf.losses.mean_squared_error(self.truth_img, self.output_img)
        self.mask_loss = tf.losses.mean_squared_error(self.truth_mask, self.output_mask)
        self.loss = self.l2_loss + self.mask_loss

        tf.summary.scalar('l2-loss', self.l2_loss)
        tf.summary.scalar('mask-loss', self.mask_loss)
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # for Batch Norm update
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = self.optimizer.minimize(self.loss,
                                           global_step=tf.train.get_or_create_global_step())
        self.train_op = tf.group([train_op, update_ops])
        self.merged_summary = tf.summary.merge_all()
        self.global_step = tf.train.get_or_create_global_step()

    def _build_down(self, img):
        # down sampling
        down_layers = []
        tensor = img
        for i, c in enumerate(self.down_channels):
            # TODO: harmonization use 4x4
            tensor = tf.layers.conv2d(tensor, c, [3, 3], activation=tf.nn.leaky_relu,
                                      padding='same')
            # TODO: no normalization?
            # tensor = tf.layers.batch_normalization(tensor, training=self.is_training)
            # only one conv for the first layer
            if i != 0:
                tensor = tf.layers.conv2d(tensor, c, [3, 3], activation=tf.nn.leaky_relu,
                                          padding='same')
                # TODO: no normalization?
                # tensor = tf.layers.batch_normalization(tensor, name=f'down_block{i}_out',
                #                                        training=self.is_training)
            else:
                # extra skip connection in the first down layer
                tensor = tf.concat([tensor, img], axis=3)
            down_layers.append(tensor)
            # no shrink for the last layer
            if i != len(self.down_channels) - 1:
                # TODO: harmonization use no pooling
                tensor = tf.layers.max_pooling2d(tensor, [2, 2], [2, 2], padding='same')
        print('down', down_layers)
        return down_layers, tensor

    def _build_up(self, img, down_layers, out_channel):
        # TODO: harmonization use flatten
        up_layers = []
        tensor = img
        for i, c in enumerate(self.up_channels):
            # print('before', tensor)
            tensor = tf.layers.conv2d_transpose(tensor, c, [3, 3], strides=[2, 2],
                                                activation=tf.nn.leaky_relu, padding='same')
            # print('after', tensor, self.down_layers[-(i + 2)])
            tensor = tf.layers.batch_normalization(tensor, training=self.is_training)
            tensor = tf.concat([tensor, down_layers[-(i + 2)]], axis=3)
            tensor = tf.layers.conv2d(tensor, c, [3, 3], activation=tf.nn.leaky_relu,
                                      padding='same')
            tensor = tf.layers.batch_normalization(tensor, training=self.is_training)
            tensor = tf.layers.conv2d(tensor, c, [3, 3], activation=tf.nn.leaky_relu,
                                      padding='same')
            tensor = tf.layers.batch_normalization(tensor, name=f'up_block{i}_out',
                                                   training=self.is_training)
            up_layers.append(tensor)
        # print('up', up_layers)
        tf.layers.conv2d(tensor, 12, [3, 3], activation=tf.nn.leaky_relu, padding='same')
        tensor = tf.layers.batch_normalization(tensor, training=self.is_training)
        output_img = tf.layers.conv2d(tensor, out_channel, [3, 3], activation=tf.nn.sigmoid, padding='same',
                                      name='reconstructed')
        return output_img

    def save(self, sess, path="tmp/paired/", global_step=None):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.saver is None:
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
        self.saver.save(sess, path, global_step=global_step)

    def restore(self, sess, path="tmp/paired/"):
        self.saver.restore(sess, path)


def tweak_foreground(image, mask):
    """
    tweak foreground by apply random factor
    """
    tweaked = mask_3d * image * np.random.uniform(0.1, 2)
    new_image = (1 - mask_3d) * image + tweaked
    new_image *= (1.0/new_image.max())
    return new_image


def train():
    np.random.seed(0)
    patch_size = 256

    images, masks = load_parsed_sod()
    images = np.array([resize(im, (patch_size, patch_size)) for im in images])
    masks = np.array([skimage.img_as_bool(resize(skimage.img_as_float(ms), (patch_size, patch_size))) for ms in masks])
    sess = tf.Session()
    net = ParameterizedUNet(patch_size=patch_size)
    net.build()
    sess.run(tf.global_variables_initializer())

    for epoch in range(500):
        index = np.arange(len(images))
        np.random.shuffle(index)
        images = images[index]
        masks = masks[index]
        out = None
        truth_img = None
        tweaked = None

        for i, (truth_img, input_mask) in enumerate(zip(images, masks)):
            tweaked = tweak_foreground(truth_img, input_mask)
            # TODO: tweaked is in (0, 1) which is good but why
            _, loss, out = sess.run([net.train_op, net.l2_loss, net.output_img], feed_dict={
                net.input_img: [tweaked],
                net.truth_img: [truth_img],
                net.truth_mask: [input_mask],
            })
            print('epoch', epoch, 'instance', i, loss, flush=True)
        else:
            plt.subplot(1, 3, 1)
            plt.imshow(tweaked)
            plt.title('Input')
            plt.subplot(1, 3, 2)
            plt.imshow(truth_img)
            plt.title('Truth')
            plt.subplot(1, 3, 3)
            plt.imshow(out[0])
            plt.title('out')
            plt.savefig(f'tmp/{epoch}.png')
            plt.close()


if __name__ == '__main__':
    train()




