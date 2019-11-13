import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from matplotlib import pyplot as plt
import skimage
from skimage.transform import resize

from load_data import load_parsed_sod


class ParameterizedUNet:
    def __init__(self, patch_size):
        # hyperparameters
        self.size = patch_size
        self.encoder_channels = [64, 64, 128, 128, 256, 256, 512]
        self.scene_parsing_decoder_channels = [512, 256, 256, 128, 128, 64, 64]
        self.harmonization_decoder_channels = [512, 256, 256, 128, 128, 64, 64]

        self.batch_size = None
        self.lr = 1e-3
        self.translator_hidden = 32

        # i/o tensor
        # input img should be patches in size 512
        self.input_img = tf.placeholder(shape=[None, self.size, self.size, 3], dtype=tf.float32)
        # mask: shape and dtype????
        self.input_mask = tf.placeholder(shape=[None, self.size, self.size, 1], dtype=tf.float32)
        self.truth_img = tf.placeholder(shape=[None, self.size, self.size, 3], dtype=tf.float32)
        self.truth_seg = tf.placeholder(shape=[None, self.size, self.size, 1], dtype=tf.float32)
        self.output_img = None
        self.output_seg = None
        self.is_training = True

        # internal tensors, set after building
        self.down_layers = None
        self.up_layers = None
        self.l2_loss = None
        self.seg_loss = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.merged_summary = None
        self.global_step = None
        self.saver = None

    def build(self):
        self.down_layers, encoder_out = self._build_encoder(self.input_img, self.input_mask)
        with tf.variable_scope('segmentation'):
            seg_layers, self.output_seg = self._build_scene_parsing_decoder(encoder_out, self.down_layers, 2)
        with tf.variable_scope('harmonization'):
            self.output_img = self._build_harmonization_decoder(encoder_out, self.down_layers, seg_layers, 3)

        self.l2_loss = tf.losses.mean_squared_error(self.truth_img, self.output_img)
        # self.seg_loss = tf.losses.mean_squared_error(self.truth_seg, self.output_seg)
        self.seg_loss = 0
        self.loss = self.l2_loss + self.seg_loss

        tf.summary.scalar('l2-loss', self.l2_loss)
        tf.summary.scalar('seg-loss', self.seg_loss)
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # for Batch Norm update
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = self.optimizer.minimize(self.loss,
                                           global_step=tf.train.get_or_create_global_step())
        self.train_op = tf.group([train_op, update_ops])
        self.merged_summary = tf.summary.merge_all()
        self.global_step = tf.train.get_or_create_global_step()

    def _build_encoder(self, img, mask):
        # down sampling
        down_layers = []
        tensor = tf.concat([img, mask], axis=3)
        tensor = tl.layers.InputLayer(tensor)
        for i, c in enumerate(self.encoder_channels):
            tensor = tl.layers.Conv2d(tensor, n_filter=c, filter_size=(4, 4), strides=(2, 2), act=tf.nn.elu,
                                      padding='SAME', name="encoder_Conv2d_{}".format(i))
            tensor = tl.layers.BatchNormLayer(tensor, is_train=self.is_training, name="encoder_batch_norm_{}".format(i))
            down_layers.append(tensor)
            # no shrink for the last layer
        tensor = tl.layers.FlattenLayer(tensor)
        tensor = tl.layers.DenseLayer(tensor, n_units=1024, name="encoder_fc")
        tensor = tl.layers.ReshapeLayer(tensor, shape=(-1, 1, 1, 1024), name="encoder_reshape")
        return down_layers, tensor

    def _build_scene_parsing_decoder(self, encoder_out, down_layers, out_channel):
        seg_layers = []
        tensor = tl.layers.ReshapeLayer(encoder_out, shape=(-1, 2, 2, 512))
        for i, c in enumerate(self.scene_parsing_decoder_channels):
            tensor = tl.layers.DeConv2d(tensor, n_filter=c, filter_size=(4, 4), strides=(2, 2),
                                        act=tf.nn.elu, padding='SAME', name="seg_decoder_DeConv2d_{}".format(i))
            # print('after', tensor, self.down_layers[-(i + 2)])
            tensor = tl.layers.BatchNormLayer(tensor, is_train=self.is_training, name="seg_batch_norm_{}".format(i))
            tensor = tl.layers.ElementwiseLayer([tensor, down_layers[-(i + 1)]], combine_fn=tf.add,
                                                name="seg_elesum_{}".format(i))
            # tensor = tf.concat([tensor, down_layers[-(i + 2)]], axis=3, name=f'scene_parse_decoder{i}_out')
            seg_layers.append(tensor)
        scene_parse = tl.layers.Conv2d(tensor, n_filter=out_channel, filter_size=(1, 1), strides=(1, 1),
                                       padding='SAME', name="get_segmentation")
        return seg_layers, scene_parse.outputs

    def _build_harmonization_decoder(self, encoder_out, down_layers, seg_layers, out_channel):
        tensor = tl.layers.ReshapeLayer(encoder_out, shape=(-1, 2, 2, 512))
        for i, c in enumerate(self.harmonization_decoder_channels):
            tensor = tl.layers.DeConv2d(tensor, n_filter=c, filter_size=(4, 4), strides=(2, 2),
                                        act=tf.nn.elu, padding='SAME', name="harm_DeConv2d_{}".format(i))
            # print('after', tensor, self.down_layers[-(i + 2)])
            tensor = tl.layers.BatchNormLayer(tensor, is_train=self.is_training, name="harm_batch_norm_{}".format(i))
            tensor = tl.layers.ElementwiseLayer([tensor, down_layers[-(i + 1)]], combine_fn=tf.add,
                                                name="harm_elesum_{}".format(i))
            tensor = tl.layers.ConcatLayer([tensor, seg_layers[i]], 3, name="concat_{}".format(i))

        harm_img = tl.layers.DeConv2d(tensor, n_filter=out_channel, filter_size=(4, 4), strides=(2, 2),
                                       padding='SAME', name='get_img')
        return harm_img.outputs

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
    mask = np.expand_dims(mask, 2)
    tweaked = mask * image * np.random.uniform(0.1, 2)
    new_image = (1 - mask) * image + tweaked
    new_image *= (1.0 / new_image.max())
    return new_image


def train():
    np.random.seed(0)
    patch_size = 512
    batch_size = 8

    images, masks = load_parsed_sod()
    images = np.array([resize(im, (patch_size, patch_size)) for im in images])
    masks = np.array(
        [np.expand_dims(skimage.img_as_bool(resize(skimage.img_as_float(ms), (patch_size, patch_size))), 2) for ms in
         masks])
    sess = tf.Session()
    net = ParameterizedUNet(patch_size=patch_size)
    net.build()
    sess.run(tf.global_variables_initializer())
    print("start training")
    for epoch in range(500):
        out_im = None
        truth_img = None
        tweaked = None
        truth_mask = None

        index = np.arange(len(images))
        np.random.shuffle(index)
        for i, batch_index in enumerate(np.array_split(index, len(index) // batch_size)):
            truth_img = images[batch_index]
            truth_mask = masks[batch_index]
            tweaked = [tweak_foreground(im, ms) for im, ms in zip(truth_img, truth_mask)]
            # TODO: tweaked is in (0, 1) which is good but why
            _, loss, out_im = sess.run([net.train_op, net.loss, net.output_img],
                                       feed_dict={net.input_img: tweaked,
                                                  net.input_mask: truth_mask,
                                                  net.truth_img: truth_img})
            print('epoch', epoch, 'batch', i, loss, flush=True)
        else:
            plt.subplot(2, 3, 1)
            plt.imshow(tweaked[0])
            plt.title('Input')
            plt.subplot(2, 3, 2)
            plt.imshow(truth_img[0])
            plt.title('Truth')
            plt.subplot(2, 3, 3)
            plt.imshow(out_im[0])
            plt.title('out')
            plt.subplot(2, 3, 5)
            plt.imshow(truth_mask[0])
            plt.savefig(f'tmp/{epoch}.png')
            plt.close()

if __name__ == '__main__':
    train()
