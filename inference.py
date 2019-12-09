import os
import cv2
import yaml
import glob
import random
import skimage
import numpy as np
import tensorflow as tf
from skimage.morphology import disk, binary_dilation
from skimage.io import imread

from ReplaceNet import ReplaceNet
from synthesize import Synthesizer
from tweak import align_image


def get_predict_session(model_path, patch_size=None, skip_connection=None):
    """
    Building inference network and session given a train network

    If patch_size is None, load from train.yaml
    """
    if patch_size is None:
        config = yaml.load(open('train.yaml', 'r'), Loader=yaml.CLoader)
        patch_size = config['patch_size']
        skip_connection = config['skip_connection']

    synthesizer = Synthesizer(patch_size=patch_size)
    sess = tf.Session()
    net = ReplaceNet(patch_size=patch_size, skip_connection=skip_connection)
    net.build(is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    return sess, net, synthesizer, patch_size


class InferenceHelper:
    def __init__(self, model_path, patch_size=None, skip_connection=None):
        self.sess, self.net, self.synthesizer, self.patch_size = get_predict_session(
            model_path, patch_size, skip_connection)

    def replace(self, fg_image, fg_mask, bg_image, bg_mask, align_mask=True, use_dilation=1):
        """
        Input should be float image in (0, 1) and float/bool masks
        Set dilation=0 to forbid dilation. Otherwise it specifies the size of the disk

        return synthesized image and fixed image (final model output)
        """
        # masks scaled to (0, 1)
        if use_dilation:
            dilation_disk = disk(use_dilation, np.bool)
            bg_mask = binary_dilation(bg_mask, dilation_disk)

        bg_image = skimage.img_as_float(cv2.resize(bg_image, (self.patch_size, self.patch_size)))
        bg_mask = cv2.resize((bg_mask > 0).astype(np.float32),
                             (self.patch_size, self.patch_size)).astype(np.bool)
        fg_image = skimage.img_as_float(cv2.resize(fg_image, (self.patch_size, self.patch_size)))
        fg_mask = cv2.resize((fg_mask > 0).astype(np.float32),
                             (self.patch_size, self.patch_size)).astype(np.bool)

        # TODO: has to have ref mask? Use zero mask now
        inpainted_bg = self.synthesizer.get_background(bg_image, bg_mask,
                                                       np.zeros_like(bg_mask, dtype=np.bool))
        if align_mask:
            fg_mask, fg_image = align_image(anchor_mask=bg_mask, movable_mask=fg_mask,
                                            movable_image=fg_image)
        # synthesized = inpainted_bg + np.expand_dims(fg_mask, 2) * fg_image
        synthesized = skimage.img_as_float(inpainted_bg.copy())
        synthesized[fg_mask.astype(np.bool)] = fg_image[fg_mask.astype(np.bool)]

        bg_mask = bg_mask.astype(np.bool)
        fg_mask = fg_mask.astype(np.bool)
        out_image = self.sess.run(self.net.output_img, feed_dict={
            self.net.input_img: [synthesized],
            self.net.input_mask: [np.stack((bg_mask, fg_mask,
                                            # fg_mask - bg_mask
                                            fg_mask & (fg_mask ^ bg_mask)
                                            ), axis=2).astype(np.float)]
        })
        return synthesized, out_image[0]


def test():
    from matplotlib import pyplot as plt
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    dataset_path = 'image_data'
    mask_path = dataset_path + '/bool_mask_sep_inst/'
    cls_dict = np.load(dataset_path + '/inst_mask_dict.npy', allow_pickle=True).take(0)

    model_path = 'tmp/model-20191203185450/model'
    inf = InferenceHelper(model_path)

    for i in range(20):
        key = np.random.choice(list(cls_dict.keys()))
        fg_mask_name, bg_mask_name = np.random.choice(cls_dict[key], 2)
        fg_mask = np.load(mask_path + fg_mask_name, allow_pickle=True)
        bg_mask = np.load(mask_path + bg_mask_name, allow_pickle=True)

        fg_image_name = dataset_path + '/img/' + os.path.basename(
            fg_mask_name[:fg_mask_name.rindex('_')]) + '.jpg'
        bg_image_name = dataset_path + '/img/' + os.path.basename(
            bg_mask_name[:bg_mask_name.rindex('_')]) + '.jpg'
        fg_image = skimage.img_as_float(imread(fg_image_name))
        bg_image = skimage.img_as_float(imread(bg_image_name))

        synthesized, out = inf.replace(fg_image, fg_mask, bg_image, bg_mask,
                                       use_dilation=5)
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 6, 1)
        plt.imshow(fg_image)
        plt.title('Foreground')
        plt.subplot(2, 6, 2)
        plt.imshow(fg_mask)
        plt.subplot(2, 6, 7)
        plt.imshow(bg_image)
        plt.title('Background')
        plt.subplot(2, 6, 8)
        plt.imshow(bg_mask)
        plt.subplot(1, 3, 2)
        plt.imshow(synthesized)
        plt.title('Synthesized')
        plt.subplot(1, 3, 3)
        plt.imshow(out)
        plt.title('out')
        plt.savefig(f'{model_path}-example-{i}.png')
        # plt.show()
        plt.close()


if __name__ == '__main__':
    test()
