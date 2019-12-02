import cv2
import yaml
import skimage
import numpy as np
import tensorflow as tf

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

    def replace(self, fg_image, fg_mask, bg_image, bg_mask, align_mask=True):
        """
        Input should be float image in (0, 1) and float/bool masks
        return synthesized image and fixed image (final model output)
        """
        # masks scaled to (0, 1)
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
        synthesized = inpainted_bg.copy()
        synthesized[fg_mask.astype(np.bool)] = fg_image[fg_mask.astype(np.bool)]

        out_image = self.sess.run(self.net.output_img, feed_dict={
            self.net.input_img: [synthesized],
            self.net.input_mask: [np.stack((bg_mask, fg_mask, fg_mask - bg_mask), axis=2)]
        })
        return synthesized, out_image[0]


def test():
    from matplotlib import pyplot as plt
    from load_data import load_parsed_sod

    images, masks = load_parsed_sod()
    fg_choice = 0
    bg_choice = 1

    model_path = 'tmp/new_model/model_20191202151110'
    inf = InferenceHelper(model_path)
    synthesized, out = inf.replace(images[fg_choice], masks[fg_choice],
                                   images[bg_choice], masks[bg_choice])
    plt.subplot(2, 6, 1)
    plt.imshow(images[fg_choice])
    plt.title('Foreground')
    plt.subplot(2, 6, 2)
    plt.imshow(masks[fg_choice])
    plt.subplot(2, 6, 7)
    plt.imshow(images[bg_choice])
    plt.title('Background')
    plt.subplot(2, 6, 8)
    plt.imshow(masks[bg_choice])
    plt.subplot(1, 3, 2)
    plt.imshow(synthesized)
    plt.title('Synthesized')
    plt.subplot(1, 3, 3)
    plt.imshow(out)
    plt.title('out')
    plt.show()


if __name__ == '__main__':
    test()
