import cv2
import yaml
import skimage
import numpy as np
import tensorflow as tf
from skimage.morphology import disk, binary_dilation

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

        out_image = self.sess.run(self.net.output_img, feed_dict={
            self.net.input_img: [synthesized],
            self.net.input_mask: [np.stack((bg_mask, fg_mask, fg_mask - bg_mask), axis=2)]
        })
        return synthesized, out_image[0]


def test():
    from matplotlib import pyplot as plt
    from load_data import load_parsed_sod
    np.random.seed(10)

    images, masks = load_parsed_sod()
    model_path = 'tmp/model-20191202162813/model'
    inf = InferenceHelper(model_path)

    for i in range(20):
        fg_choice = np.random.randint(9, len(images))
        bg_choice = np.random.randint(9, len(images))
        synthesized, out = inf.replace(images[fg_choice], masks[fg_choice],
                                       images[bg_choice], masks[bg_choice],
                                       use_dilation=5)
        plt.figure(figsize=(16, 8))
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
        plt.savefig(f'{model_path}-example-{i}.png')
        # plt.show()


if __name__ == '__main__':
    test()
