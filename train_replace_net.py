import skimage
import datetime
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from matplotlib import pyplot as plt

from load_data import load_parsed_sod
from ReplaceNet import ReplaceNet


def tweak_foreground(image, mask):
    """
    tweak foreground by apply random factor
    """
    mask_3d = np.expand_dims(mask, 2)
    foreground = mask_3d * image * np.random.uniform(0.1, 2)
    new_image = (1 - mask_3d) * image + foreground
    new_image = np.clip(new_image, 0, 1)
    return new_image


def train():
    DATETIME_STR = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    np.random.seed(0)
    patch_size = 512
    batch_size = 16

    images, masks = load_parsed_sod()
    images = np.array([resize(im, (patch_size, patch_size)) for im in images])
    masks = np.array(
        [skimage.img_as_bool(resize(skimage.img_as_float(ms), (patch_size, patch_size))) for ms in
         masks])
    sess = tf.Session()
    net = ReplaceNet(patch_size=patch_size)
    net.build(is_training=True)
    train_summary_writer = tf.summary.FileWriter(f'tmp/summary/summary-{DATETIME_STR}', sess.graph)
    sess.run(tf.global_variables_initializer())

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
            _, loss, out_im, summary, step_val = sess.run(
                [net.train_op, net.loss, net.output_img, net.merged_summary, net.global_step],
                feed_dict={net.input_img: tweaked, net.input_mask: truth_mask,
                           net.truth_img: truth_img})
            train_summary_writer.add_summary(summary, global_step=step_val)
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
