import sys
import yaml
import skimage
import logging
import datetime
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from matplotlib import pyplot as plt

from load_data import load_parsed_sod
from ReplaceNet import ReplaceNet
from synthesize import Synthesizer


DATETIME_STR = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.FileHandler('tmp/train_replace_net' + '-' + DATETIME_STR + '.log'))
logger.addHandler(logging.StreamHandler(sys.stdout))


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
    config = yaml.load(open('train.yaml', 'r'), Loader=yaml.CLoader)
    logger.info(config)

    np.random.seed(0)
    patch_size = config['patch_size']
    batch_size = config['batch_size']

    # load in-memory dataset
    images, masks = load_parsed_sod()
    images = np.array([resize(im, (patch_size, patch_size)) for im in images])
    masks = np.array(
        [skimage.img_as_bool(resize(skimage.img_as_float(ms), (patch_size, patch_size))) for ms in
         masks])

    # load synthesizer
    synthesizer = Synthesizer(patch_size=patch_size)
    _ = synthesizer.synthesize(images[0], masks[0], masks[10])  # provide size hint with image

    # build our model
    sess = tf.Session()
    net = ReplaceNet(patch_size=patch_size, skip_connection=config['skip_connection'])
    net.build(is_training=True)
    logger.info(vars(net))
    train_summary_writer = tf.summary.FileWriter(f'tmp/summary/summary-{DATETIME_STR}', sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch in range(500):
        out_im = None
        truth_img = None
        synthesized = None
        truth_mask = None
        epoch_loss = []

        index = np.arange(len(images))
        np.random.shuffle(index)
        batch_indices = np.array_split(index, len(index) // batch_size)
        for i, batch_index in enumerate(batch_indices):
            truth_img = images[batch_index]
            truth_mask = masks[batch_index]
            reference_mask_indices = np.random.choice(range(len(images)), len(truth_img))
            reference_mask = masks[reference_mask_indices]

            # apply inpaint
            synthesized = np.stack([synthesizer.synthesize(im, ms, ref_ms) for im, ms, ref_ms in
                                    zip(truth_img, truth_mask, reference_mask)])

            # TODO: tweaked is in (0, 1) which is good but why
            _, loss, out_im, summary, step_val = sess.run(
                [net.train_op, net.loss, net.output_img, net.merged_summary, net.global_step],
                feed_dict={net.input_img: synthesized, net.input_mask: truth_mask + reference_mask,
                           net.truth_img: truth_img})
            train_summary_writer.add_summary(summary, global_step=step_val)

            logger.error('epoch: ' + str(epoch) + ' batch: ' + str(i) + ' Loss: ' + str(loss))
            epoch_loss.append(loss)
        else:
            plt.subplot(2, 3, 1)
            plt.imshow(synthesized[0])
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
        print()


if __name__ == '__main__':
    train()
