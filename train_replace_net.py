import sys
import yaml
import skimage
import logging
import datetime
import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import resize
from matplotlib import pyplot as plt

from load_data import *
from ReplaceNet import ReplaceNet
from synthesize import Synthesizer
from tweak import align_mask

DATETIME_STR = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.FileHandler('tmp/train_replace_net' + '-' + DATETIME_STR + '.log'))
logger.addHandler(logging.StreamHandler(sys.stdout))


def train():
    config = dict()
    config['batch_size'] = 1
    config['patch_size'] = 256
    config['skip_connection'] = 'concat'

    logger.info(config)

    np.random.seed(0)
    patch_size = config['patch_size']
    batch_size = config['batch_size']
    dataset_size = get_dataset_size()

    # load in-memory dataset
    dataset = tf.data.Dataset.from_generator(load_larger_dataset, (tf.int64, tf.int64, tf.int64))
    it = dataset.make_initializable_iterator()
    nxt = it.get_next()

    # load synthesizer
    synthesizer = Synthesizer(patch_size=patch_size)

    # build our model
    sess = tf.Session()
    net = ReplaceNet(patch_size=patch_size, skip_connection=config['skip_connection'])
    net.build(is_training=True)
    logger.info(vars(net))
    train_summary_writer = tf.summary.FileWriter(f'tmp/summary/summary-{DATETIME_STR}', sess.graph)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for epoch in range(500):
        sess.run(it.initializer)
        epoch_loss = []

        for batch in range((dataset_size // batch_size) - 1):
            images = []
            truth_masks = []
            reference_masks = []

            for _ in range(batch_size):
                image, truth_mask, reference_mask = sess.run(nxt)  # Lazily load
                images.append(
                    cv2.resize(image / 255.0, (patch_size, patch_size)))  # int64 -> float64
                truth_masks.append((cv2.resize(truth_mask / 255.0, (patch_size, patch_size))) > 0.0)
                reference_masks.append(
                    (cv2.resize(reference_mask / 255.0, (patch_size, patch_size)) > 0.0))

            truth_img = np.stack(images)
            truth_mask = np.stack(truth_masks)
            reference_mask = np.stack(reference_masks)

            # align with current mask
            reference_mask = [align_mask(m, r) for m, r in zip(truth_mask, reference_mask)]

            # apply inpaint
            synthesized = np.stack([synthesizer.synthesize(im, ms, ref_ms) for im, ms, ref_ms in
                                    zip(truth_img, truth_mask, reference_mask)])

            _, loss, elpips_loss, out_im, summary, step_val = sess.run(
                [net.train_op, net.loss, net.elpips_distance, net.output_img, net.merged_summary,
                 net.global_step], feed_dict={net.input_img: synthesized, net.input_mask: np.stack(
                    (truth_mask, reference_mask, reference_mask - truth_mask), axis=3),
                    net.truth_img: truth_img})
            train_summary_writer.add_summary(summary, global_step=step_val)

            logger.error('epoch: ' + str(epoch) + ' batch: ' + str(batch) + ' Loss: ' + str(
                loss) + ' elpips: ' + str(elpips_loss))
            epoch_loss.append(loss)
        else:
            saver.save(sess, 'tmp/model' + '-' + DATETIME_STR + '/model')
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
            plt.subplot(2, 3, 6)
            plt.imshow(reference_mask[0])
            plt.savefig(f'tmp/model' + '-' + DATETIME_STR + '/' + str(epoch) + '.png')
            plt.close()
        print()


if __name__ == '__main__':
    train()
