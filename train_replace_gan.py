import sys
import yaml
import logging
import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from load_data import load_large_dataset_image, load_large_dataset_with_class, get_dataset_size
from ReplaceGAN import ReplaceGAN
from synthesize import Synthesizer


DATETIME_STR = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.FileHandler('tmp/train_replace_gan' + '-' + DATETIME_STR + '.log'))
logger.addHandler(logging.StreamHandler(sys.stdout))


def train():
    config = dict()
    config['batch_size'] = 64
    config['patch_size'] = 256
    config['skip_connection'] = 'concat'

    logger.info(config)

    np.random.seed(0)
    patch_size = config['patch_size']
    batch_size = config['batch_size']
    dataset_size = get_dataset_size()
    logger.info('dataset size ' + str(dataset_size))

    # load batched & prefetched dataset
    dataset = tf.data.Dataset.from_generator(
        load_large_dataset_with_class(patch_size=patch_size, align=True),
        (tf.float32, tf.bool, tf.bool))
    dataset = dataset.batch(batch_size).prefetch(16)
    it = dataset.make_initializable_iterator()
    nxt = it.get_next()

    real_image_dataset = tf.data.Dataset.from_generator(
        load_large_dataset_image(patch_size=patch_size), tf.float32)
    real_image_dataset = real_image_dataset.batch(batch_size).prefetch(16)
    real_image_it = real_image_dataset.make_initializable_iterator()
    real_image_nxt = real_image_it.get_next()

    # load synthesizer
    synthesizer = Synthesizer(patch_size=patch_size)

    # build our model
    sess = tf.Session()
    net = ReplaceGAN(patch_size=patch_size, skip_connection=config['skip_connection'])
    net.build(is_training=True)
    logger.info(vars(net))
    train_summary_writer = tf.summary.FileWriter(f'tmp/summary/summary-{DATETIME_STR}', sess.graph)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for epoch in range(500):
        sess.run(it.initializer)
        sess.run(real_image_it.initializer)
        for batch in range((dataset_size // batch_size) - 1):
            # dataset outputs are resized/aligned
            truth_img, truth_masks, reference_masks = sess.run(nxt)
            discriminator_real_img = sess.run(real_image_nxt)
            input_mask = np.stack(
                (truth_masks, reference_masks, reference_masks & (reference_masks ^ truth_masks)),
                axis=3)

            # apply inpaint
            synthesized = np.stack([synthesizer.synthesize(im, ms, ref_ms) for im, ms, ref_ms in
                                    zip(truth_img, truth_masks, reference_masks)])

            # train discriminator
            _, dis_loss, summary, step_val = sess.run(
                [net.discriminator_train_op, net.discriminator_loss, net.discriminator_summary,
                 net.global_step],
                feed_dict={
                    net.discriminator_real_input: discriminator_real_img,
                    net.input_img: synthesized,
                    net.input_mask: input_mask,
                    net.truth_img: truth_img})
            train_summary_writer.add_summary(summary, global_step=step_val)
            logger.error(
                'Dis: ' + 'epoch: ' + str(epoch) + ' batch: ' + str(batch) + ' adv loss: ' + str(
                    dis_loss))

            # train generator
            # TODO: generator trained multiple times
            _, l2_loss, elpips_loss, gen_adv_loss, out_im, summary, step_val = sess.run(
                [net.generator_train_op, net.l2_loss, net.elpips_distance,
                 net.generator_adversarial_loss, net.output_img, net.generator_summary,
                 net.global_step],
                feed_dict={
                    net.input_img: synthesized,
                    net.input_mask: input_mask,
                    net.truth_img: truth_img})
            train_summary_writer.add_summary(summary, global_step=step_val)
            logger.error(
                'Gen: ' + 'epoch: ' + str(epoch) + ' batch: ' + str(batch) + ' L2-loss: ' + str(
                    l2_loss) + ' elpips: ' + str(elpips_loss) + ' adv loss: ' + str(gen_adv_loss))
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
            plt.imshow(truth_masks[0])
            plt.subplot(2, 3, 6)
            plt.imshow(reference_masks[0])
            plt.savefig(f'tmp/model' + '-' + DATETIME_STR + '/' + str(epoch) + '.png')
            plt.close()
        print()


if __name__ == '__main__':
    train()
