import numpy as np
from pathlib import Path
from scipy.io import loadmat
from skimage.io import imread
from skimage.draw import line, polygon
import pprint
import os
from matplotlib import pyplot as plt
import tensorflow as tf

def load_larger_dataset():
    graph = tf.Graph()
    with graph.as_default():
        dataset_path = ''
        imgs_list = os.listdir(dataset_path + '/img')
        for index in range(len(imgs_list)):
            img_id = imgs_list[index].split('.')[0]
            ref_mask_index = np.random.randint(0,len(imgs_list))
            ref_mask_id = imgs_list[ref_mask_index].split('.')[0]
            image = np.array(imread(dataset_path + '/img/' +img_id+'.jpg'))
            mask = np.load(dataset_path + '/bool_mask/'+img_id+'.npy')
            ref_mask = np.load(dataset_path + '/bool_mask/'+ref_mask_id+'.npy')
            yield image, mask, ref_mask

def get_dataset_size():
    dataset_path = ''
    imgs_list = os.listdir(dataset_path + '/img')
    return len(imgs_list)


def parse_sod():
    """
    Super small dataset of SOD (Salient Objects Dataset). Load in memory
    Descriptions: http://elderlab.yorku.ca/SOD/#download

    - http://elderlab.yorku.ca/SOD/SOD.zip
    - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images
    .tgz

    Return images and boolean mask
    """
    index = loadmat('SOD/DBidx.mat', squeeze_me=True, struct_as_record=False)
    ret_images = []
    ret_masks = []

    for image_id in index['SFprefix'].flatten():
        image_path = list(Path('BSDS300/images/').rglob(str(image_id) + '.jpg'))[0]
        raw_image = imread(image_path)

        sessions = loadmat('SOD/SO' + str(image_id) + '.mat', squeeze_me=True, struct_as_record=False)['SES']
        for sess in sessions:
            # TODO: use other sessions
            # sess_id = sess.session
            if isinstance(sess.obj, np.ndarray):
                # The most salient object has imp=1.
                salient_obj = next((o for o in sess.obj if o.IMP == 1), None)
                if salient_obj is None:
                    continue
            else:
                salient_obj = sess.obj

            boundary = salient_obj.BND
            if boundary.dtype == np.object:
                # TODO: allow disconnected area
                boundary = boundary[0]
            mask = np.zeros(sess.ImSize.tolist(), dtype=np.bool)
            rr, cc = polygon(boundary[:, 0], boundary[:, 1], sess.ImSize.tolist())
            mask[rr, cc] = 1

            ret_images.append(raw_image)
            ret_masks.append(mask)
            break
    return ret_images, ret_masks


def load_parsed_sod():
    d = np.load('data/sod-pairs.npz', allow_pickle=True)
    images, masks = d['images'], d['masks']
    return np.array(images), np.array(masks)


def visualize_sod():
    images, masks = load_parsed_sod()
    for i in range(4):
        ind = np.random.randint(0, len(images))
        plt.subplot(4, 2, i * 2 + 1)
        plt.imshow(images[ind])
        plt.subplot(4, 2, i * 2 + 2)
        plt.imshow(masks[ind])
    plt.show()


if __name__ == '__main__':
    # images, masks = parse_data()
    # np.savez('data/sod-pairs.npz', images=images, masks=masks)
    visualize_sod()



