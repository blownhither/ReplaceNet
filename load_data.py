import os
import glob
import random
import skimage
import numpy as np
from skimage.transform import resize as sk_resize
from pathlib import Path
from scipy.io import loadmat
from skimage.io import imread
from skimage.draw import polygon
from collections import defaultdict

from tweak import align_mask


def load_larger_dataset():
    dataset_path = 'image_data'
    imgs_list = os.listdir(dataset_path + '/img')
    indices = list(range(len(imgs_list)))
    random.shuffle(indices)
    for index in indices:
        img_id = imgs_list[index].split('.')[0]
        ref_mask_index = np.random.randint(0, len(imgs_list))
        ref_mask_id = imgs_list[ref_mask_index].split('.')[0]
        image = np.array(imread(dataset_path + '/img/' + img_id + '.jpg'))
        mask = np.load(dataset_path + '/bool_mask/' + img_id + '.npy')
        ref_mask = np.load(dataset_path + '/bool_mask/' + ref_mask_id + '.npy')
        yield image, mask, ref_mask


def get_dataset_size():
    dataset_path = 'image_data'
    imgs_list = os.listdir(dataset_path + '/img')
    return len(imgs_list)


def load_large_dataset_with_class(patch_size=None, align=False):
    """
    Load large dataset with pixel level class info, 11355 images, 16438 masks
    Only one class of mask is returned for each image.
    If  patch_size is not None, images go through `cv2.resize`.
    If align is True, `align_mask` is called to move the ref mask to the center fo the mask

    Prepare:
        Unarchieve bool_mask_sep_inst, img into `image_data/`

    Yield:
        (an image, mask of ONE object in the image, a random ref mask)
        output images are float images in (0, 1), resized to patch_size
        output masks are bool masks, resized to patch_size
    """

    def wrapped():
        dataset_path = 'image_data'
        imgs_list = glob.glob(dataset_path + '/img/*.jpg')
        random.shuffle(imgs_list)

        # gather all corresponding masks for each image
        all_masks_files = glob.glob(dataset_path + '/bool_mask_sep_inst/*.npy')
        image_to_masks = defaultdict(list)
        for x in all_masks_files:
            x = os.path.basename(x)
            # MaskId := ImageId_MaskNum.npy
            image_id = x[:x.rindex('_')]
            image_to_masks[image_id].append(x)

        for fname in imgs_list:
            image_id = os.path.basename(fname).rstrip('.jpg')
            mask_base = random.choice(image_to_masks[image_id])
            ref_mask_path = random.choice(all_masks_files)

            image = skimage.img_as_float(imread(dataset_path + '/img/' + image_id + '.jpg'))
            mask = np.load(dataset_path + '/bool_mask_sep_inst/' + mask_base)
            ref_mask = np.load(ref_mask_path)

            if patch_size is not None:
                image = sk_resize(image, (patch_size, patch_size))
                mask = skimage.img_as_bool(sk_resize(mask * 255., (patch_size, patch_size)))
                ref_mask = skimage.img_as_bool(sk_resize(ref_mask * 255., (patch_size, patch_size)))

            if align:
                ref_mask = align_mask(mask, ref_mask)

            yield (image, mask, ref_mask)

    return wrapped


def test_load_large_dataset_with_class():
    from matplotlib import pyplot as plt

    image, mask, ref_mask = next(load_large_dataset_with_class()())
    print(image.max(), mask.max(), ref_mask.max())
    print(image.dtype, mask.dtype, ref_mask.dtype)
    print(image.shape, mask.shape, ref_mask.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.subplot(1, 3, 3)
    plt.imshow(ref_mask)
    plt.title('without resize or align')
    plt.show()

    plt.figure()
    image, mask, ref_mask = next(load_large_dataset_with_class(patch_size=1024, align=True)())
    print(image.max(), mask.max(), ref_mask.max())
    print(image.dtype, mask.dtype, ref_mask.dtype)
    print(image.shape, mask.shape, ref_mask.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.subplot(1, 3, 3)
    plt.imshow(ref_mask)
    plt.title('with resize & align')
    plt.show()


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

        sessions = \
            loadmat('SOD/SO' + str(image_id) + '.mat', squeeze_me=True, struct_as_record=False)[
                'SES']
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
    from matplotlib import pyplot as plt

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
    # visualize_sod()
    test_load_large_dataset_with_class()
