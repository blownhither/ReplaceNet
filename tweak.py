import numpy as np
from skimage.transform import warp, AffineTransform


def get_mask_center(mask):
    center = np.argwhere(mask).mean(axis=0)
    return center


def align_mask(anchor, movable):
    """
    move mass center of `movable` to align with the center of `anchor`
    """
    center_anchor = get_mask_center(anchor)
    center_movable = get_mask_center(movable)
    diff = center_movable - center_anchor
    # image coordinates use (col, row) order
    diff = diff[::-1]
    out = warp(movable, AffineTransform(translation=diff))
    return out


def align_image(anchor_mask, movable_mask, movable_image):
    center_anchor = get_mask_center(anchor_mask)
    center_movable = get_mask_center(movable_mask)
    diff = center_movable - center_anchor
    # image coordinates use (col, row) order
    diff = diff[::-1]
    result_mask = warp(movable_mask, AffineTransform(translation=diff))
    result_image = warp(movable_image, AffineTransform(translation=diff))
    return result_mask, result_image


def test_align_mask():
    from load_data import load_parsed_sod
    from matplotlib import pyplot as plt

    images, masks = load_parsed_sod()
    a, b = masks[0], masks[2]
    center_a = get_mask_center(a)
    center_b = get_mask_center(b)
    c = align_mask(a, b)
    center_c = get_mask_center(c)

    # now that `anchor` and `moved` are aligned
    plt.subplot(1, 3, 1)
    plt.imshow(a)
    plt.scatter(center_a[1], center_a[0], c='red')
    plt.title('anchor')
    plt.subplot(1, 3, 2)
    plt.imshow(b)
    plt.scatter(center_b[1], center_b[0], c='red')
    plt.title('movable')
    plt.subplot(1, 3, 3)
    plt.imshow(c)
    plt.scatter(center_c[1], center_c[0], c='red')
    plt.title('moved')
    plt.show()


if __name__ == '__main__':
    test_align_mask()




