import numpy as np
import cv2
import nibabel as nib
from skimage.io import imsave

def fillholse(im_th):
    '''
    空洞填充
    param:
        im_th:二值图像
    return:
        im_out:填充好的图像
    '''
    # Copy the thresholded image.
    im_floodfill = im_th.copy().astype(np.uint8)

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    # retval, image, mask, rect = cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out

def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data