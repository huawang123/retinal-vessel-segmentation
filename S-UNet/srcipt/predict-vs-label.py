import numpy as np
from skimage.io import imread, imsave
import glob
import os

path = ''
path1 = ''
p = glob.glob(path1)
img_segmented = imread(p[-1]).astype(np.uint8)
pre = imread(path).astype(np.uint8)
a = np.copy(img_segmented)
b = np.copy(pre)

c = a - b
a[c==1] = 2
a[c==255] = 3

f1 = np.zeros_like(a)
f1[a==1] = 255
f2 = np.zeros_like(a)
f2[a == 2] = 255
f3 = np.zeros_like(a)
f3[a == 3] = 255


g = np.stack([f1,f2,f3], axis=-1)

imsave('', g)
imsave('', img_segmented*255)