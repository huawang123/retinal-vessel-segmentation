# 读取nii数据
# 轮廓内填充，生成分割标签
# 转化为png格式
# 生成mask标签
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from nibabel import load, save
import glob
import numpy as np
from skimage.io import imread,imsave
from srcipt.fills import fillholse
import os

def cc(path, tar_path):
    basename = os.path.basename(path).split('.')[0]
    data1 = load(path)
    img_arr = data1.get_fdata()
    img_arr = np.squeeze(img_arr)
    img = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    print(np.shape(np.rot90(img)), img.dtype)
    imsave(os.path.join(tar_path, 'images', basename+'.png'), np.rot90(img))
    data = np.rot90(img)#imread(os.path.join(tar_path, 'images', basename+'.png'), as_gray=True).astype(np.float32)
    mask = data[...,0]
    mask[mask>0] = 255
    imsave(os.path.join(tar_path, 'mask', basename + '.png'), (mask/255).astype(np.uint8))
    label1 = load(path.replace('dt', 'label'))
    label_arr = label1.get_fdata()
    label_arr = np.squeeze(label_arr)

    label0 = np.copy(label_arr)
    label0[label0 > 2] = 0
    label0[label0 > 0] = 1
    label0_fill = fillholse(label0.astype(np.uint8))
    label1 = np.copy(label_arr)
    label1[label1 == 2] = 0
    label1[label1 == 3] = 1
    label1_fill = fillholse(label1.astype(np.uint8))
    label = label0_fill + label1_fill
    label[label > 0] = 1
    imsave(os.path.join(tar_path, '1st_manual', basename+'.png'), np.rot90(label))



father_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
_join = os.path.join

path = _join(father_dir, 'data/TONGREN_nii/BH_training/dt/*.nii')

files = np.sort(glob.glob(path))
target_path = _join(father_dir, 'data/TONGREN/training')

if not os.path.exists(_join(target_path, 'images')):
    os.makedirs(_join(target_path, 'images'))
if not os.path.exists(_join(target_path, 'mask')):
    os.makedirs(_join(target_path, 'mask'))
if not os.path.exists(_join(target_path, '1st_manual')):
    os.makedirs(_join(target_path, '1st_manual'))
for f in files:
    cc(f, target_path)

path = _join(father_dir, 'data/TONGREN_nii/BH_test/dt/*.nii')
files = np.sort(glob.glob(path))
target_path = _join(father_dir, 'data/TONGREN/test')
if not os.path.exists(_join(target_path, 'images')):
    os.makedirs(_join(target_path, 'images'))
if not os.path.exists(_join(target_path, 'mask')):
    os.makedirs(_join(target_path, 'mask'))
if not os.path.exists(_join(target_path, '1st_manual')):
    os.makedirs(_join(target_path, '1st_manual'))
for f in files:
    cc(f, _join(father_dir, target_path))
