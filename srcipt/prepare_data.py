import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import skimage
from skimage.io import imread, imsave
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import os
_join = os.path.join


def prepare_data(root_dir, result_file):
    print('Augmenting train image data sets ...')
    # args.logging.info('Augmenting train image data sets ...')
    train_images_dir = _join(root_dir, 'training/images')
    train_annotations_dir = _join(root_dir, 'training/1st_manual')
    train_mask_dir = _join(root_dir, 'training/mask')

    X_train1, y_train1, y_train1_, z_train1 = get_data(train_images_dir, train_annotations_dir, train_mask_dir)
    X_train, y_train, y_train_, z_train = augment_images(X_train1, y_train1, y_train1_, z_train1)

    print('Augmenting test image data sets ...')
    # args.logging.info('Augmenting test image data sets ...')
    test_images_dir = _join(root_dir, 'test/images')
    test_annotations_dir = _join(root_dir, 'test/1st_manual')
    test_mask_dir = _join(root_dir, 'test/mask')

    X_test, y_test, y_test_, z_test = get_data_t(test_images_dir, test_annotations_dir, test_mask_dir)

    X_test = np.array(X_test, dtype=np.float32)[:, :, :, np.newaxis]
    y_test = np.array(y_test, dtype=np.uint8)
    y_test_ = np.array(y_test_, dtype=np.uint8)
    z_test = np.array(z_test, dtype=np.uint8)

    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    np.savez(result_file,
             X_train=X_train, y_train=y_train, y_train_=y_train_, z_train=z_train,
             X_test=X_test, y_test=y_test, y_test_=y_test_, z_test=z_test,
             )
    print(np.shape(X_train))
    print(np.shape(X_test))



def get_data(data_path, label_path, mask_path):
    X = []
    Y = []
    Y_ = []
    Z = []
    image_list = os.listdir(data_path)
    image_list.sort()
    label_list = os.listdir(label_path)
    label_list.sort()
    mask_list = os.listdir(mask_path)
    mask_list.sort()

    for i, image in enumerate(image_list):
        # print(i, image)
        img = imread(os.path.join(data_path, image),as_gray=True).astype(np.float32)
        img_segmented = imread(os.path.join(label_path, label_list[i])).astype(np.uint8)
        img_masked = imread(os.path.join(mask_path, mask_list[i])).astype(np.uint8)
        img_segmented[img_segmented==125] = 1
        img_segmented[img_segmented==252] = 1
        img_segmented[img_segmented==255] = 1
        img_segmented[img_segmented>0] = 1

        img_label = np.copy(img_segmented)
        img_label[img_label>0] = 1

        img = normalize_image(img)
        img_masked = normalize_image(img_masked)

        size = 256*2
        if 'CHASE' in data_path:
            wh = [150,110]
        elif 'TONGREN' in data_path:
            wh = [300, 300]
        else:
            wh = [47, 47]
        for i in range(0, img.shape[0]-size, wh[0]):
            for j in range(0, img.shape[1]-size, wh[1]):
                temp_img = img[i:i+ size, j:j+size]
                temp_seg = img_segmented[i:i+size,j:j+size]
                temp_mask = img_masked[i: i + size, j: j + size]
                temp_lable = img_label[i: i + size, j: j + size]
                temp_img = normalize_image(temp_img)
                # img_segmented = normalize_image(img_segmented)
                temp_mask = normalize_image(temp_mask)

                X.append(temp_img)
                Y.append(temp_seg)
                Y_.append(temp_lable)
                Z.append(temp_mask)
    return X, Y, Y_, Z

def get_data_t(data_path, label_path, mask_path):
    X = []
    Y = []
    Y_ = []
    Z = []
    image_list = os.listdir(data_path)
    image_list.sort()
    label_list = os.listdir(label_path)
    label_list.sort()
    mask_list = os.listdir(mask_path)
    mask_list.sort()

    for i, image in enumerate(image_list):
        img = imread(os.path.join(data_path, image), as_gray=True).astype(np.float32)
        img_segmented = imread(os.path.join(label_path, label_list[i])).astype(np.uint8)
        img_masked = imread(os.path.join(mask_path, mask_list[i])).astype(np.uint8)
        img_segmented[img_segmented==125] = 1
        img_segmented[img_segmented==252] = 1
        img_segmented[img_segmented==255] = 1
        img_segmented[img_segmented>0] = 1


        img_label = np.copy(img_segmented)
        img_label[img_label>0] = 1

        p = 16
        shape = img.shape
        h_ = shape[0] / p
        h = shape[0] // p
        if h_ > h:
            pad = (h+1)*p - shape[0]
            img = np.pad(img, [[0, pad],[0,0]], 'edge')
            img_segmented = np.pad(img_segmented, [[0, pad],[0,0]], 'edge')
            img_masked = np.pad(img_masked, [[0, pad],[0,0]], 'edge')
            img_label = np.pad(img_label, [[0, pad],[0,0]], 'edge')
        w_ = shape[1] / p
        w = shape[1] // p
        if w_ > w:
            pad = (w+1)*p - shape[1]
            img = np.pad(img, [[0, 0], [0, pad]], 'edge')
            img_segmented = np.pad(img_segmented, [[0, 0], [0, pad]], 'edge')
            img_masked = np.pad(img_masked, [[0, 0], [0, pad]], 'edge')
            img_label = np.pad(img_label, [[0, 0], [0, pad]], 'edge')

        img = normalize_image(img)
        img_masked = normalize_image(img_masked)

        X.append(img)
        Y.append(img_segmented)
        Y_.append(img_label)
        Z.append(img_masked)
    return X, Y, Y_, Z

def augment_images(img_array, img_segmented_array,img_label_array, img_masked_array):
    images_augmented = []
    annotations_augmented = []
    label_augmented = []
    masks_augmented = []

    for img, img_segmented,img_label, img_masked in zip(img_array, img_segmented_array,img_label_array, img_masked_array):
        img_lr, img_ud = flip_transform_image(img)

        img_segmented_lr, img_segmented_ud = flip_transform_image(img_segmented)

        img_label_lr, img_label_ud = flip_transform_image(img_label)

        img_masked_lr, img_masked_ud = flip_transform_image(img_masked)

        images_augmented.extend([img, img_lr, img_ud])
        annotations_augmented.extend([img_segmented, img_segmented_lr, img_segmented_ud])

        label_augmented.extend([img_label, img_label_lr, img_label_ud])

        masks_augmented.extend([img_masked, img_masked_lr, img_masked_ud])

    images_augmented = np.array(images_augmented, dtype=np.float32)[:, :, :, np.newaxis]
    annotations_augmented = np.array(annotations_augmented, dtype=np.uint8)#[:, :, :, np.newaxis]
    label_augmented = np.array(label_augmented, dtype=np.uint8)#[:, :, :, np.newaxis]
    masks_augmented = np.array(masks_augmented, dtype=np.uint8)#[:, :, :, np.newaxis]

    return images_augmented, annotations_augmented,label_augmented, masks_augmented

def normalize_image(img):
    return ((img - img.min()) / (img.max() - img.min()))

def flip_transform_image(img):
    img_lr = np.fliplr(img)
    img_ud = np.flipud(img)
    return img_lr, img_ud

# root_dir = '/storage/wh/S-UNet/data/DRIVE'
# target_file = '/storage/wh/S-UNet/data/DRIVE/processed/'
#
# if not os.path.exists(target_file):
#     os.makedirs(target_file)
#
#
# prepare_data(root_dir, target_file)



