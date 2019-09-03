import numpy as np
def merge_img(tmp, h, w):
    col = []
    for i in range(h):
        row = []
        for j in range(w):
            row.append(tmp[i * w + j])
        row = np.hstack(row)
        col.append(row)
    col = np.vstack(col)
    return col

def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape)==3)  #4D arrays

    if (x >= DRIVE_masks.shape[2] or y >= DRIVE_masks.shape[1]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,y,x]>0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False

def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==3 and len(data_masks.shape)==3)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[1]==data_masks.shape[1])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    height = data_imgs.shape[1]
    width = data_imgs.shape[2]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,y,x])
                    new_pred_masks.append(data_masks[i,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

from sklearn.metrics import *
def m(a, a11,b,test_border_masks):
    y_scores, y_true = pred_only_FOV(a11, b,
                                     test_border_masks)
    confusion = confusion_matrix(y_true, y_scores)
    print(confusion)
    mcc = matthews_corrcoef(y_true, y_scores)
    print("mcc: " + str(mcc))

    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))

    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))

    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))

    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_scores, normalize=True)
    print("Jaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_scores, labels=None, average='binary', sample_weight=None)
    print("F1 score (F-measure): " + str(F1_score))

    kappa = cohen_kappa_score(y_true, y_scores)
    print("kappa: " + str(kappa))

    print(np.sum(y_true[y_scores==1])*2.0/(np.sum(y_true)+np.sum(y_scores)))


def myprint(logging, message):
    logging.info(message)
    print(message)