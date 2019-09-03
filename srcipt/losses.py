import tensorflow as tf

def cross_entropy(y_true, y_pred, *arg):
    return tf.nn.softmax_cross_entropy_with_logits_v2(y_true, y_pred)

def dice_coe(y_true: tf.Tensor, y_pred: tf.Tensor, *args, smooth=1e-10):
    y_pred = tf.nn.softmax(y_pred)
    label_shape = y_true.shape
    pred_shape = y_pred.shape
    print('label_shape', label_shape)
    print('pred_shape', pred_shape)
    # assert label_shape[1:] == pred_shape[1:], "label shape and pred shape should be same"
    rank = len(label_shape)

    num_class = int(label_shape[-1])

    ret = []
    for i in range(num_class):
        label = y_true[..., i]
        pred = y_pred[..., i]
        inse = tf.cast(tf.reduce_sum(label * pred, axis=list(range(1, rank - 1))), tf.float32)
        l = tf.cast(tf.reduce_sum(pred * pred, axis=list(range(1, rank - 1))), tf.float32)
        r = tf.cast(tf.reduce_sum(label * label, axis=list(range(1, rank - 1))), tf.float32)
        print(l.dtype, r.dtype, inse.dtype)

        dice = (2.0 * inse) / (l + r + smooth)
        ret.append(dice)

    # ret = tf.add_n(ret)
    # return float(num_class) - tf.reduce_mean(ret)
    ret = tf.reduce_mean(ret)
    return 1 - ret

