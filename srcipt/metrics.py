import tensorflow as tf


def mean_acc(label, predict):
    predict = tf.argmax(predict, axis=1)
    metric, metric_update = tf.metrics.accuracy(label, predict)
    summary_op = tf.summary.scalar('accuracy', metric)
    return metric, metric_update, summary_op


def hard_dice_coe(label, predict, num_class, smooth=1e-10):
    label_shape = label.shape
    predict_shape = predict.shape
    # assert label_shape[1:] == predict_shape[1:], "label shape and predict shape should be same"

    rank = len(label_shape)

    ret = []
    for i in range(num_class):
        gt = tf.cast(tf.equal(label, i), tf.float32)
        la = tf.cast(tf.equal(predict, i), tf.float32)
        inse = tf.reduce_sum(gt * la, axis=list(range(1, rank)))
        l = tf.reduce_sum(gt, axis=list(range(1, rank)))
        r = tf.reduce_sum(la, axis=list(range(1, rank)))
        dice = tf.reduce_mean((2.0 * inse) / (l + r + smooth))
        # summary_ops.append(tf.summary.scalar('dice_{}'.format(i), dice))
        ret.append(tf.reduce_mean(dice))

    return ret

