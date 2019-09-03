import warnings
warnings.filterwarnings('ignore')

import logging
import shutil
import argparse
import sys
import os
# 当前项目路径加入到环境变量中，让解析器能找到第一model的目录
father_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(father_dir)

from srcipt.SUnet import small_UNet
from srcipt.prepare_data import prepare_data
from srcipt.metrics import hard_dice_coe
from srcipt.record_db import start_expr
from srcipt.utils import *
from srcipt.layers import *

_join = os.path.join

os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type',
                        default='TONGREN',#'DRIVE'， 'CHASE'， 'TONGREN'
                        help='Directory for saving trained models')

    parser.add_argument('--fold',
                        default='FOLD1',  # 'FOLD1'， 'FOLD2'， 'FOLD3', 'FOLD4'
                        help='folds for cross validation')

    parser.add_argument('--log_directory',
                        default='log',
                        help='Directory for saving trained models')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--n_epoches', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Optimizer learning rate')
    parser.add_argument('--early_stopping_max_checks', type=int, default=100,
                        help='Max checks without improvement for early stopping')

    parser.add_argument('--train', action='store_true', default=True,
                        help='Set to True to train network')
    parser.add_argument('--Restore', action='store_true', default=True,
                        help='Set to True to continue train network')
    parser.add_argument('--load_checkpoint', type=str,
                        default='/home/wanghua/wh/vessel/log_c/small_UNet_20/model/model.ckpt-52',
                        help='Load saved checkpoint, arg=checkpoint_name')############################

    args = parser.parse_args()

    EXPR_NAME = 'S-UNet'

    # ===========================================
    if args.data_type == 'CHASE':
        npz_directory = 'data/%s/%s/processed/result_gray_512_16.npz' % (args.data_type, args.fold)
    else:
        npz_directory = 'data/%s/processed/result_gray_512_16.npz' % args.data_type
    data_path = _join(father_dir, npz_directory)
    if not os.path.isfile(data_path):
        source_data_dir = data_path.replace('processed/result_gray_512_16.npz', '')
        prepare_data(source_data_dir, data_path)

    data = np.load(data_path)
    train_size = np.shape(data['y_train'])[0]
    test_size = np.shape(data['y_test'])[0]
    print(train_size)
    print(test_size)
    # ===========================================

    with tf.name_scope('Train_data'):
        train_set = tf.data.Dataset.from_tensor_slices((data['X_train'], data['y_train'],
                                                        data['y_train_'], data['z_train']))
        # train_set = train_set.map(lambda x,y,z: preprocess_example((x,y,z), [512, 512, 1]))
        train_set = train_set.shuffle(100).batch(args.batch_size).repeat(args.n_epoches+1)#.prefetch(4)
        train_iterator = train_set.make_initializable_iterator()
        next_element = train_iterator.get_next()

    with tf.name_scope('Test_data'):
        test_set = tf.data.Dataset.from_tensor_slices((data['X_test'], data['y_test'],
                                                       data['y_test_'], data['z_test']))
        # test_set = test_set.map(lambda x, y, z: preprocess_example((x, y, z), [512, 512, 1]))
        test_set = test_set.batch(1)#.prefetch(4)#.repeat(args.n_epoches)
        test_iterator = test_set.make_initializable_iterator()
        one_element = test_iterator.get_next()

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=(None, None, None, 1))
        y = tf.placeholder(tf.uint8, shape=(None, None, None))
        one_hot = tf.one_hot(y, depth=2, axis=-1)

    with tf.name_scope('Net'):
        network_fn = small_UNet(num_class=2)
        aq,bq,  output = network_fn(x)

    with tf.name_scope('loss'):
        loss_a1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=aq)
        loss_b1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=bq)
        # loss_c1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=c)
        # loss_d1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=d)
        # loss_e1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=e)
        # loss_f1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=f)
        loss_o1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=output)

        l2_loss = tf.add_n([0.00002 * tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = loss_o1 + l2_loss + loss_a1 + loss_b1# + loss_c1 +loss_d1 + loss_e1 + loss_f1

    with tf.name_scope('optim'):
        # optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        optim_t = optimizer.minimize(loss)

    with tf.name_scope('evaluation'):
        predict = tf.argmax(tf.nn.softmax(output, axis=-1), axis=-1)
        ret_dice = hard_dice_coe(y, predict, num_class=2)

    if args.train:
        EXPR_ID: int = start_expr(EXPR_NAME, '', '', '')
        print('EXPR_ID', EXPR_ID)
        args.LOG_DIR = _join(_join(father_dir, args.log_directory), EXPR_NAME + '_' + str(EXPR_ID))
        os.makedirs(args.LOG_DIR, exist_ok=True)
        shutil.copytree(_join(father_dir, 'srcipt'),_join(args.LOG_DIR, 'srcipt'))

        logging.basicConfig(level=logging.INFO,
                            filename=_join(args.LOG_DIR, 'new.log'),
                            filemode='w',
                            format='%(asctime)s - : %(message)s')


        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=5)
            current_result = 0
            sess.run(tf.global_variables_initializer())
            if args.Restore:
                saver.restore(sess, args.load_checkpoint)

            sess.run([train_iterator.initializer])
            stop = 0

            for epoch in range(args.n_epoches):
                #Train Loop
                Loss = []
                Dice = []
                # for i in range(train_size // args.batch_size):
                for i in range(10):
                    image, label, mask, FOV = sess.run(next_element)

                    loss_,ret_dice_,  _ = \
                        sess.run([loss, ret_dice, optim_t], feed_dict={x:image, y:mask})
                    Loss.append(loss_)
                    Dice.append(ret_dice_[1])

                message = 'Epoch : %s Train Loss : %s Dice 1 : %s' % \
                          (epoch, np.mean(Loss), np.mean(Dice))
                myprint(logging, message)

                # Test Loop
                dice_list = []
                roc_list = []
                from sklearn.metrics import roc_auc_score
                sess.run([test_iterator.initializer])
                tmp_out = [];tmp_out_p = [];tmp_one = [];tmp_fov = []
                for j in range((test_size)):
                    image, label, mask, FOV = sess.run(one_element)
                    one_hot_, output_ = \
                        sess.run([one_hot, output], feed_dict={x: image, y: mask})
                    output_1 = (np.argmax(np.squeeze(output_), axis=-1))
                    one_hot_1 = np.squeeze(one_hot_)[..., 1]
                    tmp_one.append(one_hot_1)
                    tmp_out.append(output_1)
                    tmp_out_p.append(output_[0, :, :, 1])
                    tmp_fov.append(np.squeeze(FOV))
                    if (j + 1) % 1 == 0:
                        output_1 = merge_img(tmp_out, 1, 1)
                        output_1p = merge_img(tmp_out_p, 1, 1)
                        one_hot_1 = merge_img(tmp_one, 1, 1)
                        FOV_1 = merge_img(tmp_fov, 1, 1)
                        tmp_out = [];tmp_out_p = [];tmp_one = [];tmp_fov = []

                        dice_list.append(
                            np.sum(output_1[one_hot_1 == 1.0]) * 2.0 /
                            (np.sum(output_1) + np.sum(one_hot_1)))

                        roc_list.append(roc_auc_score(y_true=one_hot_1[np.squeeze(FOV_1) == 1].flatten(),
                                                      y_score=output_1p[np.squeeze(FOV_1) == 1].flatten()))

                me = '[***] Epoch : %s Test Loss : %s Dice 1 : %s' % (epoch, np.mean(dice_list), np.mean(roc_list))
                myprint(logging, me)

                stop += 1
                result = np.mean(dice_list)
                if stop >= args.early_stopping_max_checks:
                    myprint(logging, 'save model, performance %s' % result)
                    myprint(logging, "Early stopping!")
                    saver.save(sess, _join(args.LOG_DIR, 'model/model.ckpt'), global_step=epoch)
                    myprint(logging, '------ Checkpoint saved!')
                    break

                if result > current_result:
                    myprint(logging, 'save model, performance: %s' % result)
                    current_result = result
                    saver.save(sess, _join(args.LOG_DIR, 'model/model.ckpt'), global_step=epoch)
                    myprint(logging, '------ Checkpoint saved!')
                    stop = 0



    else:
        from sklearn.metrics import roc_auc_score
        output_path = data_path.replace('processed/result_gray_512_16.npz', 'result')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            count_trainable_params()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, args.load_checkpoint)
            # Test Loop
            dice_list = []
            roc_list = []
            label_1 = []
            pre_1p = []
            pre_1 = []
            fov = []
            sess.run([test_iterator.initializer])
            for j in range(test_size):
                image, label, mask, FOV = sess.run(one_element)
                output_ = sess.run([output], feed_dict={x: image, y: mask})
                one_hot_ = sess.run([one_hot], feed_dict={x: image, y: mask})
                output_1 = (np.argmax(np.squeeze(output_), axis=-1))
                one_hot_1 = np.squeeze(one_hot_)[..., 1]
                output_1p = output_[0][0, :, :, 1]

                label_1.append(one_hot_1)
                pre_1p.append(output_1p)
                pre_1.append(output_1)
                fov.append(np.squeeze(FOV))
                dice_list.append(
                    np.sum(output_1[one_hot_1 == 1.0]) * 2.0 /
                    (np.sum(output_1) + np.sum(one_hot_1)))
                roc_list.append(roc_auc_score(y_true=one_hot_1[np.squeeze(FOV) == 1].flatten(),
                                            y_score=output_1p[np.squeeze(FOV) == 1].flatten()))

            a1 = np.stack(label_1)
            b1 = np.stack(pre_1p)
            b11 = np.stack(pre_1)
            c1 = np.stack(fov)
            # m(b1, b11, a1, c1)
            me = '[***] Test Dice 1 : %s AUC : %s' % (np.mean(dice_list), np.mean(roc_list))
            myprint(logging, me)

if __name__ == '__main__':
    main(sys.argv)