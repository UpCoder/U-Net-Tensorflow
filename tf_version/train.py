from data import dataProcess
import numpy as np
from ISBI.Reader import Reader
import tensorflow as tf
from tf_version.model import unet_model
from model import Metrics
from keras.preprocessing.image import array_to_img
from keras.layers.noise import GaussianNoise
import os
rows = 512
cols = 512
batch_size = 5
category_num = 2
epoch_num = 10
max_iterator = int(1e5)
lr = 1e-5
save_model_path = '/home/give/PycharmProjects/unet/tf_version/parameters/model'
resume = True


def load_data():
    mydata = dataProcess(rows, cols, npy_path='/home/give/PycharmProjects/unet/data/npy')
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data()
    return imgs_train, imgs_mask_train, imgs_test


def train(inputs, gt, outputs):
    print 'softmax cross entropy', tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.cast(gt, tf.int32), 2),
                                                                           logits=outputs)
    softmax_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.cast(gt, tf.int32), 2), logits=outputs))
    binary_loss = tf.reduce_sum(
        tf.keras.backend.categorical_crossentropy(tf.one_hot(tf.cast(gt, tf.int32), 2), outputs, True))
    dice_loss = Metrics.dice_loss(gt, tf.argmax(outputs, axis=3))
    loss = binary_loss
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(tf.one_hot(tf.cast(gt, tf.int32), 2), outputs))
    dice_score = Metrics.dice_score(gt, tf.argmax(outputs, axis=3))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss, global_step=global_step)
    tf.summary.image('inputs', inputs, max_outputs=5)
    tf.summary.image('gt', tf.expand_dims(gt * 100, axis=3), max_outputs=5)
    tf.summary.image('output', tf.expand_dims(tf.cast(tf.argmax(outputs, axis=3) * 100, tf.float32), axis=3),
                     max_outputs=5)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('dice_score', dice_score)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        summary_file = tf.summary.FileWriter('./log', graph=sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if resume:
            latest = tf.train.latest_checkpoint(os.path.dirname(save_model_path))
            print latest
            saver.restore(sess, latest)
        reader = Reader('/home/give/Documents/dataset/ISBI2017/Slice_All/Train Batch1',
                        '/home/give/Documents/dataset/ISBI2017/Slice_All/Train Batch2',
                        batch_size=batch_size)
        train_generator = reader.train_generator
        val_generator = reader.val_generator
        test_generator = reader.val_generator

        for step_id in range(max_iterator):
            cur_batch_imgs, cur_batch_mask = train_generator.next()
            _, summary_value, global_step_value, accuracy_value, dice_score_value, loss_value = sess.run(
                [train_step, summary_op, global_step, accuracy, dice_score, loss], feed_dict={
                    inputs: cur_batch_imgs,
                    gt: cur_batch_mask
                })
            if step_id % 100 == 0:
                print 'Step: %d / %d' % (step_id, max_iterator)
                print 'train accuracy: %.2f, dice score: %.2f, train loss: %.2f' % (
                accuracy_value, dice_score_value, loss_value)
                saver.save(sess, save_model_path, global_step=global_step)
                cur_batch_imgs, cur_batch_mask = val_generator.next()
                accuracy_value, dice_score_value, loss_value = sess.run([accuracy, dice_score, loss], feed_dict={
                    inputs: cur_batch_imgs,
                    gt: cur_batch_mask
                })
                print 'val accuracy: %.2f, dice score: %.2f, val loss: %.2f' % (
                    accuracy_value, dice_score_value, loss_value)
            summary_file.add_summary(summary_value, global_step=global_step_value)
        summary_file.close()
        cur_batch_imgs, cur_batch_mask = test_generator.next()
        accuracy_value = sess.run(accuracy, feed_dict={
            inputs: cur_batch_imgs,
            gt: cur_batch_mask
        })
        print np.mean(accuracy_value)
        predict = sess.run(outputs, feed_dict={
            inputs: cur_batch_imgs
        })
        img = array_to_img(np.round(predict[0]))
        img.save('./predict.png')


if __name__ == '__main__':
    inputs = tf.keras.Input([512, 512, 1], batch_size=None, name='x-input', dtype=tf.float32)
    gt = tf.placeholder(dtype=tf.float32, shape=[None, rows, cols], name='y-input')
    # GaussianNoise(stddev=0.01)(inputs)
    outputs = unet_model(inputs)
    train(inputs, gt, outputs)