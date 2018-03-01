from data import dataProcess
import numpy as np
from ISBI.Reader import Reader
import tensorflow as tf
from tf_version.model import unet_model
from keras.losses import binary_crossentropy
from keras.preprocessing.image import array_to_img
rows = 512
cols = 512
batch_size = 5
epoch_num = 10


def load_data():
    mydata = dataProcess(rows, cols, npy_path='/home/give/PycharmProjects/unet/data/npy')
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data()
    return imgs_train, imgs_mask_train, imgs_test


def train(inputs, gt, outputs):
    loss = tf.reduce_mean(binary_crossentropy(gt, outputs))
    accuracy = tf.keras.metrics.binary_accuracy(gt, outputs)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=loss)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        reader = Reader('/home/give/Documents/dataset/ISBI2017/Slice/Train Batch1',
                        '/home/give/Documents/dataset/ISBI2017/Slice/Train Batch2',
                        batch_size=batch_size)
        train_generator = reader.train_generator
        val_generator = reader.val_generator
        test_generator = reader.val_generator

        for epoch_id in range(epoch_num):
            print 'Epoch: %d / %d' % (epoch_id, epoch_num)
            cur_batch_imgs, cur_batch_mask = train_generator.next()
            _, accuracy_value = sess.run([train_step, accuracy], feed_dict={
                inputs: cur_batch_imgs,
                gt: cur_batch_mask
            })
            print np.mean(accuracy_value)
        cur_batch_imgs, cur_batch_mask = test_generator.next()
        accuracy_value = sess.run(accuracy, feed_dict={
            inputs: cur_batch_imgs,
            gt: cur_batch_mask
        })
        print np.mean(accuracy_value)

if __name__ == '__main__':
    inputs = tf.keras.Input([512, 512, 1], batch_size=None, name='x-input', dtype=tf.float32)
    gt = tf.placeholder(dtype=tf.float32, shape=[batch_size, rows, cols, 1], name='y-input')
    outputs = unet_model(inputs)
    train(inputs, gt, outputs)