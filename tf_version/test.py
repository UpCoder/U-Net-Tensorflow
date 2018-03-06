import tensorflow as tf
from train import batch_size
from model import unet_model
from ISBI.Reader import Reader
import numpy as np
from ISBI.util import read_nii_file, read_mhd_image, read_dicom_series, save_mhd_image, convert2batchfirst
from PIL import Image
import os
model_save_path = '/home/give/PycharmProjects/unet/tf_version/parameters'


def test_one_scan(data_path, save_path, data_format):
    if data_format == 'mhd':
        image = read_mhd_image(data_path)
    if data_format == 'nii':
        image = read_nii_file(data_path)
        image = convert2batchfirst(image)
    if data_format == 'dicom':
        image = read_dicom_series(data_path)
        image = np.flipud(image)
    print np.shape(image), np.max(image), np.mean(image), image.dtype
    # image shape: [slice_num, w, h]
    image = np.asarray(image, np.float32) / 255.0
    slice_num = len(image)
    inputs = tf.keras.Input([512, 512, 1], batch_size=None, name='x-input', dtype=tf.float32)
    outputs = unet_model(inputs)
    softmax = tf.nn.softmax(outputs)
    pred = tf.argmax(softmax, axis=3)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        latest = tf.train.latest_checkpoint(model_save_path)
        saver.restore(sess, latest)
        start_index = 0
        pred_slices = []
        while start_index < slice_num:
            print 'Processing: %d / %d' % (start_index, slice_num)
            cur_image_batch = image[start_index: start_index + batch_size]
            cur_pred = sess.run(pred, feed_dict={
                inputs: np.expand_dims(cur_image_batch, axis=3)
            })
            pred_slices.extend(cur_pred)
            start_index += batch_size
    save_mhd_image(pred_slices, os.path.join(save_path, 'mask.mhd'))
    print np.shape(pred_slices)


def test_one_slice(data_path, save_path, data_format):
    if data_format == 'mhd':
        image = read_mhd_image(data_path)
    if data_format == 'nii':
        image = read_nii_file(data_path)
    if data_format == 'dicom':
        image = read_dicom_series(data_path)
    # image shape: [slice_num, w, h]
    image = np.squeeze(image)
    image = np.array(image) / 255.0
    if len(np.shape(image)) != 2:
        assert 'Error'
    inputs = tf.keras.Input([512, 512, 1], batch_size=None, name='x-input', dtype=tf.float32)
    outputs = unet_model(inputs)
    softmax = tf.nn.softmax(outputs)
    pred = tf.argmax(softmax, axis=3)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        latest = tf.train.latest_checkpoint(model_save_path)
        print latest
        saver.restore(sess, latest)
        cur_pred = sess.run(pred, feed_dict={
            inputs: np.expand_dims(np.expand_dims(image, axis=3), axis=0)
        })
        cur_pred = np.squeeze(cur_pred)
        pred_img = tf.keras.preprocessing.image.array_to_img(np.expand_dims(cur_pred, 2))
        pred_img.show()
    save_mhd_image(cur_pred, os.path.join(save_path, 'mask.mhd'))
    print np.shape(cur_pred)


def test_batch(inputs, outputs, data):
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        latest = tf.train.latest_checkpoint(model_save_path)
        print latest
        saver.restore(sess, latest)

        return sess.run(outputs, feed_dict={
            inputs: data
        })


def execute_test_batch():
    reader = Reader('/home/give/Documents/dataset/ISBI2017/Slice/Train Batch1',
                    '/home/give/Documents/dataset/ISBI2017/Slice/Train Batch2',
                    batch_size=batch_size)
    inputs = tf.keras.Input([512, 512, 1], batch_size=None, name='x-input', dtype=tf.float32)
    outputs = unet_model(inputs)
    test_data, _ = reader.test_generator.next()
    pred = test_batch(inputs, outputs, test_data)
    pred_img = []
    for i in range(batch_size):
        print np.shape(pred[i])
        pred_img.append(tf.keras.preprocessing.image.array_to_img(np.expand_dims(np.argmax(pred[i], axis=2), axis=2)))
        pred_img[i].show()


if __name__ == '__main__':
    test_one_scan('/home/give/PycharmProjects/unet/tf_version/test_data/1/volume-0.mhd',
                  '/home/give/PycharmProjects/unet/tf_version/test_data/1/pred', 'mhd')
    # execute_test_batch()