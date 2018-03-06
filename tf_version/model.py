# -*- coding=utf-8 -*-
import tensorflow as tf

class Metrics:
    @staticmethod
    def dice_score(y_true, y_pred, axis=[1, 2], smooth=1.0):
        '''
        计算dice_score=(2*np.sum(y_true, y_pred)) / (np.sum(y_true) + np.sum(y_pred))
        :param y_true: not one hot format
        :param axis:
        :param y_pred: not one hot format
        :param smooth:
        :return:
        '''
        with tf.name_scope('dice'):
            output = tf.cast(y_true, tf.float32)
            target = tf.cast(y_pred, tf.float32)
            inse = tf.reduce_sum(output * target, axis=axis)
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
            dice = (2. * inse + smooth) / (l + r + smooth)
            dice = tf.reduce_mean(dice)
        return dice
    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1.0):
        return 1.0 - Metrics.dice_score(y_true, y_pred, smooth=smooth)

def get_weights(name, shape, initializer):
    return tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initializer)


def do_conv(x, layer_name, kernel_size, filter_size, stride_size, padding, activation_method=None):
    with tf.variable_scope(layer_name):
        in_shape = x.get_shape().as_list()
        weights = get_weights('weights', shape=[kernel_size[0], kernel_size[1], in_shape[-1], filter_size],
                              initializer=tf.contrib.layers.xavier_initializer())
        bias = get_weights('bias', shape=[filter_size], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.conv2d(x, filter=weights,
                              strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        output = tf.nn.bias_add(output, bias)
        if activation_method is not None:
            output = activation_method(output)
        return output


def do_maxpooling(x, layer_name, kernel_size, stride_size, padding):
    with tf.variable_scope(layer_name):
        output = tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
    return output


def do_upconv(x, layer_name, kernel_size, filter_size, output_shape, stride_size, padding, activation_method=None):
    with tf.variable_scope(layer_name):
        in_shape = x.get_shape().as_list()
        weights = get_weights('weights', shape=[kernel_size[0], kernel_size[1], filter_size, in_shape[-1]],
                              initializer=tf.contrib.layers.xavier_initializer())
        output = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape,
                                        strides=[1, stride_size[0], stride_size[1], 1],
                                        padding=padding)
        bias = get_weights('bias', shape=[filter_size], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.bias_add(output, bias)
        if activation_method is not None:
            output = activation_method(output)
    return output


def unet_model(input_tensor, categories_num=2):
    batch_size = tf.shape(input_tensor)[0]
    conv1_1 = do_conv(input_tensor, layer_name='level1-conv1_1', kernel_size=[3, 3], filter_size=64, stride_size=[1, 1],
                    padding='SAME', activation_method=tf.nn.relu)
    print "conv1 shape:", conv1_1.shape
    conv1_2 = do_conv(conv1_1, layer_name='level1-conv1_2', kernel_size=[3, 3], filter_size=64, stride_size=[1, 1],
                    padding='SAME', activation_method=tf.nn.relu)
    print "conv1 shape:", conv1_2.shape
    maxpooling1 = do_maxpooling(conv1_2, layer_name='level1-maxpooling1', kernel_size=[2, 2], stride_size=[2, 2],
                                padding='VALID')
    print "pooling1 shape:", maxpooling1.shape

    conv2_1 = do_conv(maxpooling1, layer_name='level2-conv2_1', kernel_size=[3, 3], filter_size=128, stride_size=[1, 1],
                    padding='SAME', activation_method=tf.nn.relu)
    conv2_2 = do_conv(conv2_1, layer_name='level2-conv2_2', kernel_size=[3, 3], filter_size=128, stride_size=[1, 1],
                    padding='SAME', activation_method=tf.nn.relu)
    maxpooling2 = do_maxpooling(conv2_2, layer_name='level2-maxpooling2', kernel_size=[2, 2], stride_size=[2, 2],
                                padding='SAME')

    conv3_1 = do_conv(maxpooling2, layer_name='level3-conv3_1', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    conv3_2 = do_conv(conv3_1, layer_name='level3-conv3_2', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    maxpooling3 = do_maxpooling(conv3_2, layer_name='level3-maxpooling3', kernel_size=[2, 2], stride_size=[2, 2],
                                padding='SAME')

    conv4_1 = do_conv(maxpooling3, layer_name='level4-conv4_1', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    conv4_2 = do_conv(conv4_1, layer_name='level4-conv4_2', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    maxpooling4 = do_maxpooling(conv4_2, layer_name='level4-maxpooling4', kernel_size=[2, 2], stride_size=[2, 2],
                                padding='SAME')

    conv5_1 = do_conv(maxpooling4, layer_name='level5-conv5_1', kernel_size=[3, 3], filter_size=1024,
                      stride_size=[1, 1], padding='SAME', activation_method=tf.nn.relu)
    conv5_2 = do_conv(conv5_1, layer_name='level5-conv5_2', kernel_size=[3, 3], filter_size=1024,
                      stride_size=[1, 1], padding='SAME', activation_method=tf.nn.relu)
    upconv4 = do_upconv(conv5_2, layer_name='level5-upconv1', kernel_size=[2, 2], filter_size=512,
                        output_shape=[batch_size, conv4_2.get_shape().as_list()[1], conv4_2.get_shape().as_list()[2],
                                      conv4_2.get_shape().as_list()[3]], stride_size=[2, 2], padding='SAME',
                        activation_method=tf.nn.relu)

    merge4 = tf.concat([conv4_2, upconv4], axis=3)
    conv4_3 = do_conv(merge4, layer_name='level4-conv4_3', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    conv4_4 = do_conv(conv4_3, layer_name='level4-conv4_4', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    upconv3 = do_upconv(conv4_4, layer_name='level4-upconv2', kernel_size=[2, 2], filter_size=256,
                        output_shape=[batch_size, conv3_2.get_shape().as_list()[1], conv3_2.get_shape().as_list()[2],
                                      conv3_2.get_shape().as_list()[3]], stride_size=[2, 2], padding='SAME',
                        activation_method=tf.nn.relu)

    merge3 = tf.concat([conv3_2, upconv3], axis=3)
    conv3_3 = do_conv(merge3, layer_name='level3-conv3_3', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    conv3_4 = do_conv(conv3_3, layer_name='level3-conv3_4', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    upconv2 = do_upconv(conv3_4, layer_name='level3-upconv3', kernel_size=[2, 2], filter_size=128,
                        output_shape=[batch_size, conv2_2.get_shape().as_list()[1], conv2_2.get_shape().as_list()[2],
                                      conv2_2.get_shape().as_list()[3]], stride_size=[2, 2], padding='SAME',
                        activation_method=tf.nn.relu)

    merge2 = tf.concat([conv2_2, upconv2], axis=3)
    conv2_3 = do_conv(merge2, layer_name='level2-conv2_3', kernel_size=[3, 3], filter_size=128, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    conv2_4 = do_conv(conv2_3, layer_name='level2-conv2_4', kernel_size=[3, 3], filter_size=128, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    upconv1 = do_upconv(conv2_4, layer_name='level2-upconv4', kernel_size=[2, 2], filter_size=64,
                        output_shape=[batch_size, conv1_2.get_shape().as_list()[1], conv1_2.get_shape().as_list()[2],
                                      conv1_2.get_shape().as_list()[3]], stride_size=[2, 2], padding='SAME',
                        activation_method=tf.nn.relu)

    merge1 = tf.concat([conv1_2, upconv1], axis=3)
    conv1_3 = do_conv(merge1, layer_name='level1-conv1_3', kernel_size=[3, 3], filter_size=64, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)
    conv1_4 = do_conv(conv1_3, layer_name='level1-conv1_4', kernel_size=[3, 3], filter_size=64, stride_size=[1, 1],
                      padding='SAME', activation_method=tf.nn.relu)

    sg_predict = do_conv(conv1_4, layer_name='level1-conv1_5', kernel_size=[1, 1], filter_size=categories_num, stride_size=[1, 1],
                         padding='SAME', activation_method=tf.nn.relu)
    # sg_predict = do_conv(sg_predict, layer_name='level1-conv1_6', kernel_size=[1, 1], filter_size=1, stride_size=[1, 1],
    #                      padding='SAME', activation_method=tf.nn.sigmoid)
    return sg_predict

if __name__ == '__main__':
    print do_conv(tf.zeros([100, 32, 32, 10]), 'conv1', kernel_size=[3, 3], filter_size=64, stride_size=[1, 1], padding='SAME')
    print do_upconv(tf.zeros([100, 32, 32, 10]), 'upconv1', kernel_size=[3, 3], filter_size=64,
                    output_shape=[100, 64, 64, 64],
                    stride_size=[2, 2], padding='SAME')
    print unet_model(tf.zeros([100, 512, 512, 1]), 2)
