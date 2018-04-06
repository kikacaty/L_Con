import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    input           = tf.placeholder(tf.float32, shape = (None, 512, 512, 8), name = 'input')
    conv1_1_pad     = tf.pad(input, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv1_1         = convolution(conv1_1_pad, group=1, strides=[2, 2], padding='VALID', name='conv1_1')
    relu1_1         = tf.nn.relu(conv1_1, name = 'relu1_1')
    conv1_2_pad     = tf.pad(relu1_1, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv1_2         = convolution(conv1_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv1_2')
    relu1_2         = tf.nn.relu(conv1_2, name = 'relu1_2')
    conv2_1_pad     = tf.pad(relu1_2, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv2_1         = convolution(conv2_1_pad, group=1, strides=[2, 2], padding='VALID', name='conv2_1')
    conv2_1_relu    = tf.nn.relu(conv2_1, name = 'conv2_1_relu')
    conv2_2_pad     = tf.pad(conv2_1_relu, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv2_2         = convolution(conv2_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv2_2')
    conv2_2_relu    = tf.nn.relu(conv2_2, name = 'conv2_2_relu')
    conv3_1_pad     = tf.pad(conv2_2_relu, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv3_1         = convolution(conv3_1_pad, group=1, strides=[2, 2], padding='VALID', name='conv3_1')
    conv3_1_relu    = tf.nn.relu(conv3_1, name = 'conv3_1_relu')
    conv4_1_pad     = tf.pad(conv3_1_relu, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv4_1         = convolution(conv4_1_pad, group=1, strides=[2, 2], padding='VALID', name='conv4_1')
    conv4_1_relu    = tf.nn.relu(conv4_1, name = 'conv4_1_relu')
    conv5_1_pad     = tf.pad(conv4_1_relu, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv5_1         = convolution(conv5_1_pad, group=1, strides=[2, 2], padding='VALID', name='conv5_1')
    conv5_1_relu    = tf.nn.relu(conv5_1, name = 'conv5_1_relu')
    refine5_deconv_relu = tf.nn.relu(refine5_deconv, name = 'refine5_deconv_relu')
    refine4_concat  = tf.concat([conv4_1_relu, refine5_deconv_relu], 3, name = 'refine4_concat')
    refine4_conv_3x3_2_pad = tf.pad(refine4_concat, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    refine4_conv_3x3_2 = convolution(refine4_conv_3x3_2_pad, group=1, strides=[1, 1], padding='VALID', name='refine4_conv_3x3_2')
    refine4_conv_3x3_2_relu = tf.nn.relu(refine4_conv_3x3_2, name = 'refine4_conv_3x3_2_relu')
    refine4_deconv_relu = tf.nn.relu(refine4_deconv, name = 'refine4_deconv_relu')
    refine3_concat  = tf.concat([conv3_1_relu, refine4_deconv_relu], 3, name = 'refine3_concat')
    refine3_conv_3x3_2_pad = tf.pad(refine3_concat, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    refine3_conv_3x3_2 = convolution(refine3_conv_3x3_2_pad, group=1, strides=[1, 1], padding='VALID', name='refine3_conv_3x3_2')
    refine3_conv_3x3_2_relu = tf.nn.relu(refine3_conv_3x3_2, name = 'refine3_conv_3x3_2_relu')
    refine3_deconv_relu = tf.nn.relu(refine3_deconv, name = 'refine3_deconv_relu')
    refine2_concat  = tf.concat([conv2_2_relu, refine3_deconv_relu], 3, name = 'refine2_concat')
    refine2_conv_3x3_2_pad = tf.pad(refine2_concat, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    refine2_conv_3x3_2 = convolution(refine2_conv_3x3_2_pad, group=1, strides=[1, 1], padding='VALID', name='refine2_conv_3x3_2')
    refine2_conv_3x3_2_relu = tf.nn.relu(refine2_conv_3x3_2, name = 'refine2_conv_3x3_2_relu')
    refine2_deconv_relu = tf.nn.relu(refine2_deconv, name = 'refine2_deconv_relu')
    refine1_concat  = tf.concat([relu1_2, refine2_deconv_relu], 3, name = 'refine1_concat')
    refine1_conv_3x3_2_pad = tf.pad(refine1_concat, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    refine1_conv_3x3_2 = convolution(refine1_conv_3x3_2_pad, group=1, strides=[1, 1], padding='VALID', name='refine1_conv_3x3_2')
    refine1_conv_3x3_2_relu = tf.nn.relu(refine1_conv_3x3_2, name = 'refine1_conv_3x3_2_relu')
    return input, predict


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer
