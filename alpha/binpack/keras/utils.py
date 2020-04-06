import argparse
import tensorflow as tf
if hasattr(tf, 'keras'):
    from tensorflow.keras.models import *
    from tensorflow.keras.layers import *
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.backend import name_scope
else:
    from keras.models import *
    from keras.layers import *
    from keras import layers
    from keras.optimizers import *
    from keras.regularizers import l2

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False, kernel_size=(3,3), name=None):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=kernel_size, strides=_strides, padding='same', name=name + 'conv2d_1')(y)
    y = layers.BatchNormalization(name=name + 'batch_normalization_1')(y)
    y = layers.LeakyReLU(name=name + 'leaky_re_lu_1')(y)

    y = layers.Conv2D(nb_channels, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name + 'conv2d_2')(y)
    y = layers.BatchNormalization(name=name + 'batch_normalization_2')(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', name=name + 'conv2d_3')(shortcut)
        shortcut = layers.BatchNormalization(name=name + 'batch_normalization_3')(shortcut)

    y = layers.add([shortcut, y], name=name + 'add')
    y = layers.LeakyReLU(name=name + 'leaky_re_lu_2')(y)

    return y

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x
