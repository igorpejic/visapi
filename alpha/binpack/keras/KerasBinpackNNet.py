import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras import layers
from keras.optimizers import *
from keras.regularizers import l2

class KerasBinpackNNet():
    def __init__(self, game, args, predict_move_index=True):
        # game params
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        # s: batch_size x board_x x board_y
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.channels))

        # batch_size  x board_x x board_y x 1
        # x_image = Reshape((self.channels, self.board_x, self.board_y, 1))(self.input_boards)
        x = Reshape((self.board_x, self.board_y, -1))(self.input_boards)

        # https://keras.io/examples/cifar10_resnet/
        depth = 26
        num_res_blocks = int((depth - 2) / 6)
        num_filters = self.channels + 26
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=3)(x)
        y = Flatten()(x)

        if predict_move_index:
            # channels - 1 for state
            # self.pi = Dense(self.channels - 1, activation='softmax', name='pi')(y)
            self.pi = Dense(self.channels - 1, activation='sigmoid', name='pi')(y)
        else:
            self.pi = Dense(self.action_size, activation='softmax', name='pi')(y)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(y)                    # batch_size x 1

        # 2 losses
        # self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        # self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        if predict_move_index:
            self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(args.lr), metrics=['categorical_accuracy'])
            # self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr), metrics=['accuracy'])
        else:
            self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr))

    def residual_block(self, y, nb_channels, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        # down-sampling is performed with a stride of 2
        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)

        return y

    def resnet_layer(self,
                     inputs,
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
