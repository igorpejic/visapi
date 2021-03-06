import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras import layers
from keras.optimizers import *
from keras.regularizers import l2
from alpha.binpack.keras.utils import residual_block, resnet_layer

class KerasBinpackNNet():
    def __init__(self, game, args, predict_move_index=True, scalar_tiles=False):
        # game params
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # s: batch_size x board_x x board_y
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.channels))

        x = Reshape((self.board_x, self.board_y, -1))(self.input_boards)

        # https://keras.io/examples/cifar10_resnet/
        depth = 38
        num_res_blocks = int((depth - 2) / 6)
        num_filters = self.channels 
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
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
