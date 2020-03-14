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

ORIENTATIONS = 2
class ScalarKerasBinpackNNet():
    def __init__(self, game, args, predict_move_index=True, scalar_tiles=False, predict_v=False, individual_tiles=False):
        # game params
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.individual_tiles = individual_tiles

        # s: batch_size x board_x x board_y
        self.input_board = Input(shape=(self.board_x, self.board_y))

        if self.individual_tiles:
            self.input_tiles = [Input(shape=(1, ORIENTATIONS)) for x in range(self.channels-1)]
        else:
            self.input_tiles = Input(shape=(self.channels - 1, ORIENTATIONS))

        y = Concatenate(name='concatenation')([self.y_state(self.input_board), self.y_tiles(self.input_tiles)])

        if predict_move_index:
            # channels - 1 for state
            self.pi = Dense(self.channels - 1, activation='softmax', name='pi')(y)
            # self.pi = Dense(self.channels - 1, activation='sigmoid', name='pi')(y)
        else:
            self.pi = Dense(self.action_size, activation='softmax', name='pi')(y)   # batch_size x self.action_size

        self.v = Dense(1, activation='tanh', name='v')(y)                    # batch_size x 1

        # 2 losses
        if predict_v:
            if predict_move_index:
                self.model = Model(inputs=[self.input_board, self.input_tiles], outputs=[self.pi, self.v])
                self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr), metrics=['accuracy'])
        else:
            if self.individual_tiles:
                self.model = Model(inputs=[self.input_board, *self.input_tiles], outputs=[self.pi])
            else:
                self.model = Model(inputs=[self.input_board, self.input_tiles], outputs=[self.pi])
            if predict_move_index:
                self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(args.lr), metrics=['categorical_accuracy'])
                # self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr), metrics=['categorical_accuracy'])
            else:
                self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr))

    def y_tiles(self, x):
        # x = Reshape((self.channels -1, ORIENTATIONS, 1))(x)
        # # y_tiles = Conv2D(5, kernel_size=3, strides=(1, 2), padding='same')(y_tiles)

        # num_res_blocks = 3
        # num_filters = 3
        # for stack in range(3):
        #     for res_block in range(num_res_blocks):
        #         strides = 1
        #         if stack > 0 and res_block == 0:  # first layer but not first stack
        #             strides = 2  # downsample
        #         y = resnet_layer(inputs=x,
        #                          num_filters=num_filters,
        #                          strides=strides)
        #         y = resnet_layer(inputs=y,
        #                          num_filters=num_filters,
        #                          activation=None)
        #         if stack > 0 and res_block == 0:  # first layer but not first stack
        #             # linear projection residual shortcut connection to match
        #             # changed dims
        #             x = resnet_layer(inputs=x,
        #                              num_filters=num_filters,
        #                              kernel_size=1,
        #                              strides=strides,
        #                              activation=None,
        #                              batch_normalization=False)
        #         x = layers.add([x, y])
        #         x = Activation('relu')(x)
        #     num_filters *= 2

        # x = MaxPooling2D(pool_size=3)(x)
        if self.individual_tiles:
            l = []
            for i in range(self.channels - 1):
                l.append(Dense(10, activation='relu')(x[1]))
            x = Concatenate()(l)
            x = Dense(512, activation='relu')(x)
        else:
            x = Dense(1012, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
        y = Flatten()(x)
        return y


    def y_state(self, x):


        # x = Reshape((self.board_x, self.board_y, 1))(x)
        # # https://keras.io/examples/cifar10_resnet/
        # num_res_blocks = 12
        # num_filters = 3
        # # Instantiate the stack of residual units
        # for stack in range(3):
        #     for res_block in range(num_res_blocks):
        #         strides = 1
        #         if stack > 0 and res_block == 0:  # first layer but not first stack
        #             strides = 2  # downsample
        #         y = resnet_layer(inputs=x,
        #                          num_filters=num_filters,
        #                          strides=strides)
        #         y = resnet_layer(inputs=y,
        #                          num_filters=num_filters,
        #                          activation=None)
        #         if stack > 0 and res_block == 0:  # first layer but not first stack
        #             # linear projection residual shortcut connection to match
        #             # changed dims
        #             x = resnet_layer(inputs=x,
        #                              num_filters=num_filters,
        #                              kernel_size=1,
        #                              strides=strides,
        #                              activation=None,
        #                              batch_normalization=False)
        #         x = layers.add([x, y])
        #         x = Activation('relu')(x)
        #     num_filters *= 2

        # x = AveragePooling2D(pool_size=3)(x)
        x = Dense(1012, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        y_state = Flatten()(x)
        return y_state
