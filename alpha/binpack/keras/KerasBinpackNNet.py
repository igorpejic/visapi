import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras import layers
from keras.optimizers import *

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
        x_image = Reshape((self.board_x, self.board_y, -1))(self.input_boards)

        # batch_size  x board_x x board_y x num_channels
        # h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.channels, 3, padding='same', use_bias=False)(x_image))) 
        h_conv1 = self.residual_block(x_image, self.channels)

        # batch_size  x board_x x board_y x num_channels
        # h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.channels, 3, padding='same', use_bias=False)(h_conv1))) 
        h_conv2 = self.residual_block(h_conv1, self.channels)
        h_conv3 = self.residual_block(h_conv2, self.channels)
        h_conv4 = self.residual_block(h_conv3, self.channels)
        h_conv5 = self.residual_block(h_conv4, self.channels)
        h_conv6 = self.residual_block(h_conv5, self.channels)
        h_conv7 = self.residual_block(h_conv6, self.channels)
        h_conv8 = self.residual_block(h_conv7, self.channels)
        h_conv2 = self.residual_block(h_conv8, self.channels)

        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.channels, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024

        if predict_move_index:
            # channels - 1 for state
            self.pi = Dense(self.channels - 1, activation='softmax', name='pi')(s_fc2)
            # self.pi = Dense(self.channels - 1, activation='sigmoid', name='pi')(s_fc2)
        else:
            self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        # 2 losses
        # self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        # self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        if predict_move_index:
            self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(args.lr), metrics=['accuracy'])
            # self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr), metrics=['accuracy'])
        else:
            self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr))

    def residual_block(self, y, nb_channels, _strides=(1, 1), _project_shortcut=True):
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
