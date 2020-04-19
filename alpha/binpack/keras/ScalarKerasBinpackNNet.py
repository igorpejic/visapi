import sys
sys.path.append('..')
from utils import *

import argparse
import tensorflow as tf
if hasattr(tf, 'keras'):
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import *
    from tensorflow.keras import metrics
    from tensorflow.keras.layers import *
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.regularizers import l2
else:
    import keras.backend as K
    from keras.models import *
    from keras import metrics
    from keras.layers import *
    from keras import layers
    from keras.optimizers import *
    from keras.regularizers import l2

from alpha.binpack.keras.utils import residual_block, resnet_layer

def custom_accuracy(y_true, y_pred):
    '''
    Accuracy which chooses the highest predicted probability and checks if that
    probability is a solution in y_pred.
    '''
    # y_true = K.cast(y_true, K.floatx())
    # y_pred = K.cast(y_pred, K.floatx())
    y_pred_max = K.argmax(y_pred, axis=-1)
    num_examples = K.cast(tf.shape(y_pred)[0], y_pred_max.dtype)
    idx = tf.stack([tf.range(num_examples), y_pred_max], axis=-1)
    res = tf.gather_nd(y_true, idx)
    return res

def true_positives(y_true, y_pred):
    threshold = 0.5
    y_pred_s = K.cast(K.greater(y_pred, 0.5), K.floatx())
    correct_pred = y_true * y_pred_s
    return K.sum(correct_pred, axis=-1) / K.sum(y_true, axis=-1)

def false_positives(y_true, y_pred):
    threshold = 0.5
    # y_pred = K.print_tensor(y_pred, message='y_pred')
    y_pred_s = K.cast(K.greater(y_pred, 0.5), K.floatx())
    predicted_as_true = y_pred_s
    predicted_as_true_but_not_true = y_pred_s * (1 - y_true)
    # print(y_pred, K.eval(y_pred))

    # y_true_size = K.cast(K.shape(y_true), K.floatx())
    # y_true_size = K.print_tensor(y_true_size, message='y_true size')
    
    return K.sum(predicted_as_true_but_not_true, axis=-1) / K.sum(y_true, axis=-1)


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=-1) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)

    return binary_focal_loss_fixed

def weighted_cross_entropy(y_true, y_pred):

    beta = 1 # 4.5
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    if hasattr(tf, 'log'):
        y_pred = tf.log(y_pred / (1 - y_pred))
    else:
        y_pred = tf.math.log(y_pred / (1 - y_pred))
    if tf.__version__ == '2.1.0':
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)
    else:
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    return tf.reduce_mean(loss)
    

ORIENTATIONS = 2
class ScalarKerasBinpackNNet():
    def __init__(self, game, args, predict_move_index=True, scalar_tiles=False, predict_v=False, individual_tiles=False):
        # game params
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.individual_tiles = individual_tiles

        # s: batch_size x board_x x board_y
        self.input_board = Input((None, None, 1), name='board_input')

        if self.individual_tiles:
            self.input_tiles = [Input(shape=(1, ORIENTATIONS), name='tiles_input') for x in range(self.channels-1)]
        else:
            self.input_tiles = Input(shape=(self.channels - 1, ORIENTATIONS), name='tiles_input')

        y = Concatenate(name='concatenation')(
            [self.y_state(self.input_board), self.y_tiles(self.input_tiles)])

        if predict_move_index:
            # channels - 1 for state
            # self.pi = Dense(self.channels - 1, activation='softmax', name='pi')(y)
            self.pi = Dense(self.channels - 1, activation='sigmoid', name='pi')(y)
        else:
            # TODO: uncomment for MCTS
            self.pi = Dense(self.action_size, activation='softmax', name='pi')(y)   # batch_size x self.action_size

        # TODO: uncomment for MCTS
        if predict_v:
            self.v = Dense(1, activation='tanh', name='v')(y)                    # batch_size x 1

        # 2 losses
        if predict_v:
            if predict_move_index:
                self.model = Model(inputs=[self.input_board, self.input_tiles], outputs=[self.pi, self.v])
                self.model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer=Adam(args.lr), metrics=['binary_accuracy', custom_accuracy])
        else:
            if self.individual_tiles:
                self.model = Model(inputs=[self.input_board, *self.input_tiles], outputs=[self.pi])
            else:
                self.model = Model(inputs=[self.input_board, self.input_tiles], outputs=[self.pi])
            if predict_move_index:
                # the best
                self.model.compile(loss=[weighted_cross_entropy], optimizer=Adam(args.lr), metrics=['binary_accuracy', true_positives, false_positives, custom_accuracy])
                # self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr), metrics=['binary_accuracy', true_positives, false_positives, custom_accuracy])
                # self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr), metrics=['categorical_accuracy'])
            else:
                self.model.compile(loss=['binary_crossentropy'], optimizer=Adam(args.lr))

    def y_tiles(self, x):
        x = Reshape((self.channels -1, ORIENTATIONS, 1))(x)
        # y_tiles = Conv2D(5, kernel_size=3, strides=(1, 2), padding='same')(y_tiles)
        from tensorflow.keras.backend import name_scope
        with name_scope('residual_1') as scope:
            x = residual_block(x, self.channels, kernel_size=(14, 2), _project_shortcut=True, name=scope)
        with name_scope('residual_2') as scope:
            x = residual_block(x, self.channels, kernel_size=(12, 2), _project_shortcut=True, name=scope)
        with name_scope('residual_3') as scope:
            x = residual_block(x, self.channels, kernel_size=(6, 2), _project_shortcut=True, name=scope)
        with name_scope('residual_4') as scope:
            x = residual_block(x, self.channels, kernel_size=(3, 2), _project_shortcut=True, name=scope)
        with name_scope('residual_5') as scope:
            x = residual_block(x, self.channels, kernel_size=(3, 2), _project_shortcut=True, name=scope)

        #num_res_blocks = 2
        #num_filters = 6
        #for stack in range(3):
        #    for res_block in range(num_res_blocks):
        #        strides = 1
        #        if stack > 0 and res_block == 0:  # first layer but not first stack
        #            strides = 2  # downsample
        #        y = resnet_layer(inputs=x,
        #                         num_filters=num_filters,
        #                         strides=strides)
        #        y = resnet_layer(inputs=y,
        #                         num_filters=num_filters,
        #                         activation=None)
        #        if stack > 0 and res_block == 0:  # first layer but not first stack
        #            # linear projection residual shortcut connection to match
        #            # changed dims
        #            x = resnet_layer(inputs=x,
        #                             num_filters=num_filters,
        #                             kernel_size=1,
        #                             strides=strides,
        #                             activation=None,
        #                             batch_normalization=False)
        #        x = layers.add([x, y])
        #        x = Activation('relu')(x)
        #    num_filters *= 2

        # x = MaxPooling2D(pool_size=3)(x)
        if self.individual_tiles:
            l = []
            for i in range(self.channels - 1):
                l.append(Dense(10, activation='relu')(x[1]))
            x = Concatenate()(l)
            x = Dense(512, activation='relu')(x)
        else:
            # x = Dense(1, activation='relu')(x)
            # x = Dense(64, activation='relu')(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.2)(x)
            #x = Dense(212, activation='relu')(x)
        y = Flatten()(x)
        return y


    def y_state(self, x):


        #x = Reshape((self.board_x, self.board_y, 1))(x)
        ## https://keras.io/examples/cifar10_resnet/
        #num_res_blocks = 12
        #num_filters = 3
        ## Instantiate the stack of residual units
        #for stack in range(3):
        #    for res_block in range(num_res_blocks):
        #        strides = 1
        #        if stack > 0 and res_block == 0:  # first layer but not first stack
        #            strides = 2  # downsample
        #        y = resnet_layer(inputs=x,
        #                         num_filters=num_filters,
        #                         strides=strides)
        #        y = resnet_layer(inputs=y,
        #                         num_filters=num_filters,
        #                         activation=None)
        #        if stack > 0 and res_block == 0:  # first layer but not first stack
        #            # linear projection residual shortcut connection to match
        #            # changed dims
        #            x = resnet_layer(inputs=x,
        #                             num_filters=num_filters,
        #                             kernel_size=1,
        #                             strides=strides,
        #                             activation=None,
        #                             batch_normalization=False)
        #        x = layers.add([x, y])
        #        x = Activation('relu')(x)
        #    num_filters *= 2

        # x = AveragePooling2D(pool_size=3)(x)
        # x = Dense(1, activation='relu')(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dense(512, activation='relu')(x)
        #x = Dense(512, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)

        x = Conv2D(filters=20, 
             kernel_size=(15, 15), 
             padding="same", 
             activation='relu',
             )(x)
        x = Conv2D(filters=15, 
             kernel_size=(10, 10), 
             padding="same", 
             activation='relu',
             )(x)
        x = Conv2D(filters=10, 
             kernel_size=(5, 5), 
             padding="same", 
             activation='relu',
             )(x)
        x = GlobalMaxPooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        # model.add(Dense(10, activation='softmax'))
        y_state = Flatten()(x)
        return y_state
