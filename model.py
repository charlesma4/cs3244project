import sys
from collections import deque

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)


class Model:
    def __init__(self, model_name, num_frames=48, num_features=4, saved_model=None, dropout=0.2, image_shape=None):

        # Set defaults.
        self.num_frames = num_frames
        self.load_model = load_model
        self.saved_model = saved_model
        self.feature_queue = deque()
        self.dropout = dropout

        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model_name == 'lstm':
            print("Loading LSTM model...")
            self.input_shape = (num_frames, num_features)
            self.model = self.lstm()
        elif model_name == 'lrcn':
            print("Loading LRCN model...")
            self.input_shape = (num_frames, image_shape[0], image_shape[1])
            self.model = self.lrcn()

        # Now compile the network.
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def lstm(self):
        model = Sequential()
        model.add(LSTM(self.input_shape[1], return_sequences=False,
                       input_shape=self.input_shape, dropout=self.dropout))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py
        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556
        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model