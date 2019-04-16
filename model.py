from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)

def lstm(self):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(200, return_sequences=False,
                   input_shape=self.input_shape,
                   dropout=0.1))
    model.add(LSTM(200, return_sequences=False,
                   input_shape=self.input_shape,
                   dropout=0.1))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(self.nb_classes, activation='softmax'))

    return model

