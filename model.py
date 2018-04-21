from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Embedding
from keras.layers import Merge


def createNetwork(input_size, hidden_size, choose_size, choose_extract_size, layer1_size, layer2_size, learning_rate = 0.001):
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(hidden_size, input_dim=input_size, return_sequences=True, activation='sigmoid'))

    model_choose = Sequential()
    model_choose.add(Dense(choose_extract_size, input_shape=(None, choose_size)))

    merged = Merge([model_LSTM, model_choose], mode='concat')
    model_value = Sequential()
    model_value.add(merged)
    model_value.add(Dense(layer1_size))
    #model_value.add(Dense(layer2_size))
    model_value.add(Dense(1))
    myadam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_value.compile(loss='mean_squared_error', optimizer=myadam)
    return model_value
#createNetwork(10,10,10,10,10,10,0.001)