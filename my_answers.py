import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    if window_size < len(series) - 1 and window_size > 0:
        for end_idx in range(window_size-1, len(series)-1):
            start_idx = end_idx - window_size + 1
            X.append(series[start_idx:end_idx+1])
            y.append(series[end_idx+1])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, dropout=0.2, recurrent_dropout=0.2, input_shape=(window_size, 1)))
    model.add(Dense(1, activation=None))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = set(['!', ',', '.', ':', ';', '?'])
    cleaned_text = []
    for char in text:
        is_ascii = 97 <= ord(char) <= 122
        is_blank_space = ord(char) == 32
        is_punctuation = char in punctuation
        if is_ascii or is_blank_space or is_punctuation:
            cleaned_text.append(char)
        else:
            cleaned_text.append(' ')
    return "".join(cleaned_text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    if window_size < len(text) - 1 and window_size > 0:
        for end_idx in range(window_size-1, len(text)-1, step_size):
            start_idx = end_idx - window_size + 1
            inputs.append(text[start_idx:end_idx+1])
            outputs.append(text[end_idx+1])
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, dropout=0.5, recurrent_dropout=0.5, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
