from block import fusions
import sys
sys.path.append('../')
from loader import Dataset
import config as config
import numpy as np
import ipdb

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report, auc, roc_curve
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input, Add, Concatenate
from keras.optimizers import Adam, RMSprop, SGD

import tensorflow as tf


class_weights = {0: 0.3, 1: 0.7}

def get_data(split=0, location=config.csv_location, mode='valid'):
    '''
    use mode = 'test' for testing
    '''

    print('mode', mode)
    print('location', location)
    X_train, y_train, y_train2 = [], [], []
    with open(location + 'train_' + str(split) + '.csv', 'r') as curr_file:
        for row in curr_file:
            a, b, c = row.split('\t')
            X_train.append(a.strip())
            y_train.append(int(b.strip()))
            y_train2.append(int(c.strip()))

    X_test, y_test, y_test2 = [], [], []
    with open(location + mode + '_' + str(split) + '.csv', 'r') as curr_file:
        for row in curr_file:
            a, b, c = row.split('\t')
            X_test.append(a.strip())
            y_test.append(int(b.strip()))
            y_test2.append(int(c.strip()))
 
    return X_train, X_test, y_train, y_test, y_train2, y_test2

def model1():

    model = Sequential([
        # Flatten(input_shape=(1, 1024)),
        Dense(512, activation=tf.nn.relu, input_shape=(1024,)),
        Dropout(0.25),
        Dense(256, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(2, activation=tf.nn.softmax)
    ])

    return model

def model2 ():

    """ Early fusion"""

    input = Input(shape=(6292,))

    x1 = Dense(1024, activation=tf.nn.relu)(input)
    x1 = Dropout(0.25)(x1)
    x1 = Dense(1024, activation=tf.nn.relu)(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Dense(512, activation=tf.nn.relu)(x1)
    x1 = Dropout(0.25)(x1)
    x1_feat = Dense(256, activation=tf.nn.relu, name='feature')(x1)
    x1 = Dropout(0.25)(x1_feat)
    prediction = Dense(2, activation=tf.nn.softmax)(x1)

    model = Model(inputs=[input], outputs=prediction)

    return model

def model3 ():

    """ Intermediate fusion"""

    input1 = Input(shape=(1024,))
    input2 = Input(shape=(5268,))
#     input3 = Input(shape=(17,))

    x1 = Dense(512, activation=tf.nn.relu)(input1)
    x1 = Dropout(0.25)(x1)
    x1 = Dense(256, activation=tf.nn.relu)(x1)
    x1 = Dropout(0.25)(x1)

    x2 = Dense(1024, activation=tf.nn.relu)(input2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(512, activation=tf.nn.relu)(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(256, activation=tf.nn.relu)(x2)
    x2 = Dropout(0.25)(x2)

#     x3 = Dense(8, activation=tf.nn.relu)(input3)
#     x3 = Dropout(0.25)(x3)

#     x = Concatenate()([x1, x2, x3])
    x = Concatenate()([x1, x2])
    x = Dense(256, activation=tf.nn.relu)(x)
    x = Dropout(0.25)(x)
    x_feat = Dense(128, activation=tf.nn.relu, name='feature')(x)
    x = Dropout(0.25)(x_feat)
    prediction = Dense(2, activation=tf.nn.softmax)(x)

#     model = Model(inputs=[input1, input2, input3], outputs=prediction)
    model = Model(inputs=[input1, input2], outputs=prediction)


    return model

def model4 ():

    """ Late fusion"""

    input1 = Input(shape=(1024,))
    input2 = Input(shape=(5268,))
#     input3 = Input(shape=(17,))

    x1 = Dense(512, activation=tf.nn.relu)(input1)
    x1 = Dropout(0.25)(x1)
    x1 = Dense(256, activation=tf.nn.relu)(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Dense(32, activation=tf.nn.relu)(x1)
    x1 = Dropout(0.25)(x1)

    x2 = Dense(1024, activation=tf.nn.relu)(input2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(512, activation=tf.nn.relu)(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(256, activation=tf.nn.relu)(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(32, activation=tf.nn.relu)(x2)
    x2 = Dropout(0.25)(x2)

#     x3 = Dense(8, activation=tf.nn.relu)(input3)
#     x3 = Dropout(0.25)(x3)

#     x = Concatenate()([x1, x2, x3])
    x = Concatenate()([x1, x2])
    x_feat = Dense(32, activation=tf.nn.relu, name='feature')(x)
    x = Dropout(0.25)(x_feat)
    prediction = Dense(2, activation=tf.nn.softmax)(x)

#     model = Model(inputs=[input1, input2, input3], outputs=prediction)
    model = Model(inputs=[input1, input2], outputs=prediction)

    return model

def basic_densenet():
    predict_variable = 'recurrence'

    NRG = Dataset()
    X_train_list, X_test_list, y_train, y_test = get_data(predict_variable)

    X_train1 = NRG.get_densenet_features(X_train_list)
    X_test1 = NRG.get_densenet_features(X_test_list)

    scaler = MinMaxScaler()
    scaler.fit(X_train1)
    X_train1 = scaler.transform(X_train1)
    X_test1 = scaler.transform(X_test1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = model1()
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train1, y_train, epochs=200, class_weight=class_weights, verbose=0)

    test_loss, test_acc = model.evaluate(X_test1, y_test)
    print(test_loss, test_acc)

    y_pred = model.predict_classes(X_test1)

    y_test = np.argmax(y_test, axis=1)
    # y_pred = np.amax(y_pred, axis=1)

#     print(y_test)
#     print(y_pred)

#     print(y_test.shape)
#     print(y_pred.shape)

    fval = f1_score(y_pred=y_pred, y_true=y_test, average='weighted')
    print(fval)
    print(classification_report(y_true=y_test, y_pred=y_pred))

def densenet_genomics(X_train_list, X_test_list, y_train, y_test, NRG, fusion='late'):
    predict_variable = 'recurrence'

    X_train1 = np.array(NRG.get_densenet_features(X_train_list))
    X_test1 = np.array(NRG.get_densenet_features(X_test_list))

    scaler = MinMaxScaler()
    scaler.fit(X_train1)
    X_train1 = scaler.transform(X_train1)
    X_test1 = scaler.transform(X_test1)

    X_train2, gene_list = NRG.get_genomics(X_train_list)
    X_test2, gene_list = NRG.get_genomics(X_test_list)

    scaler = MinMaxScaler()
    scaler.fit(X_train2)
    X_train2 = scaler.transform(X_train2)
    X_test2 = scaler.transform(X_test2)

    X_train = [np.concatenate((X_train1[idx], X_train2[idx])) for idx in range(len(X_train1))]
    X_test = [np.concatenate((X_test1[idx], X_test2[idx])) for idx in range(len(X_test1))]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    curr_max = -10
    
#     y_preds = []
#     for _ in range(num_runs): 
    if fusion == 'early':
        model = model2()
        intermediate_output = Model(inputs=model.input, outputs=model.get_layer('feature').output)
        optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    elif fusion == 'intermediate':

        model = model3()
        intermediate_output = Model(inputs=model.input, outputs=model.get_layer('feature').output)
        # optimizer = Adam(lr=0.00001, decay=1e-6)
        optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        X_train = [X_train1, X_train2]
        X_test = [X_test1, X_test2]

    elif fusion == 'late':

        model = model4()
        intermediate_output = Model(inputs=model.input, outputs=model.get_layer('feature').output)
        optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        X_train = [X_train1, X_train2]
        X_test = [X_test1, X_test2]

    model.fit(X_train, y_train, epochs=100, class_weight=class_weights, verbose=0)
    y_pred = model.predict(X_test)
    
    del model
    return y_pred
        
