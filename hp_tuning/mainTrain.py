#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:18:42 2020

@author: kodewill
"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
import nni
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import roc_curve, precision_recall_curve


def true_acc(model, X_test, y_test):
    # predict probabilities
    yhat = model.predict_proba(X_test)
    # calculate pr curves
    precision, recall, thresholdsPR = precision_recall_curve(y_test, yhat)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ixPR = sorted(range(len(precision)), key=lambda i: precision[i], reverse=True)[:6]
    ixPR = list(fscore).index(max(fscore[ixPR]))
    return thresholdsPR[ixPR]    
    
def load_dataset(tests_size):
    X = np.load("./X.npy")
    y = np.load('./y.npy')
    total = np.shape(X)[0]
    input_shape = np.shape(X)[1]
    pos = sum(y)
    neg = total - pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=tests_size, random_state=0)
    return x_train, y_train, x_test, y_test, class_weight, input_shape


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def EmbeddingNN(tests_size, num_units, dropout, dropoutOne, lr, actOne, actTwo, vocab_size, embedding_dim, lossF):
    X_train, y_train, X_test, y_test, class_weights, input_shape = load_dataset(tests_size)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_units, activation=actOne),
        tf.keras.layers.Dropout(dropoutOne),
        tf.keras.layers.Dense(1, activation=actTwo)
    ])
    model.compile(loss=lossF,optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                  metrics= ['accuracy'])
    model.summary()

    return model

def train(params):
    num_units = params.get('num_units')
    dropout_rate = params.get('dropout_rate')
    lr = params.get('lr')
    activationOne = params.get('activationOne')
    activationTwo = params.get('activationTwo')
    batch_size = params.get('batch_size')
    test_size = params.get('test_size')
    dropout_rate = params.get('dropout_rate')
    dropout_rateOne = params.get('dropout_rateOne')
    vocab_size = params.get('vocab_size')
    embedding_dim = params.get('embedding_dim')
    lossF = params.get('lossF')
    model = EmbeddingNN(test_size, num_units, 
                        dropout_rate, dropout_rateOne, lr, activationOne, activationTwo, 
                        vocab_size, embedding_dim, lossF)
    X_train, y_train, X_test, y_test, class_weights, input_shape = load_dataset(test_size)
    
    #Model fit    
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              class_weight=class_weights, verbose=2, batch_size=int(batch_size))
    threshold = true_acc(model, X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred = np.array([1 if row > threshold else 0 for row in y_pred])
    y_test = np.array(y_test)
    acc = np.mean(y_pred == y_test)
    # print(acc)
    nni.report_final_result(acc)
    
if __name__ == '__main__':
    params = {
    'num_units': 32,
    'dropout_rate': 0.3,
    'dropout_rateOne': 0.3,
    'lr': 0.003,
    'activationOne': 'relu',
    'activationTwo': 'sigmoid',
    'batch_size': 30,
    'test_size': 0.28,
    'vocab_size': 500, 
    'embedding_dim': 64,
    'lossF': 'mse'
    }
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    
    train(params)    
