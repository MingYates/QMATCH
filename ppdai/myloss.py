# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers import Lambda, Multiply, Add


def focal_loss(y_true, y_pred):
    gamma = 2
    prop_w_0 = Lambda(lambda x: x ** gamma)(y_pred)
    prop_w_1 = Lambda(lambda x: (1.0 - x) ** gamma)(y_pred)

    reverse = Lambda(lambda x: 1 - x)
    w_0 = Multiply()([prop_w_0, reverse(y_true)])
    w_1 = Multiply()([prop_w_1, y_true])
    w = Add()([w_0, w_1])

    bce = K.binary_crossentropy(y_pred, y_true)
    floss = Multiply()([w, bce])
    return K.mean(floss)