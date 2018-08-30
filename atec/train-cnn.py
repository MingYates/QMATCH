# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv1D, TimeDistributed
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, BatchNormalization
from keras.layers import concatenate, Flatten, regularizers
from keras.layers import Add, Subtract, Multiply, Concatenate, Reshape
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu, sigmoid, tanh
from keras import backend as K
from keras import optimizers
import numpy as np
import pickle
from utils import *


c2vpath = 'char2vec.model'
c2vfile = open(c2vpath, 'rb')
cdict = pickle.load(c2vfile)
emat = pickle.load(c2vfile)
c2vfile.close()

MAXLEN = 30
VOCABLEN = len(cdict)
EMBEDDING_DIM = 200
DROPOUT = 0.5


input_q1 = Input(shape=(MAXLEN,))
input_q2 = Input(shape=(MAXLEN,))

layer_embedding_c = Embedding(VOCABLEN,
                              EMBEDDING_DIM,
                              weights=[emat],
                              input_length=MAXLEN,
                              trainable=False)

filter_size = [1, 2, 3, 5]
layer_conv_c1 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # char conv1
conv_1_pool = [MaxPooling1D(ws) for ws in [3, 5, 7]]
layer_fc_c1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # char fc1

layer_conv_c2 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # char conv2
conv_2_pool = [MaxPooling1D(ws) for ws in [7, 9, 11]]
layer_fc_c2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # char fc2
layer_conv_c3 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # char conv3

# ------ embedding -------
q1_embed_c = layer_embedding_c(input_q1)
q2_embed_c = layer_embedding_c(input_q2)

# ------ conv 1 -------
q1_embed_c = BatchNormalization()(q1_embed_c)
q2_embed_c = BatchNormalization()(q2_embed_c)
q1_embed_c = Dropout(0.2)(q1_embed_c)
q2_embed_c = Dropout(0.2)(q2_embed_c)

q1_conv_c1 = [conv(q1_embed_c) for conv in layer_conv_c1]
q1_conv_c1 = Concatenate()(q1_conv_c1)
q2_conv_c1 = [conv(q2_embed_c) for conv in layer_conv_c1]
q2_conv_c1 = Concatenate()(q2_conv_c1)

q1_pool_c1 = [Flatten()(mp(q1_conv_c1)) for mp in conv_1_pool]
q1_pool_c1 = Concatenate()(q1_pool_c1)
q2_pool_c1 = [Flatten()(mp(q2_conv_c1)) for mp in conv_1_pool]
q2_pool_c1 = Concatenate()(q2_pool_c1)

pair_sub_c1 = Subtract()([layer_fc_c1(q1_pool_c1), layer_fc_c1(q2_pool_c1)])
pair_feat_c1 = Lambda(lambda x: x ** 2)(pair_sub_c1)

# # ------ conv 2 -------
q1_conv_c1 = BatchNormalization()(q1_conv_c1)
q2_conv_c1 = BatchNormalization()(q2_conv_c1)
q1_conv_c1 = Dropout(0.5)(q1_conv_c1)
q2_conv_c1 = Dropout(0.5)(q2_conv_c1)

q1_conv_c2 = [conv(q1_conv_c1) for conv in layer_conv_c2]
q1_conv_c2 = Concatenate()(q1_conv_c2)
q2_conv_c2 = [conv(q2_conv_c1) for conv in layer_conv_c2]
q2_conv_c2 = Concatenate()(q2_conv_c2)

q1_pool_c2 = [Flatten()(mp(q1_conv_c2)) for mp in conv_2_pool]
q1_pool_c2 = Concatenate()(q1_pool_c2)
q2_pool_c2 = [Flatten()(mp(q2_conv_c2)) for mp in conv_2_pool]
q2_pool_c2 = Concatenate()(q2_pool_c2)

pair_sub_c2 = Subtract()([layer_fc_c2(q1_pool_c2), layer_fc_c2(q2_pool_c2)])
pair_feat_c2 = Lambda(lambda x: x ** 2)(pair_sub_c2)

# # ------ conv 3 -------
q1_conv_c2 = BatchNormalization()(q1_conv_c2)
q2_conv_c2 = BatchNormalization()(q2_conv_c2)
q1_conv_c2 = Dropout(0.5)(q1_conv_c2)
q2_conv_c2 = Dropout(0.5)(q2_conv_c2)

q1_conv_c3 = [conv(q1_conv_c2) for conv in layer_conv_c3]
q1_conv_c3 = Concatenate()(q1_conv_c3)
q2_conv_c3 = [conv(q2_conv_c2) for conv in layer_conv_c3]
q2_conv_c3 = Concatenate()(q2_conv_c3)

q1_pool_c3 = GlobalMaxPooling1D()(q1_conv_c3)
q2_pool_c3 = GlobalMaxPooling1D()(q2_conv_c3)

pair_sub_c3 = Subtract()([q1_pool_c3, q2_pool_c3])
pair_feat_c3 = Lambda(lambda x: x ** 2)(pair_sub_c3)

# # ------ concat all -------
pair = Concatenate()([pair_feat_c1, pair_feat_c2, pair_feat_c3])
pair = BatchNormalization()(pair)
pair = Dropout(0.5)(pair)

# ------ last hidden -------
pair_last = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1))(pair)
# pair_last = Dense(512, activation='tanh')(pair)
# pair_last = Dense(128, activation='tanh')(pair_last)

# ------ predict layer -------
pair_last = BatchNormalization()(pair_last)
pair_last = Dropout(0.5)(pair_last)
predict = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1))(pair_last)

# ------ build model -------
model = Model(inputs=[input_q1, input_q2], outputs=predict)

# ------ optimizer -------
LEARNING_RATE = 0.0025
# optimizer = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.5, nesterov=True)
# optimizer = optimizers.Adagrad(lr=LEARNING_RATE, epsilon=1e-06)
# optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.compile(loss=focal_loss,
#               optimizer=optimizer,
#               metrics=['binary_crossentropy'])
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


paths = ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']
TRAINDATA_RAW = readdata(paths)
traindata = prepare(TRAINDATA_RAW)
import random
random.shuffle(traindata)
train_split = traindata[:int(len(traindata)*0.9)]
val_split = traindata[int(len(traindata)*0.9):]

train_q1, train_q2, train_y = geninput(train_split, 'train')
val_q1, val_q2, val_y = geninput(val_split, 'train')


EPOCH = 100
BATCH_SIZE = 128
BESTSCORE = 1e-6
EARLYSTOP = 5
ES_FLAG = 0
for i in range(EPOCH):
    model.fit([train_q1, train_q2], train_y,
              validation_data=([val_q1, val_q2], val_y),
              batch_size=BATCH_SIZE)
    evalinfo = model.evaluate([val_q1, val_q2], val_y,
                              batch_size=BATCH_SIZE, verbose=1)
    print('val_loss = %f, val_c = %f' % (evalinfo[0], evalinfo[1]))
    score = evalinfo[-1]
    if score > BESTSCORE:
        BESTSCORE = score
        ES_FLAG = 0
        savepath = 'model/cnn-' + str(BESTSCORE)[:7] + '.model'
        model.save(savepath)
    else:
        ES_FLAG += 1
        if ES_FLAG > EARLYSTOP:
            if LEARNING_RATE > 1e-6:
                LEARNING_RATE *= 0.5
                K.set_value(model.optimizer.lr, LEARNING_RATE)
                print('reduce learning rate = %f' % LEARNING_RATE)
                ES_FLAG = 0
            else:
                break

