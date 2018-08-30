# -*- coding: utf-8 -*-
from ppdai_utils import *
import pickle
import csv

## collect all qid
traindata = []
qidlist_train = []
trainfile = '/files/faust/COMPETITION/ppdai/train.csv'
with open(trainfile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        qidlist_train.append(row['q1'])
        qidlist_train.append(row['q2'])
        traindata.append((row['q1'], row['q2'], row['label']))
        if not row['q1'] == row['q2']:
            traindata.append((row['q2'], row['q1'], row['label']))
qidlist_train = set(qidlist_train)
print(len(qidlist_train))


## get all vector
from keras.models import load_model
train_bowvec = [getvector_with_id(qid) for qid in qidlist_train]
train_bowvec = np.array(train_bowvec)
print(train_bowvec.shape)

# model = load_model('ppdai_autoencoder.model')
model = load_model('ppdai_mlc.model')
train_rep = model.predict(train_bowvec, batch_size=128, verbose=0)
trainvec_dict = dict((qid, vec) for qid, vec in zip(qidlist_train, train_rep))


def trans_pair(qid_0, qid_1):
    vec_0 = trainvec_dict[qid_0]
    vec_1 = trainvec_dict[qid_1]
    sub_abs = np.abs(vec_0 - vec_1)
    return sub_abs
    # return np.concatenate([vec_0, sub_abs])
traindata_new = [(trans_pair(pair[0], pair[1]), pair[2]) for pair in traindata]
print(len(traindata_new))

import random
random.shuffle(traindata_new)
count = len(traindata_new)

## model
# indim = (len(wdict) + 1) * 2
indim = len(wdict) + 1

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers

model = Sequential()
model.add(Dense(units=2048, activation='tanh', input_dim=indim))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='tanh'))
model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])


traindata = traindata_new[:int(count*0.9)]
testdata  = traindata_new[int(count*0.9):]
print(len(traindata))
print(len(testdata))
print(traindata[0])

train_x = np.array([s[0] for s in traindata])
train_y = np.array([int(s[1]) for s in traindata])
test_x = np.array([s[0] for s in testdata])
test_y = np.array([int(s[1]) for s in testdata])
print(train_x.shape)
print(train_y.shape)


EPOCH = 30
BATCH_SIZE = 128
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=EPOCH, batch_size=BATCH_SIZE)
model.save('ppdai_multilayer.model')


# train_x = np.array([s[0] for s in traindata_new])
# train_y = np.array([int(s[1]) for s in traindata_new])
#
# EPOCH = 25
# BATCH_SIZE = 128
# model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE)
# model.save('ppdai_multilayer.model')

