# -*- coding: utf-8 -*-
import numpy as np
import pickle
from ppdai_utils import *

# read train pair
data = []
import csv
trainfile = '/files/faust/COMPETITION/ppdai/train.csv'
with open(trainfile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append((row['q1'], row['q2'], row['label']))
        if not row['q1'] == row['q2']:
            data.append((row['q2'], row['q1'], row['label']))
print(len(data))


indim = len(wdict) + 1

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
model = Sequential()
model.add(Dense(units=1024, activation="tanh", input_dim=indim))
model.add(Dense(units=256, activation="tanh"))
model.add(Dense(units=1024, activation="tanh"))
model.add(Dense(units=indim))
model.add(Activation("sigmoid"))
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


## split train/test
import random
random.shuffle(data)
traindata = data[:int(len(data)*0.9)]
testdata  = data[int(len(data)*0.9):]
print(len(traindata))
print(len(testdata))

## filter unmatch pairs
traindata_pos = [q for q in traindata if q[2] == '1']
testdata_pos = [q for q in testdata if q[2] == '1']
print(len(traindata_pos))

## add copy pair
qidlist = list(questions.keys())
qid_sample = random.sample(qidlist, len(traindata_pos))
traindata_copy = [(qid, qid, 1) for qid in qid_sample]
traindata_pos.extend(traindata_copy)
random.shuffle(traindata_pos)
print(len(traindata_pos))

## generate vector
train_x = np.array([getvector_with_id(q[0]) for q in traindata_copy])
train_y = np.array([getvector_with_id(q[1]) for q in traindata_copy])
test_x = np.array([getvector_with_id(q[0]) for q in testdata_pos])
test_y = np.array([getvector_with_id(q[1]) for q in testdata_pos])


EPOCH = 20
BATCH_SIZE = 128
model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE)
score = model.evaluate(test_x, test_y, batch_size=64)
print(score)


model.save('ppdai_autoencoder.model')

