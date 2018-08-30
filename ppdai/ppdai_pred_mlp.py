# -*- coding: utf-8 -*-
from ppdai_utils import *
import pickle
import csv

## collect all qid
testdata = []
qidlist_test = []
trainfile = '/files/faust/COMPETITION/ppdai/test.csv'
with open(trainfile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        qidlist_test.append(row['q1'])
        qidlist_test.append(row['q2'])
        testdata.append((row['q1'], row['q2']))
qidlist_test = set(qidlist_test)
print(len(qidlist_test))


## get all vector
from keras.models import load_model
train_bowvec = [getvector_with_id(qid) for qid in qidlist_test]
train_bowvec = np.array(train_bowvec)
print(train_bowvec.shape)

# model = load_model('ppdai_autoencoder.model')
model = load_model('ppdai_mlc.model')
test_rep = model.predict(train_bowvec, batch_size=128, verbose=0)
testvec_dict = dict((qid, vec) for qid, vec in zip(qidlist_test, test_rep))


def trans_pair(qid_0, qid_1):
    vec_0 = testvec_dict[qid_0]
    vec_1 = testvec_dict[qid_1]
    sub_abs = np.abs(vec_0 - vec_1)
    return sub_abs
    # return np.concatenate([vec_0, sub_abs])
testdata_new = [trans_pair(pair[0], pair[1]) for pair in testdata]
test_x = np.array(testdata_new)


model = load_model('ppdai_multilayer.model')
pred_y = model.predict(test_x, batch_size=64)
print(pred_y.shape)
pred_y = np.reshape(pred_y, (len(pred_y),))
make_submission(pred_y)


