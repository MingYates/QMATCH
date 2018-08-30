# -*- coding: utf-8 -*-
import pickle
import numpy as np
import csv

# questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions_ngram_extend.pkl', 'rb'))
questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions.pkl', 'rb'))

word_count = {}
for _, wc in questions.items():
    for word in wc['cwords']:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
print(len(word_count))

word_count_filt = {}
for w, f in word_count.items():
    if f > 1:
        word_count_filt[w] = f
print(len(word_count_filt))


wdict = [w for w,_ in word_count_filt.items()]
wdict = dict((k,v) for v,k in enumerate(wdict))
# print(wdict)


def bag_of_words(sentence):
    vector = np.zeros((len(wdict) + 1,), dtype=np.int32)
    for w in sentence:
        if w in wdict:
            vector[wdict[w]] = 1
        else:
            vector[-1] = 1
    return vector


def getvector_with_id(qid):
    words = questions[qid]['cwords']
    vector =bag_of_words(words)
    return vector


def make_submission(predict_prob, path):
    with open(path, 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()


def embedding(path):
    cdict = []
    ematrix = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            piece = line.strip().split(' ')
            ic = piece[0]
            iv = np.array(piece[1:])
            cdict.append(ic)
            ematrix.append(iv)
    cdict = dict((c, i) for i, c in enumerate(cdict))
    # OOV
    cdict['OOV'] = len(cdict)
    # print(ematrix[0].shape)
    ematrix.append(np.zeros((300,), dtype=np.float32))
    ematrix = np.array(ematrix)
    return cdict, ematrix

#
# def gentable(qidx1, qidx2, embed_mat):
#


def sent2idx(sent, cdict, maxlen=None):
    result = []
    for c in sent:
        if c in cdict:
            result.append(cdict[c])
        else:
            result.append(cdict['OOV'])

    if maxlen is not None:
        pad = [cdict['OOV']] * maxlen
        result.extend(pad)
        result = result[:maxlen]

    return result


def readtrain_extend():
    traindata = []
    trainfile = '/files/faust/COMPETITION/ppdai/train.csv'
    with open(trainfile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            traindata.append((row['q1'], row['q2'], row['label']))
            if not row['q1'] == row['q2']:
                traindata.append((row['q2'], row['q1'], row['label']))
    return traindata


def readtrain():
    traindata = []
    trainfile = '/files/faust/COMPETITION/ppdai/train.csv'
    with open(trainfile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            traindata.append((row['q1'], row['q2'], row['label']))
    return traindata


def readtest():
    testdata = []
    testfile = '/files/faust/COMPETITION/ppdai/test.csv'
    with open(testfile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            testdata.append((row['q1'], row['q2']))
    return testdata
