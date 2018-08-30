# -*- coding: utf-8 -*-
import numpy as np


def readdata(paths, mode='train'):
    data = []
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line = line.split('\t')
                    q1 = line[1]
                    q2 = line[2]
                    if mode == 'train':
                        lb = line[-1]
                        data.append((q1, q2, lb))
                    else:
                        index = line[0]
                        data.append((index, q1, q2))
    return data


def sent2int(sent, cdict, maxlen=30):
    result = [0]*maxlen
    for i in range(min(maxlen, len(sent))):
        if sent[i] in cdict:
            result[i] = cdict[sent[i]]
        else:
            result[i] = cdict['OOV']
    result = np.array(result)
    return result


def prepare(rawdata, cdict, maxlen=30, mode='train'):
    result = []
    for smp in rawdata:
        if mode == 'train':
            q1_vec = sent2int(smp[0], cdict, maxlen)
            q2_vec = sent2int(smp[1], cdict, maxlen)
            label = int(smp[-1])
            result.append((q1_vec, q2_vec, label))
        else:
            index = smp[0]
            q1_vec = sent2int(smp[1], cdict, maxlen)
            q2_vec = sent2int(smp[2], cdict, maxlen)
            result.append((index, q1_vec, q2_vec))
    return result


def geninput(data, mode='train'):
    if mode == 'train':
        q0 = np.array([p[0] for p in data])
        q1 = np.array([p[1] for p in data])
        y = np.array([p[-1] for p in data])
        return q0, q1, y
    else:
        index = [p[0] for p in data]
        q0 = np.array([p[1] for p in data])
        q1 = np.array([p[2] for p in data])
        return index, q0, q1
