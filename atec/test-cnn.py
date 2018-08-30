# -*- coding: utf-8 -*-
from keras.models import load_model
from utils import *
import pickle
import sys

c2vpath = 'char2vec.model'
c2vfile = open(c2vpath, 'rb')
cdict = pickle.load(c2vfile)
emat = pickle.load(c2vfile)
c2vfile.close()


def test(inpath, outpath):
    MAXLEN = 30
    TESTDATA_RAW = readdata([inpath], mode='test')
    testdata = prepare(TESTDATA_RAW, cdict, maxlen=MAXLEN, mode='test')
    index, test_q1, test_q2 = geninput(testdata, 'test')

    mpath = 'model/cnn-0.83743.model'
    model = load_model(mpath)
    pred_y = model.predict([test_q1, test_q2], batch_size=64)
    pred_y = np.reshape(pred_y, (len(pred_y),))

    alpha = 0.5
    with open(outpath, 'w+') as f:
        for idx, _y in zip(index, pred_y):
            if _y > alpha:
                _y = 1
            else:
                _y = 0
            f.write('{}\t{}\n'.format(idx, _y))


if __name__ == '__main__':
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    test(inpath, outpath)

