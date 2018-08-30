# -*- coding: utf-8 -*-
from ppdai_utils import *
import pickle
from keras.models import load_model
from myloss import focal_loss

cdict, emat = embedding('/files/faust/COMPETITION/ppdai/char_embed.txt')
questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions.pkl', 'rb'))
testpair = readtest()


MAX_SEQUENCE_LENGTH = 60
testpair_idx = []
for q0, q1 in testpair:
    q0_chars = questions[q0]['cchars']
    q1_chars = questions[q1]['cchars']
    q0_idx = sent2idx(q0_chars, cdict, MAX_SEQUENCE_LENGTH)
    q1_idx = sent2idx(q1_chars, cdict, MAX_SEQUENCE_LENGTH)
    testpair_idx.append((q0_idx, q1_idx))
print(len(testpair_idx))

test_q1 = np.array([p[0] for p in testpair_idx])
test_q2 = np.array([p[1] for p in testpair_idx])


###############################################
# test
###############################################
model = load_model('ppdai_cnn.model')
pred_y = model.predict([test_q1, test_q2], batch_size=64)
print(pred_y.shape)
pred_y = np.reshape(pred_y, (len(pred_y),))
make_submission(pred_y, 'submission.csv')

