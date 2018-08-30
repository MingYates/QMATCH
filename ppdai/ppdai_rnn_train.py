# -*- coding: utf-8 -*-
from ppdai_utils import *
import pickle
from myloss import focal_loss

MAXLEN_CHAR = 60
MAXLEN_WORD = 40

cdict, emat_char = embedding('/files/faust/COMPETITION/ppdai/char_embed.txt')
wdict, emat_word = embedding('/files/faust/COMPETITION/ppdai/word_embed.txt')

questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions.pkl', 'rb'))
# trainpair = readtrain_extend()
trainpair = readtrain()

trainpair_idx = []
for q0, q1, l in trainpair:
    q0_words = questions[q0]['cwords']
    q1_words = questions[q1]['cwords']
    q0_widx = sent2idx(q0_words, wdict, MAXLEN_WORD)
    q1_widx = sent2idx(q1_words, wdict, MAXLEN_WORD)

    q0_chars = questions[q0]['cchars']
    q1_chars = questions[q1]['cchars']
    q0_cidx = sent2idx(q0_chars, cdict, MAXLEN_CHAR)
    q1_cidx = sent2idx(q1_chars, cdict, MAXLEN_CHAR)

    trainpair_idx.append((q0_widx, q0_cidx, q1_widx, q1_cidx, l))
print(len(trainpair_idx))


from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv1D, TimeDistributed, LSTM, Bidirectional
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, BatchNormalization
from keras.layers import concatenate, Flatten, regularizers
from keras.layers import Add, Subtract, Multiply, Concatenate, Reshape
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu, sigmoid, tanh
from keras import backend as K
from keras import optimizers


EMBEDDING_DIM = 300
DROPOUT = 0.5



###############################################
# definition of layers
###############################################
input_q1_w = Input(shape=(MAXLEN_WORD,))
input_q1_c = Input(shape=(MAXLEN_CHAR,))
input_q2_w = Input(shape=(MAXLEN_WORD,))
input_q2_c = Input(shape=(MAXLEN_CHAR,))

layer_embedding_w = Embedding(len(wdict),
                            EMBEDDING_DIM,
                            weights=[emat_word],
                            input_length=MAXLEN_WORD,
                            trainable=False)

layer_embedding_c = Embedding(len(cdict),
                            EMBEDDING_DIM,
                            weights=[emat_char],
                            input_length=MAXLEN_CHAR,
                            trainable=False)

layer_bilstm_w = Bidirectional(LSTM(256, return_sequences=False, dropout=0.2, activation='tanh'),
                               input_shape=(MAXLEN_WORD, EMBEDDING_DIM),
                               merge_mode='concat')

layer_bilstm_c = Bidirectional(LSTM(256, return_sequences=False, dropout=0.2, activation='tanh'),
                               input_shape=(MAXLEN_CHAR, EMBEDDING_DIM),
                               merge_mode='concat')

layer_dense1_c = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))


# ------ embedding -------
embed_q1_w = layer_embedding_w(input_q1_w)
embed_q1_c = layer_embedding_w(input_q1_c)
embed_q2_w = layer_embedding_w(input_q2_w)
embed_q2_c = layer_embedding_w(input_q2_c)


# ------ bilstm -------
bilstm_q1_w = layer_bilstm_w(embed_q1_w)
bilstm_q1_c = layer_bilstm_w(embed_q1_c)
bilstm_q2_w = layer_bilstm_w(embed_q2_w)
bilstm_q2_c = layer_bilstm_w(embed_q2_c)

feat_bilstm_w = Subtract()([bilstm_q1_w, bilstm_q2_w])
feat_bilstm_w = Lambda(lambda x: x ** 2)(feat_bilstm_w)

feat_bilstm_c = Subtract()([bilstm_q1_c, bilstm_q2_c])
feat_bilstm_c = Lambda(lambda x: x ** 2)(feat_bilstm_c)

# ------ last hidden -------
pair = Concatenate()([feat_bilstm_w, feat_bilstm_c])
pair = layer_dense1_c(pair)
pair = Dropout(0.5)(pair)


# ------ predict -------
predict = Dense(1, activation='sigmoid')(pair)


LEARNING_RATE = 0.005
# optimizer = optimizers.SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
optimizer = optimizers.Adagrad(lr=LEARNING_RATE, epsilon=1e-06)
model = Model(inputs=[input_q1_w, input_q1_c, input_q2_w, input_q2_c], outputs=predict)
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizer,
#               metrics=['binary_crossentropy'])
model.compile(loss=focal_loss,
              optimizer=optimizer,
              metrics=['binary_crossentropy'])


###############################################
# train
###############################################
# train.test split
import random
random.shuffle(trainpair_idx)
train_pair = trainpair_idx[:int(len(trainpair_idx)*0.9)]
test_pair = trainpair_idx[int(len(trainpair_idx)*0.9):]

# train_pair.extend([(w2, c2, w1, c1, l) for w1, c1, w2, c2, l in train_pair])
train_q1_w = np.array([p[0] for p in train_pair])
train_q1_c = np.array([p[1] for p in train_pair])
train_q2_w = np.array([p[2] for p in train_pair])
train_q2_c = np.array([p[3] for p in train_pair])
train_y = np.array([p[-1] for p in train_pair])

val_q1_w = np.array([p[0] for p in test_pair])
val_q1_c = np.array([p[1] for p in test_pair])
val_q2_w = np.array([p[2] for p in test_pair])
val_q2_c = np.array([p[3] for p in test_pair])
val_y = np.array([p[-1] for p in test_pair])


testpair = readtest()
testpair_idx = []
for q0, q1 in testpair:
    q0_words = questions[q0]['cwords']
    q1_words = questions[q1]['cwords']
    q0_widx = sent2idx(q0_words, wdict, MAXLEN_WORD)
    q1_widx = sent2idx(q1_words, wdict, MAXLEN_WORD)

    q0_chars = questions[q0]['cchars']
    q1_chars = questions[q1]['cchars']
    q0_cidx = sent2idx(q0_chars, cdict, MAXLEN_CHAR)
    q1_cidx = sent2idx(q1_chars, cdict, MAXLEN_CHAR)

    testpair_idx.append((q0_widx, q0_cidx, q1_widx, q1_cidx))
print(len(testpair_idx))

test_q1_w = np.array([p[0] for p in testpair_idx])
test_q1_c = np.array([p[1] for p in testpair_idx])
test_q2_w = np.array([p[2] for p in testpair_idx])
test_q2_c = np.array([p[3] for p in testpair_idx])

EPOCH = 100
BATCH_SIZE = 128
BESTSCORE = 1e6
EARLYSTOP = 3
ES_FLAG = 0

for i in range(EPOCH):
    # print(K.get_value(model.optimizer.lr))
    model.fit([train_q1_w, train_q1_c, train_q2_w, train_q2_c], train_y,
              validation_data=([val_q1_w, val_q1_c, val_q2_w, val_q2_c], val_y),
              batch_size=BATCH_SIZE)
    evalinfo = model.evaluate([val_q1_w, val_q1_c, val_q2_w, val_q2_c], val_y, batch_size=BATCH_SIZE)
    score = evalinfo[-1]
    if score < BESTSCORE:
        BESTSCORE = score
        ES_FLAG = 0

        pred_y = model.predict([test_q1_w, test_q1_c, test_q2_w, test_q2_c], batch_size=BATCH_SIZE)
        pred_y = np.reshape(pred_y, (len(pred_y),))
        savepath = 'submission-rnn-' + str(BESTSCORE)[:7] + '.csv'
        make_submission(pred_y, savepath)
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

