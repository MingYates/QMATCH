# -*- coding: utf-8 -*-
from ppdai_utils import *
import pickle
from myloss import focal_loss

MAXLEN_CHAR = 60
MAXLEN_WORD = 40
VECLEN = 100

cdict, emat_char = embedding('/files/faust/COMPETITION/ppdai/char_embed.txt')
wdict, emat_word = embedding('/files/faust/COMPETITION/ppdai/word_embed.txt')

questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions.pkl', 'rb'))

trainpair = readtrain()
question2vec = pickle.load(open('/files/faust/COMPETITION/ppdai/questions_lda.pkl', 'rb'))


def negpairs(questions, num=50000):
    result = []
    qlist_0 = random.sample(list(questions.keys()), num)
    qlist_1 = random.sample(list(questions.keys()), num)
    for q0, q1 in zip(qlist_0, qlist_1):
        q0_words = questions[q0]['cwords']
        q1_words = questions[q1]['cwords']
        q0_widx = sent2idx(q0_words, wdict, MAXLEN_WORD)
        q1_widx = sent2idx(q1_words, wdict, MAXLEN_WORD)

        q0_chars = questions[q0]['cchars']
        q1_chars = questions[q1]['cchars']
        q0_cidx = sent2idx(q0_chars, cdict, MAXLEN_CHAR)
        q1_cidx = sent2idx(q1_chars, cdict, MAXLEN_CHAR)
        q0_vec = question2vec[q0]
        q1_vec = question2vec[q1]
        if not q0 == q1:
            result.append((q0_widx, q0_cidx, q0_vec, q1_widx, q1_cidx, q1_vec, 0))
        else:
            result.append((q0_widx, q0_cidx, q0_vec, q1_widx, q1_cidx, q1_vec, 1))
    return result


def prepare(qpairs, state='train'):
    result = []
    if state == 'train':
        for q0, q1, l in qpairs:
            q0_words = questions[q0]['cwords']
            q1_words = questions[q1]['cwords']
            q0_widx = sent2idx(q0_words, wdict, MAXLEN_WORD)
            q1_widx = sent2idx(q1_words, wdict, MAXLEN_WORD)

            q0_chars = questions[q0]['cchars']
            q1_chars = questions[q1]['cchars']
            q0_cidx = sent2idx(q0_chars, cdict, MAXLEN_CHAR)
            q1_cidx = sent2idx(q1_chars, cdict, MAXLEN_CHAR)

            q0_vec = question2vec[q0]
            q1_vec = question2vec[q1]
            result.append((q0_widx, q0_cidx, q0_vec, q1_widx, q1_cidx, q1_vec, l))
    else:
        for q0, q1 in qpairs:
            q0_words = questions[q0]['cwords']
            q1_words = questions[q1]['cwords']
            q0_widx = sent2idx(q0_words, wdict, MAXLEN_WORD)
            q1_widx = sent2idx(q1_words, wdict, MAXLEN_WORD)

            q0_chars = questions[q0]['cchars']
            q1_chars = questions[q1]['cchars']
            q0_cidx = sent2idx(q0_chars, cdict, MAXLEN_CHAR)
            q1_cidx = sent2idx(q1_chars, cdict, MAXLEN_CHAR)

            q0_vec = question2vec[q0]
            q1_vec = question2vec[q1]

            result.append((q0_widx, q0_cidx, q0_vec, q1_widx, q1_cidx, q1_vec))
    # print(len(trainpair_idx))
    return result


def input(pairs, state='train'):
    if state == 'train':
        q0_w = np.array([p[0] for p in pairs])
        q0_c = np.array([p[1] for p in pairs])
        q0_tm = np.array([p[2] for p in pairs])
        q1_w = np.array([p[3] for p in pairs])
        q1_c = np.array([p[4] for p in pairs])
        q1_tm = np.array([p[5] for p in pairs])
        y = np.array([p[-1] for p in pairs])
        return q0_w, q0_c, q0_tm, q1_w, q1_c, q1_tm, y
    else:
        q0_w = np.array([p[0] for p in pairs])
        q0_c = np.array([p[1] for p in pairs])
        q0_tm = np.array([p[2] for p in pairs])
        q1_w = np.array([p[3] for p in pairs])
        q1_c = np.array([p[4] for p in pairs])
        q1_tm = np.array([p[5] for p in pairs])
        return q0_w, q0_c, q0_tm, q1_w, q1_c, q1_tm


from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv1D, Conv2D, TimeDistributed
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization
from keras.layers import concatenate, Flatten, regularizers
from keras.layers import Add, Subtract, Multiply, Concatenate, Reshape, RepeatVector, Permute
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu, sigmoid, tanh
from keras import backend as K
from keras import optimizers


EMBEDDING_DIM = 300
DROPOUT = 0.5
input_q1_w = Input(shape=(MAXLEN_WORD,))
input_q2_w = Input(shape=(MAXLEN_WORD,))

layer_embedding_w = Embedding(len(wdict),
                              EMBEDDING_DIM,
                              weights=[emat_word],
                              input_length=MAXLEN_WORD,
                              trainable=False)

# conv1
layer_conv_1 = [Conv2D(64, (s, s), data_format='channels_last', activation='relu', padding='same') for s in [1, 2, 3, 5]]
layer_pool_1 = [MaxPooling2D(pool_size=(s, s), data_format='channels_last') for s in [3, 5, 7]]
layer_conv_2 = [Conv2D(64, (s, s), data_format='channels_last', activation='relu', padding='same') for s in [3, 5, 7, 9]]
layer_pool_2 = [MaxPooling2D(pool_size=(s, s), data_format='channels_last') for s in [7, 9, 11]]

# ------ embedding -------
q1_embed_w = layer_embedding_w(input_q1_w)
q2_embed_w = layer_embedding_w(input_q2_w)

# ------ mix table -------
pair_table_1 = RepeatVector(MAXLEN_WORD)(Flatten()(q1_embed_w))
pair_table_2 = RepeatVector(MAXLEN_WORD)(Flatten()(q2_embed_w))
pair_table_1 = Reshape((MAXLEN_WORD, MAXLEN_WORD, EMBEDDING_DIM))(pair_table_1)
pair_table_2 = Reshape((MAXLEN_WORD, MAXLEN_WORD, EMBEDDING_DIM))(pair_table_2)
pair_table_2 = Permute((2, 1, 3))(pair_table_2)

pair_table = Subtract()([pair_table_1, pair_table_2])
pair_table = Lambda(lambda x: x ** 2)(pair_table)
pair_table = BatchNormalization()(pair_table)
pair_table = Dropout(0.5)(pair_table)
# print(pair_table)

# ------ table conv 1 and pool 1-------
conv_t1 = [conv(pair_table) for conv in layer_conv_1]
conv_t1 = Concatenate()(conv_t1)
pool_t1 = [Flatten()(pool(conv_t1)) for pool in layer_pool_1]
pair_feat_1 = Concatenate()(pool_t1)

# ------ table conv 2 and pool 2 -------
conv_t1 = BatchNormalization()(conv_t1)
conv_t1 = Dropout(0.5)(conv_t1)
conv_t2 = [conv(conv_t1) for conv in layer_conv_2]
conv_t2 = Concatenate()(conv_t2)
pool_t2 = [Flatten()(pool(conv_t2)) for pool in layer_pool_2]
pool_t2 = Concatenate()(pool_t2)
pool_tg = GlobalMaxPooling2D()(conv_t2)
pair_feat_2 = Concatenate()([pool_t2, pool_tg])

# ------ concat all features -------
pair = Concatenate()([pair_feat_1, pair_feat_2])
pair = BatchNormalization()(pair)
pair = Dropout(0.5)(pair)

# ------ last hidden -------
pair_last = Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(0.05))(pair)
# pair_last = Dense(256, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001))(pair)
# pair_last = Dense(256, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(pair)

# ------ predict -------
pair_last = BatchNormalization()(pair_last)
pair_last = Dropout(0.5)(pair_last)
predict = Dense(1, activation='sigmoid')(pair_last)

LEARNING_RATE = 0.001
# optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model = Model(inputs=[input_q1_w, input_q2_w], outputs=predict)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['binary_crossentropy'])
# model.compile(loss=focal_loss,
#               optimizer=optimizer,
#               metrics=['binary_crossentropy'])

###############################################
# prepare data
###############################################
import random
train_all = prepare(trainpair, 'train')
# print(train_all[:5])
random.shuffle(train_all)
train_pair = train_all[:int(len(train_all)*0.9)]
print(len(train_pair))
val_pair = train_all[int(len(train_all)*0.9):]

# train_pair.extend([(w2, c2, w1, c1, l) for w1, c1, w2, c2, l in train_pair])
samples_neg = negpairs(questions, num=100000)
train_pair.extend(samples_neg)
random.shuffle(train_pair)
print(len(train_pair))

train_all_q1_w, train_all_q1_c, train_all_q1_tm, train_all_q2_w, train_all_q2_c, train_all_q2_tm, train_all_y = input(train_all, 'train')
# print(train_all_q1_w[:5])
# print(train_all_q1_tm[:5])
train_q1_w, train_q1_c, train_q1_tm, train_q2_w, train_q2_c, train_q2_tm, train_y = input(train_pair, 'train')
val_q1_w, val_q1_c, val_q1_tm, val_q2_w, val_q2_c, val_q2_tm, val_y = input(val_pair, 'train')

testdata = readtest()
testpair = prepare(testdata, 'test')
test_q1_w, test_q1_c, test_q1_tm, test_q2_w, test_q2_c, test_q2_tm = input(testpair, 'test')


EPOCH = 50
BATCH_SIZE = 128
BESTSCORE = 1e6
EARLYSTOP = 3
ES_FLAG = 0


for i in range(EPOCH):
    # print(K.get_value(model.optimizer.lr))
    # model.fit([train_q1_w, train_q2_w], train_y,
    #           validation_split=0.1,
    #           batch_size=BATCH_SIZE)
    model.fit([train_q1_w, train_q2_w], train_y,
              validation_data=([val_q1_w, val_q2_w], val_y),
              batch_size=BATCH_SIZE)
    evalinfo = model.evaluate([val_q1_w, val_q2_w], val_y,
                              batch_size=BATCH_SIZE, verbose=1)
    print('val_loss = %f, val_c = %f' % (evalinfo[0], evalinfo[1]))
    score = evalinfo[-1]
    if score < BESTSCORE:
        BESTSCORE = score
        ES_FLAG = 0

        pred_y = model.predict([test_q1_w, test_q2_w],
                               batch_size=BATCH_SIZE)
        pred_y = np.reshape(pred_y, (len(pred_y),))
        savepath = 'submission/submission-tcnn-' + str(BESTSCORE)[:7] + '.csv'
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

