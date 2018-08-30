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
# trainpair = readtrain_extend()
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


###############################################
# model
###############################################
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


EMBEDDING_DIM = 300
DROPOUT = 0.5

input_q1_w = Input(shape=(MAXLEN_WORD,))
input_q1_c = Input(shape=(MAXLEN_CHAR,))
input_q1_vec = Input(shape=(VECLEN,))
input_q2_w = Input(shape=(MAXLEN_WORD,))
input_q2_c = Input(shape=(MAXLEN_CHAR,))
input_q2_vec = Input(shape=(VECLEN,))

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

filter_size = [1, 2, 3, 5]
layer_conv_c1 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # char conv1
layer_fc_c1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # char fc1
layer_conv_c2 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # char conv2
layer_fc_c2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # char fc2
layer_conv_c3 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # char conv3

layer_conv_w1 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # word conv1
layer_fc_w1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # word fc1
layer_conv_w2 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # word conv2
layer_fc_w2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # word fc2
layer_conv_w3 = [Conv1D(64, size, activation='relu', padding='same') for size in filter_size]  # word conv3

# ------ embedding -------
q1_embed_c = layer_embedding_c(input_q1_c)
q2_embed_c = layer_embedding_c(input_q2_c)
q1_embed_w = layer_embedding_w(input_q1_w)
q2_embed_w = layer_embedding_w(input_q2_w)

# ------ conv 1 -------
q1_embed_c = BatchNormalization()(q1_embed_c)
q1_embed_w = BatchNormalization()(q1_embed_w)
q2_embed_c = BatchNormalization()(q2_embed_c)
q2_embed_w = BatchNormalization()(q2_embed_w)
q1_embed_c = Dropout(0.2)(q1_embed_c)
q1_embed_w = Dropout(0.2)(q1_embed_w)
q2_embed_c = Dropout(0.2)(q2_embed_c)
q2_embed_w = Dropout(0.2)(q2_embed_w)

# multi-size pooling
# conv_1_pool = [MaxPooling1D(ws) for ws in [3, 5, 7]]
conv_1_pool = [MaxPooling1D(ws) for ws in [3, 5, 7]]
# char
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

# word
q1_conv_w1 = [conv(q1_embed_w) for conv in layer_conv_w1]
q1_conv_w1 = Concatenate()(q1_conv_w1)
q2_conv_w1 = [conv(q2_embed_w) for conv in layer_conv_w1]
q2_conv_w1 = Concatenate()(q2_conv_w1)

q1_pool_w1 = [Flatten()(mp(q1_conv_w1)) for mp in conv_1_pool]
q1_pool_w1 = Concatenate()(q1_pool_w1)
q2_pool_w1 = [Flatten()(mp(q2_conv_w1)) for mp in conv_1_pool]
q2_pool_w1 = Concatenate()(q2_pool_w1)


pair_sub_w1 = Subtract()([layer_fc_w1(q1_pool_w1), layer_fc_w1(q2_pool_w1)])
pair_feat_w1 = Lambda(lambda x: x ** 2)(pair_sub_w1)

# # ------ conv 2 -------
q1_conv_c1 = BatchNormalization()(q1_conv_c1)
q2_conv_c1 = BatchNormalization()(q2_conv_c1)
q1_conv_w1 = BatchNormalization()(q1_conv_w1)
q2_conv_w1 = BatchNormalization()(q2_conv_w1)
q1_conv_c1 = Dropout(0.5)(q1_conv_c1)
q2_conv_c1 = Dropout(0.5)(q2_conv_c1)
q1_conv_w1 = Dropout(0.5)(q1_conv_w1)
q2_conv_w1 = Dropout(0.5)(q2_conv_w1)

# conv_2_pool = [MaxPooling1D(ws) for ws in [5, 7, 11]]
conv_2_pool = [MaxPooling1D(ws) for ws in [7, 9, 11]]

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

# word
q1_conv_w2 = [conv(q1_conv_w1) for conv in layer_conv_w2]
q1_conv_w2 = Concatenate()(q1_conv_w2)
q2_conv_w2 = [conv(q2_conv_w1) for conv in layer_conv_w2]
q2_conv_w2 = Concatenate()(q2_conv_w2)

q1_pool_w2 = [Flatten()(mp(q1_conv_w2)) for mp in conv_2_pool]
q1_pool_w2 = Concatenate()(q1_pool_w2)
q2_pool_w2 = [Flatten()(mp(q2_conv_w2)) for mp in conv_2_pool]
q2_pool_w2 = Concatenate()(q2_pool_w2)

# q1_pool_w2 = Dropout(0.5)(q1_pool_w2)
# q2_pool_w2 = Dropout(0.5)(q2_pool_w2)
pair_sub_w2 = Subtract()([layer_fc_w2(q1_pool_w2), layer_fc_w2(q2_pool_w2)])
pair_feat_w2 = Lambda(lambda x: x ** 2)(pair_sub_w2)

# # ------ conv 3 -------
q1_conv_c2 = BatchNormalization()(q1_conv_c2)
q2_conv_c2 = BatchNormalization()(q2_conv_c2)
q1_conv_w2 = BatchNormalization()(q1_conv_w2)
q2_conv_w2 = BatchNormalization()(q2_conv_w2)
q1_conv_c2 = Dropout(0.5)(q1_conv_c2)
q2_conv_c2 = Dropout(0.5)(q2_conv_c2)
q1_conv_w2 = Dropout(0.5)(q1_conv_w2)
q2_conv_w2 = Dropout(0.5)(q2_conv_w2)

q1_conv_c3 = [conv(q1_conv_c2) for conv in layer_conv_c3]
q1_conv_c3 = Concatenate()(q1_conv_c3)
q2_conv_c3 = [conv(q2_conv_c2) for conv in layer_conv_c3]
q2_conv_c3 = Concatenate()(q2_conv_c3)

q1_pool_c3 = GlobalMaxPooling1D()(q1_conv_c3)
q2_pool_c3 = GlobalMaxPooling1D()(q2_conv_c3)

pair_sub_c3 = Subtract()([q1_pool_c3, q2_pool_c3])
pair_feat_c3 = Lambda(lambda x: x ** 2)(pair_sub_c3)

# word
q1_conv_w3 = [conv(q1_conv_w2) for conv in layer_conv_w3]
q1_conv_w3 = Concatenate()(q1_conv_w3)
q2_conv_w3 = [conv(q2_conv_w2) for conv in layer_conv_w3]
q2_conv_w3 = Concatenate()(q2_conv_w3)

q1_pool_w3 = GlobalMaxPooling1D()(q1_conv_w3)
q2_pool_w3 = GlobalMaxPooling1D()(q2_conv_w3)

pair_sub_w3 = Subtract()([q1_pool_w3, q2_pool_w3])
pair_feat_w3 = Lambda(lambda x: x ** 2)(pair_sub_w3)

# topic
pair_sub_vec = Subtract()([input_q1_vec, input_q2_vec])
pair_feat_vec = Lambda(lambda x: x ** 2)(pair_sub_vec)

# # ------ concat all -------
pair = Concatenate()([pair_feat_c1, pair_feat_c2, pair_feat_c3,
                      pair_feat_w1, pair_feat_w2, pair_feat_w3,
                      pair_feat_vec])
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
model = Model(inputs=[input_q1_w, input_q1_c, input_q1_vec, input_q2_w, input_q2_c, input_q2_vec], outputs=predict)

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
              metrics=['binary_crossentropy'])


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
# train_pair.extend([(w2, c2, w1, c1, l) for w1, c1, w2, c2, l in train_pair])

train_all_q1_w, train_all_q1_c, train_all_q1_tm, train_all_q2_w, train_all_q2_c, train_all_q2_tm, train_all_y = input(train_all, 'train')
# print(train_all_q1_w[:5])
# print(train_all_q1_tm[:5])
train_q1_w, train_q1_c, train_q1_tm, train_q2_w, train_q2_c, train_q2_tm, train_y = input(train_pair, 'train')
val_q1_w, val_q1_c, val_q1_tm, val_q2_w, val_q2_c, val_q2_tm, val_y = input(val_pair, 'train')

testdata = readtest()
test_pair = prepare(testdata, 'test')
test_q1_w, test_q1_c, test_q1_tm, test_q2_w, test_q2_c, test_q2_tm = input(test_pair, 'test')


###############################################
# train
###############################################
EPOCH = 100
BATCH_SIZE = 128
BESTSCORE = 1e6
EARLYSTOP = 5
ES_FLAG = 0


for i in range(EPOCH):
    # print(K.get_value(model.optimizer.lr))
    # model.fit([train_q1_w, train_q1_c, train_q1_tm, train_q2_w, train_q2_c, train_q2_tm], train_y,
    #           validation_split=0.1,
    #           batch_size=BATCH_SIZE)
    model.fit([train_q1_w, train_q1_c, train_q1_tm, train_q2_w, train_q2_c, train_q2_tm], train_y,
              validation_data=([val_q1_w, val_q1_c, val_q1_tm, val_q2_w, val_q2_c, val_q2_tm], val_y),
              batch_size=BATCH_SIZE)
    evalinfo = model.evaluate([val_q1_w, val_q1_c, val_q1_tm, val_q2_w, val_q2_c, val_q2_tm], val_y,
                              batch_size=BATCH_SIZE, verbose=1)
    print('val_loss = %f, val_c = %f' % (evalinfo[0], evalinfo[1]))
    score = evalinfo[-1]
    if score < BESTSCORE:
        BESTSCORE = score
        ES_FLAG = 0

        pred_y = model.predict([test_q1_w, test_q1_c, test_q1_tm, test_q2_w, test_q2_c, test_q2_tm],
                               batch_size=BATCH_SIZE)
        pred_y = np.reshape(pred_y, (len(pred_y),))
        savepath = 'submission/submission-cnn-' + str(BESTSCORE)[:7] + '.csv'
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


# from keras import callbacks
# cb_reducelr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,
#                                           verbose=0, mode='auto', epsilon=0.001, cooldown=0)
# cb_ckpt = callbacks.ModelCheckpoint('/files/faust/COMPETITION/ppdai/rescnn.{val_loss:.2f}.hdf5',
#                                     monitor='val_loss', verbose=1, mode='auto', period=1)
# cb_earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
#
# model.fit([train_all_q1_w, train_all_q1_c, train_all_q2_w, train_all_q2_c], train_all_y,
#           validation_split=0.2,
#           batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,
#           callbacks=[cb_reducelr, cb_ckpt, cb_reducelr])



