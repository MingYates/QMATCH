# -*- coding: utf-8 -*-
import numpy as np


# cdict = []
# ematrix = []
# with open('/files/faust/COMPETITION/ppdai/char_embed.txt', 'r', encoding='utf8') as f:
#     for line in f:
#         piece = line.strip().split(' ')
#         ic = piece[0]
#         iv = np.array(piece[1:])
#         cdict.append(ic)
#         ematrix.append(iv)
# cdict = dict((c, i) for i, c in enumerate(cdict))
# # OOV
# cdict['OOV'] = len(cdict)
# ematrix.append(np.zeros((300,), dtype=np.float32))
# ematrix = np.array(ematrix)
#
# print(len(cdict))
# print(ematrix.shape)


# import csv
#
# q1_list = []
# q2_list = []
#
# count = 0
# trainfile = '/files/faust/COMPETITION/ppdai/train.csv'
# with open(trainfile) as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         if row['label'] == '1':
#             count += 1
#         q1_list.append(row['q1'])
#         q2_list.append(row['q2'])
#
# print(len(q1_list))
# print(count)
#
# q1_list = set(q1_list)
# print(len(q1_list))
# q2_list = set(q2_list)
# print(len(q2_list))
#
# print(len(q1_list & q2_list))
# print(len(q1_list | q2_list))
#
#
# clus = {}
# with open(trainfile) as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         if row['label'] == '1':
#             qpair = [row['q1'], row['q2']]
#             for qid in qpair:
#                 if qid not in clus:
#                     clus[qid] = qpair
#                 else:
#                     clus[qid].extend(qpair)
#
# for k, v in clus.items():
#     clus[k] = set(clus[k])


from ppdai_utils import *
import pickle

# cdict, emat = embedding('/files/faust/COMPETITION/ppdai/char_embed.txt')
# questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions.pkl', 'rb'))
# trainpair = readtrain_extend()
# charlen = [len(wc['cchars']) for _,wc in questions.items()]
# print(max(charlen))
# charlen = [len(wc['cwords']) for _,wc in questions.items()]
# print(max(charlen))



from magpie import Magpie
print('imported done.')