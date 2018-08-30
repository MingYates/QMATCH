# -*- coding: utf-8 -*-

# jaccard 距离
def dis_jaccard(q1, q2):
    return len(set(q1) & set(q2)) / len(set(q1) | set(q2))


# # 词向量平均向量距离
# def dis_w2v_avg(q1, q2):
#


# #编辑距离
# def dis_edit(q1, q2):
#


# #n-gram 距离
# def dis_ngram(q1, q2):
#


# #LDA
# def sim_lda(q1, q2):
#


