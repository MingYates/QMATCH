# -*- coding: utf-8 -*-
from gensim import corpora, models
from ppdai_utils import *
numtopics = 50

questions = pickle.load(open('/files/faust/COMPETITION/ppdai/questions.pkl', 'rb'))
questions_ids = [qid for qid, _ in questions.items()]
questions_words = [text['cwords'] for _, text in questions.items()]

word_dict = corpora.Dictionary(questions_words)
print(word_dict)
doc_list = [word_dict.doc2bow(q) for q in questions_words]
print('doc size = %d' % len(doc_list))
doc_tfidf = models.TfidfModel(doc_list)[doc_list]

print('learning...')
model_lda = models.ldamodel.LdaModel(corpus=doc_tfidf, id2word=word_dict, num_topics=numtopics, alpha='auto')
print('done.')

corpus_lda = model_lda[doc_tfidf]
print(len(corpus_lda))


def tm2vec(qtm):
    vec = [.0] * numtopics
    for t, p in qtm:
        vec[t] = p
    return vec


model_lda.show_topics(num_topics=numtopics, num_words=10, formatted=True)
corpus_lda = [tm2vec(qv) for qv in corpus_lda]
corpus_lda = np.array(corpus_lda)

question_id2v = dict([(qid, vec) for qid, vec in zip(questions_ids, corpus_lda)])
import pickle
pickle.dump(question_id2v, open('/files/faust/COMPETITION/ppdai/questions_lda.pkl', 'wb'))

