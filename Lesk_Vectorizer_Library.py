import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pywsd.similarity import max_similarity
import Lemmatizer
import tfidf_vectoriser_library as vec_lib


class TfidfEmbeddingVectorizer_Lesk(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform_sent_1(self, X, vectoriser):
        # only transforms one sentence
        lesk_word_vector = tfidf_Lesk_sent_tranformer(X, vectoriser)
        X = Lemmatizer.lemmatize(X)
        # temp_vector_container = []
        # for w in X:
        #     # print(w)
        #     temp_vector_container.append(lesk_word_vector[w] * self.word2weight[w])
        # temp_mean = np.mean(temp_vector_container, axis=0)
        # print(temp_mean)
        return np.mean(
            [lesk_word_vector[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)],
            axis=0)

    def transform_sent(self, X):
        # only transforms one sentence
        test_sent = X
        X = Lemmatizer.lemmatize(X)
        try:
            return np.average([self.word2vec[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)], axis=0, weights=[self.word2weight[w] for w in X if w in self.word2vec])
        except:
            print("Error for : ", X)
            return np.zeros(self.dim)
            pass


def lesk_word_sense(tokens, word):
    # pos_tagged = nltk.pos_tag(vec_lib.tokenize(tokens), tagset='universal')
    # need to fix this
    # if pos_tagged[1][1]=='VERB':
    #     temp_pos = 'v'
    # else:
    temp_pos = 'n'
    temp_synset = max_similarity(tokens, word, pos=temp_pos)
    if temp_synset:
        return temp_synset.definition()
    else:
        return word


def tfidf_Lesk_sent_tranformer(text, vectoriser):
    tokens = Lemmatizer.lemmatize(text)
    sent_arr = {}
    for tkn in tokens:
        # get the meaning sentence from the wordnet
        temp_sent = lesk_word_sense(text, tkn)
        sent_arr[tkn] = temp_sent
    res = {}
    for i in range(len(tokens)):
        res[tokens[i]] = vectoriser.transform_sent(sent_arr[tokens[i]])
    return res
    pass
