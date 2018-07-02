import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pywsd.similarity import max_similarity
import Lemmatizer
import tfidf_vectoriser_library as vec_lib

word_sense_vector_hash = {}

class TfidfEmbeddingVectorizer_Lesk(object):
    # this hash is used to store the sense vectors of the words,
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

    def transform_sent_1(self, text, vectoriser):
        # only transforms one sentence using Lesk algorithm
        tokens = Lemmatizer.lemmatize(text)
        lesk_word_vectors = tfidf_Lesk_sent_tranformer(text,tokens,vectoriser)
        # return np.mean(
        #     [lesk_word_vectors[w] * self.word2weight[w] for w in tokens if w in self.word2vec] or [np.zeros(self.dim)],
        #     axis=0)
        return np.mean([lesk_word_vectors[w] * self.word2weight[w] for w in tokens], axis=0)

    def transform_sent(self, text):
        # only transforms one sentence using tfidf weighting and GloVe
        tokens = Lemmatizer.lemmatize(text)
        try:
            return np.average([self.word2vec[w] * self.word2weight[w] for w in tokens if w in self.word2vec] or [np.zeros(self.dim)], axis=0, weights=[self.word2weight[w] for w in tokens if w in self.word2vec])
        except:
            print("Error for : ", text)
            return np.zeros(self.dim)
            pass


def lesk_word_sense(text, word, pos_tagged):
    # get the meaaning of the word from the context
    if pos_tagged[1] == 'VERB':
        temp_pos = 'v'
    else:
        temp_pos = 'n'
    temp_synset = max_similarity(text, word, pos=temp_pos)
    if temp_synset:
        return temp_synset.definition()
    else:
        return word


def tfidf_Lesk_sent_tranformer(text, tokens,vectoriser):
    res = {}
    pos_tagged = nltk.pos_tag(tokens, tagset='universal')
    for i in range(len(tokens)):
        # get the meaning sentence from the wordnet, get its vector and return the set of vectors
        tkn = tokens[i]
        temp_sent = lesk_word_sense(text, tkn, pos_tagged[i])
        if temp_sent not in word_sense_vector_hash:
            res[tkn] = vectoriser.transform_sent(temp_sent)
            word_sense_vector_hash[temp_sent] = res[tkn]
        else:
            res[tkn] = word_sense_vector_hash[temp_sent]

    return res
    pass
