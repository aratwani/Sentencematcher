import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import os
import csv
import Lemmatizer
import string


def write_line_to_csv(line, header, filename):
    # if file is not present at all
    need_header = False
    filename = filename
    if not os.path.isfile(filename):
        f = open(filename, 'w+')
        need_header = True
    # if file is present but empty
    elif os.stat(filename).st_size == 0:
        f = open(filename, 'w')
        need_header = True
    else:
        f = open(filename, 'a')

    writer = csv.writer(f)
    if need_header:
        entries = header
        writer.writerow(entries)
    entries = line
    writer.writerow(entries)
    f.close()


def import_glove6b_pretrained_vectors():
    w2v = {}
    with open("glove.6B.300d.txt", "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode('utf-8')
            nums = np.array(parts[1:], dtype=np.float32)
            w2v[word] = nums
    return w2v


def tokenize(sent):
    # input is a sentence
    # removing punctuation
    sent = "".join([ch for ch in sent if ch not in string.punctuation])
    # tokenize
    tokens = nltk.word_tokenize(sent.lower())
    # removal of stop words
    stop_words = set(stopwords.words('english'))
    res = [word for word in tokens if word not in stop_words and word.isalnum()]
    return res
    pass


class TfidfEmbeddingVectorizer(object):
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

    def transform(self, X):
        X = tokenize(X)
        return np.array([np.average([self.word2vec[w] * self.word2weight[w] for w in tokenize(words) if w in self.word2vec] or [np.zeros(self.dim)], axis=0, weights=[self.word2weight[w] for w in X if w in self.word2vec]) for words in X])

    def transform_sent(self, X):
        # only transforms one sentence
        X = Lemmatizer.lemmatize(X)
        try:
            return np.average([self.word2vec[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)], axis=0, weights=[self.word2weight[w] for w in X if w in self.word2vec])
        except:
            print("Error for : ", X)
            return np.zeros(self.dim)
            pass


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X):
        return self

    def transform_sent(self, X):
        # transforms a sentence
        return np.array(np.mean([self.word2vec[w] for w in tokenize(X) if w in self.word2vec] or [np.zeros(self.dim)], axis=0))

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])


def cosine_similarity_vector(a, b):
    return dot(a, b) / (norm(a)*norm(b))
    pass



