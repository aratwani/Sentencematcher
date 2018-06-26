import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

w2v = {}
with open("glove.6B.50d.txt", "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode('utf-8')
        nums=np.array(parts[1:], dtype=np.float32)
        w2v[word] = nums


def tokenize(sent):
    # input is a sentence
    tokens = nltk.word_tokenize(sent.lower())
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

    def transform_1(self, X):
        X = tokenize(X)
        #return np.array([np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)for words in X])
        return np.mean([self.word2vec[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)], axis=0)

    def transform(self, X):
        X = tokenize(X)
        return np.average(
            [self.word2vec[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)], axis=0, weights=[self.word2weight[w] for w in X if w in self.word2vec])


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def cosine_similarity_vector(a, b):
    return dot(a,b) / (norm(a)*norm(b))
    pass


import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def main():
    df = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    df.dropna()
    data = df["answer"].dropna().astype(str)
    # vectoriser = MeanEmbeddingVectorizer(w2v)
    # vectoriser.fit(data)
    #
    # X = vectoriser.transform("Brake pedal is pulsating".lower())
    # Y = vectoriser.transform("Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower())
    # Z = vectoriser.transform("Amit is a rockstar".lower())
    #
    # res = cosine_similarity_vector(X, Y)
    # res1 = cosine_similarity_vector(Z, Y)
    # res2 = cosine_similarity_vector(X, Z)
    # print("1. ", res)
    # print("2. ", res1)
    # print("3. ", res2)

    tfidf_vectorizer = TfidfEmbeddingVectorizer(w2v)
    tfidf_vectorizer.fit(data)
    X1 = tfidf_vectorizer.transform("Brake pedal is pulsating".lower())
    X2 = tfidf_vectorizer.transform_1("Brake pedal is pulsating".lower())
    Y1 = tfidf_vectorizer.transform("Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower())
    Y2 = tfidf_vectorizer.transform_1("Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower())
    Z1 = tfidf_vectorizer.transform_1("its hot in texas")
    Z2 = tfidf_vectorizer.transform_1("texas is cold")

    tfidf_res = cosine_similarity(X1, X2)
    tfidf_res1 = cosine_similarity(X1, Y1)
    tfidf_res2 = cosine_similarity(X2, Y2)
    tfidf_res3 = cosine_similarity(Z1, Z2)

    print("X1 - X2", tfidf_res)
    print("X1 - Y1", tfidf_res1)
    print("X2 - Y2", tfidf_res2)
    print("Z1 - Z2", tfidf_res3)
    Z1 = tfidf_vectorizer.transform("Amit is a rockstar".lower())
    #
    # tfidf_res = cosine_similarity_vector(X1, Y1)
    # tfidf_res1 = cosine_similarity_vector(Z1, Y1)
    # tfidf_res2 = cosine_similarity_vector(X1, Z1)
    # print("1. ", tfidf_res)
    # print("2. ", tfidf_res1)
    # print("3. ", tfidf_res2)
    pass


main()