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
from pywsd.similarity import max_similarity
from pywsd.utils import lemmatize
import Lemmatizer


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
            nums=np.array(parts[1:], dtype=np.float32)
            w2v[word] = nums
    return w2v


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



    def transform_sent_1(self, X):
        # only transforms one sentence
        lesk_word_vector = tfidf_Lesk_sent_tranformer(X, self.word2vec)
        X = tokenize(X)
        temp = []
        for w in X:
            print(w)
            if lemmatize(w) in self.word2vec:
                temp.append()
        return np.mean(
            [lesk_word_vector[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)],
            axis=0)


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


    def transform_sent_1(self, X):
        # only transforms one sentence
        lesk_word_vector = tfidf_Lesk_sent_tranformer(X, self.word2vec)
        X = tokenize(X)
        temp = []
        for w in X:
            print(w)
            if lemmatize(w) in self.word2vec:
                temp.append()
        return np.mean(
            [lesk_word_vector[w] * self.word2weight[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)],
            axis=0)



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
    return dot(a,b) / (norm(a)*norm(b))
    pass




def lesk_word_sense(tokens, word):
    #text = "Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower()
    pos_tagged = nltk.pos_tag(tokenize(tokens), tagset='universal')
    print(pos_tagged)
    # need to fix this
    if pos_tagged[1][1]=='VERB':
        temp_pos = 'v'
    else:
        temp_pos = 'n'
    #print(lesk(tokens, pos_tagged[1][0], pos=temp_pos).definition())
    #temp_synset = lesk(tokens, word, pos=temp_pos)
    temp_synset = max_similarity(tokens, word, pos=temp_pos)
    if temp_synset:
        return temp_synset.definition()
    else:
        return word


def tfidf_Lesk_sent_tranformer(text, w2v):
    mean_vectorsizer = MeanEmbeddingVectorizer(w2v)
    tokens = tokenize(text)
    sent_arr = []
    for tkn in tokens:
        # get the meaning sentence from the wordnet
        temp_sent = lesk_word_sense(text, tkn)
        sent_arr.append(temp_sent)
    word_vec_arr = mean_vectorsizer.transform(sent_arr)
    res = {}
    for i in range(len(tokens)):
        res[tokens[i].lower()] = word_vec_arr[i]
    return res
    pass







def main():
    df = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    df.dropna()
    data = df["answer"].dropna().astype(str)

    pass


main()