import numpy as np
import nltk
import string
import Lemmatizer
import os
import pandas as pd
import sys
import csv
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from pywsd.similarity import max_similarity
from nltk.corpus import stopwords
from collections import defaultdict


# this hash is used to store the sense vectors of the words
word_sense_vector_hash = {}


def import_glove6b_pretrained_vectors():
    w2v = {}
    with open("glove.6B.300d.txt", "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode('utf-8')
            nums = np.array(parts[1:], dtype=np.float32)
            w2v[word] = nums
    return w2v


def import_word2vec_vectors():
    w2v = {}
    words = []
    vecs = []
    w2v_word_file = "/Users/aratwani/PycharmProjects/NLPProjects/word2vec_vectors/words_index.txt"
    w2v_vector_file = "/Users/aratwani/PycharmProjects/NLPProjects/word2vec_vectors/words_vectors.npy"
    with open(w2v_word_file, "rb") as wordfile:
        for line in wordfile:
            words.append(line.split()[0].decode('utf-8'))

    vecs = np.load(w2v_vector_file)

    if len(words) != len(vecs) :
        print("Error extracting the wor2vec files : the index does not match the vectors")
        return None
    else:
        for i in range(len(words)):
            w2v[words[i]] = vecs[i]
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


def cosine_similarity_vector(a, b):
    return dot(a, b) / (norm(a)*norm(b))
    pass


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

    def transform_sent_lesk(self, text, vectoriser):
        # only transforms one sentence using Lesk algorithm
        tokens = Lemmatizer.lemmatize(text)
        lesk_word_vectors = tfidf_Lesk_sent_tranformer(text,tokens,vectoriser)
        return np.mean([lesk_word_vectors[w] * self.word2weight[w] for w in tokens], axis=0)

    def transform_sent_glove(self, text):
        # only transforms one sentence using tfidf weighting and GloVe
        tokens = Lemmatizer.lemmatize(text)
        try:
            #return np.average([self.word2vec[w] * self.word2weight[w] for w in tokens if w in self.word2vec] or [np.zeros(self.dim)], axis=0, weights=[self.word2weight[w] for w in tokens if w in self.word2vec])
            return np.mean(
                [self.word2vec[w] * self.word2weight[w] for w in tokens if w in self.word2vec] or [np.zeros(self.dim)],
                axis=0)
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
            res[tkn] = vectoriser.transform_sent_glove(temp_sent)
            word_sense_vector_hash[temp_sent] = res[tkn]
        else:
            res[tkn] = word_sense_vector_hash[temp_sent]

    return res
    pass


def get_list_of_uniques_amd_symptoms_merged(ans_file_path):
    stop_ans = ["Yes", "No", "Not Sure", "All the time", "Front", "Rear", "When turning", ""]
    try:
        if os.path.isfile(ans_file_path):
            ans_data = pd.read_csv(ans_file_path, error_bad_lines=False)
            ans_unique = ans_data['problem_path'].unique()
            amd_unique_symptoms = np.array([sent for sent in ans_unique if sent not in stop_ans])
            amd_unique_symptoms = np.trim_zeros(amd_unique_symptoms)
            return amd_unique_symptoms
    except:
        print("Error in get_list_of_uniques_problems: \n", sys.exc_info()[0], sys.exc_info()[1])
    pass


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
