import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm as svm
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt


w2v = {}
with open("glove.6B.50d.txt", "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode('utf-8')
        nums=np.array(parts[1:], dtype=np.float32)
        w2v[word] = nums



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
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


df = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/spam.csv", encoding='ISO-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df.columns = ['labels', 'data']

df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].as_matrix()

# count_vectoriser = CountVectorizer(decode_error='ignore')
# X = count_vectoriser.fit_transform(df['data'])


tfidf_Glove_vectoriser = TfidfEmbeddingVectorizer(w2v)
tfidf_Glove_vectoriser.fit(df['data'])
X = tfidf_Glove_vectoriser.transform(df['data'])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.33)

# tfidf = TfidfVectorizer(decode_error='ignore')
# X = tfidf.fit_transform(df['data'])


model = svm.SVC()
model.fit(Xtrain, Ytrain)

print("train score :", model.score(Xtrain,Ytrain))
print("test score : ", model.score(Xtest, Ytest))

df['predictions'] = model.predict(X)

# things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
  print(msg)

# things that should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
  print(msg)








