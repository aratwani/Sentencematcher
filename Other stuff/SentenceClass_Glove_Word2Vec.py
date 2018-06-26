import numpy as np
import nltk
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
w2v = {}
with open("glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))for line in lines}


text = "In the eighteenth century it was often convenient to regard man as a clockwork automaton."

text_tokens = nltk.word_tokenize(text)
text_normalised = [word for word in text_tokens if word.isalnum()]


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.items())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w.encode("utf-8")] for w in words if w.encode("utf-8") in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])


def main():
    X = [['Brake Pads Replacement', 'Worn Front Brakes']]
    y = ['Brakes']
    vectoriser = MeanEmbeddingVectorizer(w2v)
    vectoriser.fit(X, y)


    vector_YM = vectoriser.transform("brakes worn out")
    vector_AMD = vectoriser.transform("worn front brakes")
    arr_res  = cosine_similarity(vector_YM, vector_AMD)
    print(vector_YM)
    pass




    pass

main()