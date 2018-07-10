

class TfidfEmbeddingVectorizer(object):
    def __init__(self):
        pass

    def fit(self, X):

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

def main():
    pass

main()