import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



class custom_one_hot_encoder():
    def __init__(self, vec_len):
        self.n_values = vec_len
        pass

    def fit(self):

        pass

    def transform(self):
        pass

def random_forest_implementation():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    # cancer = load_breast_cancer()
    #
    # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    ym_training_data = pd.read_csv(
        "/Users/aratwani/PycharmProjects/NLPProjects/ym_symptoms_decisiontree_training_data.csv")
    X = ym_training_data['symptom'].values
    Y = ym_training_data['job_name'].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(X_train, y_train)

    print('Accuracy on the training subset: {:.3f}'.format(forest.score(X_train, y_train)))
    print('Accuracy on teh test subset : {:.3f}'.format(forest.score(X_test, y_test)))
    pass


def single_decisiontree_implementation():
    ym_training_data = pd.read_csv(
        "/Users/aratwani/PycharmProjects/NLPProjects/ym_symptoms_decisiontree_training_data.csv")
    Y = ym_training_data['symptom'].values

    count_vectoriser = CountVectorizer(decode_error='ignore')
    X = count_vectoriser.fit_transform(ym_training_data['job_name'])
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(Xtrain, Ytrain)

    print('Accuracy on the training subset: ', tree.score(Xtrain, Ytrain))
    print('Accuracy on the training subset: ', tree.score(Xtest, Ytest))
    pass


def sentence_one_hot_encoding_try():
    from sklearn.preprocessing import OneHotEncoder
    ym_training_data = pd.read_csv(
        "/Users/aratwani/PycharmProjects/NLPProjects/ym_symptoms_decisiontree_training_data.csv")
    Y = ym_training_data['symptom'].values
    y_set = [y.lower() for y in set(Y)]
    word_to_id = {token: idx for idx, token in enumerate(y_set)}
    token_ids = [word_to_id[tokens_doc] for tokens_doc in y_set]

    vec = OneHotEncoder(n_values=len(word_to_id))
    X = vec.fit(token_ids)

    print(X.array())


def try1():
    # on hot encoding POC
    from sklearn.preprocessing import OneHotEncoder
    import itertools

    # two example documents
    docs = ["A B C", "B B D"]

    # split documents to tokens
    tokens_docs = [doc.split(" ") for doc in docs]

    # convert list of of token-lists to one flat list of tokens
    # and then create a dictionary that maps word to id of word,
    # like {A: 1, B: 2} here
    all_tokens = itertools.chain.from_iterable(tokens_docs)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

    # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
    token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_docs]

    # convert list of token-id lists to one-hot representation
    vec = OneHotEncoder(n_values=len(word_to_id), sparse=True)
    X = vec.fit(token_ids)

    print(X.toarray())


def main():
    try1()
    pass


main()
