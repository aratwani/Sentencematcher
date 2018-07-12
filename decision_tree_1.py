import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from numpy import argmax
import numpy as np


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


class custom_one_hot_encoder():
    def __init__(self):
        self.n_values = 0
        self.sent_int_dict = {}
        self.int_sent_dict = {}
        pass

    def fit(self, X):
        # create one hot encoded vectors for each of the job_names and saves it in the sent_vec_dict
        y_set = [x.lower() for x in set(X)]
        self.n_values = len(y_set)
        i = 0
        for sent in y_set:
            # temp_vector = np.zeros((self.n_values,), dtype=int)
            # temp_vector[i] = 1
            # self.sent_vec_dict[sent] = temp_vector
            self.sent_int_dict[sent] = i
            self.int_sent_dict[i] = sent
            i += 1
            pass
        pass

    def transform_one_hot(self, data):
        # converts the symptoms to one hot vectors based onto the job name they are mapped to
        jobname_vectors = {}
        for index,row in data.iterrows():
            symptom = row['symptom'].lower()
            job_name = row['job_name'].lower()
            if job_name in jobname_vectors:
                jobname_vectors[job_name][self.sent_int_dict[symptom]] += 1
                pass
            else:
                temp_vector = np.zeros((self.n_values,), dtype=int)
                temp_vector[self.sent_int_dict[symptom]] = 1
                jobname_vectors[job_name] = temp_vector
        return jobname_vectors
        pass

    def transform_int_array(self, data):
        # converts the symptoms to one hot vectors based onto the job name they are mapped to
        jobname_vectors = {}
        for index, row in data.iterrows():
            symptom = row['symptom'].lower()
            job_name = row['job_name'].lower()
            if job_name in jobname_vectors:
                jobname_vectors[job_name].append(self.sent_int_dict[symptom])
                pass
            else:
                temp_vector = [self.sent_int_dict[symptom]]
                jobname_vectors[job_name] = temp_vector
        return jobname_vectors
        pass


def sentence_one_hot_encoding():
    from sklearn.preprocessing import OneHotEncoder
    ym_training_data = pd.read_csv(
        "/Users/aratwani/PycharmProjects/NLPProjects/ym_symptoms_decisiontree_training_data.csv")
    ym_training_data = ym_training_data.head(100)
    Y = ym_training_data['symptom'].values
    vec_encoder = custom_one_hot_encoder();
    vec_encoder.fit(Y)
    print(vec_encoder.sent_int_dict)
    dict = vec_encoder.transform_one_hot(ym_training_data)
    # dict2 = vec_encoder.transform_int_array(ym_training_data)
    key = 'Engine Mount'.lower()
    print(key, dict[key])
    for idx, val in enumerate(dict[key]):
        if val == 1:
            print(idx, ". ", vec_encoder.int_sent_dict[idx])
    return dict, vec_encoder


def sent_encoding_random_forest():
    ret_value = sentence_one_hot_encoding()
    data = ret_value[0]
    x, y = [], []
    for i in data:
        y.append(i)
        x.append(data[i])
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.1)
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(Xtrain, Ytrain)

    for key in data.keys():
        print(key, "\n", [val for val in data[key]])

    print('Accuracy on the training subset: ', tree.score(Xtrain, Ytrain))
    # print('Accuracy on the training subset: ', tree.score(Xtest, Ytest))
    implementing_and_visualizing_classification_tree(tree, list(ret_value[1].int_sent_dict.values()))
    pass


def implementing_and_visualizing_classification_tree(clf, feature_names):
    import pydotplus
    from sklearn import tree
    import collections

    # prepare the data
    # X = [[180, 15, 0],
    #      [177, 42, 0],
    #      [136, 35, 1],
    #      [174, 65, 0],
    #      [141, 28, 1]]
    # Y = ['man', 'woman', 'woman', 'man', 'woman']
    #
    # data_feature_names = ['height', 'hair length', 'voice pitch']
    #
    # # Train the classifier
    #
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X, Y)
    dot_data = tree.export_graphviz(clf, feature_names=feature_names, out_file=None, filled=True, rounded=True)
    #dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png('tree_3.png')
    pass

def main():
    sent_encoding_random_forest()
    #implementing_and_visualizing_classification_tree()
    pass


main()
