import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def random_forest_implementation():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    # cancer = load_breast_cancer()
    #
    # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    ym_training_data = pd.read_csv(
        "/Users/aratwani/PycharmProjects/NLPProjects/ym_symptoms_decisiontree_training_data.csv")
    Y = ym_training_data['symptom'].values

    count_vectoriser = CountVectorizer(decode_error='ignore')
    X = count_vectoriser.fit_transform(ym_training_data['job_name'])

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


def main():
    random_forest_implementation()
    pass


main()
