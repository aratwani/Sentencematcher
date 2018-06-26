from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('spambase.data').values
np.random.shuffle(data)


X = data[:, :48]
Y = data[:, -1]

xtrain = X[:-100,]
ytrain = Y[:-100,]

xtest = X[-100:,]
ytest = Y[-100:,]

model = MultinomialNB()
model.fit(xtrain, ytrain)

print("Accuracy:" + str(round(model.score(xtest, ytest), 4)))

from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier()
adaboost.fit(xtrain, ytrain)
print("Adaboost Accuracy:" + str(round(adaboost.score(xtest, ytest), 4)))

