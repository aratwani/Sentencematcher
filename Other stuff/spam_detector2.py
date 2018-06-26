import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
      msg = msg.lower()
      words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()



df = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/spam.csv", encoding='ISO-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df.columns = ['labels', 'data']

df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].as_matrix()

count_vectoriser = CountVectorizer(decode_error='ignore')
X = count_vectoriser.fit_transform(df['data'])

# tfidf = TfidfVectorizer(decode_error='ignore')
# X = tfidf.fit_transform(df['data'])


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.33)

model = MultinomialNB()
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








