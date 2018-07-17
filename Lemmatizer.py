import nltk
import string
from nltk.corpus import wordnet
import tfidf_vectoriser_library

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
tokenizer = tfidf_vectoriser_library
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def get_wordnet_pos(pos_tag):
    if pos_tag[1].startswith('J'):
        return (pos_tag[0], wordnet.ADJ)
    elif pos_tag[1].startswith('V'):
        return (pos_tag[0], wordnet.VERB)
    elif pos_tag[1].startswith('N'):
        return (pos_tag[0], wordnet.NOUN)
    elif pos_tag[1].startswith('R'):
        return (pos_tag[0], wordnet.ADV)
    else:
        return (pos_tag[0], wordnet.NOUN)


# Lemmatizes a sentence
def lemmatize(a):
    try:
        a = a.lower()
        a = a.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        pos_a = map(get_wordnet_pos, nltk.pos_tag(tokenizer.tokenize(a)))
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        lemmae_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_a if token.lower().strip(string.punctuation) not in stopwords]
        return lemmae_a
    except:
        print("Error in custom Lemmatizer for sent", a)

def lemmatize_data_frame(data_frame):
    res = []
    for symp in data_frame:
        res.append(lemmatize(symp))
    return res
