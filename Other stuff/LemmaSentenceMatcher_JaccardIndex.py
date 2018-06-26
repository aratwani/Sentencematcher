import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
from nltk.corpus import wordnet
import string
from nltk.tokenize import WordPunctTokenizer


# refer article:  https://bommaritollc.com/2014/06/12/fuzzy-match-sentences-python/
# Get default English stopwords and extend with punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')


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


# Create tokenizer and stemmer
tokenizer = WordPunctTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def is_ci_token_stopword_lemma_match(a, b, threshold):
    """Check if a and b are matches."""
    pos_a = map(get_wordnet_pos, nltk.pos_tag(tokenizer.tokenize(a)))
    pos_b = map(get_wordnet_pos, nltk.pos_tag(tokenizer.tokenize(b)))
    lemmae_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_a \
                    if token.lower().strip(string.punctuation) not in stopwords]
    lemmae_b = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_b \
                    if token.lower().strip(string.punctuation) not in stopwords]
    index = len(set(lemmae_a).intersection(set(lemmae_b)))/float(len(set(lemmae_a).union(set(lemmae_b))))
    return (index>=threshold)


def main():
    x = is_ci_token_stopword_lemma_match("In the eighteenth century it was often convenient to regard man as a clockwork automaton.", "The eighteenth century was characterized by man as a clockwork automaton.", 0.5)
    print(x)
    pass


main()
