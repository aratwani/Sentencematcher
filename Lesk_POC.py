import nltk
import pandas as pd
import numpy as np
import tfidf_vectoriser_library as lib
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from pywsd.similarity import max_similarity
import spacy
from pywsd.utils import lemmatize


# made changes to similarity.py

def test_lesk():
    #text = "Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower()
    text = "push brake pedal on the vehicle"
    tokens = lib.tokenize(text)
    pos_tagged = nltk.pos_tag(tokens, tagset='universal')
    print(pos_tagged)
    if pos_tagged[1][1]=='VERB':
        temp_pos = 'v'
    else:
        temp_pos = 'n'
    print(lesk(tokens, pos_tagged[1][0], pos=temp_pos).definition())
    print(max_similarity(text, 'brake', pos=temp_pos).definition())
    for ss in wn.synsets('pedal'):
        print(ss, ss.definition())
    pass


def main():
    # test_lesk()
    # nlp = spacy.load('en')
    # doc1 = nlp("Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower())
    # doc2 = nlp("Brake pedal is pulsating".lower())
    # print(doc1.similarity(doc2))
    str1 = "Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower()
    str2 = "Brake pedal is pulsating".lower()
    tkn1 = lib.tokenize(str1)
    tkn2 = lib.tokenize(str2)
    tkn1 = tkn1 + tkn2
    for tkn in tkn1:
        print(tkn, " - ", lemmatize(tkn))

    pass

main()