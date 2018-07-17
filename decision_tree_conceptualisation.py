import pandas as pd
import numpy as np
import nltk
l1_what = ['smell', 'hear', 'see', 'feel']
l2_when = ['all the time','starting','stopping','driving','turning']
l3_where = ['inside', 'outside', 'rear', 'front', 'under']
l1_1_hear = ['knock', 'rattle', 'hiss', 'squeal', 'chirp', 'clunk', 'tap']
l1_1_see = ['smoke','leak/puddle','warning light']  # removed 'flat tire'
l1_1_feel = ['vibrates', 'drifts', 'leans', 'sways',
             'rapidly shimmies']


class node:
    l1_vec = []
    l2_vec = []
    text = ""


def main():
    symptom_data = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/ym_symptoms_decisiontree_training_data.csv")
    for idx, row in symptom_data.iterrows():
        tokens = nltk.word_tokenize(row['symptom'])
        print(nltk.pos_tag(tokens, tagset='universal', lang='eng'))
    pass


main()