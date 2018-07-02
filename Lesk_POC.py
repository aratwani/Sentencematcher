import nltk
import pandas as pd
import numpy as np
import tfidf_vectoriser_library as lib
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from pywsd.similarity import max_similarity
import Lesk_Vectorizer_Library as lesk_vec_lib
import tfidf_vectoriser_library as vec_lib
import Lemmatizer


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
    for ss in wn.synsets('brake'):
        print(ss, ss.definition())
    pass


def sent_matcher_tfidf_Lesk():
    df_ym_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/symptoms_YM.csv")
    df_ym_symptoms.dropna()
    symptoms_YM = df_ym_symptoms["Symptoms"].replace('', np.nan).dropna().astype(str)

    df_automd_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    df_automd_symptoms.dropna()
    symptoms_automd = df_automd_symptoms["answer"].replace('', np.nan).dropna().astype(str)

    # GET TRAINED VECTORS
    w2v = vec_lib.import_glove6b_pretrained_vectors()
    # CREATE VECTORISER OBJECT
    vectoriser = lesk_vec_lib.TfidfEmbeddingVectorizer_Lesk(w2v)
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_automd_symptoms))
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_ym_symptoms))
    vec_ym_hash = {}
    output_file = "lesk_test2.csv"
    for vec_amd in symptoms_automd:
        vec_amd_vector = vectoriser.transform_sent_1(vec_amd, vectoriser)
        sent_vector_dict = {}
        for vec_ym in symptoms_YM:
            if vec_ym not in vec_ym_hash:
                vec_ym_vector = vectoriser.transform_sent_1(vec_ym, vectoriser)
                vec_ym_hash[vec_ym] = vec_ym_vector
            else:
                vec_ym_vector = vec_ym_hash[vec_ym]
            similarity = vec_lib.cosine_similarity_vector(vec_amd_vector, vec_ym_vector)
            sent_vector_dict[vec_ym] = similarity
            print(vec_amd, vec_ym, similarity)
            pass
        max_similarity_vector = max(sent_vector_dict, key=sent_vector_dict.get)
        vec_lib.write_line_to_csv([vec_amd, max_similarity_vector, sent_vector_dict[max_similarity_vector]],
                                  ["amd", "ym", "similarity"], output_file)
        print(vec_amd, ',', max_similarity_vector, ",", sent_vector_dict[max_similarity_vector])

        pass
    pass

def main():
    # test_lesk()

    sent_matcher_tfidf_Lesk()
    pass

main()