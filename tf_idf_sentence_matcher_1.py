import pandas as pd
import sys
import tfidf_vectoriser_library as vec_lib
import nltk
from nltk.wsd import lesk
import numpy as np
from nltk.corpus import wordnet as wn

# 200  dimensions
# diff method
# stemming
# Join automd sentences

# GET TRAINED VECTORS






# first implementation
def tfidf_sent_matcher(symptoms_automd, symptoms_YM, vectoriser):
    res = pd.DataFrame()
    vec_ym_hash = {}
    for vec_amd in symptoms_automd:
        vec_amd_vector = vectoriser.transform_sent(vec_amd)
        for vec_ym in symptoms_YM:
            if vec_ym not in vec_ym_hash:
                vec_ym_vector = vectoriser.transform_sent(vec_ym)
                vec_ym_hash[vec_ym] = vec_ym_vector
            else:
                vec_ym_vector = vec_ym_hash[vec_ym]
            similarity = vec_lib.cosine_similarity_vector(vec_amd_vector, vec_ym_vector)
            if similarity > 0.85:
                print(vec_amd, ',', vec_ym, ",", similarity)
                vec_lib.write_line_to_csv([vec_amd, vec_ym, similarity], ["amd", "ym", "similarity"], "test2.csv")
            pass
        pass


def main():
    w2v = vec_lib.import_glove6b_pretrained_vectors()
    df_automd_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    df_automd_symptoms.dropna()
    symptoms_automd = df_automd_symptoms["answer"].dropna().astype(str)

    df_ym_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/symptoms_YM.csv")
    df_ym_symptoms.dropna()
    symptoms_YM = df_ym_symptoms["Symptoms"].dropna().astype(str)


    # CREATE VECTORISER OBJECT
    vectoriser = vec_lib.TfidfEmbeddingVectorizer(w2v)
    vectoriser.fit(df_automd_symptoms)
    vectoriser.fit(df_ym_symptoms)

    # tfidf_Lesk_sent_tranformer("Brake pedal is pulsating")
    # print("YadaYadaYada- test text")

    X2 = vectoriser.transform_sent("Brake pedal is pulsating".lower())
    print(X2)
    Y1 = vectoriser.transform_sent_1("Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower())
    tfidf_res1 = vec_lib.cosine_similarity_vector(X2, Y1)
    print("X1 - Y1", tfidf_res1)

    Z1 = vectoriser.transform_sent_1("texas is hot")
    Z2 = vectoriser.transform_sent("texas is cold")
    tfidf_res3 = vec_lib.cosine_similarity_vector(Z1, Z2)
    print("Z1 - Z2", tfidf_res3)
    tfidf_res2 = vec_lib.cosine_similarity_vector(X2, Z2)
    print(tfidf_res2)
    pass


main()