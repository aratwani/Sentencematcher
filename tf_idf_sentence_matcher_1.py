import pandas as pd
import sys
import tfidf_vectoriser_library as vec_lib
import Lemmatizer
from pywsd.utils import lemmatize
import nltk

from nltk.wsd import lesk
import numpy as np
from nltk.corpus import wordnet as wn

# 200  dimensions
# diff method
# stemming
# Join automd sentences


def main():
    # w2v = vec_lib.import_glove6b_pretrained_vectors()
    # df_automd_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    # df_automd_symptoms.dropna()
    # symptoms_automd = df_automd_symptoms["answer"].dropna().astype(str)
    #
    # df_ym_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/symptoms_YM.csv")
    # df_ym_symptoms.dropna()
    # symptoms_YM = df_ym_symptoms["Symptoms"].dropna().astype(str)
    #
    # # CREATE VECTORISER OBJECT
    # vectoriser = vec_lib.TfidfEmbeddingVectorizer_Lesk(w2v)
    # print(df_automd_symptoms)
    # vectoriser.fit(df_automd_symptoms)
    # vectoriser.fit(df_ym_symptoms)
    #
    #
    # X2 = vectoriser.transform_sent("Brake pedal is pulsating".lower())
    # print(X2)
    # Y1 = vectoriser.transform_sent_1("Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower())
    # tfidf_res1 = vec_lib.cosine_similarity_vector(X2, Y1)
    # print("X1 - Y1", tfidf_res1)
    #
    # Z1 = vectoriser.transform_sent_1("texas is hot")
    # Z2 = vectoriser.transform_sent("texas is cold")
    # tfidf_res3 = vec_lib.cosine_similarity_vector(Z1, Z2)
    # print("Z1 - Z2", tfidf_res3)
    # tfidf_res2 = vec_lib.cosine_similarity_vector(X2, Z2)
    # print(tfidf_res2)


    pass


main()