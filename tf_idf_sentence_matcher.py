import pandas as pd
import sys
import tfidf_vectoriser_library as vec_lib
import numpy as np
import Lemmatizer
import LemmaSentenceMatcher_JaccardIndex as lemma_ji
# 200  dimensions - done
# diff

# stemming -
# Join automd sentences


def sent_matcher_tfidf_GloVe():
    df_ym_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/symptoms_YM.csv")
    df_ym_symptoms.dropna()
    symptoms_YM = df_ym_symptoms["Symptoms"].replace('', np.nan).dropna().astype(str)

    df_automd_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    df_automd_symptoms.dropna()
    symptoms_automd = df_automd_symptoms["answer"].replace('', np.nan).dropna().astype(str)
    # GET TRAINED VECTORS
    w2v = vec_lib.import_glove6b_pretrained_vectors()
    # CREATE VECTORISER OBJECT
    vectoriser = vec_lib.TfidfEmbeddingVectorizer(w2v)
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_automd_symptoms))
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_ym_symptoms))

    vec_ym_hash = {}
    for vec_amd in symptoms_automd:
        vec_amd_vector = vectoriser.transform_sent(vec_amd)
        sent_vector_dict = {}
        for vec_ym in symptoms_YM:
            if vec_ym not in vec_ym_hash:
                vec_ym_vector = vectoriser.transform_sent(vec_ym)
                vec_ym_hash[vec_ym] = vec_ym_vector
            else:
                vec_ym_vector = vec_ym_hash[vec_ym]
            similarity = vec_lib.cosine_similarity_vector(vec_amd_vector, vec_ym_vector)
            sent_vector_dict[vec_ym] = similarity
            pass
        max_similarity_vector = max(sent_vector_dict, key=sent_vector_dict.get)
        output_file = "test5.csv"
        if sent_vector_dict[max_similarity_vector] > 0.75:
            vec_lib.write_line_to_csv([vec_amd, max_similarity_vector, sent_vector_dict[max_similarity_vector]], ["amd", "ym", "similarity"], output_file)
            print(vec_amd, ',', max_similarity_vector, ",", sent_vector_dict[max_similarity_vector])
        else:
            vec_lib.write_line_to_csv([vec_amd, "", 0], ["amd", "ym", "similarity"], output_file)
        pass
    pass


def sent_matcher_tfidf_GloVe_1():
    df_ym_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/symptoms_YM.csv")
    df_ym_symptoms.dropna()
    symptoms_YM = df_ym_symptoms["Symptoms"].replace('', np.nan).dropna().astype(str)

    df_automd_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    df_automd_symptoms.dropna()
    symptoms_automd = df_automd_symptoms["answer"].replace('', np.nan).dropna().astype(str)

    # GET TRAINED VECTORS
    w2v = vec_lib.import_glove6b_pretrained_vectors()
    # CREATE VECTORISER OBJECT
    vectoriser = vec_lib.TfidfEmbeddingVectorizer(w2v)
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_automd_symptoms))
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_ym_symptoms))
    vec_ym_hash = {}
    output_file = "test7.csv"
    for vec_amd in symptoms_automd:
        vec_amd_vector = vectoriser.transform_sent(vec_amd)
        sent_vector_dict = {}
        sent_jci_dict = {}
        for vec_ym in symptoms_YM:
            if vec_ym not in vec_ym_hash:
                vec_ym_vector = vectoriser.transform_sent(vec_ym)
                vec_ym_hash[vec_ym] = vec_ym_vector
            else:
                vec_ym_vector = vec_ym_hash[vec_ym]
            similarity = vec_lib.cosine_similarity_vector(vec_amd_vector, vec_ym_vector)
            if similarity > 0.65:
                jaccard_index = lemma_ji.lemma_match_jaccard_index(vec_ym, vec_amd)
                sent_jci_dict[vec_ym] = jaccard_index
                sent_vector_dict[vec_ym] = similarity
            pass
        if len(sent_jci_dict):
            max_jaccard_index = max(sent_jci_dict, key=sent_jci_dict.get)
            max_similarity_vector = max(sent_vector_dict, key=sent_vector_dict.get)
            vec_lib.write_line_to_csv([vec_amd, max_jaccard_index, sent_vector_dict[max_jaccard_index]],
                                      ["amd", "ym", "similarity"], output_file)
            print(vec_amd, ',', max_jaccard_index, ",", sent_vector_dict[max_jaccard_index])
        else:
            vec_lib.write_line_to_csv([vec_amd, "", 0], ["amd", "ym", "similarity"], output_file)
    pass


def main():
    sent_matcher_tfidf_GloVe_1()

    pass


main()