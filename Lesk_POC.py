import nltk
import pandas as pd
import numpy as np
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from pywsd.similarity import max_similarity
import Lesk_Vectorizer_Library as lesk_vec_lib
import Lemmatizer
import LemmaSentenceMatcher_JaccardIndex as lemma_ji


# made changes to similarity.py

def test_lesk():
    #text = "Pulsation- Fluctuation of the brake pedal when the brakes are applied".lower()
    text = "push brake pedal on the vehicle"
    tokens = lesk_vec_lib.tokenize(text)
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

    df_automd_symptoms = lesk_vec_lib.get_list_of_uniques_amd_symptoms_merged(
        "/Users/aratwani/PycharmProjects/NLPProjects/automd_answers_merged.csv")
    # df_automd_symptoms.dropna()
    # symptoms_automd = df_automd_symptoms["answer"].replace('', np.nan).dropna().astype(str)
    symptoms_automd = df_automd_symptoms

    # GET TRAINED VECTORS
    w2v = lesk_vec_lib.import_glove6b_pretrained_vectors()
    # CREATE VECTORISER OBJECT
    vectoriser = lesk_vec_lib.TfidfEmbeddingVectorizer_Lesk(w2v)
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_automd_symptoms))
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_ym_symptoms))
    vec_ym_lesk_hash = {}
    vec_amd_lesk_hash = {}
    output_file = "lesk_test_merged_2.csv"

    for vec_amd in symptoms_automd:
        if vec_amd not in vec_amd_lesk_hash:
            vec_amd_vector = vectoriser.transform_sent_lesk(vec_amd, vectoriser)
            vec_amd_lesk_hash[vec_amd] = vec_amd_vector
        else:
            vec_amd_vector = vec_amd_lesk_hash[vec_amd]
        sent_lesk_similarity_dict = {}
        sent_jci_similarity_dict = {}
        for vec_ym in symptoms_YM:
            if vec_ym not in vec_ym_lesk_hash:
                vec_ym_vector = vectoriser.transform_sent_lesk(vec_ym, vectoriser)
                vec_ym_lesk_hash[vec_ym] = vec_ym_vector
            else:
                vec_ym_vector = vec_ym_lesk_hash[vec_ym]
            similarity = lesk_vec_lib.cosine_similarity_vector(vec_amd_vector, vec_ym_vector)
            if similarity > 0.7:
                sent_lesk_similarity_dict[vec_ym] = similarity
                jaccard_index = lemma_ji.lemma_match_jaccard_index(vec_ym, vec_amd)
                sent_jci_similarity_dict[vec_ym] = jaccard_index
            print(vec_amd, vec_ym, similarity)
            pass
        if len(sent_jci_similarity_dict):
            max_jaccard_index = max(sent_jci_similarity_dict, key=sent_jci_similarity_dict.get)
            lesk_vec_lib.write_line_to_csv([vec_amd, max_jaccard_index, sent_lesk_similarity_dict[max_jaccard_index]],
                                      ["amd", "ym", "similarity"], output_file)
            print(vec_amd, ',', max_jaccard_index, ",", sent_lesk_similarity_dict[max_jaccard_index])
            # max_similarity_vector = max(sent_lesk_similarity_dict, key=sent_lesk_similarity_dict.get)
            # lesk_vec_lib.write_line_to_csv([vec_amd, max_similarity_vector, sent_lesk_similarity_dict[max_similarity_vector]],
            #                           ["amd", "ym", "similarity"], output_file)
            # print(vec_amd, ',', max_similarity_vector, ",", sent_lesk_similarity_dict[max_similarity_vector])
        else:
            lesk_vec_lib.write_line_to_csv([vec_amd, "", 0], ["amd", "ym", "similarity"], output_file)

        pass
    pass


def sent_matcher_lesk_glove_jaccard(ym_symptoms_path, amd_symptoms_path):
    # define threshold for both the algorithms
    lesk_threshold = 0.89
    glove_threshlold = 0.65

    df_ym_symptoms = pd.read_csv(ym_symptoms_path)
    df_ym_symptoms.dropna()
    symptoms_YM = df_ym_symptoms["Symptoms"].replace('', np.nan).dropna().astype(str)

    df_automd_symptoms = lesk_vec_lib.get_list_of_uniques_amd_symptoms_merged(amd_symptoms_path)
    symptoms_automd = df_automd_symptoms

    # GET TRAINED VECTORS
    # w2v = lesk_vec_lib.import_glove6b_pretrained_vectors()
    w2v = lesk_vec_lib.import_word2vec_vectors()
    # CREATE VECTORISER OBJECT
    vectoriser = lesk_vec_lib.TfidfEmbeddingVectorizer_Lesk(w2v)
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_automd_symptoms))
    vectoriser.fit(Lemmatizer.lemmatize_data_frame(df_ym_symptoms))

    vec_ym_lesk_hash = {}
    vec_amd_lesk_hash = {}
    vec_ym_glove_hash = {}
    vec_amd_glove_hash = {}
    output_file = "benchmark_file_w2v_1.csv"

    for vec_amd in symptoms_automd:
        try:
            if vec_amd not in vec_amd_lesk_hash:
                vec_amd_lesk_vector = vectoriser.transform_sent_lesk(vec_amd, vectoriser)
                vec_amd_lesk_hash[vec_amd] = vec_amd_lesk_vector
            else:
                vec_amd_lesk_vector = vec_amd_lesk_hash[vec_amd]
            sent_lesk_similarity_dict = {}
            for vec_ym in symptoms_YM:
                try:
                    if vec_ym not in vec_ym_lesk_hash:
                        vec_ym_lesk_vector = vectoriser.transform_sent_lesk(vec_ym, vectoriser)
                        vec_ym_lesk_hash[vec_ym] = vec_ym_lesk_vector
                    else:
                        vec_ym_lesk_vector = vec_ym_lesk_hash[vec_ym]
                    similarity_lesk = lesk_vec_lib.cosine_similarity_vector(vec_amd_lesk_vector, vec_ym_lesk_vector)
                    # if similarity_lesk > lesk_threshold:
                    sent_lesk_similarity_dict[vec_ym] = similarity_lesk
                    print(vec_amd, vec_ym, similarity_lesk)
                    pass
                except:
                    print("Error occured in Lesk vectorisation for amd: " + vec_amd + " and ym: " + vec_ym)
                pass
            max_lesk_similarity_vector = max(sent_lesk_similarity_dict, key=sent_lesk_similarity_dict.get)
            max_lesk_similarity_value = sent_lesk_similarity_dict[max_lesk_similarity_vector]
            # use standard glove + tfidf + Jaccard index
            print("***--******--***match not found using lesk, proceeding with Glove and Jaccard***--******--***")
            if vec_amd not in vec_amd_glove_hash:
                vec_amd_glove_vector = vectoriser.transform_sent_glove(vec_amd)
                vec_amd_glove_hash[vec_amd] = vec_amd_glove_vector
            else:
                vec_amd_glove_vector = vec_amd_glove_hash[vec_amd]
            sent_glove_similarity_dict = {}
            sent_jci_similarity_dict = {}
            for vec_ym in symptoms_YM:
                try:
                    if vec_ym not in vec_ym_glove_hash:
                        vec_ym_vector = vectoriser.transform_sent_glove(vec_ym)
                        vec_ym_glove_hash[vec_ym] = vec_ym_vector
                    else:
                        vec_ym_vector = vec_ym_glove_hash[vec_ym]
                    similarity_glove = lesk_vec_lib.cosine_similarity_vector(vec_amd_glove_vector, vec_ym_vector)
                    if similarity_glove > glove_threshlold:
                        jaccard_index_glove = lemma_ji.lemma_match_jaccard_index(vec_ym, vec_amd)
                        sent_jci_similarity_dict[vec_ym] = jaccard_index_glove
                    sent_glove_similarity_dict[vec_ym] = similarity_glove
                    print(vec_amd, vec_ym, similarity_glove)
                except:
                    print("Error occured in Glove vectorisation for amd: " + vec_amd + " and ym: " + vec_ym)
                pass
            # if len(sent_jci_similarity_dict):
            max_jaccard_index_glove_vector = max(sent_jci_similarity_dict, key=sent_jci_similarity_dict.get)
            max_jaccard_index_glove_value = sent_glove_similarity_dict[max_jaccard_index_glove_vector]
            max_glove_vector = max(sent_glove_similarity_dict, key=sent_glove_similarity_dict.get)
            max_glove_value = sent_glove_similarity_dict[max_glove_vector]
            lesk_vec_lib.write_line_to_csv([vec_amd, max_lesk_similarity_vector, max_lesk_similarity_value, max_glove_vector, max_glove_value, max_jaccard_index_glove_vector, max_jaccard_index_glove_value],
                                      ["amd", "ym_lesk", "lesk_similarity", "tfidf_glove_ym",
                                       "tfidf_glove_similarity", "glove_ji_ym", "golve_ji_similarity"], output_file)
            print(vec_amd, ',', max_jaccard_index_glove_vector, ",", max_jaccard_index_glove_value)
            pass
            pass
        except:
            print("Error: the following statement will be skipped! : ", vec_amd)
        pass
    pass


def main():
    # test_lesk()

    sent_matcher_lesk_glove_jaccard("/Users/aratwani/PycharmProjects/NLPProjects/symptoms_YM.csv", "/Users/aratwani/PycharmProjects/NLPProjects/automd_answers_merged2018-07-10T16-37-58-32.csv")
    pass

main()