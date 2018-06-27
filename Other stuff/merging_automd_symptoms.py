import pandas as pd
import numpy as np
import os
id_ans_dict = {}


def get_list_of_uniques_problems(ans_file_path):
    try:
        if os.path.isfile(ans_file_path):
            ans_data = pd.read_csv(ans_file_path, error_bad_lines=False)
            ans_id_unique = ans_data['id'].unique()
            ans_parentid_unique = ans_data['parent_id'].unique()
            ans_leaf_problems = np.setdiff1d(ans_id_unique, ans_parentid_unique)
            ans_leaf_problems = np.trim_zeros(ans_leaf_problems)
            return ans_leaf_problems
    except:
        print("Error in get_list_of_uniques_problems: \n", sys.exc_info()[0], sys.exc_info()[1])
    pass


def main():
    df_automd_symptoms = pd.read_csv("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    leaf_nodes = get_list_of_uniques_problems("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")
    child_parent_dict = {}
    for item, row in df_automd_symptoms.iterrows():
        #print(row["id"], row["parent_id"], row["answer"])
        if row["id"] not in id_ans_dict:
            id_ans_dict[row["id"]] = row["answer"]
        if row["id"] not in leaf_nodes:
            child_parent_dict[row["id"]] = row["parent_id"]

    for item, row in df_automd_symptoms.iterrows():
        # found a leaf node traverse to parent
        if row["id"] in leaf_nodes:
            temp_text = id_ans_dict[row["id"]] + "-" + get_parent_path(row["parent_id"], id_ans_dict, child_parent_dict)
            print(temp_text)
            pass
        pass

    print(len(id_ans_dict))
    pass


def get_parent_path(parent_id, id_ans_dict, child_parent_dict):
    if str(parent_id) != '1' and str(parent_id) != '2':
        return get_parent_path(child_parent_dict[parent_id], id_ans_dict, child_parent_dict) + "-" + id_ans_dict[parent_id]
    else:
        return ""
    pass


main()