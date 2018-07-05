import os
import sys
import numpy as np
import pandas as pd


def get_parent_path(slice_df, temp_parent_id):
    if temp_parent_id == 1 or temp_parent_id==2:
        return ''
    for i, row in slice_df.iterrows():
        if row[0] == temp_parent_id:
            return get_parent_path(slice_df,row[1]) + "--" + str(row[3])

def get_list_of_uniques_problems(ans_file_path):
    try:
        if os.path.isfile(ans_file_path):
            ans_data = pd.read_csv(ans_file_path, error_bad_lines=False)
            leaf_ids = ans_data[ans_data.difficulty.notnull()]
            leaf_indexes = leaf_ids.index
            #print(leaf_indexes)
            curr_parent_index = 0;
            # for i, row in enumerate(leaf_ids.values):
            #     print(leaf_ids.index[i])
            #     # temp_id, temp_parent_id, temp_answer, temp_difficulty = row
            #     print(row)
            for i,row in ans_data.iterrows():
                # check if its top level parent row
                if row[1] == 1 or row[1] == 2:
                    curr_parent_index = ans_data.index[i]
                    continue
                #print("child : " + str(row[0]) + "  parent id: "+str(ans_data.iloc[curr_parent_index][0]))

                # check if its leaf node
                if ans_data.index[i] in leaf_indexes:
                    slice_df = ans_data.iloc[curr_parent_index:i]
                    # print(slice_df)
                    # one sliced, trace back the parent
                    temp_parent_id = row[1]
                    leaf_text = row[3] if isinstance(row[3],str) else ''
                    try:
                        print(get_parent_path(slice_df, temp_parent_id) + "--" + leaf_text)
                    except:
                        print("Error in get_list_of_uniques_problems: \n", sys.exc_info()[0], sys.exc_info()[1])



    except:
        print("Error in get_list_of_uniques_problems: \n", sys.exc_info()[0], sys.exc_info()[1])
    pass


get_list_of_uniques_problems("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")