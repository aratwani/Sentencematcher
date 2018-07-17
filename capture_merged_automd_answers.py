import os
import sys
import pandas as pd
import tfidf_vectoriser_library as vec_lib

# this code is imply to find all the paths followed by the automd crawler

def get_parent_path(slice_df, temp_parent_id):
    for i, row in slice_df.iterrows():
        if row[0] == temp_parent_id:
            if row[1] == 1 or row[1] == 2:
                return str(row[3])
            else:
                return get_parent_path(slice_df,row[1]) + "--" + str(row[3])


def get_list_of_uniques_problems(ans_file_path):
    try:
        opfilename = 'automd_answers_merged' + vec_lib.get_timestamp() + '.csv'
        if os.path.isfile(ans_file_path):
            ans_data = pd.read_csv(ans_file_path, error_bad_lines=False)
            leaf_ids = ans_data[ans_data.difficulty.notnull()]
            leaf_indexes = leaf_ids.index
            curr_parent_index = 0;
            for i,row in ans_data.iterrows():
                # check if its top level parent row
                if row[1] == 1 or row[1] == 2:
                    curr_parent_index = ans_data.index[i]
                    continue
                # check if its leaf node
                if ans_data.index[i] in leaf_indexes:
                    slice_df = ans_data.iloc[curr_parent_index:i]
                    # one sliced, trace back the parent
                    temp_parent_id = row[1]
                    leaf_text = row[3] if isinstance(row[3],str) else ''
                    try:
                        # temp_path = get_parent_path(slice_df, temp_parent_id) + "--" + leaf_text

                        # temp_path = get_parent_path(slice_df, temp_parent_id)

                        temp_path = leaf_text
                        print(temp_path)
                        vec_lib.write_line_to_csv([temp_path], ['problem_path'], opfilename)

                        # for txt in temp_path.split('--'):
                        #     vec_lib.write_line_to_csv([txt,leaf_text], ['problem_symptom','service'], opfilename)
                    except:
                        print("Error in get_list_of_uniques_problems: \n", sys.exc_info()[0], sys.exc_info()[1])



    except:
        print("Error in get_list_of_uniques_problems: \n", sys.exc_info()[0], sys.exc_info()[1])
    pass


get_list_of_uniques_problems("/Users/aratwani/PycharmProjects/NLPProjects/answers2018-06-12.csv")