import numpy as np
import pickle
import pandas as pd
import os
import random
import glob

######### FOLLOWING CODE PROCESS THE SONGS #############
def process_songs(song_data):
    hash_to_id = {}
    counter = 0

    rows = []
    for row in song_data:
        new_row = np.copy(row)
        hash = new_row[0]

        hash_to_id[hash] = counter
        new_row[0] = counter

        if new_row[16] == "minor":
            new_row[16] = 0
        elif new_row[16] == "major":
            new_row[16] = 1


        counter += 1
        rows.append(new_row)

    new_array = np.vstack(rows)
    return new_array, hash_to_id

def pickle_songs(path,file_name_1,file_name_2):
    track_pd_1 = pd.read_csv(path + file_name_1)
    track_data_1 = track_pd_1.values

    track_pd_2 = pd.read_csv(path + file_name_2)
    track_data_2 = track_pd_2.values

    track_data =  np.vstack([track_data_1,track_data_2])

    track_data, dict_track = process_songs(track_data)
    pickle.dump(track_data, open(path + "track_data.pickle", "wb"))
    pickle.dump(dict_track, open(path + "track_dict.pickle", "wb"))

#Pickle the track and make a mapping from haskey to integer which is used when processing the playback track data.
#We replace hask key with int key to save space
#There are two files, simply because the track data came in two files
def process_tracks(path_files, file_name_1, file_name_2):
    pickle_songs(path_files, file_name_1, file_name_2)




###Used for shuffling the produced list of sessions accross files, as the session in the same file seem to have som temporal correlation,
#we want to remove for training
def shuffle_on_file_level(path_to_files, n_files_to_split, path_to_put):
    files = os.listdir(path_to_files)
    random.shuffle(files)

    indexes = np.linspace(0,len(files), n_files_to_split+1, dtype=np.int32)
    counter = 0
    for i in range(len(indexes) - 1):
        files_to_load = files[indexes[i]:indexes[i+1]]
        print(files_to_load)

        sessions = []
        for f in files_to_load:
            print(len(sessions))
            sessions = sessions + pickle.load(open(path_to_files+f,"rb"))
        print(len(sessions))
        random.shuffle(sessions)

        for f in files_to_load:
            os.remove(path_to_files+f)

        ses_indexes = np.linspace(0,len(sessions), len(files_to_load) + 1, dtype=np.int32)
        for j in range(len(ses_indexes) - 1):
            ses = sessions[ses_indexes[j]:ses_indexes[j+1]]
            pickle.dump(ses,open(path_to_put + str(counter) + ".p","wb"))
            counter += 1



import shutil
import time
#Number of shuffles need to be even to end in same folder. n_files_to_split is chosen based on avaible ram, if low ram choose high
def call_shuffle_on_file(path_to_files, path_to_put, n_files_to_split, shuffles):
    #shuffle_on_file_level(path_to_files,n_files_to_split,path_to_put)
    #path_to_files = path_to_put
    #path_to_put = path_to_files
    for i in range(shuffles):
        shuffle_on_file_level(path_to_files, n_files_to_split, path_to_put)
        time.sleep(5)
        temp = path_to_files
        path_to_files = path_to_put
        path_to_put = temp


#used for shuffling the item between the files, assume the directories exists
def shuffle_files(path_to_files, path_to_put, n_files_to_split = 20, shuffles=2):
    call_shuffle_on_file(path_to_files, path_to_put, n_files_to_split,shuffles)


#Code for processing the train data
#takes as argument the path to folder for all the playback tracks, path to put it, path to the dictionary make when processing the tracks
# path to a set of dictionaries which are used for translating the categorical strings to integers and lastly path to where to dump dictionary mapping session hashkey to new int
def process_train(path_to_csv, path_to_processed, path_to_track_dict, path_to_cat_var_dict, path_session_dict):
    '''
    path_to_csv = "../data_2/train_csv/"
    path_to_processed = "../data_2/1/"
    path_to_track_dict = "../data_2/track_data/track_dict.pickle"
    path_to_cat_var_dict = "../data_2/dicts/cat_session_dicts.p"
    path_session_dict = 
    '''

    track_dict = pickle.load(open(path_to_track_dict,"rb"))
    (context_type_dict, hist_user_behavior_reason_start_dict, hist_user_behavior_reason_end) = pickle.load(open(path_to_cat_var_dict,"rb"))

    csv_files = os.listdir(path_to_csv)

    COUNTER = 0
    session_dict = {}
    file_counter = 0
    for csv_file in csv_files:
        session_pd = pd.read_csv(path_to_csv + csv_file)
        session_data = session_pd.values

        rows = []
        for row in session_data:
            new_row = np.copy(row)
            hash_session = new_row[0]

            if hash_session not in session_dict:
                session_dict[hash_session] = COUNTER
                COUNTER = COUNTER + 1

            new_row[0] = session_dict[hash_session]
            new_row[3] = track_dict[new_row[3]]

            #replace categorical variables
            new_row[-3] = context_type_dict[row[-3]]
            new_row[-2] = hist_user_behavior_reason_start_dict[row[-2]]
            new_row[-1] = hist_user_behavior_reason_end[row[-1]]
            rows.append(new_row)

        sessions = []
        current = 0
        while current < len(rows):
            l = rows[current][2]
            sessions.append(rows[current:current + l])
            current += l


        pickle.dump(sessions,open(path_to_processed + str(file_counter) + ".p","wb"))
        file_counter += 1

    pickle.dump(session_dict, open(path_session_dict + "session_hash_dict.p","wb"))

## This is the code for processing test files
#first argument is the path to the test csv files, second is where to put the processed files,
# third is to the track dictionary, and fourth is where to load the dictionary for categorical variables
def process_test_files(test_path,submission_path,path_to_track_dict,path_to_cat_var_dict):
    '''
    data_path = "C:\\Users\\christian\\Dropbox\\phd\\Projects\\spotify\\"
    test_path = data_path + "testing\\"
    submission_path = data_path + "testing_processed\\"
    test_input_logs = sorted(glob.glob(test_path + "log_input_*.csv"))
    test_input_hists = sorted(glob.glob(test_path + "log_prehistory_*.csv"))

    path_to_track_dict = "../data/track_data/track_dict.pickle"
    path_to_cat_var_dict = "..\\data\\dicts\\cat_session_dicts.p"
    '''

    submission_path =  "../_validation_test_proc/"
    test_path = "../_validation_test/"
    path_to_track_dict = "../track_data/track_dict.pickle"
    path_to_cat_var_dict = "../track_data/cat_session_dicts.p"

    test_input_logs = sorted(glob.glob(test_path + "inp_*.csv"))
    test_input_hists = sorted(glob.glob(test_path + "pre_*.csv"))



    #test_input_logs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    #test_input_hists.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    track_dict = pickle.load(open(path_to_track_dict,"rb"))
    (context_type_dict, hist_user_behavior_reason_start_dict, hist_user_behavior_reason_end) = pickle.load(open(path_to_cat_var_dict,"rb"))

    file_counter = 0
    session_dict = {}
    COUNTER = 0
    for i in range(len(test_input_logs)):
        input_log = test_input_logs[i]
        input_hist = test_input_hists[i]
        log = pd.read_csv(input_log,header=None).values
        hist = pd.read_csv(input_hist,header=None).values

        rows_hist = []
        for row in hist:
            new_row = np.copy(row)
            hash_session = new_row[0]

            if hash_session not in session_dict:
                session_dict[hash_session] = COUNTER
                COUNTER += 1

            new_row[0] = session_dict[hash_session]
            new_row[3] = track_dict[new_row[3]]

            #replace categorical variables
            new_row[-3] = context_type_dict[row[-3]]
            new_row[-2] = hist_user_behavior_reason_start_dict[row[-2]]
            new_row[-1] = hist_user_behavior_reason_end[row[-1]]
            rows_hist.append(new_row)


        rows_log = []
        for row in log:
            new_row = np.copy(row)
            hash_session = new_row[0]

            new_row[0] = session_dict[hash_session]
            new_row[1] = track_dict[new_row[1]]

            rows_log.append(new_row)


        current_log = 0
        current_hist = 0
        sessions_hist = []
        sessions_log = []
        while current_hist < len(rows_hist):
            l = rows_hist[current_hist][2]
            l_hist = rows_log[current_log][2]-1


            session_hist = rows_hist[current_hist:current_hist+l_hist]
            session_log = rows_log[current_log:current_log+(l-l_hist)]

            current_hist += l_hist
            current_log += (l-l_hist)

            sessions_hist.append(session_hist)
            sessions_log.append(session_log)


        combined = (sessions_hist,sessions_log)
        pickle.dump(combined,open(submission_path + str(file_counter) + ".p","wb"))
        file_counter += 1

    pickle.dump(session_dict,open(submission_path + "test_session_dict.p" + ".p","wb"))

#for during all the preprocessing, this can take a long time for the full dataset
#Fill the data folder up with the competition data,
#track data into the folder defined in  path_files_track
#train data into the file defined in path_to_train
#test data into the file defined in path_to_test
if __name__ == "__main__":
    #first process the tracks
    path_files_track = "data/track_features/"
    file_name_1 = "tf_mini.csv"
    file_name_2 = "tf_mini_2.csv"
    process_tracks(path_files_track, file_name_1, file_name_2)


    #process the train data
    path_to_train = "data/training_set/"
    path_to_processed_train = "data/training_set_proc/"
    path_to_track_dict = path_files_track + "track_dict.pickle"
    path_to_cat_var_dict = "cat_dict/cat_session_dicts.pickle"
    path_session_dict = ""
    process_train(path_to_train, path_to_processed_train, path_to_track_dict, path_to_cat_var_dict, path_session_dict)


    #OPTIONAL, shuffle the train data
    path_to_put = "data/training_set_proc_shuffled/" #is only used during this function call and emptied before call is over
    shuffle_files(path_to_processed_train, path_to_put, n_files_to_split=3, shuffles=2)


    #process the test data
    path_to_test = "data/test_set"
    path_to_processed_test = "data/test_set_proc"
    #There was no small example of the test files, but is again just putting all the test files into the folder at test_path
    #process_test_files(path_to_test, path_to_processed_test, path_to_track_dict, path_to_cat_var_dict)



