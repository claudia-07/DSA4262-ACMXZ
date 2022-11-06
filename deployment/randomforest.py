"""
Random Forest script for predictions
To use:
cd deployment
python randomforest.py -i A549_rep5_run1.csv

"""
import joblib
import argparse
import pandas as pd

def parse_args():
    """ Parse the command line arguments.
        :return: argument object
    """
    parser = argparse.ArgumentParser(description='randomforest.py')
    parser.add_argument('-i', type=str, help='input test csv file', default='A549_rep5_run1.csv') # to change the default 
    args = parser.parse_args()
    return args

def prediction(csv_fpath, pickled_model):
    ## read csv file
    data = pd.read_csv(csv_fpath)
    print(data.shape)

    ## save transcript and position col for concatenation later
    data_id_col = data[["transcript", "position"]]

    ## predict using rfc
    data_pred = pickled_model.predict_proba(data[rfe_features])[:,1]
    print(len(data_pred))

    ## convert predictions to dataframe
    data_pred_df = pd.DataFrame(data_pred, columns = ['score'])

    ## 
    data_pred_df = pd.concat([data_id_col, data_pred_df], axis = 1)
    print(f"Prediction file is of shape: {data_pred_df.shape}")

    return data_pred_df

if __name__=='__main__':

    rfe_features = ['std_-1_25', 'std_-1_50', 'std_-1_75', 'std_-1_mean', 'std_-1_min',
        'mean_-1_25', 'mean_-1_50', 'mean_-1_75', 'mean_-1_mean', 'mean_-1_min',
        'dwelling_time_0_50', 'dwelling_time_0_mean', 'std_0_25', 'std_0_50',
        'std_0_75', 'std_0_mean', 'std_0_min', 'std_0_max', 'mean_0_25',
        'mean_0_50', 'mean_0_75', 'mean_0_mean', 'mean_0_min', 'mean_0_max',
        'dwelling_time_+1_mean', 'std_+1_25', 'std_+1_50', 'mean_+1_25',
        'mean_+1_50', 'mean_+1_75', 'mean_+1_mean', 'mean_+1_min',
        'mean_+1_max', 'relative_position', 'position_1_G', 'position_5_T']
    args = parse_args()
    path_to_test = args.i # test file is a csv
    dataset = pd.read_csv(path_to_test)

    pickled_model = joblib.load('rf.pkl')

    predictions_df = prediction(path_to_test, pickled_model)

    predictions_df.to_csv('y_pred_proba.csv')

