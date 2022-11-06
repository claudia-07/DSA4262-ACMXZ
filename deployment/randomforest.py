import joblib
import argparse
import pandas as pd

def parse_args():
    """ Parse the command line arguments.
        :return: argument object
    """
    parser = argparse.ArgumentParser(description='randomforest.py')
    parser.add_argument('-i', type=str, help='input test file', default='df_pred.csv') # to change the default 
    args = parser.parse_args()
    return args

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

    rf_model = joblib.load('rf.pkl')

    y_test_pred_proba = rf_model.predict_proba(dataset[rfe_features])[:,1]
    res0 = pd.DataFrame(y_test_pred_proba,columns=['score'])

    res0.to_csv('y_pred_proba.csv', index= False)

