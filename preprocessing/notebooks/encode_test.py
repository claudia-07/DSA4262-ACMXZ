import pandas as pd
import numpy as np
import sys
import os 

sys.path.append(os.path.abspath('../../util/model'))
from training import get_percent

# Read in encoded test data
X_train_path = "../data/raw_data/X_raw_enc.parquet"
X_train = pd.read_parquet(X_train_path)
y_train_path = "../data/raw_data/y_raw.parquet"
y_train = pd.read_parquet(y_train_path)

# Read in raw data
X_test = 
y_test = 


# Apply the functions to the train data


percentiles_df = feature_eng(cleaned_df)
percentiles_df.head()





# Function to get aggregated values 
def feature_eng(df):
    temp = pd.DataFrame(df.groupby(['gene_id', 'transcript', 'position', 'nucleotides', 'reads_count', 'label'], as_index=False)
                           .agg({'dwellingtime_-1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'std_-1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'mean_-1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'dwellingtime_0': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'std_0': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'mean_0': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'dwellingtime_+1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'std_+1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                'mean_+1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max]}))
    temp.columns = ['gene_id', 'transcript', 'position', 'nucleotides', 'reads_count', 'label',
                        'dwelling_time_-1_25', 'dwelling_time_-1_50', 'dwelling_time_-1_75', 'dwelling_time_-1_mean','dwelling_time_-1_min', 'dwelling_time_-1_max',
                        'std_-1_25', 'std_-1_50', 'std_-1_75', 'std_-1_mean','std_-1_min', 'std_-1_max',
                        'mean_-1_25', 'mean_-1_50', 'mean_-1_75', 'mean_-1_mean','mean_-1_min', 'mean_-1_max',
                        'dwelling_time_0_25', 'dwelling_time_0_50', 'dwelling_time_0_75', 'dwelling_time_0_mean','dwelling_time_0_min','dwelling_time_0_max',
                        'std_0_25', 'std_0_50', 'std_0_75', 'std_0_mean','std_0_min', 'std_0_max',
                        'mean_0_25', 'mean_0_50', 'mean_0_75', 'mean_0_mean','mean_0_min', 'mean_0_max',
                        'dwelling_time_+1_25', 'dwelling_time_+1_50', 'dwelling_time_+1_75', 'dwelling_time_+1_mean','dwelling_time_+1_min','dwelling_time_+1_max',
                        'std_+1_25', 'std_+1_50', 'std_+1_75', 'std_+1_mean','std_+1_min', 'std_+1_max',
                        'mean_+1_25', 'mean_+1_50', 'mean_+1_75', 'mean_+1_mean','mean_+1_min', 'mean_+1_max']
    return temp

# Function to encode test data
def encoding_test_val(X_test, X_val, columns_to_map, test=True):

    # applying encoding on columns in df
    if test == True:  # for test
        for i in range(7):
            X_test['position_' + str(i)] = X_test['nucleotides'].apply(lambda x: x[i])
        X_test_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(X_test).T, columns_to_map)})


        return X_test_enc
    else:  # for validation
        for i in range(7):
            X_val['position_' + str(i)] = X_val['nucleotides'].apply(lambda x: x[i])
        X_val_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(X_val).T, columns_to_map)})

        return X_val_enc
