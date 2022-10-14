import pandas as pd
import numpy as np

##### preprocessing #####
# features - model
model_features_list = ['transcript', 'position', 'nucleotides', 'reads_count', 'gene_id',
                            'dwellingtime_-1', 'std_-1', 'mean_-1',
                            'dwellingtime_0', 'std_0', 'mean_0',
                            'dwellingtime_+1', 'std_+1', 'mean_+1','label']

# features - preprocessing
key_columns = ['label'] # columns that cannot be null
non_nan_cols = []

# features - encoding
target_col = 'label'
one_hot_col = ['nucleotides']
numeric_cols =  ['reads_count','dwelling_time_-1_25',
                    'dwelling_time_-1_50', 'dwelling_time_-1_75', 'dwelling_time_-1_mean',
                    'std_-1_25', 'std_-1_50', 'std_-1_75', 'std_-1_mean', 'mean_-1_25',
                    'mean_-1_50', 'mean_-1_75', 'mean_-1_mean', 'dwelling_time_0_25',
                    'dwelling_time_0_50', 'dwelling_time_0_75', 'dwelling_time_0_mean',
                    'std_0_25', 'std_0_50', 'std_0_75', 'std_0_mean', 'mean_0_25',
                    'mean_0_50', 'mean_0_75', 'mean_0_mean', 'dwelling_time_+1_25',
                    'dwelling_time_+1_50', 'dwelling_time_+1_75', 'dwelling_time_+1_mean',
                    'std_+1_25', 'std_+1_50', 'std_+1_75', 'std_+1_mean', 'mean_+1_25',
                    'mean_+1_50', 'mean_+1_75', 'mean_+1_mean']

# oversampling / undersampling
undersampling_strategy = 0.17
oversampling_strategy = 0.22

# data splitting
stratify_col = 'label' # column used to stratify data
seed = 42 # random state
id_col = ['gene_id']
position_col = ['transcript', 'position']
    # the following 3 percentages should sum to 1
train_percent = 0.8 # percentage of train data
validation_percent = 0.1 # percentage of validation data
test_percent = 0.1 # percentage of test data


##### model #####
num_splits = 5 # number of splits for kfold
max_n_iter = 5 # max number of iterations for search space
scale_pos_weight = 5 # weight of scale pos
scoring = 'roc_auc' # scoring method







