import pandas as pd
import numpy as np

# features - model
model_features_list = ['transcript', 'position', 'nucleotides', 'dwelling_time', 'std', 'mean', 'label']

# features - preprocessing
key_columns = ['label'] # columns that cannot be null
non_nan_cols = []

# features - encoding
target_encode_cols = ['nucleotides']
target_col = 'label'
numeric_cols = ['dwelling_time', 'std', 'mean']

# oversampling / undersampling
undersampling_strategy = 0.15
oversampling_strategy = 0.20

# data splitting
stratify_col = 'label' # column used to stratify data
seed = 42 # random state
id_col = ['transcript', 'position']
    # the following 3 percentages should sum to 1
train_percent = 0.8 # percentage of train data
validation_percent = 0.1 # percentage of validation data
test_percent = 0.1 # percentage of test data










