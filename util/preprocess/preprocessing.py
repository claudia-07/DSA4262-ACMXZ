import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
import pathlib
import sys

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

THIS_DIR = pathlib.Path(__file__).resolve()
PROJ_DIR = THIS_DIR.parents[2]
sys.path.append(PROJ_DIR.as_posix())

from util.model.training import get_percent, relative_position
from util.preprocess.parameters import (model_features_list, key_columns, non_nan_cols, stratify_col,
                             train_percent, validation_percent, test_percent, seed, id_col,
                             numeric_cols, target_col, position_col, one_hot_col,
                             undersampling_strategy, oversampling_strategy)


class Preprocessing:
    def __init__(self, df):
        self.df = df

        self.columns_to_map = None
        self.df_train = None
        self.df_test = None
        self.df_val = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.pipe = None
        self.X_train_enc = None
        self.X_val_enc = None
        self.X_test_enc = None
        self.X_train_oversampled = None

    def model_features_and_clean(self):
        '''Function to select features for modelling and clean the initial raw data. 
            Removes rows which contain NAN values in key_columns, and converts all NAN values to 0 for non_nan_cols

        :Parameters:
        ------------
            self.df: DataFrame
                DataFrame containing raw data

        :Returns:
        ---------
            self.df: DataFrame
                DataFrame containing features required for modelling
        '''
        self.df = self.df[model_features_list]
        self.df[non_nan_cols] = self.df[non_nan_cols].fillna(0)
        self.df = self.df.dropna(subset=key_columns)
        return self.df

    def feature_eng(self):
        self.df = pd.DataFrame(self.df.groupby(['gene_id', 'transcript', 'position', 'nucleotides', 'reads_count', 'label'], as_index=False)
                                .agg({'dwellingtime_-1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'std_-1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'mean_-1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'dwellingtime_0': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'std_0': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'mean_0': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'dwellingtime_+1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'std_+1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max],
                                        'mean_+1': [get_percent(25), get_percent(50), get_percent(75), np.mean, np.min, np.max]}))
        self.df.columns = ['gene_id', 'transcript', 'position', 'nucleotides', 'reads_count', 'label',
                            'dwelling_time_-1_25', 'dwelling_time_-1_50', 'dwelling_time_-1_75', 'dwelling_time_-1_mean','dwelling_time_-1_min', 'dwelling_time_-1_max',
                            'std_-1_25', 'std_-1_50', 'std_-1_75', 'std_-1_mean','std_-1_min', 'std_-1_max',
                            'mean_-1_25', 'mean_-1_50', 'mean_-1_75', 'mean_-1_mean','mean_-1_min', 'mean_-1_max',
                            'dwelling_time_0_25', 'dwelling_time_0_50', 'dwelling_time_0_75', 'dwelling_time_0_mean','dwelling_time_0_min','dwelling_time_0_max',
                            'std_0_25', 'std_0_50', 'std_0_75', 'std_0_mean','std_0_min', 'std_0_max',
                            'mean_0_25', 'mean_0_50', 'mean_0_75', 'mean_0_mean','mean_0_min', 'mean_0_max',
                            'dwelling_time_+1_25', 'dwelling_time_+1_50', 'dwelling_time_+1_75', 'dwelling_time_+1_mean','dwelling_time_+1_min','dwelling_time_+1_max',
                            'std_+1_25', 'std_+1_50', 'std_+1_75', 'std_+1_mean','std_+1_min', 'std_+1_max',
                            'mean_+1_25', 'mean_+1_50', 'mean_+1_75', 'mean_+1_mean','mean_+1_min', 'mean_+1_max']
        
        self.df = relative_position(self.df)
        return self.df

    def split_stratified_into_train_val_test(self, random_state=None):
        '''Function to split the data into train,test and validation datasets. 

        :Parameters:
        ------------
            self.df: DataFrame
                DataFrame containing gold and non gold customers

        :Returns:
        ---------
            df_train: DataFrame 
                DataFrame containing training features and target
            df_test:   DataFrame
                DataFrame containing test features and target 
            df_val:    DataFrame
                DataFrame containing validation features and target 
            X_train:   DataFrame
                DataFrame containing training features
            y_train:   DataFrame
                DataFrame containing target variable for training data
            X_val: DataFrame
                DataFrame containing validation features
            y_val: DataFrame
                DataFrame containing target variable for validation data
            X_test: DataFrame
                DataFrame containing test features
            y_test: DataFrame
                DataFrame containing target variable for test data
            df_val_id: DataFrame
                DataFrame containing validation features and target, including id_cols
            list_train: List
                List containing id_cols for training 
        '''
        # getting unique id_col and stratify_col for splitting
        split_col = id_col + [stratify_col]
        df_target = self.df[split_col].drop_duplicates()
        X = df_target  # Contains all columns.
        # Dataframe of just the column on which to stratify.
        y = df_target[[stratify_col]]

        # parameter checks
        if train_percent + validation_percent + test_percent != 1.0:
            raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                             (train_percent, validation_percent, test_percent))

        if stratify_col not in df_target.columns:
            raise ValueError('%s is not a column in the dataframe' %
                             (stratify_col))

        # Split original dataframe into train and temp dataframes.
        self.df_train, df_temp, self.y_train, y_temp = train_test_split(X, y, stratify=y,
                                                                        test_size=(
                                                                            1.0 - train_percent),
                                                                        random_state=random_state)

        # Split the temp dataframe into val and test dataframes.
        relative_test_percent = test_percent / (validation_percent + test_percent)
        self.df_val, self.df_test, self.y_val, self.y_test = train_test_split(df_temp, y_temp,
                                                                              stratify=y_temp,
                                                                              test_size=relative_test_percent,
                                                                              random_state=random_state)

        assert len(df_target) == len(self.df_train) + len(self.df_val) + len(self.df_test)

        # getting list of splitting columns of each df
        list_train = np.array(self.df_train[split_col])
        list_train = [tuple(i) for i in list_train]
        list_test = np.array(self.df_test[split_col])
        list_test = [tuple(i) for i in list_test]
        list_val = np.array(self.df_val[split_col])
        list_val = [tuple(i) for i in list_val]

        # creating train/test/val data.
        self.df_train = self.df[self.df[split_col].apply(tuple, axis=1).isin(list_train)]
        self.df_test = self.df[self.df[split_col].apply(tuple, axis=1).isin(list_test)]
        self.df_val = self.df[self.df[split_col].apply(tuple, axis=1).isin(list_val)]
        # df_val_id contains all features + idenifying columns
        df_val_id = self.df_val.copy()

        # printing percentages
        print("train target percentage:", len(self.df_train[self.df_train[target_col] == '1'])/len(self.df_train))
        print("test target percentage:", len(self.df_test[self.df_test[target_col] == '1'])/len(self.df_test))
        print("val target percentage:", len(self.df_val[self.df_val[target_col] == '1'])/len(self.df_val))

        # printing df shape
        print("train data shape:", self.df_train.shape)
        print("validation data shape:", self.df_val.shape)
        print("test data shape:", self.df_test.shape)

        # separating df from target column: features -> X | target -> y
        # removing identifying columns as they are not features
        temp_col = split_col + position_col
        self.X_train = self.df_train.drop(columns=temp_col).reset_index(drop=True)
        self.y_train = pd.DataFrame(self.df_train[target_col]).reset_index(drop=True)
        self.X_val = self.df_val.drop(columns=temp_col).reset_index(drop=True)
        self.y_val = pd.DataFrame(self.df_val[target_col]).reset_index(drop=True)
        self.X_test = self.df_test.drop(columns=temp_col).reset_index(drop=True)
        self.y_test = pd.DataFrame(self.df_test[target_col]).reset_index(drop=True)

        return(self.df_train, self.df_test, self.df_val, self.X_train, self.y_train,
               self.X_val, self.y_val, self.X_test, self.y_test, df_val_id, list_train)

    def oversample_undersample(self, sampling=True):
        '''Function to oversample minority and undersample majority.

        :Returns:
        ---------
            self.X_train_oversampled: DataFrame
                DataFrame contiaining oversampled and understampled training data only for first purchase date excluding target variable
        '''
        self.X_train_oversampled = self.X_train.copy()

        if sampling:
            undersample = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=seed)
            oversample = SMOTENC(categorical_features=[0], sampling_strategy=oversampling_strategy, random_state=seed)
            self.X_train_oversampled, self.y_train = undersample.fit_resample(self.X_train_oversampled, self.y_train)
            self.X_train_oversampled, self.y_train = oversample.fit_resample(self.X_train_oversampled, self.y_train)

        return self.X_train_oversampled, self.y_train

    def encoding_train(self):
        '''Function to encode categorical features for training data.

        :Parameters:
        -----------

        :Returns:
        ---------
            x_df: DataFrame
                DataFrame containing all encoded and standardized columns
            pipe: Pipeline
                Pipeline for encoding and standardization
        '''
        # piping the encoding
        numeric_encoder = Pipeline([('scale', StandardScaler())])
        one_hot_encoder = Pipeline([('one_hot_enocde', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(transformers=[("num", numeric_encoder, numeric_cols), 
                                                        ("one_hot_encode", one_hot_encoder, one_hot_col)], remainder='passthrough')

        # getting list of column names to map
        for i in range(7):
            self.X_train_oversampled['position_' + str(i)] = self.X_train_oversampled['nucleotides'].apply(lambda x: x[i])
        ref_df = self.X_train_oversampled # df used as reference for encoding
        self.columns_to_map = numeric_cols
        one_hot = OneHotEncoder(handle_unknown='ignore',use_cat_names=True)
        x_df_ohe = pd.DataFrame(one_hot.fit_transform(ref_df[one_hot_col]))
        x_df_ohe.columns = one_hot.get_feature_names()
        self.columns_to_map = numeric_cols + list(x_df_ohe.columns)

        print('columns after preprocessing :', self.columns_to_map,  '\n')

        # applying encoding on columns in df and creating pipeline
        self.X_train_enc = pd.DataFrame({col: vals for vals, col in zip(preprocessor.fit_transform(ref_df, self.y_train).T, self.columns_to_map)})
        self.pipe = Pipeline(steps=[("preprocessor", preprocessor)])
        self.pipe = self.pipe.fit(ref_df, self.y_train)

        # # encoding nucleotides
        # for i in range(7):
        #     self.X_train_enc['position_' + str(i) + '_A'] = 0
        #     self.X_train_enc['position_' + str(i) + '_C'] = 0
        #     self.X_train_enc['position_' + str(i) + '_G'] = 0
        #     self.X_train_enc['position_' + str(i) + '_T'] = 0
        #     temp = self.X_train_enc['nucleotides'].apply(lambda x: x[i])
        #     self.X_train_enc['position_' + str(i) + '_A'][temp == 'A'] = 1
        #     self.X_train_enc['position_' + str(i) + '_C'][temp == 'C'] = 1
        #     self.X_train_enc['position_' + str(i) + '_G'][temp == 'G'] = 1
        #     self.X_train_enc['position_' + str(i) + '_T'][temp == 'T'] = 1

        # # dropping nucleotides column
        # self.X_train_enc = self.X_train_enc.drop(columns=['nucleotides'])

        return self.X_train_enc, self.pipe

    def encoding_test_val(self, pipe, test=True):
        '''Function to encode categorical features for test and validation data. 

        :Parameters:
        -----------
            test: Boolean
                if test = True, encode for X_test. if False, encode for X_val

        :Returns:
        ---------
            x_output: DataFrame
                DataFrame containing all encoded and standardized columns
        '''

        # applying encoding on columns in df
        if test == True:  # for test
            for i in range(7):
                self.X_test['position_' + str(i)] = self.X_test['nucleotides'].apply(lambda x: x[i])
            self.X_test_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(self.X_test).T, self.columns_to_map)})
            # # encoding nucleotides
            # for i in range(7):
            #     self.X_test_enc['position_' + str(i) + '_A'] = 0
            #     self.X_test_enc['position_' + str(i) + '_C'] = 0
            #     self.X_test_enc['position_' + str(i) + '_G'] = 0
            #     self.X_test_enc['position_' + str(i) + '_T'] = 0
            #     temp = self.X_test_enc['nucleotides'].apply(lambda x: x[i])
            #     self.X_test_enc['position_' + str(i) + '_A'][temp == 'A'] = 1
            #     self.X_test_enc['position_' + str(i) + '_C'][temp == 'C'] = 1
            #     self.X_test_enc['position_' + str(i) + '_G'][temp == 'G'] = 1
            #     self.X_test_enc['position_' + str(i) + '_T'][temp == 'T'] = 1

            # # dropping nucleotides column
            # self.X_test_enc = self.X_test_enc.drop(columns=['nucleotides'])

            return self.X_test_enc
        else:  # for validation
            for i in range(7):
                self.X_val['position_' + str(i)] = self.X_val['nucleotides'].apply(lambda x: x[i])
            self.X_val_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(self.X_val).T, self.columns_to_map)})
            # # encoding nucleotides
            # for i in range(7):
            #     self.X_val_enc['position_' + str(i) + '_A'] = 0
            #     self.X_val_enc['position_' + str(i) + '_C'] = 0
            #     self.X_val_enc['position_' + str(i) + '_G'] = 0
            #     self.X_val_enc['position_' + str(i) + '_T'] = 0
            #     temp = self.X_val_enc['nucleotides'].apply(lambda x: x[i])
            #     self.X_val_enc['position_' + str(i) + '_A'][temp == 'A'] = 1
            #     self.X_val_enc['position_' + str(i) + '_C'][temp == 'C'] = 1
            #     self.X_val_enc['position_' + str(i) + '_G'][temp == 'G'] = 1
            #     self.X_val_enc['position_' + str(i) + '_T'][temp == 'T'] = 1

            # # dropping nucleotides column
            # self.X_val_enc = self.X_val_enc.drop(columns=['nucleotides'])

            return self.X_val_enc
