import pandas as pd
import numpy as np

from parameters import (model_features_list, key_columns, non_nan_cols, stratify_col, 
                        train_percent, validation_percent, test_percent, seed, id_col, 
                        target_encode_cols, numeric_cols, target_col, position_col, 
                        undersampling_strategy, oversampling_strategy)
from utilities import get_percent

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from category_encoders import TargetEncoder


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
        self.df = pd.DataFrame(self.df.groupby(['gene_id', 'transcript', 'position', 'nucleotides', 'label'], as_index=False)
                            .agg({'dwelling_time': [get_percent(25), get_percent(50), get_percent(75), np.mean], 
                                    'std': [get_percent(25), get_percent(50), get_percent(75), np.mean], 
                                    'mean': [get_percent(25), get_percent(50), get_percent(75), np.mean]}))
        self.df.columns = ['gene_id', 'transcript', 'position', 'nucleotides', 'label',
                            'dwelling_time_25', 'dwelling_time_50', 'dwelling_time_75', 'dwelling_time_mean', 
                            'std_25', 'std_50', 'std_75', 'std_mean', 
                            'mean_25', 'mean_50', 'mean_75', 'mean_mean']
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
        temp_col = id_col + [stratify_col]
        df_target = self.df[temp_col].drop_duplicates()
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
                                                                test_size=(1.0 - train_percent),
                                                                random_state=random_state)

        # Split the temp dataframe into val and test dataframes.
        relative_test_percent = test_percent / (validation_percent + test_percent)
        self.df_val, self.df_test, self.y_val, self.y_test = train_test_split(df_temp, y_temp,
                                                            stratify=y_temp,
                                                            test_size=relative_test_percent,
                                                            random_state=random_state)

        assert len(df_target) == len(self.df_train) + len(self.df_val) + len(self.df_test)

        # getting list of id_cols of each df
        list_train = np.array(self.df_train[id_col])
        list_train = [tuple(i) for i in list_train]
        list_test = np.array(self.df_test[id_col])
        list_test = [tuple(i) for i in list_test]
        list_val = np.array(self.df_val[id_col])
        list_val = [tuple(i) for i in list_val]

        # creating train/test/val data, removing identifying columns as they are not features. 
        # identifying columns include id_col and position_col
        temp_col = id_col + position_col
        self.df_train = self.df[self.df[id_col].apply(tuple, axis = 1).isin(list_train)]
        self.df_train.drop(columns=temp_col, inplace=True)
        self.df_test = self.df[self.df[id_col].apply(tuple, axis = 1).isin(list_test)]
        self.df_test.drop(columns=temp_col, inplace=True)
        self.df_val = self.df[self.df[id_col].apply(tuple, axis = 1).isin(list_val)]
        df_val_id = self.df_val.copy()
        self.df_val.drop(columns=temp_col, inplace=True)


        # printing percentages
        print("train target percentage:", len(self.df_train[self.df_train[target_col] == '1'])/len(self.df_train))
        print("test target percentage:", len(self.df_test[self.df_test[target_col] == '1'])/len(self.df_test))
        print("val target percentage:", len(self.df_val[self.df_val[target_col] == '1'])/len(self.df_val))

        # printing df shape
        print("train data shape:", self.df_train.shape)
        print("validation data shape:", self.df_val.shape)
        print("test data shape:", self.df_test.shape)

        # separating df from target column: features -> X | target -> y
        self.X_train = self.df_train.drop(columns=target_col).reset_index(drop=True)
        self.y_train = pd.DataFrame(self.df_train[target_col]).reset_index(drop=True)
        self.X_val = self.df_val.drop(columns=target_col).reset_index(drop=True)
        self.y_val = pd.DataFrame(self.df_val[target_col]).reset_index(drop=True)
        self.X_test = self.df_test.drop(columns=target_col).reset_index(drop=True)
        self.y_test = pd.DataFrame(self.df_test[target_col]).reset_index(drop=True)

        return(self.df_train, self.df_test, self.df_val, self.X_train, self.y_train,
                self.X_val, self.y_val, self.X_test, self.y_test, df_val_id, list_train)


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
        numeric_encoder = Pipeline([('scale', MinMaxScaler())])
        target_encoder = Pipeline([('target_encode', 
                                        TargetEncoder(cols = target_encode_cols, 
                                        handle_unknown='return_nan'))])
        preprocessor = ColumnTransformer(transformers = [
                                            ("num", numeric_encoder, numeric_cols),
                                            ("target_encode", target_encoder, target_encode_cols)], 
                                            remainder = 'passthrough')

        # getting list of column names to map
        self.columns_to_map = numeric_cols + target_encode_cols

        print('columns after preprocessing :', self.columns_to_map,  '\n')

        # applying encoding on columns in df and creating pipeline
        self.X_train = pd.DataFrame({col: vals for vals, col in zip(preprocessor.fit_transform(self.X_train, self.y_train).T, self.columns_to_map)})
        self.pipe = Pipeline(steps=[("preprocessor", preprocessor)])
        self.pipe = self.pipe.fit(self.X_train, self.y_train)

        return self.X_train, self.pipe
    

    def encoding_test_val(self, test = True):
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
            # for test
        if test == True:
            self.X_test = pd.DataFrame({col:vals for vals,col in zip(self.pipe.transform(self.X_test).T, self.columns_to_map)})
            return self.X_test
            # for validation
        else:
            self.X_val = pd.DataFrame({col:vals for vals,col in zip(self.pipe.transform(self.X_val).T, self.columns_to_map)})
            return self.X_val
        
        

    def oversample_undersample(x_df, y_df, sampling=True):
        '''Function to oversample minority and undersample majority.

        :Parameters:
        ------------
            x_df: DataFrame
                DataFrame containing training data only for first purchase date excluding the target variable 
            y_df: DataFrame
                DataFrame containing target variable for training data only for first purchase date

        :Returns:
        ---------
            x_df: DataFrame
                DataFrame contiaining oversampled and understampled training data only for first purchase date excluding target variable
            y_df: DataFrame
                DataFrame containing oversampled and undersampled target variable for training data only for first purchase date

        '''

        if sampling:
            undersample = RandomUnderSampler(sampling_strategy = undersampling_strategy, random_state=seed)
            oversample = SMOTE(sampling_strategy = oversampling_strategy, random_state=seed)
            x_df, y_df = undersample.fit_resample(x_df, y_df)
            x_df, y_df = oversample.fit_resample(x_df, y_df)
        return x_df, y_df
