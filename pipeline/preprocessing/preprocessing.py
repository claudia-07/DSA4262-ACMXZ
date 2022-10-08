import pandas as pd
import numpy as np

from parameters import (model_features_list, key_columns, non_nan_cols, stratify_col, 
                        train_percent, validation_percent, test_percent, seed, id_col, 
                        target_encode_cols, one_hot_encode_cols, numeric_cols, target_col,
                        undersampling_strategy, oversampling_strategy)
from utilities import get_first_transaction

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from category_encoders import OneHotEncoder, TargetEncoder


class Preprocessing:
    def __init__(self, df):
        self.df = df

        self.columns_to_map = []

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

    def split_stratified_into_train_val_test(self, random_state=None):
        '''Function to split the data into train,test and validation datasets. 

        :Parameters:
        ------------
            self.df: DataFrame
                DataFrame containing gold and non gold customers

        :Returns:
        ---------
            df_train_all_tx: DataFrame 
                DataFrame containing training features and target for all transactions
            df_train_first_tx:  DataFrame
                DataFrame containing training features and target for first transaction
            df_test:   DataFrame
                DataFrame containing test features and target 
            df_val:    DataFrame
                DataFrame containing validation features and target 
            X_train:   DataFrame
                DataFrame containing training features
            y_train:   DataFrame
                DataFrame containing target variable for training data
            X_train_first_tx: DataFrame
                DataFrame containing training features for training first transaction
            y_train_first_tx: DataFrame
                DataFrame containing target variable for training first transaction
            X_val: DataFrame
                DataFrame containing validation features
            y_val: DataFrame
                DataFrame containing target variable for validation data
            X_test: DataFrame
                DataFrame containing test features
            y_test: DataFrame
                DataFrame containing target variable for test data
            df_val_id: DataFrame
                DataFrame containing validation features and target, including membership_card_id
            list_train_members: List
                List containing membership_card_id for training 
        '''

        # parameter checks
        if train_percent + validation_percent + test_percent != 1.0:
            raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                            (train_percent, validation_percent, test_percent))

        if stratify_col not in df_target.columns:
            raise ValueError('%s is not a column in the dataframe' %
                            (stratify_col))

        # getting df for splitting
        df_target = self.df[[id_col, stratify_col]].drop_duplicates()
        X = df_target  # Contains all columns.
        # Dataframe of just the column on which to stratify.
        y = df_target[[stratify_col]]

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(X, y, stratify=y,
                                                                test_size=(1.0 - train_percent),
                                                                random_state=random_state)

        # Split the temp dataframe into val and test dataframes.
        relative_test_percent = test_percent / (validation_percent + test_percent)
        df_val, df_test, y_val, y_test = train_test_split(df_temp, y_temp,
                                                            stratify=y_temp,
                                                            test_size=relative_test_percent,
                                                            random_state=random_state)

        assert len(df_target) == len(df_train) + len(df_val) + len(df_test)

        # getting list of members of each df
        list_train_members = np.array(df_train[id_col])
        list_test_members = np.array(df_test[id_col])
        list_val_members = np.array(df_val[id_col])

        # creating train data with all transactions, removing membership_card_id and transaction_date
        df_train_all_tx = self.df[self.df[id_col].isin(list_train_members)]
        df_train_all_tx.drop(columns=['transaction_date', id_col], inplace=True)

        # get df with first transaction
        df_train_first_tx = get_first_transaction(self.df, list_train_members)
        df_test = get_first_transaction(self.df, list_test_members)
        df_val, df_val_id = get_first_transaction(self.df, list_val_members, with_id = True)

        # printing percentages
        print("train percentage:", len(df_train_all_tx[df_train_all_tx[target_col] == 1])/len(df_train_all_tx))
        print("train percentage first timers:", len(df_train_first_tx[df_train_first_tx[target_col] == 1])/len(df_train_first_tx))
        print("test percentage:", len(df_test[df_test[target_col] == 1])/len(df_test))
        print("val percentage:", len(df_val[df_val[target_col] == 1])/len(df_val))

        # printing df shape
        print("train data shape:", df_train_all_tx.shape)
        print("train data first timers shape:", df_train_first_tx.shape)
        print("validation data shape:", df_val.shape)
        print("test data shape:", df_test.shape)

        # separating df from target column: features -> X | target -> y
        X_train = df_train_all_tx.drop(columns=target_col)
        y_train = pd.DataFrame(df_train_all_tx[target_col])
        X_train_first_tx = df_train_first_tx.drop(columns=target_col)
        y_train_first_tx = pd.DataFrame(df_train_first_tx[target_col])
        X_val = df_val.drop(columns=target_col)
        y_val = pd.DataFrame(df_val[target_col])
        X_test = df_test.drop(columns=target_col)
        y_test = pd.DataFrame(df_test[target_col])

        return(df_train_all_tx, df_train_first_tx, df_test, df_val, X_train, y_train, X_train_first_tx,
            y_train_first_tx, X_val, y_val, X_test, y_test, df_val_id, list_train_members)


    def encoding_train(self, x_df, y_df):
        '''Function to encode categorical features for training data. 
            Features with fewer than 10 unique variables will be one hot encoded, and the rest will be target encoded

        :Parameters:
        -----------
            x_df: DataFrame
                DataFrame containing raw features
            y_df: DataFrame
                Dataframe containing target column
        :Returns:
        ---------
            x_df: DataFrame
                DataFrame containing all encoded and standardized columns
            pipe: Pipeline
                Pipeline for encoding and standardization
        '''

        # piping the encoding
        numeric_encoder = Pipeline([('scale', MinMaxScaler())])
        one_hot_encoder = Pipeline([('one_hot', OneHotEncoder(handle_unknown='ignore'))])
        target_encoder = Pipeline([('target_encode', 
                                        TargetEncoder(cols = target_encode_cols, 
                                        handle_unknown='return_nan'))])
        preprocessor = ColumnTransformer(transformers = [
                                            ("num", numeric_encoder, numeric_cols),
                                            ("target_encode", target_encoder, target_encode_cols), ("one_hot_encode", one_hot_encoder, one_hot_encode_cols)], 
                                            remainder = 'passthrough')

        # getting list of column names to map
        self.columns_to_map = numeric_cols + target_encode_cols
        one_hot = OneHotEncoder(handle_unknown='ignore',use_cat_names=True)
        x_df_ohe = pd.DataFrame(one_hot.fit_transform(x_df[one_hot_encode_cols]))
        x_df_ohe.columns = one_hot.get_feature_names()
        self.columns_to_map = numeric_cols + target_encode_cols + list(x_df_ohe.columns)

        # for f in one_hot_encode_cols:
        #     cols = pd.get_dummies(x_df[f]).columns
        #     columns_to_map.extend(f + '_' + str(col)
        #                                 for col in cols)

        # columns_to_map.extend('gender_' + str(col)
        #                             for col in pd.get_dummies(x_df['gender']).columns)
        # columns_to_map.extend('customer_most_spent_merchandise_grp_' + str(col)
        #                             for col in pd.get_dummies(x_df['customer_most_spent_merchandise_grp']).columns)
        print('columns after preprocessing :', self.columns_to_map,  '\n')

        # applying encoding on columns in df and creating pipeline
        x_df = pd.DataFrame({col: vals for vals, col in zip(preprocessor.transform(x_df, y_df).T, self.columns_to_map)})
        pipe = Pipeline(steps=[("preprocessor", preprocessor)])
        pipe = pipe.fit(x_df, y_df)

        return x_df, pipe
    

    def encoding_test_val(self, x_input, pipe):
        '''Function to encode categorical features for test and validation data. 
            Features with fewer than 10 unique variables will be one hot encoded, and the rest will be target encoded

        :Parameters:
        -----------
            y_input: DataFrame
                Dataframe containing test/validation features
            pipe: Pipeline
                Pipeline created from training data
        :Returns:
        ---------
            x_output: DataFrame
                DataFrame containing all encoded and standardized columns
        '''

        # applying encoding on columns in df
        x_output = pd.DataFrame({col:vals for vals,col in zip(pipe.transform(x_input).T, self.columns_to_map)})
        return x_output
        

    def oversample_undersample(X_train_first_timers, y_train_first_timers, sampling=True):
        '''Function to oversample minority and undersample majority.

        :Parameters:
        ------------
            X_train_first_timers: DataFrame
                DataFrame containing training data only for first purchase date excluding the target variable 
            y_train_first_timers: DataFrame
                DataFrame containing target variable for training data only for first purchase date

        :Returns:
        ---------
            X_train_first_timers: DataFrame
                DataFrame contiaining oversampled and understampled training data only for first purchase date excluding target variable
            y_train_first_timers: DataFrame
                DataFrame containing oversampled and undersampled target variable for training data only for first purchase date

        '''

        if sampling:
            undersample = RandomUnderSampler(
                sampling_strategy = undersampling_strategy, random_state=seed)
            oversample = SMOTE(
                sampling_strategy = oversampling_strategy, random_state=seed)
            X_train_first_timers, y_train_first_timers = undersample.fit_resample(
                X_train_first_timers, y_train_first_timers)
            X_train_first_timers, y_train_first_timers = oversample.fit_resample(
                X_train_first_timers, y_train_first_timers)
        return X_train_first_timers, y_train_first_timers
