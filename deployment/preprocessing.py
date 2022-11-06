import json
import pandas as pd
import sys
import pickle
import pathlib
import csv

THIS_DIR = pathlib.Path(__file__).resolve()
PROJ_DIR = THIS_DIR.parents[1]
sys.path.append(PROJ_DIR.as_posix())

from util.model import (parse_line)
from util.preprocess import (Preprocessing)

# taking in raw df from command line and parsing them into dataframe
n = len(sys.argv)
data_df = pd.DataFrame()
for i in range(1, n):
    data = [json.loads(line) for line in open(sys.argv[i], 'r')]
    if len(data)==1:
        data = data[0]
    reads_df = [parse_line(data[j]) for j in range(len(data))]
    data_df = pd.concat(reads_df, axis = 0)

# importing info
with open("../data/model_training/raw_data/data.info", 'r') as f:
    info = f.read().splitlines()

# importing encoding pipeline and columns for mapping
filename = '../data/model_training/raw_data/encoding_pipeline.pkl'
pipe = pickle.load(open(filename, 'rb'))
filename = '../data/model_training/raw_data/columns.pkl'
columns = pickle.load(open(filename, 'rb'))

# info
info_list = [info[i].split(",") for i in range(len(info))]
info_df = pd.DataFrame(info_list[1:]) 
info_df.columns = info_list[0]
df = data_df.merge(info_df, how = "left", left_on = ["transcript", "position"], right_on = ["transcript_id", "transcript_position"])
df = df.drop(['transcript_id', 'transcript_position'],  axis = 1)

### preprocessing
df_pre = Preprocessing(df)
df_pre.model_features_and_clean()
df = df_pre.feature_eng()

# encoding
df_id = df[['transcript', 'position']]
for i in range(7):
    df['position_' + str(i)] = df['nucleotides'].apply(lambda x: x[i])
df_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(df).T, columns)})
df_enc[['transcript', 'position']] = df_id

# returning df
df_enc.to_csv('preprocessed_data.csv', index=False)