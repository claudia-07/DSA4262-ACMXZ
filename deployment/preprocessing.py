import json
import pandas as pd
import sys
import pickle
import pathlib

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
    print(type(data))
    print(len(data))
    reads_df = [parse_line(data[j]) for j in range(len(data))]
    data_df = pd.concat(reads_df, axis = 0)
    print("parsed " + str(n-1) + " number of files")
print("parsed all data")
print(data_df.columns)

# importing info
with open("data/raw_data/data.info", 'r') as f:
    info = f.read().splitlines()

# importing encoding pipeline
filename = 'data/raw_data/encoding_pipeline.pkl'
pipe = pickle.load(open(filename, 'rb'))
print("imported all files")

# info
info_list = [info[i].split(",") for i in range(len(info))]
info_df = pd.DataFrame(info_list[1:]) 
info_df.columns = info_list[0]
df = data_df.merge(info_df, how = "left", left_on = ["transcript", "position"], right_on = ["transcript_id", "transcript_position"])
df = df.drop(['transcript_id', 'transcript_position'],  axis = 1)
print("merged data with info")

### preprocessing
df_pre = Preprocessing(df)
df_pre.model_features_and_clean()
df = df_pre.feature_eng()
print("data cleaned and feature engineering completed")

# encoding
for i in range(7):
    df['position_' + str(i)] = df['nucleotides'].apply(lambda x: x[i])
df_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(df).T, df_pre.columns_to_map)})
print("encoding completed")

# returning df
with open('df.csv', 'w') as sys.stdout:
    print(df)
