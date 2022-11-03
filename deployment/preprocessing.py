import json
import pandas as pd
import sys
import pickle

sys.path.append("..")
from util.model import (parse_line)
from util.preprocess import (Preprocessing)

# taking in raw df from command line and parsing them into dataframe
n = len(sys.argv)
data_df = pd.DataFrame()
for i in range(1, n):
    data = sys.argv[i]
    reads_df = [parse_line(data[j]) for j in range(len(data))]
    data_df = pd.concat(reads_df, axis = 0)

# importing info
with open("../data/raw_data/data.info", 'r') as f:
    info = f.read().splitlines()

# importing encoding pipeline
filename = '../data/raw_data/encoding_pipeline.pkl'
pipe = pickle.load(open(filename, 'rb'))

# info
info_list = [info[i].split(",") for i in range(len(info))]
info_df = pd.DataFrame(info_list[1:]) 
info_df.columns = info_list[0]
df = data_df.merge(info_df, how = "left", left_on = ["transcript", "position"], right_on = ["transcript", "transcript_position"])
df = df.drop(['transcript_id', 'transcript_position'],  axis = 1)

### preprocessing
df_pre = Preprocessing(df)
df_pre.model_features_and_clean()
df = df_pre.feature_eng()

# encoding
for i in range(7):
    df['position_' + str(i)] = df['nucleotides'].apply(lambda x: x[i])
df_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(df).T, df_pre.columns_to_map)})

# returning df
with open('df.csv', 'w') as sys.stdout:
    print(df)
