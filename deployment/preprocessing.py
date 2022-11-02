import json
import pandas as pd
import sys
import pickle

from util.model import (parse_line)
from util.preprocess import (Preprocessing)
from util.model import (get_percent, Modelling)

# taking in raw df from command line
data = sys.argv

# importing info
with open("../data/raw_data/data.info", 'r') as f:
    info = f.read().splitlines()

# importing encoding pipeline
filename = 'encoding_pipeline.pkl'
pipe = pickle.load(open(filename, 'rb'))

### parse all lines in json file into dataframe
reads_df = [parse_line(data[i]) for i in range(len(data))]
data_df = pd.concat(reads_df, axis = 0)

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
# splitting in df and df_id (with id columns)
id_col = ['transcript', 'position', 'gene_id', 'label']
# df_id = df
df = df.drop(columns=id_col)
# encoding
for i in range(7):
    df['position_' + str(i)] = df['nucleotides'].apply(lambda x: x[i])
df_enc = pd.DataFrame({col: vals for vals, col in zip(pipe.transform(df).T, df_pre.columns_to_map)})

# returning df
with open('df', 'w') as sys.stdout:
    print(df)
