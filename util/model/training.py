import numpy as np
import pandas as pd

## function to get key of a dictionary
def get_key(dictionary):
    key_object = dictionary.keys()
    key = list(key_object)[0]
    return key

## function to help concatenate columns to get transcript, position, nucleotides
def concat_col(transcript, position, nucleotide, n):
    t_df = pd.DataFrame([transcript]*n)
    p_df = pd.DataFrame([position]*n)
    nu_df = pd.DataFrame([nucleotide]*n)
    n_df = pd.DataFrame([n]*n)

    ## concat columns together
    final_df = pd.concat([t_df, p_df, nu_df, n_df], axis = 1)
    final_df.columns = ['transcript', 'position', 'nucleotides', 'reads_count']
    return final_df

## function to parse line in json file
def parse_line(line):
    ## get transcript
    t = get_key(line)

    ## get position
    p = get_key(line[t])

    ## get nucleotide seq
    nu = get_key(line[t][p])

    ## get number of reads
    reads_count = len(line[t][p][nu])

    ## get dataframe of list of reads
    reads = pd.DataFrame(line[t][p][nu])

    ## concat columns together to get transcript, position, nucleotides and all dwelling time, std, mean
    df = pd.concat([concat_col(t, p, nu, reads_count), reads], axis = 1)
    df.columns = ['transcript', 'position', 'nucleotides', 'reads_count', 'dwellingtime_-1', 'std_-1', 'mean_-1', 'dwellingtime_0', 'std_0', 'mean_0', 'dwellingtime_+1', 'std_+1', 'mean_+1']

    return df

def get_first_transaction(df_all, list_members, with_id = False):
    # filter for customers in list_members
    df_first_tx = df_all[df_all['membership_card_id'].isin(list_members)]

    # getting first purchase transactions
    df_temp_tx = pd.DataFrame(df_first_tx.groupby('membership_card_id')['transaction_date'].unique())
    df_temp_tx['first_purchase_date'] = df_temp_tx['transaction_date'].apply(lambda x: x[0])
    df_first_tx = df_first_tx.merge(df_temp_tx['first_purchase_date'], on='membership_card_id', how='left')
    df_first_tx = df_first_tx[df_first_tx['transaction_date']== df_first_tx['first_purchase_date']]
    if with_id == True:
        df_first_tx_id = df_first_tx.copy()
        df_first_tx.drop(columns=['first_purchase_date', 'transaction_date', 'membership_card_id'], inplace=True)
        return (df_first_tx, df_first_tx_id)
    else:
        df_first_tx.drop(columns=['first_purchase_date', 'transaction_date', 'membership_card_id'], inplace=True)
        return (df_first_tx)

def get_percent(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def onehote(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq2 = [mapping[i] for i in sequence]
    return np.eye(4)[seq2]

