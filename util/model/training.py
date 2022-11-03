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

def relative_position(percentile_df):
    percentile_df['position'] = percentile_df['position'].astype(int)
    # find the relative position of each read in each transcript
    percentile_df['relative_position'] = percentile_df.groupby(['transcript','gene_id'])['position'].transform(lambda x: (x - x.min())/(x.max()-x.min()))
    # note: have NAs because there's transcripts with only one position
    # fill the NAs with 0
    percentile_df['relative_position'] = percentile_df['relative_position'].fillna(0)
    return percentile_df

def get_percent(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def onehote(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq2 = [mapping[i] for i in sequence]
    return np.eye(4)[seq2]

