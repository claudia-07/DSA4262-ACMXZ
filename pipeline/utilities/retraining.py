import numpy as np
import pandas as pd




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



