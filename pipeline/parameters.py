import pandas as pd
import numpy as np

# features - model
model_features_list=['membership_card_id','transaction_date','country', 'gender',
                        'avg_items_bought_prestige_diamond',
                        'avg_items_bought_loyal_t',
                        'avg_items_bought_prestige_ruby',
                        'avg_items_bought_jade',
                        'avg_items_bought_non_member', 
                        'avg_prestige_diamond_total_amount_spent',
                        'avg_loyal_t_total_amount_spent',
                        'avg_prestige_ruby_total_amount_spent', 
                        'avg_jade_total_amount_spent',
                        'avg_non_member_total_amount_spent',
                        'customer_most_spent_prod_cat',
                        'customer_most_spent_brand',
                        'customer_most_spent_merchandise_grp',
                        'customer_most_popular_brand',
                        'customer_most_popular_merch_group',
                        'customer_most_popular_prod_cat',
                        'customer_average_discount_amount',
                        'customer_average_discount_percentage',
                        'customer_minimum_discount_amount',
                        'customer_maximum_discount_amount',
                        'customer_minimum_discount_percentage',
                        'customer_maximum_discount_percentage',
                        'sa_items_sold_last_30_days',
                        'sa_sales_avg',
                        'sa_repeaters', 
                        'sa_repeaters_percentage', 
                        'sa_prestige_diamond_members_percentage',
                        'sa_loyal_t_members_percentage', 
                        'sa_prestige_ruby_members_percentage',
                        'sa_jade_members_percentage',
                        'sa_non_members_percentage',
                        'sa_very_high_spenders_percentage', 
                        'sa_high_spenders_percentage',
                        'sa_medium_spenders_percentage', 
                        'sa_low_spenders_percentage',
                        'customer_curr_total_spent_usd',
                        'customer_curr_most_expensive',
                        'customer_curr_item_qty',
                        'customer_curr_spent_wja',
                        'customer_curr_spent_fashion',
                        'customer_curr_spent_beauty', 
                        'customer_curr_spent_swt',
                        'customer_curr_item_qty_wja', 
                        'customer_curr_item_qty_fashion',
                        'customer_curr_item_qty_beauty',
                        'customer_curr_item_qty_swt',
                        'customer_relative_purchasing_capacity_beauty',
                        'customer_relative_purchasing_capacity_wja',
                        'customer_relative_purchasing_capacity_fashion',
                        'customer_relative_purchasing_capacity_swt',
                        'target']


non_repeaters_features_list=['membership_card_id','transaction_date','country', 'gender',
                        'avg_items_bought_prestige_diamond',
                        'avg_items_bought_loyal_t',
                        'avg_items_bought_prestige_ruby',
                        'avg_items_bought_jade',
                        'avg_items_bought_non_member', 
                        'avg_prestige_diamond_total_amount_spent',
                        'avg_loyal_t_total_amount_spent',
                        'avg_prestige_ruby_total_amount_spent', 
                        'avg_jade_total_amount_spent',
                        'avg_non_member_total_amount_spent',
                        'customer_most_spent_prod_cat',
                        'customer_most_spent_brand',
                        'customer_most_spent_merchandise_grp',
                        'customer_most_popular_brand',
                        'customer_most_popular_merch_group',
                        'customer_most_popular_prod_cat',
                        'customer_average_discount_amount',
                        'customer_average_discount_percentage',
                        'customer_minimum_discount_amount',
                        'customer_maximum_discount_amount',
                        'customer_minimum_discount_percentage',
                        'customer_maximum_discount_percentage',
                        'sa_items_sold_last_30_days',
                        'sa_sales_avg',
                        'sa_repeaters', 
                        'sa_repeaters_percentage', 
                        'sa_prestige_diamond_members_percentage',
                        'sa_loyal_t_members_percentage', 
                        'sa_prestige_ruby_members_percentage',
                        'sa_jade_members_percentage',
                        'sa_non_members_percentage',
                        'sa_very_high_spenders_percentage', 
                        'sa_high_spenders_percentage',
                        'sa_medium_spenders_percentage', 
                        'sa_low_spenders_percentage',
                        'customer_curr_total_spent_usd',
                        'customer_curr_most_expensive',
                        'customer_curr_item_qty',
                        'customer_curr_spent_wja',
                        'customer_curr_spent_fashion',
                        'customer_curr_spent_beauty', 
                        'customer_curr_spent_swt',
                        'customer_curr_item_qty_wja', 
                        'customer_curr_item_qty_fashion',
                        'customer_curr_item_qty_beauty',
                        'customer_curr_item_qty_swt',
                        'customer_relative_purchasing_capacity_beauty',
                        'customer_relative_purchasing_capacity_wja',
                        'customer_relative_purchasing_capacity_fashion',
                        'customer_relative_purchasing_capacity_swt',
                        ]

# features - preprocessing
key_columns = ['country'] # columns that cannot be null

non_nan_cols = ['avg_prestige_diamond_total_amount_spent', 'avg_prestige_ruby_total_amount_spent', # columns that have null values that should be converted to 0
                   'avg_jade_total_amount_spent', 'avg_non_member_total_amount_spent',
                   'avg_items_bought_prestige_diamond', 'avg_items_bought_prestige_ruby', 
                   'avg_items_bought_jade','avg_items_bought_loyal_t','avg_items_bought_non_member', 'customer_curr_spent_fashion',
                   'customer_curr_spent_wja', 'customer_curr_spent_swt', 'customer_curr_spent_beauty',
                   'customer_relative_purchasing_capacity_beauty', 'customer_relative_purchasing_capacity_wja', 
                   'customer_relative_purchasing_capacity_fashion', 'customer_relative_purchasing_capacity_swt',
                   'customer_curr_item_qty_wja','customer_curr_item_qty_fashion', 'customer_curr_item_qty_swt', 'customer_curr_item_qty_beauty',
                   'customer_average_discount_amount','customer_average_discount_percentage', 'customer_minimum_discount_amount',
                   'customer_maximum_discount_amount', 'customer_minimum_discount_percentage','customer_maximum_discount_percentage']

# features - encoding
target_encode_cols = ['country', 'customer_most_spent_prod_cat', 'customer_most_spent_brand','customer_most_popular_brand','customer_most_popular_prod_cat']
one_hot_encode_cols = ['gender', 'customer_most_spent_merchandise_grp', 'customer_most_popular_merch_group']
target_col = 'target'
numeric_cols = ['avg_items_bought_prestige_diamond','avg_items_bought_loyal_t','avg_items_bought_prestige_ruby',
                           'avg_items_bought_jade', 'avg_items_bought_non_member','avg_prestige_diamond_total_amount_spent', 
                           'avg_loyal_t_total_amount_spent', 'avg_prestige_ruby_total_amount_spent', 'avg_jade_total_amount_spent', 
                           'avg_non_member_total_amount_spent', 'sa_items_sold_last_30_days', 'sa_sales_avg', 'sa_repeaters', 
                           'sa_repeaters_percentage', 'sa_prestige_diamond_members_percentage', 'sa_loyal_t_members_percentage',
                           'sa_prestige_ruby_members_percentage', 'sa_jade_members_percentage', 'sa_non_members_percentage', 
                           'sa_very_high_spenders_percentage', 'sa_high_spenders_percentage', 'sa_medium_spenders_percentage',
                           'sa_low_spenders_percentage', 'customer_curr_total_spent_usd', 'customer_curr_most_expensive', 'customer_curr_item_qty',
                           'customer_curr_spent_wja', 'customer_curr_spent_fashion', 'customer_curr_spent_beauty',
                           'customer_curr_spent_swt', 'customer_curr_item_qty_wja', 'customer_curr_item_qty_fashion', 
                           'customer_curr_item_qty_beauty', 'customer_curr_item_qty_swt',
                           'customer_relative_purchasing_capacity_beauty', 'customer_relative_purchasing_capacity_wja',
                           'customer_relative_purchasing_capacity_fashion', 'customer_relative_purchasing_capacity_swt',
                           'customer_average_discount_amount','customer_average_discount_percentage',
                           'customer_minimum_discount_amount', 'customer_maximum_discount_amount',
                           'customer_minimum_discount_percentage','customer_maximum_discount_percentage']

# oversampling / undersampling
undersampling_strategy = 0.15
oversampling_strategy = 0.20

# data splitting
stratify_col = 'target' # column used to stratify data
seed = 42 # random state
id_col = 'membership_card_id'
    # the following 3 percentages should sum to 1
train_percent = 0.8 # percentage of train data
validation_percent = 0.1 # percentage of validation data
test_percent = 0.1 # percentage of test data










