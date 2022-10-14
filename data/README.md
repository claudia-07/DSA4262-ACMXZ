## archuve
Merging parsed data with data.info, which contains classification labels

## merged data
Parsing of JSON and info files into dataframes

## Preprocessed data
### archive
Raw data before encoding
### test
Encoded data for testing 
Note: Require dropping of columns for X_test_enc before using
### training
Encoded data for training, separated by quantiles (25th, 50th, 75th) and mean
### validation
Encoded data for validation
Note: Require dropping of columns for X_val_enc before using

## raw data