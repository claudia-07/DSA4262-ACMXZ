# DSA4262-ACMXZ-
DSA4262 Genomics Project

To explore how we preprocessed our data and performed modelling, you can look into our deployment and modelling folder.
You can test the scripts from deployment folder.

Our general project directory is as follows:

```
├── archive # our notebooks for testing of functions

├── data
    └── aws_predictions # csv files for predictions of aws data
    └── codes # consists of notebook to parse data
    └── model_training
        └── merged_data
        └── preprocessed_data
            └── archive
            └── test
            └── training
            └── validation
        └── raw_data 
    └── README.md

├── deployment
    └── preprocessing.py # To preprocess json data
    └── random_forest.py # To be run for predictions
    └── rf.pkl # pickled random forest model 

├── modelling
    └── lightgbm.ipynb
    └── random_forest.ipynb

├── prediction
    └── test_predictions.ipynb # finalised notebook used for predictions

├── util # consists of util scripts used in other scripts and notebooks
    └── model
        └── modelling.py
        └── training.py
    └── preprocess
        └── parameters.py
        └── preprocessing.py

└── info.parquet

└── requirements.txt

```
