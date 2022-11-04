# Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

# Load in encoded train 

X_train_path = "../data/raw_data/X_raw_enc.parquet"
X_train = pd.read_parquet(X_train_path)
y_train_path = "../data/raw_data/y_raw.parquet"
y_train = pd.read_parquet(y_train_path)

# Load in encoded test

X_test = 
y_test = 

# Best Model

best_model = RandomForestClassifier(n_estimators = 220, max_features = "sqrt", max_depth = 30,
                                    min_samples_split = 2, min_samples_leaf = 1, bootstrap = False,
                                    random_state= 42)


best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)