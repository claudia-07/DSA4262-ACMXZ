# Importing Libraries
import numpy as np
import datetime
from datetime import datetime, timedelta
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
import lightgbm
import shap

from parameters import (seed, num_splits, max_n_iter, scale_pos_weight, scoring)
from preprocessing import Preprocessing

class Modelling(Preprocessing):
    def __init__(self, df):
        super().__init__(df)

    def plot_feature_importance(model, X_val, max_display=20):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val,check_additivity=False)
        shap.summary_plot(shap_values, X_val, plot_type="bar",max_display=max_display)

    def modelling_lightgbm(self):
        '''Function for finding optimal parameters for lightgbm using hyperparameter tuning

        :Returns:
        ---------
            df_val: DataFrame
                DataFrame containing validation data 
        '''
        
        # initializing parameters for hyperparameter tuning
        trials = Trials()
        lgbm_tune_kwargs = {
            'learning_rate': hp.loguniform('learning_rate',-6.9, 0),
            'max_depth': hp.quniform('max_depth',2,8,1),
            'num_leaves':hp.quniform('num_leaves',30,50,1),
            'colsample_bytree': hp.uniform('colsample_bytree',0.1,0.5),
            'reg_alpha': hp.quniform('reg_alpha',1.1,1.5,0.1),
            'reg_lambda': hp.uniform('reg_lambda',1.1,1.5),
            'n_estimators': hp.quniform('n_estimators',100,200,1),
            'min_child_weight': hp.quniform('min_child_weight',0,10,1) 

        }

        # always use same k-folds for reproducibility
        kfolds = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
        
        # function to find optimal parameters
        def my_lgbm(config):

            config['n_estimators'] = int(config['n_estimators'])  # pass float eg loguniform distribution, use int
            # hyperopt needs left to start at 0 but we want to start at 2    
            config['max_depth'] = int(config['max_depth'])
            config['learning_rate'] = config['learning_rate']
        #     config['gamma'] = config['gamma']
            config['num_leaves'] = int(config['num_leaves'])
            config['reg_alpha'] = config['reg_alpha']
            config['reg_lambda'] = config['reg_lambda']
            config['colsample_bytree'] =config['colsample_bytree']
            config['min_child_weight'] =config['min_child_weight']


            lgbm_model = lightgbm.LGBMClassifier(
                n_jobs=-1,
                random_state=seed,
                scale_pos_weight=scale_pos_weight, 
                **config,
            )
            scores = cross_val_score(lgbm_model, self.X_train_enc, self.y_train, scoring=scoring, cv=kfolds)
            auc = np.mean(scores)
            return {"loss":-auc,"status":'ok'}

        start_time = datetime.now()
        print("%-20s %s" % ("Start Time", start_time))

        # search space
        best=fmin(fn = my_lgbm, # function to optimize
                space = lgbm_tune_kwargs, 
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = max_n_iter, # maximum number of iterations
                trials = trials, # logging
                rstate = np.random.default_rng(seed)  # fixing random state for reproducibility
                )
        
        print('-'*50)
        print('The best params:')
        print( best )
        print('\n\n')
        
        ## must make type correct
        best['n_estimators'] = int(best['n_estimators'])  # pass float eg loguniform distribution, use int
        best['max_depth'] = int(best['max_depth'])
        best['num_leaves'] = int(best['num_leaves'])


        end_time = datetime.now()
        print("%-20s %s" % ("Start Time", start_time))
        print("%-20s %s" % ("End Time", end_time))
        print(str(timedelta(seconds=(end_time-start_time).seconds)))

        # refit using best parameters
        lgbm_model = lightgbm.LGBMClassifier(
            n_jobs=-1,
            random_state=seed,    
            scale_pos_weight=scale_pos_weight,
            verbosity=1,
            **best,
        )
        print(lgbm_model)
        
        # scoring
        scores =cross_val_score(lgbm_model, self.X_train_enc, self.y_train, scoring=scoring, cv=kfolds)
        auc = np.mean(scores)
        print("auc:",auc)

        # feature importance
        lgbm_model.fit( self.X_train_enc, self.y_train) 
        self.plot_feature_importance(lgbm_model, self.X_val, max_display=10)
        
        # creating pipeline of model
        pipeline = make_pipeline(lgbm_model)
        filename = "genomics_lightgbm" + '.sav'
        pickle.dump(pipeline, open(filename, 'wb'))
        pipeline = pickle.load(open(filename, 'rb'))


        # concate train val numbers    
        y_pred_val = lgbm_model.predict(self.X_val)
        self.df_val['prediction'] = y_pred_val

        # confusion matrix
        CM = confusion_matrix(self.y_val,y_pred_val)
        print(CM)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        print("Total customers:",len(self.df_val))
        print("Target segment customers:",len(self.df_val[self.df_val['target']==1]))
        print("True Positives:",TP)
        print("True Positive rate:",TP/(TP+FN))
        print("False Positives:",FP)
        print("False Positive rate:",FP/(FP+TN))
        print("False Negatives:",FN)
        print("False Negative rate:",FN/(FN+TP))
        
        # evaluation metrics: using MCC for minority target
        mcc = matthews_corrcoef(self.y_val,y_pred_val)
        print("mcc:",mcc)
        accuracy = accuracy_score(self.y_val,y_pred_val)
        print ("Accuracy:", accuracy)
        report = classification_report(self.y_val,y_pred_val)
        print(report)
        
        return self.df_val    

