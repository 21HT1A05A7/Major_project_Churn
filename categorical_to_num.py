import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import logging_code
from logging_code import setup_logging
logger=setup_logging("categorical_to_num")
import sys
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

def c_t_n(X_train_cat,X_test_cat):
    try:
        logger.info(f"Before X_train_cat Column : {X_train_cat.shape} : \n : {X_train_cat.columns}")
        logger.info(f"Before X_test_cat Column : {X_test_cat.shape} : \n : {X_test_cat.columns}")

        X_train_cat = X_train_cat.drop(['customerID'], axis=1)
        X_test_cat = X_test_cat.drop(['customerID'], axis=1)
        oh = OneHotEncoder(drop='first')
        oh.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'telecom_partner']])
        values_train = oh.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'telecom_partner']]).toarray()
        values_test = oh.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'telecom_partner']]).toarray()
        t1 = pd.DataFrame(values_train)
        t2 = pd.DataFrame(values_test)
        t1.columns = oh.get_feature_names_out()
        t2.columns = oh.get_feature_names_out()
        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, t1], axis=1)
        X_test_cat = pd.concat([X_test_cat, t2], axis=1)
        X_train_cat = X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'telecom_partner'], axis=1)
        X_test_cat = X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'telecom_partner'], axis=1)
        logger.info(f"After Nominal X_train_cat Column : {X_train_cat.shape} : \n : {X_train_cat.columns}")
        logger.info(f"After Nominal X_test_cat Column : {X_test_cat.shape} : \n : {X_test_cat.columns}")

        od = OrdinalEncoder()
        od.fit(X_train_cat[['Contract']])
        results_train = od.transform(X_train_cat[['Contract']])
        results_test = od.transform(X_test_cat[['Contract']])
        p1 = pd.DataFrame(results_train)
        p2 = pd.DataFrame(results_test)
        p1.columns = od.get_feature_names_out() + "_od"
        p2.columns = od.get_feature_names_out() + "_od"
        p1.reset_index(drop=True, inplace=True)
        p2.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, p1], axis=1)
        X_test_cat = pd.concat([X_test_cat, p2], axis=1)
        X_train_cat = X_train_cat.drop(['Contract'], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)
        logger.info(f"After Odinal X_train_cat Column : {X_train_cat.shape} : \n : {X_train_cat.columns}")
        logger.info(f"After Odinal X_test_cat Column : {X_test_cat.shape} : \n : {X_test_cat.columns}")

        logger.info(f"Train NUll values : {X_train_cat.isnull().sum()}")
        logger.info(f"Test NUll values : {X_test_cat.isnull().sum()}")
        return X_train_cat, X_test_cat

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
