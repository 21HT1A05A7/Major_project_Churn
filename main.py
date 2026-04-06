'''
in this file i am going to call related function for data cleaning and model development
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import logging_code
from logging_code import setup_logging
logger=setup_logging("main")
from sklearn.model_selection import train_test_split
from mode_imputation import missing_values
from var_out import vt_outliers
from filter_methods import fm
from categorical_to_num import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs
import sys

class CHURN:
    def __init__(self,path):
        try:
          self.path=path
          self.df=pd.read_csv(self.path)#loading the data into  variable
          logger.info(f"Total data_size: {self.df.shape}")
          self.df['telecom_partner'] = self.df['PaymentMethod'].map({
              'Electronic check': 'Reliance Jio',
              'Mailed check': 'Airtel',
              'Bank transfer (automatic)': 'VI-Idea',
              'Credit card (automatic)': 'BSNL'
          })
          logger.info(self.df['telecom_partner'])
          logger.info(f"Null values : \n : {self.df.isnull().sum()}")
          self.df['TotalCharges'] =self.df['TotalCharges'].replace("", np.nan)
          self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
          logger.info(f"Null values : \n : {self.df['TotalCharges'].isnull().sum()}")
          logger.info(f"Null values : \n : {self.df.isnull().sum()}")
          self.X = self.df.drop(['Churn'], axis=1)  # Independent variables
          self.y = self.df['Churn']  # Dependent variable
          self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                  random_state=42)
          self.y_train = self.y_train.map({'Yes': 1, 'No': 0}).astype(int)
          self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)
          logger.info(
              f'Train data size :{len(self.X_train)}:{len(self.y_train)} \n{self.X_train.shape} : {self.y_train.shape}')
          logger.info(
              f'Test data size :{len(self.X_test)}:{len(self.y_test)} \n{self.X_test.shape} : {self.y_test.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def missing_values(self):
        try:
            self.X_train['TotalCharges'] = pd.to_numeric(self.X_train['TotalCharges'])
            self.X_test['TotalCharges'] = pd.to_numeric(self.X_test['TotalCharges'])
            self.X_train, self.X_test =missing_values(self.X_train, self.X_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def data_seperation(self):
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')

            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')
            logger.info(f'X_train_num_cols.columns = {self.X_train_num_cols.columns} : {self.X_train_num_cols.shape}')
            logger.info(f'X_test_num_cols.columns = {self.X_test_num_cols.columns}: {self.X_test_num_cols.shape}')
            logger.info(f"---------------------------------------------------------")
            logger.info(f'X_train_cat_cols.columns = {self.X_train_cat_cols.columns} : {self.X_train_cat_cols.shape}')
            logger.info(f'X_test_cat_cols.columns = {self.X_test_cat_cols.columns}: {self.X_test_cat_cols.shape}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def variable_transformation(self):
        try:
           logger.info(f"Before Train Column Name : {self.X_train_num_cols.columns}")
           logger.info(f"Before Test Column Name : {self.X_test_num_cols.columns}")
           self.X_train_num_cols,self.X_test_num_cols=vt_outliers(self.X_train_num_cols,self.X_test_num_cols)
           logger.info(f"After Train Column Name : {self.X_train_num_cols.columns}")
           logger.info(f"AFter Test Column Name : {self.X_test_num_cols.columns}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def feature_selection(self):
        try:
            self.X_train_num_cols,self.X_test_num_cols = fm(self.X_train_num_cols, self.X_test_num_cols, self.y_train,
                                                             self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def cat_to_num(self):
        try:
            self.X_train_cat_cols, self.X_test_cat_cols = c_t_n(self.X_train_cat_cols, self.X_test_cat_cols)
            # combine the data
            self.X_train_num_cols.reset_index(drop=True, inplace=True)
            self.X_train_cat_cols.reset_index(drop=True, inplace=True)
            self.X_test_num_cols.reset_index(drop=True, inplace=True)
            self.X_test_cat_cols.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num_cols, self.X_train_cat_cols], axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols, self.X_test_cat_cols], axis=1)

            logger.info(f"===========================================================")
            logger.info(f"Final Trainining data : {self.training_data.shape}")
            logger.info(f"{self.training_data.columns}")
            logger.info(f"{self.training_data.isnull().sum()}")

            logger.info(f"Final Testing data : {self.testing_data.shape}")
            logger.info(f"{self.testing_data.columns}")
            logger.info(f"{self.testing_data.isnull().sum()}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def data_balancing(self):
        try:
            logger.info(f"Number of Rows for Yes Customer {1} : {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for No Customer {0} : {sum(self.y_train == 0)}")
            logger.info(f"Training data size : {self.training_data.shape}")

            sm = SMOTE(random_state=42)

            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

            logger.info(f"Number of Rows for Good Customer {1} : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad Customer {0} : {sum(self.y_train_bal == 0)}")
            logger.info(f"Training data size : {self.training_data_bal.shape}")
            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
if __name__== "__main__":
    try:
       obj = CHURN('Customer-Churn.csv')
       obj.missing_values()
       obj.data_seperation()
       obj.variable_transformation()
       obj.feature_selection()
       obj.cat_to_num()
       obj.data_balancing()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
