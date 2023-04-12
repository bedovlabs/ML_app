import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


class fish_weight_prediction(object):
    # First, let's load the data

    def fish_weight_load_explore_split_data():
        df_fishw= pd.read_csv('data/fish.csv')
        #df_RPP_Original=df_fishw.copy(deep=True)
        #df_fishw.drop('No',inplace=True,axis=1)
        df_fishw.columns = ['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight']
        #Lets seperate the label from data set.

        #label_updated=df_fishw[df_fishw.columns[-1]]

        #df_fishw.drop('transaction date',inplace=True,axis=1)
        X, y = df_fishw[df_fishw.columns[1:-1]].values, df_fishw[df_fishw.columns[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        return df_fishw, X_train, X_test, y_train, y_test 
#*******************************************************************************
        #30% test data and 70% train data
    def  fish_weight_linear_reg(X_train,X_test,y_train,y_test):
   
        pipe_LR = Pipeline(steps=[ ('scaler', StandardScaler()),  ('estimator', LinearRegression())  ])
        model_LR=pipe_LR.fit(X_train,y_train)
        predictions_LR=model_LR.predict(X_test)
        # Display metrics
        mse_LR = mean_squared_error(y_test, predictions_LR)
        rmse_LR = np.sqrt(mse_LR)
        r2_LR = r2_score(y_test, predictions_LR)
        filename = 'savedmodels/fish_weight_linear_reg_model.sav'
        pickle.dump(model_LR, open(filename, 'wb'))
        return mse_LR,rmse_LR,r2_LR


    #Lasso model: LASSO is good at generalization(L1) of features where it sets some of the least correlation 
    #features to value near to zero, which enables the model to depend more on other features.
    #****************************************************************************************
    def fish_weight_lasso_reg(X_train,X_test,y_train,y_test):

        pipe_LASSO = Pipeline(steps=[('scaler', StandardScaler()),   ('estimator', Lasso())  ])

        model_LASSO=pipe_LASSO.fit(X_train,y_train)
        predictions_LASSO=model_LASSO.predict(X_test)

        mse_LASSO = mean_squared_error(y_test, predictions_LASSO)
        rmse_LASSO = np.sqrt(mse_LASSO)
        r2_LASSO = r2_score(y_test, predictions_LASSO)
        filename = 'savedmodels/fish_weight_lasso_reg_model.sav'
        pickle.dump(model_LASSO, open(filename, 'wb'))
        return mse_LASSO,rmse_LASSO,r2_LASSO
#***************************************************************************
    def fish_weight_gradient_bossting(X_train,X_test,y_train,y_test):

        # define the pipeline
        pipe_GBR = Pipeline(steps=[('scaler', StandardScaler()),     ('estimator', GradientBoostingRegressor(loss='squared_error',learning_rate=0.04,n_iter_no_change=10))   ])

        model_GBR=pipe_GBR.fit(X_train,y_train)
        predictions_GBR=model_GBR.predict(X_test)

        mse_GBR = mean_squared_error(y_test, predictions_GBR)
        rmse_GBR = np.sqrt(mse_GBR)
        r2_GBR = r2_score(y_test, predictions_GBR)
        filename = 'savedmodels/fish_weight_gradient_bossting_model.sav'
        pickle.dump(model_GBR, open(filename, 'wb'))
        return mse_GBR,rmse_GBR,r2_GBR
       

   

