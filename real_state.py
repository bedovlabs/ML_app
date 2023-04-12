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
import streamlit as st

class realstate_prediction(object):
    # First, let's load the data

    def  realstate_load_explore_split_data():
        df_RPP = pd.read_csv('data/Real_estate.csv')
       
        df_RPP_Original=df_RPP.copy(deep=True)
        df_RPP.drop('No',inplace=True,axis=1)
        df_RPP.columns = ['transaction date', 'house age', 'distance to the nearest MRT station', 'number of convenience stores', 'latitude', 'longitude', 'house price of unit area']
        #Lets seperate the label from data set.

        label_updated=df_RPP[df_RPP.columns[-1]]

        df_RPP.drop('transaction date',inplace=True,axis=1)
        X, y = df_RPP[df_RPP.columns[0:-1]].values, df_RPP[df_RPP.columns[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        
        return df_RPP, X_train, X_test, y_train, y_test 

        #30% test data and 70% train data
    def  realstate_linear_reg(X_train,X_test,y_train,y_test):
   
        pipe_LR = Pipeline(steps=[ ('scaler', StandardScaler()),  ('estimator', LinearRegression())  ])
        model_LR=pipe_LR.fit(X_train,y_train)
        predictions_LR=model_LR.predict(X_test)
        # Display metrics
        mse_LR = mean_squared_error(y_test, predictions_LR)
        rmse_LR = np.sqrt(mse_LR)
        r2_LR = r2_score(y_test, predictions_LR)
        filename = 'savedmodels/realstate_linear_reg_model.sav'
        pickle.dump(model_LR, open(filename, 'wb'))
        return mse_LR,rmse_LR,r2_LR


    #Lasso model: LASSO is good at generalization(L1) of features where it sets some of the least correlation 
    #features to value near to zero, which enables the model to depend more on other features.
    def realstate_lasso_reg(X_train,X_test,y_train,y_test):

        pipe_LASSO = Pipeline(steps=[('scaler', StandardScaler()),   ('estimator', Lasso())  ])

        model_LASSO=pipe_LASSO.fit(X_train,y_train)
        predictions_LASSO=model_LASSO.predict(X_test)

        mse_LASSO = mean_squared_error(y_test, predictions_LASSO)
        rmse_LASSO = np.sqrt(mse_LASSO)
        r2_LASSO = r2_score(y_test, predictions_LASSO)
        filename = 'savedmodels/realstate_lasso_reg_model.sav'
        pickle.dump(model_LASSO, open(filename, 'wb'))

        return mse_LASSO,rmse_LASSO,r2_LASSO

    def realstate_gradient_bossting(X_train,X_test,y_train,y_test):

        # define the pipeline
        pipe_GBR = Pipeline(steps=[('scaler', StandardScaler()),     ('estimator', GradientBoostingRegressor(loss='squared_error',learning_rate=0.04,n_iter_no_change=10))   ])

        model_GBR=pipe_GBR.fit(X_train,y_train)
        predictions_GBR=model_GBR.predict(X_test)

        mse_GBR = mean_squared_error(y_test, predictions_GBR)
        rmse_GBR = np.sqrt(mse_GBR)
        r2_GBR = r2_score(y_test, predictions_GBR)
        filename = 'savedmodels/realstate_gradient_bossting_model.sav'
        pickle.dump(model_GBR, open(filename, 'wb'))
        return mse_GBR,rmse_GBR,r2_GBR
    
    def user_input_feature_realstate(real_col1,real_col2):
        #[ 'house age', 'distance to the nearest MRT station', 'number of convenience stores', 'latitude', 'longitude',]
        house_age = real_col1.slider('House Age', 0.0, 50.0 , step=0.1)
        distance_to_MRT = real_col1.slider('distance to the nearest MRT', 120, 990, 1)
        number_convenience_stores = real_col1.slider('number of convenience stores',10,263, 1)
        latitude = real_col2.slider('latitude', 20.0000, 30.0000, step=0.0001)
        longitude=real_col2.slider('longitude',121.47353, 121.56627, step=0.0001)
        data = {'house age': house_age,'distance to the nearest MRT': distance_to_MRT,
            'number of convenience stores': number_convenience_stores,'latitude': latitude,
            'longitude': longitude, }
        features = pd.DataFrame(data, index=[0])
        return features


   

#print("Linear Regression R2 ", r2_score(y_test, predictions_LR))
#print("LASSO R2 ", r2_score(y_test, predictions_LASSO))
#print("Gradient boosting regressor Regression R2 ", r2_score(y_test, predictions_GBR))
#print("Ridge Regression R2 ", r2_score(y_test, predictions_Ridge))
#print("Stochastic Graident descent R2 ", r2_score(y_test, predictions_SGD))

"""
 #Ridge regressor: This model also like Lasso where Generalization happens(L2)
    from sklearn.linear_model import Ridge

    # define the pipeline
    pipe_Ridge = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('estimator', Ridge(alpha=0.0001))
    ])

    model_Ridge=pipe_Ridge.fit(X_train,y_train)
    predictions_Ridge=model_Ridge.predict(X_test)

    mse_Ridge = mean_squared_error(y_test, predictions_Ridge)
    print("MSE_Ridge:", mse_Ridge)
    rmse_Ridge = np.sqrt(mse_Ridge)
    print("RMSE_Ridge:", rmse_Ridge)
    r2_Ridge = r2_score(y_test, predictions_Ridge)
    print("R2_Ridge:", r2_Ridge)

    #Stochastic Gradient Descent
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import PolynomialFeatures

    # define the pipeline
    pipe_SGD = Pipeline(steps=[
        ('scaler', StandardScaler()), # feature engineering to scale the features
        ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)), # 2 degree poly 
        ('estimator', SGDRegressor(max_iter=10000)) # learning rate 0.001
    ])

    model_SGD=pipe_SGD.fit(X_train,y_train)
    predictions_SGD=model_SGD.predict(X_test)

    mse_SGD = mean_squared_error(y_test, predictions_SGD)
    print("MSE_SGD:", mse_SGD)
    rmse_SGD = np.sqrt(mse_SGD)
    print("RMSE_SGD:", rmse_SGD)
    r2_SGD = r2_score(y_test, predictions_SGD)
    print("R2_SGD:", r2_SGD)
"""