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
from sklearn import preprocessing


class fish_weight_prediction(object):
    # First, let's load the data

    def fish_weight_load_explore_split_data():
        df_fishw= pd.read_csv('data/fish.csv')
        df_fishw.columns = ['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight']
        le = preprocessing.LabelEncoder()
        df_fishw['Species']=le.fit_transform(df_fishw['Species'])
        X, y = df_fishw[df_fishw.columns[0:-1]].values, df_fishw[df_fishw.columns[-1]].values
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
       
    def user_input_feature_fish(Fish_co1,Fish_col2):
        #['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight']
        options=['Bream','Parkki', 'Perch', 'Pike','Roach','Smelt','Whitefish']
        Species = Fish_co1.selectbox('Species', options=options)
        Length1 = Fish_co1.slider('Length1', 0, 1650, 1)
        Length2 = Fish_co1.slider('Length2',5.0,65.0, 0.1)
        Length3 = Fish_co1.slider('Length3', 5.0,65.0, 0.1)
        Height=Fish_co1.slider('Height', 5.0,85.0, 0.1)
        Width=Fish_co1.slider('Width',1.000, 25.000, step=0.001)
        data = {'Species': options.index(Species),'Length1': Length1,
            'Length2': Length2,'Length3': Length3,
            'Height': Height, 'Width': Width}
        features = pd.DataFrame(data, index=[0])
        return features



