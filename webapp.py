import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import seaborn as sns
import numpy as np
from support_vector import svm,supervised_svm
import pickle
from Page_layout import main_page
from logisticreg import supervised_logistic
from Naive_Bayes import supervised_naive
from decession_tree import Supervised_decession_tree
from KNN import Supervised_knn
from unsupervied import unsupervised_kmeans

main_page=main_page()
col1, col2 = st.columns((3,3))



#basic class working

#st.write(algorithm)
if main_page.learntype=='Tutorials':
    if main_page.algorithm=='Logistic Regression':
            logistic_param=supervised_logistic() 
            logistic_param.logitic_tut()  
    elif main_page.algorithm=='Support Vector Machine':
            supervied_svm1=supervised_svm()
            supervied_svm1.svm_tut()
    elif main_page.algorithm=='Naive Bayes':
            supervised_naive=supervised_naive()
            supervised_naive.naive_tut()
    elif main_page.algorithm=='Decession Tree':
            supervised_dec_tree=Supervised_decession_tree()
            supervised_dec_tree.dec_tree_tut()
    elif main_page.algorithm=='K Nearset Neighpors KNN':
            supervised_knn=Supervised_knn()
            supervised_knn.knn_tut()
    elif  main_page.algorithm=='K Means':
            unsupervised_kmeans.kmeans_tut()

    elif main_page.algorithm=='Linear Regression':
            from regression import regression_algorithms
            regression_algorithms.linear_reg_tut()
            

      
elif main_page.learntype=='Algorithm in Action':
    
    if main_page.algorithm=='Logistic Regression':
            logistic_param=supervised_logistic() 
            logistic_param.logiticaction()
    elif  main_page.algorithm=='Support Vector Machine':
            supervied_svm1=supervised_svm()
            supervied_svm1.svm_inaction()
    elif main_page.algorithm=='Naive Bayes':
            supervised_naive=supervised_naive()
            supervised_naive.naive_inaction()
    elif main_page.algorithm=='Decession Tree':
            supervised_dec_tree=Supervised_decession_tree()
            supervised_dec_tree.dec_tree_inaction()
    elif  main_page.algorithm=='K Nearset Neighpors KNN':
            supervised_knn=Supervised_knn()
            supervised_knn.knn_inaction()
   # elif main_page.algorithm=='K Means':
         #   un_kmeans=kmeans()
          #  un_kmeans.kmeantu    
    elif main_page.algorithm=='Linear Regression':
           from regression import regression_algorithms
           regression_algorithms.linearinaction()
    elif main_page.algorithm=='Lasso Regression':
           from regression import regression_algorithms
           regression_algorithms.Lassoinaction()
    elif main_page.algorithm=='Gradient Boasting Regression':
           from regression import regression_algorithms
           regression_algorithms.Gradientinaction()    
    
elif  main_page.learntype=='Benchmark Algorithms':
            #st.header(main_page.algorithm )
            from benchmark import benchmark_classifiers
            multi_classifier=benchmark_classifiers()
   


    




   



       

# %%
