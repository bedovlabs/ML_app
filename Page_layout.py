
import streamlit as st
from PIL import Image
from sklearn import datasets
import pandas as pd

class main_page(object):
    def __init__(self):
        st.set_page_config(layout="wide")
        #####Main Page s
        self.image=Image.open('media/bedo2.png')
        st.image(self.image,width=250)
        st.title('Bedo AI Virtual labs')
        #sidebar Content
        st.sidebar.image('media/bedo2.png',width=50)
        self.supervised_Regression_algorithms=['Linear Regression','K Nearset Neighpors KNN','Random Forest','Support Vector Machine','Decession Tree']
        self.supervised_classification=['Logistic Regression','K Nearset Neighpors KNN','Support Vector Machine','Decession Tree','Naive Bayes']
        self.unsupervised_classification=['K Means','K Nearset Neighpors KNN']
        self.all_algo=['Logistic Regression','K Nearset Neighpors KNN','K Means','Support Vector Machine','Decession Tree','Naive Bayes','Linear Regression']
        self.learntype=st.sidebar.radio("Choose learning enviornment",options=['Tutorials','Algorithm in Action'])
        if self.learntype=='Tutorials':
            self.algorithm=st.sidebar.selectbox("Agorithm ",options=self.all_algo)
        elif self.learntype=='Algorithm in Action':
            self.Algotypes=st.sidebar.selectbox("Agorithm Type",options=['Classification'])
            if self.Algotypes=="Regression":
               self.algorithm=st.sidebar.radio('Algorithm',options=self.supervised_Regression_algorithms )
            elif self.Algotypes=="Classification":
                 self.classification_type=st.sidebar.selectbox("Classification Type",options=['Supervised Calssification'])   
                 self.algorithm=st.sidebar.radio('Algorithm',options=self.supervised_classification if self.classification_type=='Supervised Calssification'else  self.unsupervised_classification )


        #st.sidebar.header("What do you want to learn today")
        #self.categories=st.sidebar.selectbox("Ml Category",options=['Supervised Learning','Unsupervised Learning'])
      
       # new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">'+  " 👉" + self.Algotypes + '</p>'
        #st.markdown(new_title, unsafe_allow_html=True)
        #self.algorithm=st.sidebar.radio('Algorithm',options=self.classification_algorithms if self.Algotypes=='Classification'else self.Regression_algorithms )

    def gen_new_data():
        Feature,Target = datasets.make_classification(
        n_samples=100,  # row number
        n_features=3, # feature numbers
        n_classes = 2, # The number of classes 
        n_redundant=0,
        n_clusters_per_class=1,#The number of clusters per class
        weights=[0.5,0.5] )    
        df=pd.DataFrame(Feature,Target)
        df=df.drop(columns=2)
        df[2]=df.index
        df=df.reset_index(drop=True)
        #scaler=MinMaxScaler()
        #df[[0, 1]] = scaler.fit_transform(df[[0, 1]])
        df.to_csv('data/new.txt',sep=',',index=False)
        return df
    def alignh(lines,colname):
       for i in range (1 , lines):
         colname.markdown("#")