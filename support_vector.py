

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.close("all")

from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC
from celluloid import Camera
import streamlit.components.v1 as components
from Page_layout import main_page
import streamlit as st
import seaborn as sns


class svm(object):
    def __init__(self,Kernel,C,Gammas,df):
        self.kernel=Kernel
        self.C=C
        self.Gammas=Gammas
        df=df
         #generate an SVM model
        features = ["sepal width (cm)", "petal length (cm)"]
        flowers = [0,1]
        target_names=['Iris-virginica','Iris-setosa']
        SVM_model1 = SVC(kernel=self.kernel,C=self.C,gamma=self.Gammas)

        
       # SVM_model2 = SVC(kernel="rbf",C=C,gamma=gamma)

        #train the SVM model with the first 2 instances
        #otherwise SVM will throw error when tried to train SVM feeding samples one-by-one
        #(the permutation seed above is determined for this)
        init_rows = 2
        X = df.iloc[:init_rows,:2]
        y = df["Targets"][:init_rows]
        SVM_model1.fit(X,y)
        #SVM_model2.fit(X,y)

        classes = df.Targets.value_counts().index
        scores1 = [SVM_model1.score(X,y)]
       # scores2 = [SVM_model2.score(X,y)]

        # Dynamic plotting part
        fig = plt.figure(figsize=(8,5))
        self.camera = Camera(fig)
        ax1 = fig.add_subplot(111)
       # ax2 = fig.add_subplot(122)

        #train SVM dynamically in the for loop by adding data in the training set
        for i in range(init_rows+1,df.shape[0],3):
            
        #    ax1.cla()
        #    ax2.cla()
            
            #extend the data with new entries in each loop and train SVM
            X = df.iloc[:i,:2]
            y = df["Targets"][:i]
            SVM_model1.fit(X,y)
           # SVM_model2.fit(X,y)
            
            f1 = df[features[0]][:i]
            f2 = df[features[1]][:i]
            t = df["Targets"][:i]
            
            f11 = ax1.scatter(f1[t==classes[0]],f2[t==classes[0]],c="cornflowerblue",marker="o")
            f12 = ax1.scatter(f1[t==classes[1]],f2[t==classes[1]],c="sandybrown",marker="^")

            #f21 = ax2.scatter(f1[t==classes[0]],f2[t==classes[0]],c="cornflowerblue",marker="o")
            #f22 = ax2.scatter(f1[t==classes[1]],f2[t==classes[1]],c="sandybrown",marker="^")
            
            #calculate the model accuracy
            scores1.append(SVM_model1.score(X,y))
            #scores2.append(SVM_model2.score(X,y))
            
            #draw the SVM boundary line
            #prepare data for decision boundary plotting
            x_min = X.iloc[:,0].min()
            x_max = X.iloc[:,0].max()
            y_min = X.iloc[:,1].min()
            y_max = X.iloc[:,1].max()
            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z1 = SVM_model1.decision_function(np.c_[XX.ravel(), YY.ravel()])
            Z1 = Z1.reshape(XX.shape)
            #Z2 = SVM_model2.decision_function(np.c_[XX.ravel(), YY.ravel()])
           # Z2 = Z2.reshape(XX.shape)

            #plot the decision boundary
            ax1.contour(XX, YY, Z1, colors=['darkgrey','dimgrey','darkgrey'],
                        linestyles=[':', '--', ':'], levels=[-.5, 0, .5])
          #  ax2.contour(XX, YY, Z2, colors=['darkgrey','dimgrey','darkgrey'],
          #              linestyles=[':', '--', ':'], levels=[-.5, 0, .5])
            
            #title wont work with celluloid package, text is an alternative to workaround
           # ax1.text(0.25, 1.03, "SVM (Linear) Training Accuracy: %.2f"%(SVM_model1.score(X,y)), 
            #        fontweight="bold", transform=ax1.transAxes)
            ax1.text(0.25, 1.03, f"SVC {self.kernel}  Training Accuracy: %.2f"%(SVM_model1.score(X,y)), 
                    fontweight="bold", transform=ax1.transAxes)
            ax1.set_xlim([-3,3])
            ax1.set_ylim([-3,3])
            
            #ax2.text(0.25, 1.03, "SVM (RBF) Training Accuracy: %.2f"%(SVM_model2.score(X,y)), 
           #         fontweight="bold", transform=ax2.transAxes)
           # ax2.set_xlim([-3,3])
           # ax2.set_ylim([-3,3])
            
            #x-y labels
            ax1.set_xlabel(features[0])
            ax1.set_ylabel(features[1])
           # ax2.set_xlabel(features[0])
           # ax2.set_ylabel(features[1])
            
        #    #dynamic text position for the boundary function
        #    if func_text_pos_y < 2.95:
        #        ax.text(-2, func_text_pos_y, r"$y = \frac{%.2f}{%.2f}+\frac{%.2f}{%.2f}x$"%(-w0,w2,-w1,w2), 
        #                fontweight="bold")
                
            ax1.legend([f11,f12],target_names,fontsize=9)
           # ax2.legend([f21,f22],iris.target_names[flowers],fontsize=9)
        #    plt.pause(0.1)
        
            #take a snapshot of the figure
            self.camera.snap()
        #animation=self.anim()
        return 
    def anim(self):#create animation
        anim =self.camera.animate()
        #save the animation as a gif file
        #anim.save("SVM_Boundary_RBF_vs_Linear.gif",writer="pillow")
        return anim
# %%

class supervised_svm():
    def __init__(self):
        global timer


    def svm_tut(self):
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            kernel = Try_algorithm.selectbox("Kernel",options=["linear","rbf","poly"])
            C = Try_algorithm.selectbox("C",options=[0.1, 1, 10, 100, 1000])
            gammas= Try_algorithm.selectbox("Gammas",options=[0.1, 1, 10, 100])
            Try_col1,Try_col2=Try_algorithm.columns((5,5))
            
            Go_model_BTN=Try_col1.button("Go")

            Generate_dataBTN =Try_col2.button("Generate New Data")
        
            if Go_model_BTN:
                from sklearn.datasets import load_iris
                from sklearn import preprocessing

                iris = load_iris()

        #features to use
                features = ["sepal width (cm)", "petal length (cm)"]

                #only versicolor and virginica are the targets
                flowers = [1,2]
                target_cond = (iris.target == flowers[0]) | (iris.target == flowers[1])

                #construct a dataframe with the conditions
                df_features = pd.DataFrame(preprocessing.scale(iris.data[target_cond,:]),
                                        columns = iris.feature_names)
                df_features = df_features[features]
                df_targets = pd.DataFrame(iris.target[target_cond],columns=["Targets"])
                df = pd.concat([df_features,df_targets],axis=1)

                #shuffle the dataset
                df = df.reindex(np.random.RandomState(seed=2).permutation(df.index))
                df = df.reset_index(drop=True)
            #%% train SVM dynamically by adding new data in the loop
                svm1=svm(kernel,float(C),float(gammas),df)
                svmanim=svm1.anim()
                components.html(svmanim.to_jshtml(), height=600,width=1000,scrolling=True)
              
            if Generate_dataBTN:
                df=main_page.gen_new_data()
                cols=["sepal width (cm)", "petal length (cm)","Targets"]
                df.columns=cols
                df = df.reindex(np.random.RandomState(seed=2).permutation(df.index))
                df = df.reset_index(drop=True)
                
                svm1=svm(kernel,float(C),float(gammas),df)
                svmanim=svm1.anim()
                components.html(svmanim.to_jshtml(), height=600,width=1000,scrolling=True)



                
                df.columns=cols
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.subheader("Support vector Machine Component")
            how_col1.image("media/svm2.png")
            how_col1.subheader("Support vector Machine General Model structure")
            how_col1.image("media/svm general model.png")
            how_col1.subheader("Support vector Machine Data piplining")
            how_col1.image("media/svmboundry.png")

    def svm_inaction(self):
        tab_text,tab_images,tab_irrig,tab_Disease=st.tabs(("Articl Classification","Images Classification","Smart Irrigation",'Disease Diagnostic'))
        tab_textData,tab_textGraphs=tab_text.tabs(("Data","Graphs"))
        tab_imagesData,tab_imagesGraph=tab_images.tabs(("Data","Graphs"))
        tab_irrigData,tab_irrigGraph=tab_irrig.tabs(("Data","Graphs"))
        tab_DiseaseData,tab_DiseaseGraph=tab_Disease.tabs(("Data","Graphs"))


        with tab_DiseaseData:
            from Diseses import svm_disese
            Disease_classifier=svm_disese()
            df,X,y,max_feat=Disease_classifier.load_data()
            tab_DiseaseData.subheader("Sample Data")
            tab_DiseaseData.write(df.sample(10))
            heart_col1,heart_col2=tab_DiseaseData.columns((6,6))
            heart_col2.subheader("Missing Values ")
            heart_col2.write(df.isnull().sum())
            heart_col1.subheader(" Classes Representation")
            heart_col1.write( df['Disease'].value_counts())
           
            xtrain,xtest,ytrain,ytest=Disease_classifier.train_test(X,y)
            
            
            Disease_classifier.fit_model(xtrain,ytrain)
            accuracy=Disease_classifier.test_model(xtest,ytest)
            heart_col2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")
            heart_col1.subheader("Predict Disease")
                    
            xs=heart_col1.multiselect("select Symptoms",options= max_feat)
            prediction = Disease_classifier.predict_input(xs)
            heart_col2.subheader("The predicted Disease is "+ prediction)
            



        with tab_DiseaseGraph:
           
            #graphcol1,graphcol2=tab_DiseaseGraph.columns((5,5))
            df1=Disease_classifier.visualize_word_freq(df['symptoms'],50)
            fig,ax=plt.subplots()
            ax=df1.plot(kind='bar',title="TITLE")
            st.write(df1)
            st.pyplot(fig)
            
            







        with tab_textData:
            
            from Articles_classification import svc_text
            text_classifioer=svc_text()    
            df=text_classifioer.load_articles()
            tab_textData.subheader("Sample Data")
            tab_textData.write(df.sample(10))
            text_col1,text_col2=tab_textData.columns((6,6))
            text_col1.subheader("Missing values ")
            text_col1.write(df.isnull().sum())
            text_col2.subheader(" Equal Class representation")
            text_col2.write( df['category'].value_counts())
            text_col1.subheader("predict article  Category ")
            article_to_predict=text_col2.text_area("Put Your Text Here",height=200)
            text_col1.write(text_classifioer.predict_input(article_to_predict))
       
        with tab_textGraphs:
            graphcol1,graphcol2=tab_textGraphs.columns((6,6))    
            fig, ax = plt.subplots()  
            ax = df['category'].value_counts().plot(kind='bar',
                                    figsize=(6,4),
                                    title="Ctegories distribution")
            ax.set_xlabel("Category Names")
            ax.set_ylabel("Frequency")           
          
            graphcol1.pyplot(fig)


            count = df['body'].str.split().str.len()
            count.index = count.index.astype(str) + ' words:'
            count.sort_index(inplace=True)

            fig, ax = plt.subplots()  
            ax = count.describe().plot(kind='line',
                                    figsize=(6,4),
                                    title="Article statistics")
            ax.set_xlabel("Metric")
            ax.set_ylabel("No of words")           

            graphcol2.pyplot(fig)


            
        with tab_imagesData :   

            from PIL import Image
            from skimage.transform import resize
            from skimage.io import imread
            from image_recognition import svm_image
            import os
           
            tab_imagesData.subheader("Sample Data ")
            img_col1,img_col2=tab_imagesData.columns((6,6))
            cat_dog_classifier=svm_image()
            x,y,df=cat_dog_classifier.load_data()
            datadir='media/animals/' 
            for i in ['Cats','Dogs']:
                path=os.path.join(datadir,i)
                for img,j in zip(os.listdir(path),range(1,4)):
                  if i=='Cats':
                    image = Image.open(str(path+"/"+img))
                    new_image = image.resize((200, 200))
                    img_col1.image(new_image,width=150)
                  else:
                    image = Image.open(str(path+"/"+img))
                    new_image = image.resize((200, 200))
                    img_col2.image(new_image, width=150)

            img_col1.subheader("Classes Representation")
            img_col1.write(df['Target'].value_counts())

            img_col1.subheader("Predict Animal Type")
              
            bottom_image = st.file_uploader('', type=['jpg','png'])
            if bottom_image is not None:
                 image = Image.open(bottom_image)
                 new_image = image.resize((150, 150))
                 st.image(new_image)
                 img_array = np.array(image)
                 img_resized=resize(img_array,(250,270,3))
                 img=img_resized.transpose(2,0,1).reshape(3,-1)
                 prediction=cat_dog_classifier.predict_input(img)
                 st.write(prediction)
                 st.write("Dog" if np.bincount(prediction).argmax()==1 else "Cat")
                
                                 
        with tab_irrigData :
            from smart_irrigation import svm_irr
            irrigation_svm=svm_irr()
            df=irrigation_svm.load_data()
           
            tab_irrigData.subheader("Sample Data")
            tab_irrigData.write(df.sample(10))
            tab_irrigData.subheader("Irrigation Statistics")
            tab_irrigData.write(df.describe())
            irr_col1,irr_col2=tab_irrigData.columns((6,6))
            corr=df.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("dark"):
                f, ax = plt.subplots(figsize=(6, 6))
                ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
            irr_col1.subheader("Heat Map")
            irr_col1.pyplot(f)
            irr_col2.subheader("Missing Values ")
            irr_col2.write(df.isnull().sum())
            irr_col2.subheader(" Classes Representation")
            irr_col2.write( df['Irrigation'].value_counts())
           
            xtrain,xtest,ytrain,ytest=irrigation_svm.train_test(df)
            #irrigation_svm.fit_model(xtrain,xtest,ytrain,ytest)
            accuracy=irrigation_svm.test_model(xtest,ytest)
            irr_col2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")
            irr_col1.subheader("Predict Irrigation State")
            user_input=irrigation_svm.user_input_smart_irr(irr_col1,irr_col2)
            prediction=irrigation_svm.predict_input(user_input)
            main_page.alignh(10,irr_col2)
            irr_col2.subheader("   predicted irrigation state is   "+ str(prediction))


        with tab_irrigGraph:
            graphcol1,graphcol2=tab_irrigGraph.columns((5,5))
            x=df.iloc[:,:5]
            x.hist()
            plt.show()
            graphcol1.header("Features Histogram")
            graphcol1.pyplot(plt)
            #fig, ax = plt.subplots()
            #ax.hist(x, bins=20)
           # tabirisGraphs.pyplot(fig)
            x.plot()
            plt.show()
            graphcol2.header("Features Plot")
            graphcol2.pyplot(plt)

            #redvize
            pd.plotting.radviz(df, 'Irrigation')
            graphcol1.header("Radviz Plot")
            graphcol1.pyplot(plt)
            #Andrews_curves plot
            pd.plotting.andrews_curves(df, 'Irrigation')
            graphcol2.header("Andrews Curves Plot")
            graphcol2.pyplot(plt)
            #Scatter Matrix plot
            axes=pd.plotting.scatter_matrix(df, alpha=0.2)
            graphcol1.header("Scatter Matrix")
            for ax in axes.flatten():
                ax.xaxis.label.set_rotation(90)
                ax.yaxis.label.set_rotation(0)
                ax.yaxis.label.set_ha('right')
            graphcol1.pyplot(plt)
            #feature ploting
            x=df.iloc[:100,:]
            y=np.arange(0,100)
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            x['y']=y
            ax1 = x.plot(kind='scatter', y='CropType', x='y', color=color[0],rot=90)    
            ax2 = x.plot(kind='scatter', y='CropDays', x='y', color=color[1], ax=ax1)    
            ax3 = x.plot(kind='scatter', y='SoilMoisture', x='y', color=color[2], ax=ax1)
            ax4 = x.plot(kind='scatter', y='temperature', x='y', color=color[3],ax=ax1)    
            ax5 = x.plot(kind='scatter', y='Humidity', x='y', color=color[4],ax=ax1)  

            graphcol2.header("Feature Scatter Matrix")
            plt.xlabel("Features")
            plt.ylabel("Irrigation")
            graphcol2.pyplot(plt)
            #CropType,CropDays,SoilMoisture,temperature,Humidity,Irriga

            
 
    def svm_applied(self):
        pass
# %%
