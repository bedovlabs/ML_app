
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from vega_datasets import data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
import streamlit.components.v1 as components
import seaborn as sns
from Page_layout import main_page
import pandas as pd
import pickle
class regression_algorithms(object):
   ###########################################################Tutorial#############################################################
    def linear_reg_tut():
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
        
            interval=st.slider("Interval",10,200,step=10)
            go_button=st.button("Go")
            if go_button :
                df = data.cars()
                # Drop rows with NaN
                df.dropna(subset=['Horsepower', 'Miles_per_Gallon'], inplace=True)
                x_data = []
                y_data = []
                # Transform data
                x = df['Horsepower'].to_numpy().reshape(-1, 1)
                y = df['Miles_per_Gallon'].to_numpy().reshape(-1, 1)

                fig, ax = plt.subplots()     # A tuple unpacking to unpack the only fig
                ax.set_xlim(30, 250)
                ax.set_ylim(5, 50)
                # Plotting 
                scatter, = ax.plot([], [], 'go', label='Horsepower vs. Miles_per_Gallon')
                line, = ax.plot([], [], 'r', label='Linear Regression')
                ax.legend()

                reg = LinearRegression()

                def animate(frame_num):
                    # Adding data
                    x_data.append(x[frame_num])
                    y_data.append(y[frame_num])
                    # Convert data to numpy array
                    x_train = np.array(x_data).reshape(-1, 1)
                    y_train = np.array(y_data).reshape(-1, 1)
                    # Fit values to a linear regression
                    reg.fit(x_train, y_train)

                    # update data for scatter plot
                    scatter.set_data((x_data, y_data))
                    # Predict value and update data for line plot
                    line.set_data((list(range(250)), reg.predict(np.array([entry for entry in range(250)]).reshape(-1, 1))))

                anim = FuncAnimation(fig, animate, frames=len(x), interval=interval)
                components.html(anim.to_jshtml(), height=600,width=600,scrolling=True)
                return 
        with How_it_work:
            col1,col2=st.columns((5,5))
            col1.image('media/linear1.png')
            col2.image('media/cost linear.png')
            col2.image('media/linearcost2.png')
            col1.image('media/gardient_linear.png')
            col1.image('media/lnear4.png')
    def lasso_regr_tut():
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
        
            interval=st.slider("Interval",10,200,step=10)
            go_button=st.button("Go")
            if go_button :
                df = data.cars()
                # Drop rows with NaN
                df.dropna(subset=['Horsepower', 'Miles_per_Gallon'], inplace=True)
                x_data = []
                y_data = []
                # Transform data
                x = df['Horsepower'].to_numpy().reshape(-1, 1)
                y = df['Miles_per_Gallon'].to_numpy().reshape(-1, 1)

                fig, ax = plt.subplots()     # A tuple unpacking to unpack the only fig
                ax.set_xlim(30, 250)
                ax.set_ylim(5, 50)
                # Plotting 
                scatter, = ax.plot([], [], 'go', label='Horsepower vs. Miles_per_Gallon')
                line, = ax.plot([], [], 'r', label='Linear Regression')
                ax.legend()

                reg = LinearRegression()

                def animate(frame_num):
                    # Adding data
                    x_data.append(x[frame_num])
                    y_data.append(y[frame_num])
                    # Convert data to numpy array
                    x_train = np.array(x_data).reshape(-1, 1)
                    y_train = np.array(y_data).reshape(-1, 1)
                    # Fit values to a linear regression
                    reg.fit(x_train, y_train)

                    # update data for scatter plot
                    scatter.set_data((x_data, y_data))
                    # Predict value and update data for line plot
                    line.set_data((list(range(250)), reg.predict(np.array([entry for entry in range(250)]).reshape(-1, 1))))

                anim = FuncAnimation(fig, animate, frames=len(x), interval=interval)
                components.html(anim.to_jshtml(), height=600,width=600,scrolling=True)
                return 
        with How_it_work:
            col1,col2=st.columns((5,5))
            col1.image('media/linear1.png')
            col2.image('media/cost linear.png')
            col2.image('media/linearcost2.png')
            col1.image('media/gardient_linear.png')
            col1.image('media/lnear4.png')
 
 
 
    def Gradient_boast_regr_tut():
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
        
            interval=st.slider("Interval",10,200,step=10)
            go_button=st.button("Go")
            if go_button :
                df = data.cars()
                # Drop rows with NaN
                df.dropna(subset=['Horsepower', 'Miles_per_Gallon'], inplace=True)
                x_data = []
                y_data = []
                # Transform data
                x = df['Horsepower'].to_numpy().reshape(-1, 1)
                y = df['Miles_per_Gallon'].to_numpy().reshape(-1, 1)

                fig, ax = plt.subplots()     # A tuple unpacking to unpack the only fig
                ax.set_xlim(30, 250)
                ax.set_ylim(5, 50)
                # Plotting 
                scatter, = ax.plot([], [], 'go', label='Horsepower vs. Miles_per_Gallon')
                line, = ax.plot([], [], 'r', label='Linear Regression')
                ax.legend()

                reg = LinearRegression()

                def animate(frame_num):
                    # Adding data
                    x_data.append(x[frame_num])
                    y_data.append(y[frame_num])
                    # Convert data to numpy array
                    x_train = np.array(x_data).reshape(-1, 1)
                    y_train = np.array(y_data).reshape(-1, 1)
                    # Fit values to a linear regression
                    reg.fit(x_train, y_train)

                    # update data for scatter plot
                    scatter.set_data((x_data, y_data))
                    # Predict value and update data for line plot
                    line.set_data((list(range(250)), reg.predict(np.array([entry for entry in range(250)]).reshape(-1, 1))))

                anim = FuncAnimation(fig, animate, frames=len(x), interval=interval)
                components.html(anim.to_jshtml(), height=600,width=600,scrolling=True)
                return 
        with How_it_work:
            col1,col2=st.columns((5,5))
            col1.image('media/linear1.png')
            col2.image('media/cost linear.png')
            col2.image('media/linearcost2.png')
            col1.image('media/gardient_linear.png')
            col1.image('media/lnear4.png')
 ###############################################################Algorithm In action ######################################
 
    def linearinaction():
        Realstate,Fish=st.tabs(("Real state price ","Fish weight prediction"))
        with Realstate:
            real_Data,real_Graphs=Realstate.tabs(("Data","Graphs"))
            from real_state import realstate_prediction
            df,xtrain,ytrain,xtest,ytest=realstate_prediction.realstate_load_explore_split_data()
            with real_Data:
                real_Data.subheader("Sample Data")    
                real_Data.write(df.head())
                real_Data.header("Data Statistics")
                real_Data.write(df.describe())
                #main_page.alignh(6,diabetescol2)
                Real_statecol1,Real_statecol2=real_Data.columns((6,6))
                Real_statecol2.header("Missing values ")
                Real_statecol2.write(df.isnull().sum())
                #main_page.alignh(3,diabetescol2)
                Real_statecol1.subheader("Correlation Heat Map")
                corr=df.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(7, 5))
                    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                Real_statecol1.pyplot(f)
                mse_LR,rmse_LR,r2_LR=realstate_prediction.realstate_linear_reg(xtrain,ytrain,xtest,ytest)
                main_page.alignh(3,Real_statecol2)
                Real_statecol2.subheader("Mean Square Error")
                Real_statecol2.write(mse_LR)
                Real_statecol2.subheader("Root Mean Square Error")
                Real_statecol2.write(rmse_LR)
                Real_statecol2.subheader("R-Squared")
                Real_statecol2.write(r2_LR)
                user_input =realstate_prediction.user_input_feature_realstate(Real_statecol1,Real_statecol2)
                filename = 'savedmodels/realstate_linear_reg_model.sav'
                model=pickle.load(open(filename,'rb'))
                prediction =model.predict(user_input)
                Real_statecol2.write(prediction)
            with real_Graphs:
                real_Graphscol1,real_Graphscol2=real_Graphs.columns((6,6))
                label=df['house price of unit area']
                fig, ax=plt.subplots(2,1,figsize=(9,12))
                # Plot histogram
                ax[0].hist(label,bins=100)
                ax[0].set_ylabel('Frquency')
                ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
                ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
                # Plot the boxplot
                ax[1].boxplot(label, vert=False)
                ax[1].set_xlabel('Label')
                # Add a title to the Figure
                fig.suptitle('Label Distribution')
                # Show the figure
                real_Graphscol1.pyplot(fig)
               
                df.hist(xlabelsize=5)
                plt.show()
                real_Graphscol2.header("Features Histogram")
                real_Graphscol2.pyplot(plt)
                #fig, ax = plt.subplots()
                #ax.hist(x, bins=20)
            # tabirisGraphs.pyplot(fig)
                df.plot()
                plt.show()
                real_Graphscol2.header("Features plot")
                real_Graphscol2.pyplot(plt)
               
                axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                real_Graphscol1.header("Scatter Matrix")
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(90)
                    ax.yaxis.label.set_rotation(0)
                    ax.yaxis.label.set_ha('right')
                real_Graphscol1.pyplot(plt)
                #feature ploting
                x=df.iloc[:100,:]
                y=np.arange(0,100)
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                x['y']=y
                ax1 = x.plot(kind='scatter', y='house age', x='y', color=color[0],rot=90)    
                ax2 = x.plot(kind='scatter', y='distance to the nearest MRT station', x='y', color=color[1], ax=ax1)    
                ax3 = x.plot(kind='scatter', y='number of convenience stores', x='y', color=color[2], ax=ax1)
                ax4 = x.plot(kind='scatter', y='latitude', x='y', color=color[3],ax=ax1)    
                plt.xlabel("Features")
                plt.ylabel("Unit price")
                real_Graphscol2.header("feature Scatter Matrix")
                real_Graphscol2.pyplot(plt)
        
##################################################################### Fish weights prediction dataset ####################################
      # ['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight']
        with Fish:
            Fish_Data,Fish_Graphs=Fish.tabs(("Data","Graphs"))
            from Fish import fish_weight_prediction
            df,xtrain,ytrain,xtest,ytest=fish_weight_prediction.fish_weight_load_explore_split_data()
            with Fish_Data:
                Fish_Data.subheader("Sample Data")    
                Fish_Data.write(df.head())
                Fish_Data.header("Data Statistics")
                Fish_Data.write(df.describe())
                #main_page.alignh(6,diabetescol2)
                Fishcol1,Fishcol2=Fish_Data.columns((6,6))
                Fishcol2.header("Missing values ")
                Fishcol2.write(df.isnull().sum())
                #main_page.alignh(3,diabetescol2)
                Fishcol1.subheader("Correlation Heat Map")
                corr=df.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(7, 5))
                    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                Fishcol1.pyplot(f)
                mse_LR,rmse_LR,r2_LR=fish_weight_prediction.fish_weight_linear_reg(xtrain,ytrain,xtest,ytest)
                main_page.alignh(3,Fishcol2)
                Fishcol2.subheader("Mean Square Error")
                Fishcol2.write(mse_LR)
                Fishcol2.subheader("Root Mean Square Error")
                Fishcol2.write(rmse_LR)
                Fishcol2.subheader("R-Squared")
                Fishcol2.write(r2_LR)
                Fishcol1.subheader("Pridict fish weight")
                user_input=fish_weight_prediction.user_input_feature_fish(Fishcol1,Fishcol2)
                filename = 'savedmodels/fish_weight_linear_reg_model.sav'
                model=pickle.load(open(filename,'rb'))
                prediction =model.predict(user_input)
                Fishcol2.subheader("predicted weight ")
                Fishcol2.write(prediction)
            with Fish_Graphs:
                FishGraphscol1,FishGraphscol2=Fish_Graphs.columns((6,6))
                label=df['Weight']
                fig, ax=plt.subplots(2,1,figsize=(9,12))
                # Plot histogram
                ax[0].hist(label,bins=100)
                ax[0].set_ylabel('Frquency')
                ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
                ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
                # Plot the boxplot
                ax[1].boxplot(label, vert=False)
                ax[1].set_xlabel('Label')
                # Add a title to the Figure
                fig.suptitle('Label Distribution')
                # Show the figure
                FishGraphscol1.pyplot(fig)
                
                df.hist(xlabelsize=5)
                plt.show()
                FishGraphscol2.header("Features Histogram")
                FishGraphscol2.pyplot(plt)
                df.plot()
                plt.show()
                FishGraphscol2.header("Features plot")
                FishGraphscol2.pyplot(plt)
               
                axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                FishGraphscol1.header("Scatter Matrix")
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(90)
                    ax.yaxis.label.set_rotation(0)
                    ax.yaxis.label.set_ha('right')
                FishGraphscol1.pyplot(plt)
                #feature ploting
                x=df.iloc[:100,:]
                y=np.arange(0,100)
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                x['y']=y
                ax1 = x.plot(kind='scatter', y='Species', x='y', color=color[0],rot=90)    
                ax2 = x.plot(kind='scatter', y='Length2', x='y', color=color[1], ax=ax1)    
                ax3 = x.plot(kind='scatter', y='Length1', x='y', color=color[2], ax=ax1)
                ax4 = x.plot(kind='scatter', y='Width', x='y', color=color[3],ax=ax1)    
                plt.xlabel("Features")
                plt.ylabel("Weight")
                FishGraphscol2.header("feature Scatter Matrix")
                FishGraphscol2.pyplot(plt)
       
    def Lassoinaction():
        Realstate,Fish=st.tabs(("Real state price ","Fish Weight prediction "))
        with Realstate:
            real_Data,real_Graphs=Realstate.tabs(("Data","Graphs"))
            from real_state import realstate_prediction
            df,xtrain,ytrain,xtest,ytest=realstate_prediction.realstate_load_explore_split_data()
            with real_Data:
                real_Data.subheader("Sample Data")    
                real_Data.write(df.head())
                real_Data.header("Data Statistics")
                real_Data.write(df.describe())
                #main_page.alignh(6,diabetescol2)
                Real_statecol1,Real_statecol2=real_Data.columns((6,6))
                Real_statecol2.header("Missing values ")
                Real_statecol2.write(df.isnull().sum())
                #main_page.alignh(3,diabetescol2)
                Real_statecol1.subheader("Correlation Heat Map")
                corr=df.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(7, 5))
                    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                Real_statecol1.pyplot(f)
                mse_LR,rmse_LR,r2_LR=realstate_prediction.realstate_lasso_reg(xtrain,ytrain,xtest,ytest)
                main_page.alignh(3,Real_statecol2)
                Real_statecol2.subheader("Mean Square Error")
                Real_statecol2.write(mse_LR)
                Real_statecol2.subheader("Root Mean Square Error")
                Real_statecol2.write(rmse_LR)
                Real_statecol2.subheader("R-Squared")
                Real_statecol2.write(r2_LR)
                user_input =realstate_prediction.user_input_feature_realstate(Real_statecol1,Real_statecol2)
                filename = 'savedmodels/realstate_lasso_reg_model.sav'
                model=pickle.load(open(filename,'rb'))
                prediction =model.predict(user_input)
                Real_statecol2.write(prediction)
            with real_Graphs:
                real_Graphscol1,real_Graphscol2=real_Graphs.columns((6,6))
                label=df['house price of unit area']
                fig, ax=plt.subplots(2,1,figsize=(9,12))
                # Plot histogram
                ax[0].hist(label,bins=100)
                ax[0].set_ylabel('Frquency')
                ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
                ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
                # Plot the boxplot
                ax[1].boxplot(label, vert=False)
                ax[1].set_xlabel('Label')
                # Add a title to the Figure
                fig.suptitle('Label Distribution')
                # Show the figure
                real_Graphscol1.pyplot(fig)
                
                df.hist(xlabelsize=5)
                plt.show()
                real_Graphscol2.header("Features Histogram")
                real_Graphscol2.pyplot(plt)
                #fig, ax = plt.subplots()
                #ax.hist(x, bins=20)
            # tabirisGraphs.pyplot(fig)
                df.plot()
                plt.show()
                real_Graphscol2.header("Features plot")
                real_Graphscol2.pyplot(plt)
               
                axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                real_Graphscol1.header("Scatter Matrix")
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(90)
                    ax.yaxis.label.set_rotation(0)
                    ax.yaxis.label.set_ha('right')
                real_Graphscol1.pyplot(plt)
                #feature ploting
                x=df.iloc[:100,:]
                y=np.arange(0,100)
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                x['y']=y
                ax1 = x.plot(kind='scatter', y='house age', x='y', color=color[0],rot=90)    
                ax2 = x.plot(kind='scatter', y='distance to the nearest MRT station', x='y', color=color[1], ax=ax1)    
                ax3 = x.plot(kind='scatter', y='number of convenience stores', x='y', color=color[2], ax=ax1)
                ax4 = x.plot(kind='scatter', y='latitude', x='y', color=color[3],ax=ax1)    
                plt.xlabel("Features")
                plt.ylabel("Unit price")
                real_Graphscol2.header("feature Scatter Matrix")
                real_Graphscol2.pyplot(plt)
  ###########################################################################Fish Lasso #############################     
        with Fish:
            Fish_Data,Fish_Graphs=Fish.tabs(("Data","Graphs"))
            from Fish import fish_weight_prediction
            df,xtrain,ytrain,xtest,ytest=fish_weight_prediction.fish_weight_load_explore_split_data()
            with Fish_Data:
                Fish_Data.subheader("Sample Data")    
                Fish_Data.write(df.head())
                Fish_Data.header("Data Statistics")
                Fish_Data.write(df.describe())
                #main_page.alignh(6,diabetescol2)
                Fishcol1,Fishcol2=Fish_Data.columns((6,6))
                Fishcol2.header("Missing values ")
                Fishcol2.write(df.isnull().sum())
                #main_page.alignh(3,diabetescol2)
                Fishcol1.subheader("Correlation Heat Map")
                corr=df.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(7, 5))
                    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                Fishcol1.pyplot(f)
                mse_LR,rmse_LR,r2_LR=fish_weight_prediction.fish_weight_lasso_reg(xtrain,ytrain,xtest,ytest)
                main_page.alignh(3,Fishcol2)
                Fishcol2.subheader("Mean Square Error")
                Fishcol2.write(mse_LR)
                Fishcol2.subheader("Root Mean Square Error")
                Fishcol2.write(rmse_LR)
                Fishcol2.subheader("R-Squared")
                Fishcol2.write(r2_LR)
                Fishcol1.subheader("Pridict fish weight")
                user_input=fish_weight_prediction.user_input_feature_fish(Fishcol1,Fishcol2)
                filename = 'savedmodels/fish_weight_lasso_reg_model.sav'
                model=pickle.load(open(filename,'rb'))
                prediction =model.predict(user_input)
                Fishcol2.subheader("predicted weight ")
                Fishcol2.write(prediction)
               
            with Fish_Graphs:
                FishGraphscol1,FishGraphscol2=Fish_Graphs.columns((6,6))
                label=df['Weight']
                fig, ax=plt.subplots(2,1,figsize=(9,12))
                # Plot histogram
                ax[0].hist(label,bins=100)
                ax[0].set_ylabel('Frquency')
                ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
                ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
                # Plot the boxplot
                ax[1].boxplot(label, vert=False)
                ax[1].set_xlabel('Label')
                # Add a title to the Figure
                fig.suptitle('Label Distribution')
                # Show the figure
                FishGraphscol1.pyplot(fig)
                
                df.hist(xlabelsize=5)
                plt.show()
                FishGraphscol2.header("Features Histogram")
                FishGraphscol2.pyplot(plt)
                #fig, ax = plt.subplots()
                #ax.hist(x, bins=20)
            # tabirisGraphs.pyplot(fig)
                df.plot()
                plt.show()
                FishGraphscol2.header("Features plot")
                FishGraphscol2.pyplot(plt)
               
                axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                FishGraphscol1.header("Scatter Matrix")
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(90)
                    ax.yaxis.label.set_rotation(0)
                    ax.yaxis.label.set_ha('right')
                FishGraphscol1.pyplot(plt)
                #feature ploting
                x=df.iloc[:100,:]
                y=np.arange(0,100)
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                x['y']=y
                ax1 = x.plot(kind='scatter', y='Species', x='y', color=color[0],rot=90)    
                ax2 = x.plot(kind='scatter', y='Length2', x='y', color=color[1], ax=ax1)    
                ax3 = x.plot(kind='scatter', y='Length1', x='y', color=color[2], ax=ax1)
                ax4 = x.plot(kind='scatter', y='Width', x='y', color=color[3],ax=ax1)    
                plt.xlabel("Features")
                plt.ylabel("Weight")
                FishGraphscol2.header("feature Scatter Matrix")
                FishGraphscol2.pyplot(plt)









            
    def Gradientinaction():
        Realstate,Fish=st.tabs(("Real state price ","Fish weight prediction "))
        with Realstate:
            real_Data,real_Graphs=Realstate.tabs(("Data","Graphs"))
            from real_state import realstate_prediction
            df,xtrain,ytrain,xtest,ytest=realstate_prediction.realstate_load_explore_split_data()
            with real_Data:
                real_Data.subheader("Sample Data")    
                real_Data.write(df.head())
                real_Data.header("Data Statistics")
                real_Data.write(df.describe())
                #main_page.alignh(6,diabetescol2)
                Real_statecol1,Real_statecol2=real_Data.columns((6,6))
                Real_statecol2.header("Missing values ")
                Real_statecol2.write(df.isnull().sum())
                #main_page.alignh(3,diabetescol2)
                Real_statecol1.subheader("Correlation Heat Map")
                corr=df.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(7, 5))
                    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                Real_statecol1.pyplot(f)
                mse_LR,rmse_LR,r2_LR=realstate_prediction.realstate_gradient_bossting(xtrain,ytrain,xtest,ytest)
                main_page.alignh(3,Real_statecol2)
                Real_statecol2.subheader("Mean Square Error")
                Real_statecol2.write(mse_LR)
                Real_statecol2.subheader("Root Mean Square Error")
                Real_statecol2.write(rmse_LR)
                Real_statecol2.subheader("R-Squared")
                Real_statecol2.write(r2_LR)
                user_input =realstate_prediction.user_input_feature_realstate(Real_statecol1,Real_statecol2)
                filename = 'savedmodels/realstate_gradient_bossting_model.sav'
                model=pickle.load(open(filename,'rb'))
                prediction =model.predict(user_input)
                Real_statecol2.write(prediction)
            with real_Graphs:
                real_Graphscol1,real_Graphscol2=real_Graphs.columns((6,6))
                label=df['house price of unit area']
                fig, ax=plt.subplots(2,1,figsize=(9,12))
                # Plot histogram
                ax[0].hist(label,bins=100)
                ax[0].set_ylabel('Frquency')
                ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
                ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
                # Plot the boxplot
                ax[1].boxplot(label, vert=False)
                ax[1].set_xlabel('Label')
                # Add a title to the Figure
                fig.suptitle('Label Distribution')
                # Show the figure
                real_Graphscol1.pyplot(fig)
                
                df.hist(xlabelsize=5)
                plt.show()
                real_Graphscol2.header("Features Histogram")
                real_Graphscol2.pyplot(plt)
                #fig, ax = plt.subplots()
                #ax.hist(x, bins=20)
            # tabirisGraphs.pyplot(fig)
                df.plot()
                plt.show()
                real_Graphscol2.header("Features plot")
                real_Graphscol2.pyplot(plt)
               
                axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                real_Graphscol1.header("Scatter Matrix")
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(90)
                    ax.yaxis.label.set_rotation(0)
                    ax.yaxis.label.set_ha('right')
                real_Graphscol1.pyplot(plt)
                #feature ploting
                x=df.iloc[:100,:]
                y=np.arange(0,100)
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                x['y']=y
                ax1 = x.plot(kind='scatter', y='house age', x='y', color=color[0],rot=90)    
                ax2 = x.plot(kind='scatter', y='distance to the nearest MRT station', x='y', color=color[1], ax=ax1)    
                ax3 = x.plot(kind='scatter', y='number of convenience stores', x='y', color=color[2], ax=ax1)
                ax4 = x.plot(kind='scatter', y='latitude', x='y', color=color[3],ax=ax1)    
                plt.xlabel("Features")
                plt.ylabel("Unit price")
                real_Graphscol2.header("feature Scatter Matrix")
                real_Graphscol2.pyplot(plt)
 ########################################################### Fish  Gradients ##################################################
        with Fish:
            Fish_Data,Fish_Graphs=Fish.tabs(("Data","Graphs"))
            from Fish import fish_weight_prediction
            df,xtrain,ytrain,xtest,ytest=fish_weight_prediction.fish_weight_load_explore_split_data()
            with Fish_Data:
                Fish_Data.subheader("Sample Data")    
                Fish_Data.write(df.head())
                Fish_Data.header("Data Statistics")
                Fish_Data.write(df.describe())
                #main_page.alignh(6,diabetescol2)
                Fishcol1,Fishcol2=Fish_Data.columns((6,6))
                Fishcol2.header("Missing values ")
                Fishcol2.write(df.isnull().sum())
                #main_page.alignh(3,diabetescol2)
                Fishcol1.subheader("Correlation Heat Map")
                corr=df.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(7, 5))
                    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                Fishcol1.pyplot(f)
                mse_LR,rmse_LR,r2_LR=fish_weight_prediction.fish_weight_gradient_bossting(xtrain,ytrain,xtest,ytest)
                main_page.alignh(3,Fishcol2)
                Fishcol2.subheader("Mean Square Error")
                Fishcol2.write(mse_LR)
                Fishcol2.subheader("Root Mean Square Error")
                Fishcol2.write(rmse_LR)
                Fishcol2.subheader("R-Squared")
                Fishcol2.write(r2_LR)
                Fishcol1.subheader("Pridict fish weight")
                user_input=fish_weight_prediction.user_input_feature_fish(Fishcol1,Fishcol2)
                st.write(user_input.info())
                filename = 'savedmodels/fish_weight_gradient_bossting_model.sav'
                model=pickle.load(open(filename,'rb'))
                prediction =model.predict(user_input)
                Fishcol2.subheader("predicted weight ")
                Fishcol2.write(prediction)
            with Fish_Graphs:
                FishGraphscol1,FishGraphscol2=Fish_Graphs.columns((6,6))
                label=df['Weight']
                fig, ax=plt.subplots(2,1,figsize=(9,12))
                # Plot histogram
                ax[0].hist(label,bins=100)
                ax[0].set_ylabel('Frquency')
                ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
                ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
                # Plot the boxplot
                ax[1].boxplot(label, vert=False)
                ax[1].set_xlabel('Label')
                # Add a title to the Figure
                fig.suptitle('Label Distribution')
                # Show the figure
                FishGraphscol1.pyplot(fig)
                
                df.hist(xlabelsize=5)
                plt.show()
                FishGraphscol2.header("Features Histogram")
                FishGraphscol2.pyplot(plt)
                #fig, ax = plt.subplots()
                #ax.hist(x, bins=20)
            # tabirisGraphs.pyplot(fig)
                df.plot()
                plt.show()
                FishGraphscol2.header("Features plot")
                FishGraphscol2.pyplot(plt)
               
                axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                FishGraphscol1.header("Scatter Matrix")
                for ax in axes.flatten():
                    ax.xaxis.label.set_rotation(90)
                    ax.yaxis.label.set_rotation(0)
                    ax.yaxis.label.set_ha('right')
                FishGraphscol1.pyplot(plt)
                #feature ploting
                x=df.iloc[:100,:]
                y=np.arange(0,100)
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                x['y']=y
                ax1 = x.plot(kind='scatter', y='Species', x='y', color=color[0],rot=90)    
                ax2 = x.plot(kind='scatter', y='Length2', x='y', color=color[1], ax=ax1)    
                ax3 = x.plot(kind='scatter', y='Length1', x='y', color=color[2], ax=ax1)
                ax4 = x.plot(kind='scatter', y='Width', x='y', color=color[3],ax=ax1)    
                plt.xlabel("Features")
                plt.ylabel("Weight")
                FishGraphscol2.header("feature Scatter Matrix")
                FishGraphscol2.pyplot(plt)
                

                
                                