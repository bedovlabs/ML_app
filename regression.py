
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
        Realstate,Cancer=st.tabs(("Real state price ","Cancer mortality prediction "))
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
        
    def Lassoinaction():
        Realstate,Cancer=st.tabs(("Real state price ","Cancer mortality prediction "))
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
            
    def Gradientinaction():
        Realstate,Cancer=st.tabs(("Real state price ","Cancer mortality prediction "))
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


                
                                