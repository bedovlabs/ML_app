
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from vega_datasets import data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
import streamlit as st
import streamlit.components.v1 as components
class linear_reg(object):
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
