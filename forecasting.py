import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

def linear_regression(df): 
    x = np.arange(1, len(df) + 6).reshape(-1, 1)  # Expand x range by 5 units 
    plt.figure(figsize=(10, 5))  # Create a new figure 
    plt.xticks(np.arange(1, len(df) + 6, 1))  # Set the x-axis tick marks with step size 1 
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan'] 
    pred_dfs = [] 
    for i, column in enumerate(df.columns[3:]): 
        y = df[column].values 
        # Create a linear regression model 
        model = LinearRegression() 
        # Fit the model to the data 
        model.fit(x[:-5], y)  # Fit the model to the original data 
        # Predict the y values using the model 
        y_pred = model.predict(x) 
        # Plot the result 
        sns.scatterplot(x=np.arange(1, len(df) + 1), y=y, color=colors[i], label='Actual' + ' ' + column) 
        sns.lineplot(x=np.arange(1, len(df) + 6), y=y_pred, color=colors[i], label='Predicted' + ' ' + column)  # Expand x range for prediction 
        # Show the y value at each of the last 5 predicted points 
        predicted_values = [] 
        dct = {} 
        for j in range(len(df) + 1, len(df) + 6): 
            plt.text(j, y_pred[j-1], f'{y_pred[j-1]:.2f}', ha='center', va='bottom') 
            predicted_values.append(y_pred[j-1]) 
            dct[column] = predicted_values 
        pred_df = pd.DataFrame(dct, index=np.arange(len(df) + 1, len(df) + 6)) 
        pred_dfs.append(pred_df) 
    combined_df = pd.concat(pred_dfs, axis=1) 
    st.write(combined_df) 
    plt.xlabel('Test ID') 
    plt.ylabel('Test Score') 
    plt.legend() 
    with PdfPages("temp/plot3.pdf") as pdf:
        pdf.savefig()
    st.pyplot(plt)
    plt.close()