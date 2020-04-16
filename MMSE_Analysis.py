import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from matplotlib.pyplot import figure
from scipy import ndimage
import numpy as np
from tkinter import *
import csv
import pandas as pd
import cv2
from sklearn import linear_model
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("MMSE_final.csv")
df.isnull().any()
df = df.fillna(method='ffill')

X = df[['MMDATE','MMYEAR','MMMONTH','MMDAY','MMSEASON',
        'MMHOSPIT','MMFLOOR','MMCITY','MMAREA','MMSTATE','MMBALL','MMFLAG','MMTREE',
        'MMTRIALS','MMD','MML','MMR','MMO','MMW','MMBALLDL','MMFLAGDL','MMTREEDL','MMWATCH','MMPENCIL',
        'MMREPEAT','MMHAND','MMFOLD','MMONFLR','MMREAD','MMWRITE','MMDRAW']].values
Y = df['MMSCORE'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df2 = df1.head(25)

    
def InstancePlot():
    plt.figure(figsize=(10,8))
    plt.tight_layout()
    seabornInstance.distplot(df['MMSCORE'])
    
def Actual_Predicted():
    print(df2)

    df2.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
   
def multi_regression():    
    InstancePlot()
    Actual_Predicted()
    
multi_regression()
