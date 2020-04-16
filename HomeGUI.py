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

def destroy():
    root.destroy()
    
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
    root.destroy()
    root2 = Tk()
    root2.title(" MMSE ")
    
    photo = PhotoImage(file = "brain.gif")
    w2 = Label(root2,image = photo)
    w2.place(x=0,y=0)
    
    def destroy1():
        root2.destroy()
    
    button_11 = Button(root2, text = "      Instance Plot      ", bg='#50c878', command = InstancePlot)
    button_12 = Button(root2, text = "   Actual Vs Predicted   ", bg='#50c878', command = Actual_Predicted)
    button_13 = Button(root2, text ="   Exit   ",bg= '#ff4c4c', command = destroy1)
    button_11.bind()
    button_12.bind()
    button_13.bind()
    button_11.pack(padx=10, pady=50)
    button_12.pack(padx=10, pady=50)    
    button_13.pack(padx=10, pady=50)
    
    root2.geometry("700x475")
    root2.mainloop()

def img_segmentation():
    image = plt.imread('brain1.jpg')
    image.shape
    plt.imshow(image)
    
    gray = rgb2gray(image)
    plt.imshow(gray, cmap='gray')
    
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])
    plt.imshow(gray, cmap='gray')
    
    gray = rgb2gray(image)
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 3
        elif gray_r[i] > 0.5:
            gray_r[i] = 2
        elif gray_r[i] > 0.25:
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])
    plt.imshow(gray, cmap='gray')
    
def oasis():
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set()
    
    df = pd.read_csv("C://Users//Karunya V//Documents//sem4//Predictive//Image_recognition//PredPack//oasis_longitudinal.csv")
    df.head()
    
    df = df.loc[df['Visit']==1]
    df = df.reset_index(drop=True)
    #print(df)
    df['M/F'] = df['M/F'].replace(['F','M'], [0,1]) # M/F column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented']) # Target variable
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) # Target variable
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1) # Drop unnecessary columns
    
    df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
    print(df)
    
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    # Dataset with imputation
    Y = df['Group'].values # Target for the model
    X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']] # Features we use
    
    # splitting into three sets
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, random_state=0)
    
    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler
    
    # Define the scaler 
    scaler = StandardScaler().fit(X_trainval)
    
    # Scale the train set
    X_trainval = scaler.transform(X_trainval)
    
    # Scale the test set
    X_test = scaler.transform(X_test)
    
    Y_trainval=np.ravel(Y_trainval)
    X_trainval=np.asarray(X_trainval)
    
    Y_test=np.ravel(Y_test)
    X_test=np.asarray(X_test)
    
    #LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_trainval, Y_trainval)
    prediction = classifier.predict(X_test)
    print("Logistic Regression: ")
    print(classifier.score(X_test, Y_test))
    
    #DECISION TREES
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=12)
    classifier.fit(X_trainval, Y_trainval)
    prediction = classifier.predict(X_test)
    #print (classifier.score(X_trainval, Y_trainval))
    print("Decision trees: ")
    print (classifier.score(X_test, Y_test))
    
    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_trainval, Y_trainval)
    #print(knn.score(X_train, y_train))
    prediction = knn.predict(X_test)
    print("KNN: ")
    print(knn.score(X_test, Y_test))
    
    #SVC
    from sklearn.svm import SVC
    svc=SVC(kernel="linear", C=0.01)
    svc.fit(X_trainval, Y_trainval)
    prediction = svc.predict(X_test)
    print("SVC: ")
    print(svc.score(X_test, Y_test))
    
def edge_detection():
    img = cv2.imread('brain1.jpg',0)
    edges = cv2.Canny(img,100,200)
    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def knn():
    image = cv2.imread('brain1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3)) #2D and 3 colors
    pixel_values = np.float32(pixel_values) #making the matrix float
    #print(pixel_values.shape)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2) #define the criteria
    _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #asking for 3 classes - black, white and grey
    centers = np.uint8(centers)
    labels = labels.flatten()
    final_image = centers[labels.flatten()]
    
    final_image = final_image.reshape(image.shape)
    
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(final_image)
    plt.title('KNN (3) Image'), plt.xticks([]), plt.yticks([])
    plt.show()

            
def Mainpage():
    button_1 = Button(root, text = "    Image Segmentation (Thresholding)     ", bg = "#ffd27f", command = img_segmentation)
    button_2 = Button(root, text = " Edge Detection  ", bg = "#ffd27f", command = edge_detection)
    button_3 = Button(root, text = "  Early Alzheimer's detection using Longitudinal MRI data  ", bg = "#ffd27f" , command=oasis)
    button_4 = Button(root, text = "  Classification of Brain MRI images using KNN Algorithm  ", bg = "#ffd27f", command=knn)
    button_5 = Button(root, text = "  Multi-Variate Analysis of MMSE data  ", bg = "#ffd27f" , command = multi_regression)
    button_6 = Button(root, text = "     Quit     ",bg = "light blue", command = destroy)
    button_1.bind()
    button_2.bind()
    button_3.bind()
    button_4.bind()
    button_5.bind()
    button_6.bind()
    button_1.pack(padx=10, pady=20)
    button_2.pack(padx=10, pady=20)
    button_3.pack(padx=10, pady=20)
    button_4.pack(padx=10, pady=20)
    button_5.pack(padx=10, pady=20)
    button_6.pack(padx=20, pady=20)
    

    
    
root = Tk()
root.title(" Alzheimer Dataset ")

photo = PhotoImage(file = "alzhe.gif")
w = Label(root,image = photo)
w.place(x=0,y=0)

Mainpage()
root.geometry("700x475")
root.mainloop()
    
