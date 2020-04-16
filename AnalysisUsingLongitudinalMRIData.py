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
