import numpy as np 
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
import math



df = pd.read_csv("/Users/anamuuenishi/Desktop/dataEntryEnv/practiceCSVML/diabetes.csv")

#Replacing all zeros
zero_not_accepted = ['Glucose', 'BloodPressure','SkinThickness', 'BMI', 'Insulin']
for columns in zero_not_accepted: 
    df[columns] = df[columns].replace(0, np.nan)
    mean = int(df[columns].mean(skipna=True)) #Skipping all na and finding mean 
    df[columns] = df[columns].replace(np.nan, mean) #Replace all nan with mean 

''' Testing if mean works 
print(f"Skin Thickness row 5: {df.loc[5,"SkinThickness"]}")
print(f"Skin Thickness row 3: {df.loc[2, "SkinThickness"]}")'''

#Splitting Data 
X = df.iloc[:,0:8] #Non inclusive
y = df.iloc[:, 8] #Diabetties or not (1 or 0)

Xtrain, Xtest, ytrain, yTest = train_test_split(X, y, random_state=0, test_size=0.2)


#Standard Scailing 
sc_X = StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain) #Finding SD and Mean and Transforming Data
Xtest = sc_X.transform(Xtest) #Transforming Test Input with fit learnt from training set 

"""
K = sqrt(n), where n is the number of samples model (rows) is trained on 
K odd prevents tie breaks 
"""
#print(math.sqrt(len(Xtrain))) = 24 - 1 = 23 
classifer = KNeighborsClassifier(n_neighbors=23, p=2, metric='euclidean')
classifer.fit(Xtrain, ytrain)

ypred = classifer.predict(Xtest)

#Confusion Matrix - F1 score// Perfromance evaluation tool 
cm = confusion_matrix(yTest, ypred)
print(cm) #TN, FP, FN, TP (Left - right / Top - Bottom)
print(accuracy_score(yTest, ypred))
print(f1_score(yTest, ypred))
