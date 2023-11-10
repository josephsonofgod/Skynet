'''
Skynet Machine Learning Algorithm
Securing Networks Assignment 4
__version__ = 10/11/23
__author__  = Sam Watson
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import accuracy_score, confusion_matrix


# load into dataframe:
data = pd.read_csv(r"data.csv", encoding = "ISO-8859-1")

# fill missing values with 0:
data = data.fillna(0)

# transform data into string:
encoder = LabelEncoder ()
data['ID'] = encoder.fit_transform(data['No.'].astype ('str'))
data['Dataset'] = encoder.fit_transform(data['Dataset'].astype ('str'))
data['Time'] = encoder.fit_transform(data['Time'].astype ('str'))
data['Source IP'] = encoder.fit_transform(data['Source'].astype ('str'))
data['Destination IP'] = encoder.fit_transform(data['Destination'].astype ('str'))
data['Length'] = encoder.fit_transform(data['Length'].astype ('int'))
data['Info'] = encoder.fit_transform(data['Info'].astype ('str'))
data['Attack?'] = encoder.fit_transform(data['Attack'].astype ('str'))

# Separate informational data and expected output data
x = data. drop ('Label', axis =1).values # features
y = data['Label'].values # labels

# separate into training and testing:
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42) # <-- might need to change the last 2 parameters

# train:
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(xTrain , yTrain)
prediction = model.predict(xTest)

# accuracy: 
print("Accuracy", accuracy_score(prediction , yTest)*100, "%")
print("Confusion Matrix", confusion_matrix(prediction , yTest))
