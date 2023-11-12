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
from sklearn.metrics import accuracy_score, confusion_matrix


# load into dataframe:
data = pd.read_csv(r"src/network_traffic_monitoring_system.csv", encoding = "ISO-8859-1")

# fill missing values with 0:
data = data.fillna(0)

# Translate non-integer values into encoded values:
encoder = LabelEncoder ()
data['ID'] = encoder.fit_transform(data['Id'].astype('str'))
data['Source'] = encoder.fit_transform(data['Source'].values.ravel())
data['Destination'] = encoder.fit_transform(data["Destination"].values.ravel())
data['Info'] = encoder.fit_transform(data["Info"].astype('str').values.ravel())
data['Label'] = encoder.fit_transform(data["Attack"].values.ravel())

# Separate informational data and expected output data
x = data.drop('Label', axis = 1).values # features
y = data['Label'].values # labels

# separate into training and testing:
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# train:
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(xTrain , yTrain)
prediction = model.predict(xTest)

# accuracy: 
print("Accuracy", accuracy_score(prediction , yTest)*100, "%")
print("Confusion Matrix", confusion_matrix(prediction , yTest))
