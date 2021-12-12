#Importing all libraries
import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("AdultData.csv", sep = ", ", engine ="python")

#preprocessing all non-numeric attributes
le = preprocessing.LabelEncoder()
Workclass = le.fit_transform(list(data["Workclass"]))
Education = le.fit_transform(list(data["Education"]))
MarriageStatus = le.fit_transform(list(data["MarriageStatus"]))
Occupation = le.fit_transform(list(data["Occupation"]))
Relationship = le.fit_transform(list(data["Relationship"]))
Race = le.fit_transform(list(data["Race"]))
Sex = le.fit_transform(list(data["Sex"]))
NativeCountry = le.fit_transform(list(data["NativeCountry"]))
Salary = le.fit_transform(list(data["Salary"]))

#Setting whst is being predicted
predict = "Salary"

#Sets x and y values
x = list(zip(data["Age"],data["Hours"],Workclass,Education,MarriageStatus,Occupation,Relationship,Race,Sex,NativeCountry))
y= list(Salary)

#Grabs 10% of data to train model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)






