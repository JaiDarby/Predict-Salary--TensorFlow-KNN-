#Importing all libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import pickle

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

#Set how many neighbors you're lookign for
model = KNeighborsClassifier(n_neighbors = 5)

BestAcc = 0
for i in range(10):
    #Train Model
    model.fit(x_train,y_train)

    #Test Model
    Accuracy = model.score(x_test,y_test)

    #Prints Accuracy
    print("Accuracy:", round(Accuracy, 2))

    #Updates pickle file if hiigher accuracy
    if Accuracy > BestAcc:
        BestAcc = Accuracy
        with open("SalaryModel.pickle", "wb") as f:
            pickle.dump(model, f)

#Print Accuracy
print("Best Accuracy:", round(Accuracy, 2))





