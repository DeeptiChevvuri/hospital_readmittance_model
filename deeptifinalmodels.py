# -*- coding: utf-8 -*-

"""
Created on Wed Sep 13 10:09:16 2017

@author: deeptichevvuri
"""    
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import random
import numpy as np
import matplotlib.pyplot as plt

diabetesReadmissionDataSet = pandas.read_csv("diabetic_data_initial.csv") 

diabetesReadmissionDataArray = diabetesReadmissionDataSet.values
# avoid time depedancy in data, if removed accuracy falls by 20%
random.shuffle(diabetesReadmissionDataArray)

X = diabetesReadmissionDataArray[:,0:41]
y = diabetesReadmissionDataArray[:,41]
scaler = StandardScaler()


xTrain, xTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.25, random_state=42)
# feature Scaling !
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain) 
xTest = scaler.transform(xTest) 


print("Decision Tree(DT) CLassifier")
dtClassifier=DecisionTreeClassifier(random_state=100, max_features='auto')
dtClassifier=dtClassifier.fit(xTrain,yTrain)
scores = cross_val_score(dtClassifier, xTest, yTest, cv=300, scoring='accuracy')
yhat = dtClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

importances = dtClassifier.feature_importances_
std = np.std(dtClassifier.feature_importances_,
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("dtClassifier Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(20,10))
plt.title("Feature importances of dtClassifier")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

print("Random Forest")
kfold = model_selection.KFold(n_splits=50, random_state=5)#n=500 has lower accuracy but predicts class 3 better
rfModel = RandomForestClassifier(n_estimators=50, max_features='auto')#nestimators=numtrees=50, maxfeatures=maxfeatures=2   nestimators=50, maxfeatures=2
rfModel = rfModel.fit(xTrain,yTrain)
rfResults = model_selection.cross_val_score(rfModel, xTest, yTest, cv=kfold)
yhat = rfModel.predict(xTest)
confusion_matrix(yTest,yhat)
print("Accuracy= {} ".format(rfResults.mean()))
print("Confusion Matrix : \n{}".format(confusion_matrix(yTest,yhat)))

importances = rfModel.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfModel.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("rfModel Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances of rfModel")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


print("KNN CLassifier")
knnClassifier= KNeighborsClassifier(n_neighbors=3 )
knnClassifier=knnClassifier.fit(xTrain,yTrain)
scores = cross_val_score(knnClassifier, xTest, yTest, cv=1000, scoring='accuracy')
yhat = knnClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

n_feats = X.shape[1]

print('Feature  Accuracy')
for i in range(n_feats):
    X = X.data[:, i].reshape(-1, 1)
    scores = cross_val_score(knnClassifier, X, y)
    print('%d        %g' % (i, scores.mean()))


print("Unweigthed Majority Vote Ensemble")
eclfunweighted = VotingClassifier(estimators=[('dt', dtClassifier), ('rf', rfModel), ('knn', knnClassifier)], voting='hard')
eclfunweighted=eclfunweighted.fit(xTrain,yTrain)
scores = cross_val_score(eclfunweighted, xTest, yTest, cv=100, scoring='accuracy')
print("Ensemble Unweigthed Accuracy: {}\n".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

