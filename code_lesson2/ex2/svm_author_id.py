#!/usr/bin/python

""" 
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
'''
 #features_train = features_train[:len(features_train) / 100]
 #labels_train = labels_train[:len(labels_train) / 100]

clf = SVC(kernel="rbf", C=10000.0)
clf.fit(features_train[:len(features_train) / 100], labels_train[:len(labels_train) / 100])

t = time()

prediction = clf.predict(features_test)

print round(time() - t, 3), "seconds"

answer = prediction[100]

print "the answer extracted: ", answer

clf = SVC(kernel="rbf", C=1000.0)
clf.fit(features_train[:len(features_train) / 100], labels_train[:len(labels_train) / 100])

t = time()

prediction = clf.predict(features_test)

print round(time() - t, 3), "seconds"

clf = SVC(kernel="rbf", C=100.0)
clf.fit(features_train[:len(features_train) / 100], labels_train[:len(labels_train) / 100])

t = time()

prediction = clf.predict(features_test)

print round(time() - t, 3), "seconds"

clf = SVC(kernel="rbf", C=10.0)
clf.fit(features_train[:len(features_train) / 100], labels_train[:len(labels_train) / 100])

t = time()

prediction = clf.predict(features_test)

print round(time() - t, 3), "seconds"

t = time()

clf = SVC(kernel="rbf", C=10000.0)
clf.fit(features_train, labels_train)

print round(time() - t, 3), "seconds"

t = time()

prediction = clf.predict(features_test)

print "Accuracy Score: ", accuracy_score(labels_test, prediction)

print round(time() - t, 3), "seconds"

print "number of chris's mails", sum(prediction)

t = time()

clf = SVC(kernel="rbf", C=1000.0)
clf.fit(features_train, labels_train)

print round(time() - t, 3), "seconds"

t = time()

prediction = clf.predict(features_test)

print "Accuracy Score: ", accuracy_score(labels_test, prediction)

print round(time() - t, 3), "seconds"

t = time()

clf = SVC(kernel="rbf", C=100.0)
clf.fit(features_train, labels_train)

print round(time() - t, 3), "seconds"

t = time()

prediction = clf.predict(features_test)

print "Accuracy Score: ", accuracy_score(labels_test, prediction)

print round(time() - t, 3), "seconds"

t = time()

clf = SVC(kernel="rbf", C=10.0)
clf.fit(features_train, labels_train)

print round(time() - t, 3), "seconds"

t = time()

prediction = clf.predict(features_test)

print "Accuracy Score: ", accuracy_score(labels_test, prediction)

print round(time() - t, 3), "seconds"
'''

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier



#########################################################


