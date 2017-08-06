#!/usr/bin/python

""" 
 	Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

t = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)

print time() - t, "seconds"

t = time()

prediction = clf.predict(features_test)

print "Accuracy Score: ", accuracy_score(labels_test, prediction)

print time() - t, "seconds"

#########################################################
