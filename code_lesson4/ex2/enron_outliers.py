#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import numpy

from outlier_cleaner import outlierCleaner

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
target, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)

train_color = "b"
test_color = "r"

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)

print "Slope And Intercept: {0}, {1}".format(round(reg.coef_[0], 3), round(reg.intercept_, 3))
print "Regression Score: {0}".format(round(reg.score(feature_test, target_test), 3))

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass
plt.xlabel(features[1])
plt.ylabel(features[0])
plt.legend()
plt.show()

cleaned = []
try:
    predictions = reg.predict(feature_train)
    cleaned = outlierCleaner(predictions, feature_train, target_train)
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"

if len(cleaned) > 0:
    ages, net_worths, errors = zip(*cleaned)
    ages       = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print "After the Removal of Outliers:"
        print "\tSlope: {0}".format(round(reg.coef_[0], 3))
        print "\tRegression Score: {0}".format(round(reg.score(feature_test, target_test), 3))
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, net_worths)
    plt.xlabel(features[1])
    plt.ylabel(features[0])
    plt.show()
