#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
target, features = targetFeatureSplit(data)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(features, target)

pred = reg.predict(features)

errors = target - pred

import numpy
threshold = numpy.percentile(numpy.absolute(errors), 90)

cleaned_data = []
for feature, tango, error in zip(features, target, errors):
    if error < threshold:
        cleaned_data.append((feature, tango, error))

features, target, errors = zip(*cleaned_data)

features = numpy.reshape(numpy.array(features), (len(features), 1))
target = numpy.reshape(numpy.array(target), (len(target), 1))

reg.fit(features, target)

import matplotlib.pyplot as plt

plt.scatter(features, target)
plt.plot(features, reg.predict(features))
plt.show()
