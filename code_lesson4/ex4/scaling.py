import pickle
import numpy as np
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

# the features to be used
features_list = ['poi', 'salary', 'exercised_stock_options']

data = featureFormat(data_dict, features_list)

poi, salary, stock = zip(*data)

# put the features into 2-D numpy arrays
salary = np.array(salary).reshape((len(salary), 1))
stock = np.array(stock).reshape((len(stock), 1))

# your code goes here
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
salary_transformed = scaler.fit_transform(salary)
stock_transformed = scaler.fit_transform(stock)

finance_features =[]
for sal, sto in zip(salary_transformed, stock_transformed):
    finance_features.append(np.array([sal[0], sto[0]]))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(finance_features)
pred = kmeans.predict(finance_features)

n = 0.0
for pre, po in zip(pred, poi):
    if pre == po:
        n += 1
print "Accuracy: {0}".format(n / len(pred))

import matplotlib.pyplot as plt

for i in range(len(pred)):
    if pred[i] == 1:
        plt.scatter(finance_features[i][0], finance_features[i][1], color="r")
    else:
        plt.scatter(finance_features[i][0], finance_features[i][1], color="b")
plt.show()
