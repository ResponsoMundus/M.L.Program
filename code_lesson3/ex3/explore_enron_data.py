#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Size Of The Enron DataSet: {0}".format(len(enron_data))
print "Features In The Enron DataSet: {0}".format(len(enron_data.values()[0]))

pois = [key for key, value in enron_data.items() if value['poi']]
print "Number Of POIs: {0}".format(len(pois))

print "Stock Held By James Prentice: {0}".format(enron_data["PRENTICE JAMES"]["total_stock_value"])

names = ["LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"]
payments = [(name, enron_data[name]["total_payments"]) for name in names]
print "{0} took most money.".format(sorted(payments, key=lambda x: x[1], reverse=True)[0][0])

salaries = [key for key, value in enron_data.items() if value["salary"] != "NaN"]
print "Number Of Folks Having A Quantified Salary: {0}".format(len(salaries))

emails = [key for key, value in enron_data.items() if value["email_address"] != "NaN"]
print "Number Of Folks Having A Known E-Mail Address: {0}".format(len(emails))

NaNPayments = [key for key, value in enron_data.items() if value["total_payments"] == "NaN"]
print "Number And Percentage Of Folks Having A 'NaN' For Total Payments: {0}, {1}%".format(len(NaNPayments), round(100.0 * len(NaNPayments) / len(enron_data), 3))
