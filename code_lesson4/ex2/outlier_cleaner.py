#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    ### your code goes here
    import numpy
    cleaned_data = []
    errors = net_worths - predictions

    threshold = numpy.percentile(numpy.absolute(errors), 90)

    for age, net_worth, error in zip(ages, net_worths, errors):
        if abs(error) <= threshold:
            cleaned_data.append((age, net_worth, error))
    
    return cleaned_data

