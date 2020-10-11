# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)    
    random_pick = np.random.default_rng().choice(x.shape[0], int(x.shape[0]*ratio), replace=False)
    
    x_train = x[random_pick]
    y_train = y[random_pick]
    x_test = np.delete(x, random_pick)
    y_test = np.delete(y, random_pick)
    
    return x_train, y_train, x_test, y_test
