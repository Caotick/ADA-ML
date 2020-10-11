# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    diag = np.identity(tx.shape[1]) * lambda_ * 2 * y.shape[0]
    
    w = np.linalg.solve(np.transpose(tx).dot(tx) + diag, np.transpose(tx).dot(y))
    e = y - tx.dot(w)
    return w, np.sqrt(np.transpose(e).dot(e) / y.shape[0])
