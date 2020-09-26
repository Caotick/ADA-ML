# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    loss = 0
    txw = np.dot(tx, w)
    for i in range(y.shape[0]) :
        loss += (y[i] - txw[i])**2
    return loss / (2*y.shape[0])