# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w, mae=False):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    loss = 0
    txw = np.dot(tx, w)
    if(not mae) :
        for i in range(y.shape[0]) :
            loss += (y[i] - txw[i])**2
    else :
        e = np.absolute(y - txw)
        for i in range(y.shape[0]) :
            loss += e[i]
    return loss / (2*y.shape[0])