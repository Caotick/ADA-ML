# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    sto_grad = np.zeros(w.shape)
    nb_grad = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size) :
        sto_grad += compute_gradient(minibatch_y, minibatch_tx, w)
        nb_grad += 1
    return sto_grad / nb_grad
    


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters) :
        
        sto_grad = compute_stoch_gradient(y, tx, w, batch_size)
        
        w = w - gamma * sto_grad
        loss = compute_loss(y, tx, w)
        
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws