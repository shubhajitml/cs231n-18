import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # get shapes
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # compute vector of scores
    for i in range(num_train):
        f_i = np.dot(X[i,:],W)
        
        # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
        log_c = np.max(f_i)
        f_i -= log_c
        
        # Compute loss 
        # L_i = -f(x_i){y_i} + log \sum_j e^{f(x_i)_j}
        sum_i = 0.0
        for j in range(num_classes):
            sum_i += np.exp(f_i[j])
#         for f_i_j in f_i:
#             sum_i += np.exp(f_i_j)
        loss += -f_i[y[i]] + np.log(sum_i)

        # Compute gradient
        # dw_j = \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
        # Here we are computing the contribution to the inner sum for a given i.
        for j in range(num_classes):
            sftmx_scr = np.exp(f_i[j])/sum_i
            dW[:, j] += (sftmx_scr-(j == y[i])) * X[i,:]
#             import pdb;pdb.set_trace()
#             _ = 0


    # Compute average
    loss /= num_train
    dW /= num_train

    # add regularization
    loss += reg*np.sum(W*W)
    dW += 2*reg*W
        
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    # get shapes
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # Compute scores
    f = np.dot(X,W)
    
    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    log_c = np.max(X)
    f -= log_c
    
    # Loss: L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
    # Total Loss: 1/N * sum(L_i)
    # Compute vector of stacked correct f-scores: [f(x_1)_{y_1}, ..., f(x_N)_{y_N}] (where N = num_train)
    f_correct = f[range(num_train),y]
    loss = -np.mean(np.log(np.exp(f_correct) / np.sum(np.exp(f))))
    
    # Compute gradient: dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
    p = np.exp(f)/np.sum(np.exp(f), axis=0)
    ind = np.zeros(p.shape)
    ind[range(num_train), y] = 1
    dW = np.dot(X.T, (p-ind))
    dW /= num_train
#     import pdb;pdb.set_trace()

    
    
    # Add regularization 
    loss += reg*np.sum(W*W)
    dW += 2*reg*W

    

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

