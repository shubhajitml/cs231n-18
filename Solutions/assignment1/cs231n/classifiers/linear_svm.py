import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
#     import pdb;pdb.set_trace()
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
        # Compute gradients (one inner and one outer sum)
        dW[:, y[i]] -= X[i,:] # this is the sum over j != y_i
        dW[:,j] += X[i,:] # sum each contribution x_i s
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # same for gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # gradient regularization
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # get dims
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W)
    correct_class_score = np.choose(y, scores.T) # np.choose uses y to select elements from scores.T
    margin = scores.T - correct_class_score + 1
    margin[y, np.arange(num_train)] = 0 # accounting for the j=y_i term we shouldn't count (subtracting 1 makes up for it since w_j = w_{y_j} in this case)
    
    thresh = np.maximum(np.zeros((num_classes, num_train)), margin)
    
    # loss 
    loss = np.sum(thresh)
    loss /= num_train
    
    # add regularization
    loss += reg * np.sum(W * W)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Binarize into integers
    binary = thresh
    binary[thresh > 0] = 1

    # Perform the two operations simultaneously
    # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i]
    # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i]
    col_sum = np.sum(binary, axis=0)
    binary[y, range(num_train)] = -col_sum[range(num_train)]
    dW = np.dot(X.T, binary.T)
    
    # Divide
    dW /= num_train

    # Regularize
    dW += 2*reg*W
    
#     import pdb;pdb.set_trace()
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
