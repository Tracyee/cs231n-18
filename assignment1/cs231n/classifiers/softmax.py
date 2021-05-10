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
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    # Normalization trick to avoid numerical instability
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    #softmax:
    probs = exp_scores / np.sum(exp_scores, keepdims = True)
    #loss update
    loss += -(np.log(probs[y[i]]))
    for j in range(num_classes):
      # gradient update
      dscores = X[i].T
      dW[:, j] += (probs[j] - (j == y[i])) * dscores 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W # regularize the weights
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
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
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis = 1, keepdims = True)
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
  loss = -np.log(probs[range(num_train),y])
  loss = np.sum(loss)
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # gradient update
  dscores = probs
  dscores[range(num_train),y] -= 1
  dW = X.T.dot(dscores)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W # regularize the weights

  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

