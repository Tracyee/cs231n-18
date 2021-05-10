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
    inc_count = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        inc_count += 1
        dW[:, j] += X[i] # gradient update for incorrect rows
        loss += margin
    # gradient update for correct row
    dW[:, y[i]] += -inc_count * X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss /= num_train
  dW /= num_train
  dW += reg*W # regularize the weights
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

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
  # result in loss.                                                                                 #
  #############################################################################

  num_train = X.shape[0]
  delta = 1.0
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  #for i in xrange(num_train):
  #  scores = X[i].dot(W)
  #  # compute the margins for all classes in one vector operation
  #  margins = np.maximum(0, scores - scores[y[i]] + delta)
  ## on y-th position scores[y] - scores[y] canceled and gave delta. We want
  ## to ignore the y-th position and only consider margin on max wrong class
  #margins[y[i]] = 0
  #loss += np.sum(margins)
  #loss /= num_train
  #loss += 0.5 * reg * np.sum(W * W) # regularization

  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 50,000 x 3073 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 3073 x 10)
  """
  scores = X.dot(W)
  correct_class_score = scores[range(num_train), y]
  margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + delta)
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) # regularization
  pass
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

  X_mask = np.zeros(margins.shape)
  # column maps to class, row maps to sample; a value v in X_mask[i, j]
  # adds a row sample i to column class j with multiple of v
  X_mask[margins > 0] = 1
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train # average out weights
  dW += reg*W # regularize the weights

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
