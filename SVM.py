import numpy as np

# this matrix will store the data
labels = np.array ([])

# and this vector will store the labels
points = np.array ([])

# open up the input text file
with open('bc.txt') as f:
     #
     # read in the lines and init the data and labels
     lines = f.readlines ()
     labels = np.zeros (len (lines))
     points = np.zeros ((len (lines), 30))
     counter = 0
     #
     # loop through each of the lines
     for line in lines:
          #
          # get all of the items on the line
          array = [x for x in line.split (',')]
          #
          # get the data point
          for index in range (2, 32):
               points[counter,index - 2] = float (array[index])
          #
          # if cancerous, 1, else -1
          if (array[1] == 'M'):
               labels [counter] = 1
          else:
               labels [counter] = -1
          counter = counter + 1


# evaluates the loss function and returns the loss
#
# x is the data set
# y is the labels
# w is the current set of weights
# c is the weight of the slack variables
#
def f (x, y, w, c):
     num_points = y.size
     # print("num points %d" % num_points)

     pairwise_max = pairwise_max_sum_term(x,y,w)
     summation = np.sum(pairwise_max) * (1. / num_points)
     # print("summation %d" % summation)

     w_norm_squared = np.square(np.absolute(w)).sum()
     # print("term: %f" % (num_points * c))

     wns_lambda = .5 / (num_points * c)
     # print("wns lambda %f " % wns_lambda)
     w_norm_squared = wns_lambda * w_norm_squared
     # print("w norm squared, %d" % w_norm_squared)
     # fill in missing code here!!
     return w_norm_squared + summation


def pairwise_max_sum_term(x, y, w):
     """
     Gets us the numpy array representing the max
     in the expression we're minimizing.
     NOTE: this does NOT include the sum or division by n

     :param x: 
     :param y: 
     :param w: 
     :return: 
     """
     second_term = get_2nd_term_in_max(x, y, w)
     # print("checking value in max:")
     # print(second_term)
     dim = second_term.shape
     zeros = np.zeros(dim)
     # returns dimension (n,1) -- same dimensionality as invoked function
     result = np.maximum(second_term, zeros)
     # print("result of summation: %d" % result.sum())
     # print(result)
     return result.sum()


def get_2nd_term_in_max(x, y,w):
     """
     Gets us the second term in the expression we're minimizing, not including
     the max call. 
     :param x: 
     :param y: 
     :param w: 
     :return: 
     """
     # assume the matrices have the following dimensions
     # w - (30,1)
     # x - (n,30)
     # y - (n,1)
     num_points = y.size

     # (n,1) matrix
     w_times_x = np.dot(x, w)
     # (1) matrix
     y_w_x = np.multiply(y, w_times_x)

     ones = np.full(y_w_x.shape, 1)
     # returns dimension (n,1)
     return np.subtract(ones, y_w_x)



def get_greater_than_0(np_arr):
     """
     creates an np array of same dimension, 
     filling in a 0 for all elements less than 0, and 1 otherwise
     :param np_arr: 
     :return: 
     """
     trim_to_zero_1 = np_arr.copy()
     trim_to_zero_1[trim_to_zero_1 > 0] = 1
     trim_to_zero_1[trim_to_zero_1 < 0] = 0
     return trim_to_zero_1


def partial_L_dw_sum_no_max(x,y,w):
     """
     gets the partial corresponding to the second term in the 
     minimized equation, but disregarding the maximum. 
     Note: we return a vector of size (n,d), 
     where n is our number of points and d is the dimensionality of w.
     The rows can be summed up to get one vector dL/dw
     
     :param w: 
     :param x: 
     :param y: 
     :return: 
     """

     greater_than_0 = get_greater_than_0(get_2nd_term_in_max(x,y,w))
     greater_than_0_repeated = np.tile(greater_than_0, (30,1)).transpose()

     # gets us dimension (n,30)
     repeat_y = np.tile(y, (30,1)).transpose()
     repeat_y *= -1.
     # gets us dimension (n,30)
     y_times_x = np.multiply(repeat_y,x)
     result =  (1/y.size) * np.multiply(y_times_x,greater_than_0_repeated)
     # print("greater_than_0.shape")
     # print(greater_than_0.shape)
     # print("repeat_y.shape")
     # print(repeat_y.shape)
     # print("y time x shape")
     # print(y_times_x.shape)
     # print("result")
     # print(result.shape)

     return result
# evaluates and returns the gradient 
#
# x is the data set
# y is the labels
# w is the current set of weights
# c is the weight of the slack variables
#
def gradient(x, y, w, c):
     # assume the matrices have the following dimensions
     # w - (30,1)
     # x - (n,30)
     # y - (n,1)

     lbda = .5 / (y.size * c)

     first_term = lbda * 2 * w

     incomplete_2nd_partial = partial_L_dw_sum_no_max(x,y,w)
     second_partial = np.sum(incomplete_2nd_partial,0)

     # Note that the gradient has 30 dims because the data has 30 dims

     return np.add(first_term, second_partial)

# make predictions using all of the data points in x
# print ‘success’ or ‘failure’ depending on whether the 
# prediction is correct 
#
# x is the data set
# y is the labels
# w is the current set of weights
#
def predict (x, y, w):
     correct = 0
     claimed_positives = 0
     actual_positives = 0
     true_positives = 0

     for index in range(len (y)):
          if ((np.dot (x[index], w) > 0) and (y[index] > 0)):
               # true positive
               claimed_positives += 1
               actual_positives += 1
               true_positives += 1
               print ('success - true positive')
               correct = correct + 1
          elif ((np.dot (x[index], w) < 0) and (y[index] < 0)):
               # true negative
               print ('success - true negative')
               correct = correct + 1
          elif ((np.dot(x[index], w) > 0) and (y[index] < 0)):
               claimed_positives += 1
               # false positive
               print ('failure - - false negative ')
          else:
               actual_positives += 1
               # ((np.dot(x[index], w) < 0) and (y[index] > 0)):
               # false negative
               print('failure - false negative')
     recall = true_positives * 1. / actual_positives
     precision = true_positives * 1. / claimed_positives
     print("True positives: %d. Actual positives: %d .claimed positives: %d"
           % (true_positives, actual_positives, claimed_positives))
     # print(true_positives * 1. / claimed_positives)
     print("Precision: %f . Recall: %f" % (precision, recall))
     f1_score = (2 * precision * recall) / (precision + recall)
     print ('%d out of %d correct.' % (correct, len(y)))
     print("f1 score: %f" % f1_score)
                 
# performs gradient descent optimization, returns the learned set of weights
# uses the bold driver to set the learning rate
#
# x is the data set
# y is the labels
# w is the current set of weights  to start with
# c is the weight of the slack variable
#
def gd_optimize (x, y, w, c):
    rate = 1
    w_last = w + np.full (30, 1.0)
    # print("x dimensions")
    # print(x.shape)
    # print("y dimensions")
    # print(y.shape)
    # # print(y)
    # print("w dimensions")
    # print(w.shape)
    # print("slack variable value: %d" % c)

    while (abs(f (x, y, w, c) - f (x, y, w_last, c)) > 2e-6):
    # while (abs(f (x, y, w, c) - f (x, y, w_last, c)) > 10e-4):
         w_last = w 
         w = w - rate * gradient (x, y, w, c)
         if f (x, y, w, c) > f (x, y, w_last, c):
              rate = rate * .5
         else:
              rate = rate * 1.1
         print (f (x, y, w, c))
    return w

######## RUN ONCE YOU'RE READY TO TEST #########

w = np.zeros (30)

points_in = points[0:400]
labels_in = labels[0:400]
# Original is c = .1:
c = .1
# c = .02

w = gd_optimize (points_in, labels_in, w, c)

# output = f(points_in, labels_in, np.ones(30), c)
# output = f(points_in, labels_in, w, c)
# print("output of f: %f" % output)
predict (points[400:], labels[400:], w)

# OUTPUT FROM TRAINED W:
# c = .1
# error threshold = 2e-6
# Output: 46.708858
#
# TESTING RESULTS:
# True positives: 35. Actual positives: 39 .claimed positives: 48
# Precision: 0.729167 . Recall: 0.897436
# 152 out of 169 correct.
# f1 score: 0.804598
