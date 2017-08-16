import numpy as np
#Input 
X = np.array([[10,1],
              [5,2],
              [9,4]])
#Weights
W1 = np.array([[0.6,0.4,0.9],
              [0.4,0.6,0.9]])
W2 = np.array([[0.6,0.4,0.3],
               [0.5,0.2,0.4],
               [0.7,0.4,0.2]])
W3 = np.array([[0.4],
               [0.6],
               [0.7]])
#Biases
b1 = 0.16
b2 = 0.43
b3 = 0.61

#Output
y = 0.75

#Activation Function and its derivative
def sigmoid(x,deriv=False):
    if deriv==True:
        return x*(1-x)
    return 1/(1 + np.exp(-x))

#
for i in range(1,100):
#Forward Propagation
    l0 = X
    l1 = sigmoid(np.dot(X,W1) + b1)
    l2 = sigmoid(np.dot(l1,W2) + b2)
    l3 = sigmoid(np.dot(l2,W3)+ b3)   #Output
    
#Backward Propagation
#Error Calculation
    l3_error = y - l3
    l3_del = l3_error * sigmoid(l3,True)
    l3_delta = np.dot(l2.T,l3_del)
    l2_error = l3_error.dot(W3.T)
    l2_del = l2_error * sigmoid(l2,True)
    l2_delta = np.dot(l1.T,l2_del)
    l1_error = l2_error.dot(W2.T)
    l1_del = l1_error * sigmoid(l1,True)
    l1_delta = np.dot(l0.T,l1_del)
    
#Weights and Biases Update
    W1 = W1 + l1_delta
    W2 = W2 + l2_delta
    W3 = W3 + l3_delta
    b1 = b1 + l1_del
    b2 = b2 + l2_del
    b3 = b3 + l3_del
    
    