# Demo artificial neural net with one hidden layer
# Originally Created by Professor Walker - Adapted by Liptack, Lee, Holman, Perry, and Germanakos
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

###ROBBY EDIT

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from sklearn import preprocessing

#%%
#Read text file for training data
train_raw = []
f = open('TrainingData.txt', 'r')
for line in f:
    train_raw.append(line.split())
train_raw=np.array(train_raw)
train_raw=train_raw.astype(float)

#Read text file for test data
test_raw = []
f = open('TestingData.txt', 'r')
for line in f:
    test_raw.append(line.split())
test_raw=np.array(test_raw)
test_raw=test_raw.astype(float)


#%%
#Define T's and Q's from imported data
T_train_original = train_raw[:,[0]]
Q_train_original = train_raw[:,[1]]

T_test_original = test_raw[:,[0]]
Q_test_original = test_raw[:,[1]]


#%%
#Set Inputs to One Time Step Difference
T_train_col_1 = T_train_original[1:]
T_train_col_2 = T_train_original[:-1]
T_train = np.concatenate((T_train_col_1,T_train_col_2), axis=1)

T_test_col_1 = T_test_original[1:]
T_test_col_2 = T_test_original[:-1]
T_test = np.concatenate((T_test_col_1,T_test_col_2), axis=1)

#Normalize Inputs
#T_train = preprocessing.scale(T_train)
T_train = T_train/40
T_train = np.array(T_train)

#T_test = preprocessing.scale(T_test)
T_test = T_test/40
T_test = np.array(T_test)


#%%
#Define Outputs
Q_train = Q_train_original[:-1]

Q_test = Q_test_original[:-1]

#Normalize Outputs
Q_train = preprocessing.scale(Q_train)
Q_train = np.array(Q_train)

Q_test = preprocessing.scale(Q_test)
Q_test = np.array(Q_test)


#%%
# define the data size (normally we might read this from a file?)
N_input = 2
N_output = 1

# Define the ANN structure
N_node = 10        # number of nodes in the hidden layer
N_pass = 10000    # iterations for convergence
gamma = 0.0002    # step size for gradient descent
alpha = 0.05      # regularization

#Define sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
#Define ReLU
def relu(x):
    return np.maximum(0,x)

    
#%%
# Solve the forward problem with given weights (parms) and inputs (x).
def forward(parms, x):
    
    W1, b1, W2, b2 = parms['W1'], parms['b1'], parms['W2'], parms['b2']

    # Forward propagation (using tanh activation b/c this is a regression
    # problem, not classification).
    
    z1 = np.dot( x, W1 ) + b1
    z2 = np.tanh( z1 )
    #z2 = relu( z1 )
    z3 = np.dot( z2, W2 ) + b2
    
    return z3 # This is yhat


# Train the ANN.  This will create the parms dictionary too.
def training( X, Y ):

    # W are weights, b are offsets
    W1 = np.random.rand( N_input, N_node )
    b1 = np.zeros( (1, N_node) )
    W2 = np.random.rand( N_node, N_output )
    b2 = np.zeros( (1, N_output) )

    parms = {}

    for i in range( N_pass ):

        # Forward propagation (can't use forward function because we need the
        # intermediate results)
        z1 = np.dot( X, W1 ) + b1
        z2 = np.tanh( z1 )
        #z2 = relu( z1 )
        z3 = np.dot( z2, W2 ) + b2


        # Backpropagation
        err3 = z3 - Y
        err2 = np.dot( err3, W2.T ) * (1 - np.power( z2, 2 ))

        dW2 = np.dot( z2.T, err3 )
        db2 = np.sum( err3, axis=0, keepdims=True )
        dW1 = np.dot( X.T, err2 )
        db1 = np.sum( err2, axis=0 )

 
        # Regularization for stability
        dW2 += alpha * W2
        dW1 += alpha * W1

        
        # Gradient descent parameter update
        W1 -= gamma * dW1
        b1 -= gamma * db1
        W2 -= gamma * dW2
        b2 -= gamma * db2
         
        # Assign new parameters to the parms
        parms = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Show progress 
        if (i%1000) == 0:
            print( "residual %i: %f"%(i, np.linalg.norm(z3-Y)) )
     
    return parms


if __name__== "__main__":

    # train and produce the estimates
    parms = training( T_train, Q_train )
    print( parms['W1'], parms['b1'], parms['W2'], parms['b2'] )
#    q_out = np.zeros(len(T_test))
#    for i in range( len(q_out) ):
#        q_out[i]=forward( parms, T_test[i] )
    q_out = forward( parms, T_test )

    #Plot
    plt.plot(q_out, 'r--', label='NN')
    plt.plot(Q_test, 'b-', label='PID')
    plt.legend()
    plt.show()
    
    # output the results of the ANN
    datfile = open( "ann.out", "w" )
    for i in range( len(T_test) ):
        datfile.write( "%g %g %g %g \n"%
            #(X[i,0], X[i,1], yhat[i,0], Y[i,0], yhat[i,1], Y[i,1]) )
            (T_test[i,0], T_test[i,1], q_out[i], Q_test[i]) )
    datfile.close()


