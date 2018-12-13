# Demo artificial neural net with one hidden layer
# Greg Walker 2018 (based largely on wildml demo found online)
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/


import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


###
# Make some training data ...

np.random.seed( 0 )

a = np.loadtxt(open('testdata.csv', "rb"), delimiter=",", skiprows=1)
p = np.shape(a)

# define the data size (normally we might read this from a file?)
N = p[0]
N_input = 2
N_output = 1

# inputs
X = np.zeros( (N, N_input) )
X[:,0] = a[:,0]
X[:,1] = a[:,1]
X = 0.05*X

# outputs
Y = np.zeros( (N, N_output) )
Y[:,0] = a[:,2]

###
# Define the ANN structure

N_node = 10       # number of nodes in the hidden layer
N_pass = 80000    # iterations for convergence
gamma = 0.0002    # step size for gradient descent
alpha = 0.05      # regularization


# Solve the forward problem with given weights (parms) and inputs (x).
def forward(parms, x):
    
    W1, b1, W2, b2 = parms['W1'], parms['b1'], parms['W2'], parms['b2']

    # Forward propagation (using tanh activation b/c this is a regression
    # problem, not classification).
    
    z1 = np.dot( x, W1 ) + b1
    z2 = np.tanh( z1 )
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
    parms = training( X, Y )
    print('W1')
    print(parms['W1'])
    print('b1')
    print(parms['b1'])
    print('W2')
    print(parms['W2'])
    print('b2')
    print(parms['b2'])
    yhat = forward( parms, X )

    # output the results of the ANN
    datfile = open( "ann.out", "w" )
    for i in range( N ):
        datfile.write( "%g %g %g %g\n"%
            (X[i,0], X[i,1], yhat[i,0], Y[i,0]) )
    datfile.close()

if __name__== "__main__":
    
    ## Building Expected Q's with already trained neural network
    
    #Creating Data Set for Testing
    b = np.loadtxt(open('trial_data.csv', "rb"), delimiter=",", skiprows=1)
    c = np.shape(b)
    
    # define the data size (normally we might read this from a file?)
    N = c[0]
    N_input = 2
    N_output = 1

    # inputs
    T_test = np.zeros( (N, N_input) )
    T_test[:,0] = b[:,0]
    T_test[:,1] = b[:,1]
    T_test = 0.05*T_test

    # outputs
    Y_test = np.zeros( (N, N_output) )
    Y_test[:,0] = b[:,2]
    
    q_hat = forward(parms, T_test)
    
    plt.plot(Y_test)
    plt.plot(q_hat)
    plt.legend(['PID Q Values', 'Neural Net Q Values'])
    plt.xlabel('Time')
    plt.ylabel('Q')
