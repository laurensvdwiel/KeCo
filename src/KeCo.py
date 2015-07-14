'''
Description:
    Contains the KeCo algorithm implementation

Tested in python 2.7.9, with the following list of installed python packages and versions:
    numpy==1.9.2, scikit-learn==0.16.1, scipy==0.15.1

@authors: Laurens van de Wiel, Evgeni Levin
'''

import numpy as np

def training(X, Y, V, order_of_samples, kernel, kernel_parameter, _lambda):
    """Simulates the  training of the KeCo algorithm
    X : training set containing the examples
    Y : training set labels (for unlabelled 0, labelled is +1 or -1)
    V : number of views
    order_of_samples : array with indices (to X and Y) representing the order of learning from the dataset
    kernel : kernel method that is to be used
    kernel_parameter : value of c (constant) for linear kernel, sigma for gaussian kernel, gamma for rbf_kernel
    _lambda : the regularization parameter"""
    
    # initialize alphas and predictions as a dictionary, with keys representing the views
    alphas = {}
    predictions = {}
    for n in range(V):
        # values for alphas and predictions are a dictionary, with keys representing the sample_index
        alphas[n] = {}
        predictions[n] = {}
    
    t = 0
    for i in order_of_samples: # retrieve the index of the sample that is used for current iteration
        t += 1
        # train a single iteration
        train_iteration(X, Y, V, predictions, alphas, i, t, kernel, kernel_parameter, _lambda)
        
    return alphas, predictions

def train_iteration(X, Y, V, predictions, alphas, i, t, kernel, kernel_parameter, _lambda):
    """Simulates a single training iteration for the KeCo algorithm
    X : training set containing the examples
    Y : training set labels (for unlabelled 0, labelled is +1 or -1)
    V : number of views
    predictions : predictions made so far (optimization, so predictions do not need to be recalculated each iteration)
    alphas : the alpha vectors (sparse vector representing weight per sample)
    i : index for the current sample that is to be learned from
    t : number of the current iteration, with 1<= t <= T
    kernel : kernel method that is to be used
    kernel_parameter : value of c (constant) for linear kernel, sigma for gaussian kernel, gamma for rbf_kernel
    _lambda : the regularization parameter"""
    
    # retrieve the predictions for sample i
    for n in range(V):
        if not predictions[n].has_key(i):
            predictions[n][i] = 0.0
            
        predictions[n][i] = coagreement_prediction_for_view_n(X[n][i], X, Y, V, n, t, predictions, alphas, kernel, kernel_parameter, _lambda)
    
    
    for n in range(V):
        # compute loss
        z = z_j(Y, V, i, n, predictions)*predictions[n][i]
        if np.maximum(0, 1.-z) > 0: 
            # loss within threshold, increment alpha weighth
            if not(alphas[n].has_key(i)):
                alphas[n][i] = 0.0 # ensure we are not receiving key not found error
            alphas[n][i] += 1.0
    

def coagreement_prediction_for_view_n(x_i, X, Y, V, n, t, predictions, alphas, kernel, kernel_parameter, _lambda):
    """Performs a single prediction for a sample x_i for the view n
    x_i : the sample, whose label is to be predicted
    X : training set containing the examples
    Y : training set labels (for unlabelled 0, labelled is +1 or -1)
    V : number of views
    n : the index of the view this prediction is for
    t : number of the current iteration, with 1<= t <= T
    predictions : predictions made so far (optimization, so predictions do not need to be recalculated each iteration)
    alphas : the alpha vectors (sparse vector representing weight per sample)
    kernel : kernel method that is to be used
    kernel_parameter : value of c (constant) for linear kernel, sigma for gaussian kernel, gamma for rbf_kernel
    _lambda : the regularization parameter"""
    
    # initialize the predition as 0
    y_pred_i =0.
    
    # iterate over the alphas and form predictions
    for j in alphas[n].keys():
        if kernel_parameter is None:
            y_pred_i += alphas[n][j]*z_j(Y, V, j, n, predictions)*(kernel(x_i, X[n][j])[0][0])
        else:
            y_pred_i += alphas[n][j]*z_j(Y, V, j, n, predictions)*(kernel(x_i, X[n][j], kernel_parameter)[0][0])
    
    # Apply the regularization
    y_pred_i /= (_lambda * t)
    y_pred_i = y_pred_i
    
    # return the prediction
    return y_pred_i


def z_j(Y, V, j, n, predictions):
    """ retrieve_label_j_for_view_n
    Returns the label for labelled examples and the signed
    co-agreement for unlabelled examples."""
    if Y[j] == 0.:
        return c_j(n, j, V, predictions)
    else:
        return Y[j]
    

def c_j(n, j, V, predictions):
    """Perform the coagreement for view n
    Co-agreement represents the agreement between the
    different views in order to label an example."""
    # ensure we have at least multiple views
    assert V >= 2
    
    c = 0.0
    for v in range(V):
        if v != n:
            c += predictions[v][j]
    c = np.sign(c)
    
    return c