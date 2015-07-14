'''
Description:
    Purpose of this module is to evaluate the coagreement algorithm trough various strategies 

Tested in python 2.7.9, with the following list of installed python packages and versions:
    numpy==1.9.2, scikit-learn==0.16.1, scipy==0.15.1

@author: Laurens van de Wiel
'''

import numpy as np
from sklearn.grid_search import ParameterGrid
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from KeCo import training, coagreement_prediction_for_view_n
from sklearn.metrics.ranking import roc_curve, auc
from sklearn.externals.joblib.parallel import Parallel, delayed, cpu_count

def perform_experiment(T, V, experiment_parameters, evaluation_strategy, dataset_location, n_folds=10, labelled_percentage=1.0, random_seed=None, use_unlabelled=True, use_parallel=True):
    """Each experiment we split the data into three parts, where two parts are used
for training and remaining one is used for testing, we repeat this three times,
until all parts have been considered as testing. The result of an experiment is
the average performance over the three test parts."""
    X, Y = load_svmlight_file(dataset_location)
    
    # ensure the dataset gets split into multiple views
    X_views = split_dataset_into_random_views(X, V, random_seed)
    
    # retrieve the train-test folds
    folds = StratifiedShuffleSplit(Y, test_size=0.3, random_state=random_seed)
    for train_index, test_index in folds:
        X_train = {n:X_views[n][train_index] for n in X_views.keys()}
        X_test = {n:X_views[n][test_index] for n in X_views.keys()}
        y_train, y_test = Y[train_index], Y[test_index]
    
    # unlabel the trainingset
    np.random.seed(random_seed)
    unlabel = np.random.random(len(y_train))
    for i in range(len(unlabel)):
        if unlabel[i] > labelled_percentage:
            y_train[i] = 0.0
    
    # grid search for the best grid
    best_grid = gridsearch(X_train, y_train, T, V, experiment_parameters, n_folds, evaluation_strategy, use_unlabelled, use_parallel, random_seed)
    
    # predetermine the order of samples
    order_of_samples = evaluation_strategy(y_train, T, use_unlabelled, random_seed)

    # generate the model
    alphas, predictions = training(X_train, y_train, V, order_of_samples, best_grid['grid']['kernel_method'], best_grid['grid']['kernel_parameters'], best_grid['grid']['lambdas'])
    
    # test the model    
    y_preds_est = []
    y_preds = []
    for i in range(len(y_test)):
        y_pred = {}
        y_pred_est = 0.0
        for n in range(V):
            y_pred[n] = coagreement_prediction_for_view_n(X_test[n][i], X_train, y_train, V, n, T+1, predictions, alphas, best_grid['grid']['kernel_method'], best_grid['grid']['kernel_parameters'], best_grid['grid']['lambdas'])
            y_pred_est += y_pred[n]
        y_preds.append(y_pred)
        y_preds_est.append(y_pred_est/V)
    
    # retrieve the metrics
    AUC, fpr, tpr = area_under_the_roc_curve(y_test, y_preds_est)
    
    print 'Achieved in validation '+str(AUC)+' AUC, and in training '+str(best_grid['AUC'])+' over '+str(n_folds)+' folds'
        
    return {"auc":AUC, "fpr":fpr, "tpr":tpr, "model":alphas, "best_grid":best_grid}

def crossvalidate(T, V, X_views, Y, folds, n_folds, use_unlabelled, evaluation_strategy, parameter_set, random_seed):
    AUC = 0.0
    # single fold:
    for train_index, test_index in folds:
        X_train = {n:X_views[n][train_index] for n in X_views.keys()}
        X_test = {n:X_views[n][test_index] for n in X_views.keys()}
        y_train, y_test = Y[train_index], Y[test_index]
    
        order_of_samples = evaluation_strategy(y_train, T, use_unlabelled, random_seed)

        alphas, predictions = training(X_train, y_train, V, order_of_samples, parameter_set['kernel_method'], parameter_set['kernel_parameters'], parameter_set['lambdas'])
        
        y_preds_est = []
        y_preds = []
        
        # ensure we only use the labelled test set
        labelled_y_test = [item for item in y_test if item != 0.0]
        for i in range(len(labelled_y_test)):
            y_pred = {}
            y_pred_est = 0.0
            for n in range(V):
                y_pred[n] = coagreement_prediction_for_view_n(X_test[n][i], X_train, y_train, V, n, T+1, predictions, alphas, parameter_set['kernel_method'], parameter_set['kernel_parameters'], parameter_set['lambdas'])
                y_pred_est += y_pred[n]
            y_preds.append(y_pred)
            y_preds_est.append(y_pred_est/V)
        
        AUC_fold, _, _ = area_under_the_roc_curve(labelled_y_test, y_preds_est)
        AUC+= AUC_fold
        
    AUC /= n_folds
    
    print 'grid search with set '+str(parameter_set)+' achieving '+str(AUC)+' over '+str(n_folds)+' folds'
    
    current_grid = {}
    current_grid['AUC'] = AUC
    current_grid['grid'] = parameter_set
    
    return current_grid
    

def gridsearch(X_views, Y, T, V, experiment_parameters, n_folds, evaluation_strategy, use_unlabelled, use_parallel, random_seed):
    # retrieve grid
    parameterGrid = ParameterGrid(experiment_parameters)
    
    # compute permutations on the sample sets for the number of folds
    folds = StratifiedKFold(Y, n_folds=n_folds, random_state=random_seed)
    
    # search for the best grid
    best_grid = dict()
    best_grid['AUC'] = 0.0    
    
    # execute grid search in a parallel setting
    if use_parallel:
        grid_performance = Parallel(n_jobs=calculate_number_of_active_threads(len(parameterGrid)))(delayed(crossvalidate)(T, V, X_views, Y, folds, n_folds, use_unlabelled, evaluation_strategy, grid, random_seed) for grid in parameterGrid)
    else:
        grid_performance = [crossvalidate(T, V, X_views, Y, folds, n_folds, use_unlabelled, evaluation_strategy, grid, random_seed) for grid in parameterGrid]

    # check which grid performed best
    for current_grid in grid_performance:
        if best_grid['AUC'] < current_grid['AUC']:
            best_grid = current_grid 
    
    print 'Best parameter set '+str(best_grid['grid'])+' achieving '+str(best_grid['AUC'])+' over '+str(n_folds)+' folds'
    
    # return best grid
    return best_grid
    
def area_under_the_roc_curve(yTrue, yPred):
    fpr, tpr, _ = roc_curve(yTrue, yPred)
    AUC = auc(fpr, tpr)
    return AUC, fpr, tpr
    
def split_dataset_into_random_views(X, V, percentage_columns_per_view = 0.75, random_seed=None):
    """Splits a dataset column based equally based on the number of views
    remainders are added at the end"""
    if V == 1:
        # only single view is used, return X
        return {0:X}
    
    # ensure the seed is fresh
    np.random.seed(random_seed)
    
    # retrieve the number of features
    n_features = X.shape[1]
    
    # create a cutoff threshold for the max features per view
    columns = int( n_features * percentage_columns_per_view)
    
    # initialize the value that is to be returned
    X_with_views = {}
    
    # assign the features to the views
    for n in range(V):
        # create a random view permutaon
        random_features_for_view = range(n_features)
        np.random.shuffle(random_features_for_view)
        # use a cutoff
        indices_for_view = random_features_for_view[:columns]
        
        X_with_views[n] = X[:, indices_for_view]
         
    return X_with_views

def split_dataset_into_views(X, V):
    """Splits a dataset column based equally based on the number of views
    remainders are added at the end"""
    if V == 1:
        # only single view is used, return X
        return {0:X}
    
    # retrieve the number of features
    n_features = X.shape[1]
    
    # initialize the value that is to be returned
    X_with_views = {}
    
    # assign the features to the views
    split_size = n_features / V
    remainder = n_features % V
    for n in range(V):
        if n == V-1:
            a = n*split_size
            b = (n+1)*split_size + remainder
        else:
            a = n*split_size
            b = (n+1)*split_size
        X_with_views[n] = X[:, a:b]
        
    return X_with_views

def strategy_50L_then_1L1UL(Y, T, use_unlabelled, random_seed=None):
    """Y = list of labels, expected to be -1 or +1 when labelled and 0 when unlabelled
    T = number of iterations for training
    First evaluates 50% of numvewr of iterations T as labelled samples,
    then, If use_unlabbeled = True:
    sequentially sample 1 labelled followed by one unlabelled sample
    else use_unlabbeled = False: 
    sequentially sample 1 labelled followed by skipping the unlabelled sample"""
    # ensure the seed is fresh
    np.random.seed(random_seed)
    
    # retrieve the indices of the unlabelled training samples
    unlabelled_indices = [index for index, value in enumerate(Y) if value == 0.0]
    labelled_indices = [index for index, value in enumerate(Y) if value != 0.0]
    
    # generate what examples will be used to train on and in what order
    order_of_samples = []
    for t in range(T):
        # randomly choose i
        if t % 2 == 0 and t > T/2:
            if len(unlabelled_indices) > 0:
                i = np.random.randint(0, len(unlabelled_indices)) # ensure we are generating the index regardles wether we are considering unlabelled samples
                if use_unlabelled:
                    # add a random unlabelled sample
                    order_of_samples.append(unlabelled_indices[i])
                # else: attempting to add an unlabelled sample's index to the order_of_samples list, but there are no unlabelled samples provided
        else:
            # add a random labelled sample
            order_of_samples.append(labelled_indices[np.random.randint(0, len(labelled_indices))])
    return order_of_samples

def calculate_number_of_active_threads(numberOfTasks):
    """
    Calculates the number of threads possible, given the number
    of processor cores and the number of tasks needed to be 
    parallelized
    """
    if(cpu_count() == 2):
        return cpu_count()
    elif numberOfTasks < cpu_count():
        return numberOfTasks
    else:
        return cpu_count()