'''
Description:
    Sets up the experiments, resulting in the paper's results

Tested in python 2.7.9, with the following list of installed python packages and versions:
    numpy==1.9.2, scikit-learn==0.16.1, scipy==0.15.1

@author: Laurens van de Wiel
'''
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import numpy as np
import time
from evaluation import strategy_50L_then_1L1UL,  perform_experiment
import os
import sys


def convertSigmaTogamma(sigma):
    """Used for converting gamma values for the gaussian kernel
    to gammas for the RBF kernel (used from the sklearn package)
    RBF:
    K(x, y) = exp(-gamma ||x-y||^2)
    Instead of
    K(x, y) = exp(- 1/(2 sigma^2) ||x-y||^2)"""
    return 1.0/(2*(sigma**2))

def run_experiment(experimental_setup, n_experiments, random_seed):
    """ Runs a single experimental setup, as many as n_experiments
    times with each a different random_seed, generated from the 
    initial random_seed
    """
    dataset_location = experimental_setup['dataset_location'] # which dataset is being evaluated
    labelled_percentage = experimental_setup['labelled_percentage'] # determines the percentage of unlabelled data
    use_unlabelled = experimental_setup['use_unlabelled'] # determines if the unlabelled parts are used for learning (e.g. semi-supervised setting)
    evaluation_strategy = experimental_setup['evaluation_strategy'] # determines the learning strategy
    parameter_grid = experimental_setup['parameter_grid'] # determines on which parameters a grid search is performed
    V = experimental_setup['views']  # The number of views, if V > 2 if S_UL != []
    n_crossvalidation_folds = experimental_setup['n_crossvalidation_folds'] # number of folds used for cross validation
    
    
    print "Starting experiment according to setup: "+str(experimental_setup)
    results = [] # will contain the results
    for i in range(n_experiments):
        # track duration of training
        start = time.clock()
        
        # set the seed used throughout the experiment
        np.random.seed(random_seed)
        random_seed = np.random.randint(0, 2147483647) # maxint on windows system
        
        print "Starting experiment "+str(i+1)+" out of "+str(n_experiments)+" with seed: "+str(random_seed)
        
        results.append(perform_experiment(T=T, V=V, experiment_parameters=parameter_grid, evaluation_strategy=evaluation_strategy, dataset_location=dataset_location, n_folds=n_crossvalidation_folds, labelled_percentage=labelled_percentage, random_seed=random_seed, use_unlabelled=use_unlabelled, use_parallel=use_parallel))
        
        # compute duration of iteration
        end = time.clock()
        print "Experiment "+str(i+1)+" out of "+str(n_experiments)+", for dataset "+dataset_location+", finished in "+ str(end - start) +" seconds"
    
    # check the total auc
    total_auc = 0.0
    for result in results:
        print 'auc: '+str(result['auc'])
        total_auc += result['auc']
    
    print "For dataset "+dataset_location+", with semi-supervised="+str(use_unlabelled)+", T: "+str(T)+", V: "+str(V)+", achieved average auc: "+str(total_auc/n_experiments)
    
    return results


if __name__ == "__main__":
    # retrieve which dataset you would like to evaluate (e.g. "../datasets/ionosphere_scale.txt")
    dataset_location = sys.argv[1]
    
    if not(os.path.isfile(dataset_location)):
        # not a valid dataset location dataset locations
        print 'argument '+dataset_location+' is not an existing file, included dataset is "../datasets/ionosphere_scale.txt"'
    else:
        T = 1000 # The maximum number of iterations
        lambdas = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1,] # The regularization parameter for labelled examples
        sigmas = [2.0**-4, 2.0**-3, 2.0**-2, 2.0**-1, 2.0**-0, 2.0**1, 2.0**2, 2.0**3,] # The standard deviation value for the Gaussian Kernel
        gammas = [convertSigmaTogamma(sigma) for sigma in sigmas] # convert the sigmas to gammas, so the gaussian kernel may be computed with the rbf kernel
        n_crossvalidation_folds = 10 # number of folds to cross validate on
        n_experiments = 5 # number of runs for each experiment
        random_seed = 1 # seed used for all randomness troughout the project
        use_parallel = True # determines if we should use threading throughout the code
        
        linear_setup = {"kernel_method": [linear_kernel],"lambdas":lambdas,"kernel_parameters":[None]}
        gaussian_setup = {"kernel_method": [rbf_kernel],"lambdas": lambdas,"kernel_parameters":gammas}
        
        evaluation_strategy = strategy_50L_then_1L1UL # strategy with a 'warm start'
        
        percent_labelled = 0.2
        
        # specifiyng the experiments
        experiment_linear_baseline = {"dataset_location":dataset_location,
                                                  "views": 1, 
                                                  "use_unlabelled":False, 
                                                  "parameter_grid":linear_setup,
                                                  "n_crossvalidation_folds":n_crossvalidation_folds,
                                                  "labelled_percentage":percent_labelled,
                                                  "evaluation_strategy":evaluation_strategy,}
        experiment_linear_2_view = {"dataset_location":dataset_location,
                                                  "views": 2, 
                                                  "use_unlabelled":True, 
                                                  "parameter_grid":linear_setup,
                                                  "n_crossvalidation_folds":n_crossvalidation_folds,
                                                  "labelled_percentage":percent_labelled,
                                                  "evaluation_strategy":evaluation_strategy,}
        experiment_gaussian_baseline = {"dataset_location":dataset_location,
                                                  "views": 1, 
                                                  "use_unlabelled":False, 
                                                  "parameter_grid":gaussian_setup,
                                                  "n_crossvalidation_folds":n_crossvalidation_folds,
                                                  "labelled_percentage":percent_labelled,
                                                  "evaluation_strategy":evaluation_strategy,}
        experiment_gaussian_2_view  = {"dataset_location":dataset_location,
                                                "views": 2, 
                                                "use_unlabelled":True, 
                                                "parameter_grid":gaussian_setup,
                                                "n_crossvalidation_folds":n_crossvalidation_folds,
                                                "labelled_percentage":percent_labelled,
                                                "evaluation_strategy":evaluation_strategy,}
        
        # determine order of the experiments
        experiments = [experiment_linear_baseline,
                       experiment_linear_2_view,
                       experiment_gaussian_baseline,
                       experiment_gaussian_2_view,]
        
        # track duration of training
        start_of_experiments = time.clock()
        for experiment in experiments:
            run_experiment(experimental_setup=experiment, n_experiments=n_experiments, random_seed=random_seed)
            
        # compute duration of iteration
        end_of_experiments = time.clock()
        print "Total time to compute all experiments: "+ str(end_of_experiments - start_of_experiments) +" seconds"