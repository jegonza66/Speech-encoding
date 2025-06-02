# Standard libraries
from typing import Union
import numpy as np
import os


# Modules
from mtrf_models import ReceptiveFieldAdaptation, TorchMtrf
from utils.funciones import load_pickle
import config

def fold_model(
    fold:int, 
    alpha:Union[float, np.ndarray],  
    stims:np.ndarray, 
    eeg:np.ndarray, 
    relevant_indexes:np.ndarray,
    train_indexes:np.ndarray, 
    test_indexes:np.ndarray, 
    validation:bool=False, 
    shuffle:bool=False,
    statistical_test:bool=False, 
    path_null:str=None, 
    session:int=None, 
    subject:int=None
    ) -> tuple:
    """
    Perform parallel fold model training and evaluation. 
    This function is design to be run in parallel over folds.
    
    Parameters
    ----------
    fold : int
        The fold number.
    alpha : float or np.ndarray
        Regularization parameter for the model. If validation is True, it can be an array of alphas.
    stims : np.ndarray
        Stimuli data array.
    eeg : np.ndarray
        EEG data array.
    relevant_indexes : np.ndarray
        Array of relevant indexes.
    train_indexes : np.ndarray
        Array of training indexes.
    test_indexes : np.ndarray
        Array of test indexes.
    validation : bool, optional
        Whether to perform validation (default is False).
    shuffle : bool, optional
        Whether to perform permutations in order to construct null model (default is False).
    statistical_test : bool, optional
        Whether to perform statistical tests (default is False).
    path_null : str, optional
        Path to null data for statistical tests (default is None).
    session : int, optional
        Session number (default is None).
    subject : int, optional
        Subject number (default is None).
    
    Returns
    -------
    tuple
        If statistical_test is True and shuffle is False, returns (fold, weights, correlation_matrix, root_mean_square_error, p_corr, p_rmse, significant_corr_count, significant_rmse_count, null_correlation_per_channel).
        Else if statistical_test is False and shuffle is False, returns (fold, weights, correlation_matrix, root_mean_square_error).
        Otherwise (statistical_test False, shuffle True), returns (iteration, fold, weights, correlation_matrix, root_mean_square_error).
    """
    if shuffle:
        mtrf = TorchMtrf(
            relevant_indexes=np.array(relevant_indexes),
            stims_preprocess=config.stims_preprocess, 
            eeg_preprocess=config.eeg_preprocess,
            train_indexes=train_indexes, 
            test_indexes=test_indexes, 
            use_gpu=config.use_gpu,
            validation=False,
            fit_intercept=False,
            shuffle=True, 
            alpha=alpha
        )
            
        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
        weights, correlation_matrix = mtrf.fit( # n_iterations, n_chans, feats, delays; # n_iterations, n_chans
            stims, 
            eeg
            )
        return weights, correlation_matrix 
    elif validation:
        mtrf = TorchMtrf(
                relevant_indexes=np.array(relevant_indexes),
                stims_preprocess=config.stims_preprocess, 
                eeg_preprocess=config.eeg_preprocess,
                train_indexes=train_indexes, 
                test_indexes=test_indexes, 
                use_gpu=config.use_gpu,
                fit_intercept=False,
                validation=True,
                shuffle=False, 
                alpha=alpha, 
                )
        # Returns directly correlations per alpha
        return mtrf.fit(stims, eeg)
    else:
        # Implement mne model
        if config.model=='mtrf_ridge':
            weights, correlation_matrix, root_mean_square_error = old_functions(
                relevant_indexes=np.array(relevant_indexes),
                train_indexes=train_indexes, 
                test_indexes=test_indexes, 
                fit_intercept=False,
                estimator='ridge', #timedelayingridge falta config # TODO
                validation=False,
                shuffle=False,
                alpha=alpha, 
                n_jobs=1
            )
            
        else:
            mtrf = TorchMtrf(
                relevant_indexes=np.array(relevant_indexes),
                stims_preprocess=config.stims_preprocess, 
                eeg_preprocess=config.eeg_preprocess,
                train_indexes=train_indexes, 
                test_indexes=test_indexes, 
                use_gpu=config.use_gpu,
                fit_intercept=False,
                validation=False,
                shuffle=False, 
                alpha=alpha, 
            )
            
            # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
            weights, correlation_matrix, root_mean_square_error = mtrf.fit(stims, eeg)
            # weights, correlation_matrix, root_mean_square_error = mtrf.fit2(stims, eeg)
            # weights, correlation_matrix, root_mean_square_error = mtrf.fit3(stims, eeg)
            # weights, correlation_matrix, root_mean_square_error = mtrf.fit4(stims, eeg) #TODO CORRER
            # weights, correlation_matrix, root_mean_square_error = mtrf.fit5(stims, eeg) #TODO CORRER
            
            
        
        # Perform statistical test
        if statistical_test:
            # Null Hypothesis (H0): There is no significant relationship between the predicted and actual EEG data. The test statistic (e.g., correlation or RMSE) follows the null distribution.
            # Alternative Hypothesis (H1): There is a significant relationship between the predicted and actual EEG data. The test statistic follows the alternative distribution.
            null_data = load_pickle(path=os.path.join(path_null, f'null_metrics_ses_{session}_sub_{subject}_{config.random_permutations}.pkl'))
            null_correlation_per_channel, null_errors = null_data['null_correlation_per_channel_per_fold'], null_data['null_errors_per_fold']
            iterations =  null_correlation_per_channel.shape[1]

            # Correlation and RMSE (n_iterations, n_channels)
            null_correlation_matrix = null_correlation_per_channel[fold]
            null_root_mean_square_error = null_errors[fold]

            # p-values for both tests: probability of getting a value equal or greater than the measured value, given the null hypothesis distribution (P(X>=X_obs|H0))
            # (null_correlation_matrix > correlation_matrix) is the number of iterations that surpasses the measured values for each channel (n_channels)
            p_corr = ((null_correlation_matrix > correlation_matrix).sum(axis=0) + 1) / (iterations + 1) # +1 to avoid division by zero, right tail test
            p_rmse = ((null_root_mean_square_error < root_mean_square_error).sum(axis=0) + 1) / (iterations + 1) # left tail test
            return fold, weights, correlation_matrix, root_mean_square_error, p_corr, p_rmse, null_correlation_per_channel
        else:
            return fold, weights, correlation_matrix, root_mean_square_error

def old_functions(
    relevant_indexes:np.ndarray,
    train_indexes:np.ndarray,
    test_indexes:np.ndarray,
    alpha:Union[float, np.ndarray],
    estimator:str,
    stims:np.ndarray,
    eeg:np.ndarray,
    fold:int,
    ) -> tuple:
    """
    This function is a wrapper for the ReceptiveFieldAdaptation model, which is used to fit and predict EEG data based on stimuli.
    It handles the preprocessing of stimuli and EEG data, fitting the model, and making predictions.
    
    Parameters
    ----------
    relevant_indexes : np.ndarray
        Array of indexes indicating which features/channels are relevant for the analysis.
    train_indexes : np.ndarray
        Array of indexes specifying which data points to use for training the model.
    test_indexes : np.ndarray
        Array of indexes specifying which data points to use for testing the model.
    alpha : Union[float, np.ndarray]
        Regularization parameter(s) for the ridge regression. Can be a single float value or an array of values.
    estimator : str
        Type of estimator to use for the model (currently unused in implementation, defaults to 'ridge').
    stims : np.ndarray
        Stimulus data array with shape (n_samples, n_features).
    eeg : np.ndarray
        EEG data array with shape (n_samples, n_channels).
    fold : int
        Current fold number for cross-validation (used for logging purposes).
    
    Returns
    -------
    tuple
        If validation is True, returns (weights, correlation_matrix).
        Otherwise, returns (weights, correlation_matrix, root_mean_square_error).
    """
       
    mtrf = ReceptiveFieldAdaptation(
                relevant_indexes=np.array(relevant_indexes),
                stims_preprocess=config.stims_preprocess, 
                eeg_preprocess=config.eeg_preprocess,
                train_indexes=train_indexes, 
                test_indexes=test_indexes, 
                sample_rate=config.sr, 
                fit_intercept=False,
                tmin=config.tmin, 
                tmax=config.tmax, 
                estimator=estimator, #timedelayingridge falta config# todo
                validation=False,
                shuffle=False,
                alpha=alpha, 
                n_jobs=1
            )
            
    # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
    mtrf.fit(stims, eeg)
    
    weights = mtrf.coefs # Coefficients shape n_chans, feats, delays

    # Predict and save
    predicted, eeg_test = mtrf.predict(stims)

    if (predicted==0).all():
        print(f'\n\t\tFold {fold+1}/{config.n_folds} prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.')

    # Calculates and saves correlation of each channel # TODO HACER SOLO DE 0  EN ADELANTE
    try:
        correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
    except RuntimeWarning:
        correlation_matrix = np.zeros(eeg_test.shape[1])

    # Calculates and saves root mean square error of each channel
    root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
    
    return weights, correlation_matrix, root_mean_square_error