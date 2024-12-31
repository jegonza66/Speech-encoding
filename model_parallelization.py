# Standard libraries
import numpy as np, os

# Specific libraries
from joblib import Parallel, delayed

# Modules
from mtrf_models import Receptive_field_adaptation
from mtrf_models import TorchMtrf
from processing import block_bootstrap
from funciones import load_pickle
import config

def parallel_fold_model(
    fold:int, alpha:float, stims:np.ndarray, eeg:np.ndarray, relevant_indexes:np.ndarray,
    train_indexes:np.ndarray, test_indexes:np.ndarray, validation:bool=False, shuffle:bool=False,
    statistical_test:bool=False, path_null:str=None, session:int=None, subject:int=None, iteration:int=0
    ) -> tuple:
    """
    Perform parallel fold model training and evaluation. 
    This function is design to be run in parallel over folds.
    
    Parameters
    ----------
    fold : int
        The fold number.
    alpha : float
        Regularization parameter for the model.
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
    iteration : int, optional
        Iteration number for permutations, used when shuffle is True (default is 0).
    
    Returns
    -------
    tuple
        If statistical_test is True and shuffle is False, returns (fold, weights, correlation_matrix, root_mean_square_error, p_corr, p_rmse, significant_corr_count, significant_rmse_count, null_correlation_per_channel).
        Else if statistical_test is False and shuffle is False, returns (fold, weights, correlation_matrix, root_mean_square_error).
        Otherwise (statistical_test False, shuffle True), returns (iteration, fold, weights, correlation_matrix, root_mean_square_error).
    """
    # Implement mne model
    if config.model=='mtrf_ridge' or config.model=='mtrf':
        mtrf = Receptive_field_adaptation(
                                    tmin=config.tmin, 
                                    tmax=config.tmax, 
                                    sample_rate=config.sr, 
                                    alpha=alpha, 
                                    relevant_indexes=np.array(relevant_indexes),
                                    train_indexes=train_indexes, 
                                    test_indexes=test_indexes, 
                                    stims_preprocess=config.stims_preprocess, 
                                    eeg_preprocess=config.eeg_preprocess,
                                    fit_intercept=False,
                                    # n_jobs=n_jobs, 
                                    n_jobs=1,
                                    estimator=config.estimator,
                                    validation=validation,
                                    shuffle=shuffle
                                    )
        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
        mtrf.fit(stims, eeg)
        
        weights = mtrf.coefs # Coefficients shape n_chans, feats, delays

        # Predict and save
        predicted, eeg_test = mtrf.predict(stims)
    else:
        mtrf = TorchMtrf(
                alpha=alpha, 
                relevant_indexes=np.array(relevant_indexes),
                train_indexes=train_indexes, 
                test_indexes=test_indexes, 
                stims_preprocess=config.stims_preprocess, 
                eeg_preprocess=config.eeg_preprocess,
                fit_intercept=False,
                validation=validation,
                shuffle=shuffle, 
                use_gpu=config.use_gpu,
                )
        
        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
        mtrf.fit(stims, eeg)
        
        weights = mtrf.coefs # Coefficients shape n_chans, feats, delays

        # Predict and save
        predicted, eeg_test = mtrf.predict()
    
    
    if (predicted==0).all():
        print(f'\n\t\tFold {fold+1}/{config.n_folds} prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.')

    # Calculates and saves correlation of each channel
    try:
        correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
    except RuntimeWarning:
        correlation_matrix = np.zeros(eeg_test.shape[1])

    # Calculates and saves root mean square error of each channel
    root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
    
    # Perform statistical test
    if statistical_test:
        # Null Hypothesis (H0): There is no significant relationship between the predicted and actual EEG data. The test statistic (e.g., correlation or RMSE) follows the null distribution.
        # Alternative Hypothesis (H1): There is a significant relationship between the predicted and actual EEG data. The test statistic follows the alternative distribution.
        null_data = load_pickle(path=os.path.join(path_null, f'null_metrics_ses_{session}_sub_{subject}.pkl'))
        null_correlation_per_channel, null_errors = null_data['null_correlation_per_channel_per_fold'], null_data['null_errors_per_fold']
        iterations =  null_correlation_per_channel.shape[1]

        # Correlation and RMSE (n_iterations_, n_channels)
        null_correlation_matrix = null_correlation_per_channel[fold]
        null_root_mean_square_error = null_errors[fold]

        # p-values for both tests: probability of getting a value equal or greater than the measured value, given the null hypothesis distribution (P(X>=X_obs|H0))
        # (null_correlation_matrix > correlation_matrix) is the number of iterations that surpasses the measured values for each channel (n_channels)
        p_corr = ((null_correlation_matrix > correlation_matrix).sum(axis=0) + 1) / (iterations + 1) # +1 to avoid division by zero, right tail test
        p_rmse = ((null_root_mean_square_error < root_mean_square_error).sum(axis=0) + 1) / (iterations + 1) # left tail test
        
        # Calculate power of the test: probability of measuring H1 when H1 is true. It's usefull to know if the test is sensitive enough
        significant_corr_count = 0
        significant_rmse_count = 0

        # We make a bootstrap distribution of the H1, using blocks of correlation length
        for _ in range(config.power_n_bootstrap_samples):
            # Generate blocks of bootstraped samples
            bootstrap_eeg_test = block_bootstrap(eeg_test, block_size=config.correlation_length_samples)
            bootstrap_predicted = block_bootstrap(predicted, block_size=config.correlation_length_samples)

            # Calculate correlation and RMSE for bootstrap samples
            bootstrap_correlation_matrix = np.array([np.corrcoef(bootstrap_eeg_test[:, j], bootstrap_predicted[:, j])[0,1] for j in range(bootstrap_eeg_test.shape[1])])
            bootstrap_rmse = np.array(np.sqrt(np.power((bootstrap_predicted - bootstrap_eeg_test), 2).mean(0)))

            # Calculate p-values for bootstrap samples
            bootstrap_p_corr = ((null_correlation_matrix > bootstrap_correlation_matrix).sum(axis=0) + 1) / (iterations + 1)
            bootstrap_p_rmse = ((null_root_mean_square_error < bootstrap_rmse).sum(axis=0) + 1) / (iterations + 1)

            # Count significant results
            significant_corr_count += (bootstrap_p_corr < config.significance_threshold).sum()
            significant_rmse_count += (bootstrap_p_rmse < config.significance_threshold).sum()
        
        return fold, weights, correlation_matrix, root_mean_square_error, p_corr, p_rmse, significant_corr_count, significant_rmse_count, null_correlation_per_channel
    else:
        if shuffle:
            return iteration, fold, weights, correlation_matrix, root_mean_square_error
        else:
            return fold, weights, correlation_matrix, root_mean_square_error

def simulation_mtrf(
    iterations:int,
    fold:int,
    stims:np.ndarray, 
    eeg:np.ndarray,
    sr:int, 
    tmin:float, 
    tmax:float,
    relevant_indexes:list,
    alpha:float,
    train_indexes:np.ndarray,
    test_indexes:np.ndarray, 
    stims_preprocess:str,
    eeg_preprocess:str,
    null_correlation:np.ndarray, 
    null_weights:np.ndarray, 
    null_errors:np.ndarray,
    n_feats:list=[1]
    )-> tuple:
    """
    Perform mTRF simulation by running multiple iterations of the permutation test.

    Parameters:
        iterations (int): Number of iterations to run.
        fold (int): Current fold number.
        stims (np.ndarray): Stimuli data.
        eeg (np.ndarray): EEG data.
        sr (int): Sample rate.
        tmin (float): Minimum time.
        tmax (float): Maximum time.
        relevant_indexes (list): List of relevant indexes.
        alpha (float): Regularization parameter.
        train_indexes (np.ndarray): Training indexes.
        test_indexes (np.ndarray): Testing indexes.
        stims_preprocess (str): Preprocessing method for stimuli.
        eeg_preprocess (str): Preprocessing method for EEG.
        null_correlation (np.ndarray): Array to store null correlations.
        null_weights (np.ndarray): Array to store null weights.
        null_errors (np.ndarray): Array to store null errors.
        n_feats (list): Number of features, if it exceeds the limit (16/18), then 
        it doesn't perform parallel computation. Default is 1.

    Returns:
        tuple: Updated null_weights, null_correlation, and null_errors.
    """
    # Define the iterations array
    iterations = np.arange(iterations)
    
    # Whethet to perform parallel computation
    if sum(n_feats) < 16:
        results = Parallel(n_jobs=-1, verbose=0)(delayed(parallel_fold_model)(
                                                                            fold=fold, 
                                                                            alpha=alpha, 
                                                                            stims=stims, 
                                                                            eeg=eeg, 
                                                                            relevant_indexes=relevant_indexes, 
                                                                            train_indexes=train_indexes, 
                                                                            test_indexes=test_indexes, 
                                                                            validation=False, 
                                                                            shuffle=True, 
                                                                            statistical_test=False, 
                                                                            path_null=None, 
                                                                            session=None, 
                                                                            subject=None, 
                                                                            iteration=iteration
                                                                            ) for iteration in iterations)
        for i, result in enumerate(results):
            _, fold, null_weights[fold, i], null_correlation[fold, i], null_errors[fold, i] = result
    else:
        for i in iterations:
            _, fold, null_weights[fold, i], null_correlation[fold, i], null_errors[fold, i] = parallel_fold_model(
                                                                                                                fold=fold, 
                                                                                                                alpha=alpha, 
                                                                                                                stims=stims, 
                                                                                                                eeg=eeg, 
                                                                                                                relevant_indexes=relevant_indexes, 
                                                                                                                train_indexes=train_indexes, 
                                                                                                                test_indexes=test_indexes, 
                                                                                                                validation=False, 
                                                                                                                shuffle=True, 
                                                                                                                statistical_test=False, 
                                                                                                                path_null=None, 
                                                                                                                session=None, 
                                                                                                                subject=None, 
                                                                                                                iteration=i
                                                                                                                )
            if (len(iterations)>=10) and (i in iterations[::int(len(iterations)/10)]):
                print("\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
            elif len(iterations)<10:
                print("\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
    return null_weights, null_correlation, null_errors