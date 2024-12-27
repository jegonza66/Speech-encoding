# Standard libraries
import numpy as np

# Specific libraries
from joblib import Parallel, delayed

# Modules
from mtrf_models import Receptive_field_adaptation
from processing import block_bootstrap
from funciones import load_pickle
import config

def parallel_fold_model(
    fold:int, alpha:float, stims:np.ndarray, eeg:np.ndarray, relevant_indexes:np.ndarray,
    train_indexes:np.ndarray, test_indexes:np.ndarray, validation:bool=False, 
    statistical_test:bool=False, path_null:str=None, session:int=None, subject:int=None
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
        If statistical_test is True, returns (fold, weights, correlation_matrix, root_mean_square_error, p_corr, p_rmse, significant_corr_count, significant_rmse_count, null_correlation_per_channel).
        Otherwise, returns (fold, weights, correlation_matrix, root_mean_square_error).
    """
    # Implement mne model
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
                                    validation=validation
                                    )
    
    # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
    mtrf.fit(stims, eeg)
    
    # Get weights coefficients shape n_chans, feats, delays
    weights = mtrf.coefs
    
    # Predict and save
    predicted, eeg_test = mtrf.predict(stims)
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
        if config.statistical_test:
            null_data = load_pickle(path=path_null + f'null_metrics_ses_{session}_sub_{subject}.pkl')
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
        return fold, weights, correlation_matrix, root_mean_square_error

def permutations(iteration:int,
                 eeg:np.ndarray, 
                 stims:np.ndarray, 
                 tmin:float, 
                 tmax:float, 
                 sr:int,
                 alpha:float, 
                 relevant_indexes:list, 
                 train_indexes:np.ndarray, 
                 test_indexes:np.ndarray,
                 stims_preprocess:float,
                 eeg_preprocess:float, 
                 n_jobs:int=-1, 
                 fold:int=0):
        """Perform permutations to fit a null model and evaluate its performance.
        Parameters:
        -----------
            iteration : int
                The current iteration number.
            eeg : np.ndarray
                The EEG data array.
            stims : np.ndarray
                The stimuli data array.
            tmin : float
                The minimum time value for the receptive field.
            tmax : float
                The maximum time value for the receptive field.
            sr : int
                The sample rate of the data.
            alpha : float
                The regularization parameter for the model.
            relevant_indexes : list
                List of relevant indexes for the data.
            train_indexes : np.ndarray
                Array of indexes for the training data.
            test_indexes : np.ndarray
                Array of indexes for the test data.
            stims_preprocess : float
                Preprocessing parameter for the stimuli.
            eeg_preprocess : float
                Preprocessing parameter for the EEG data.
            n_jobs : int, optional
                The number of jobs to run in parallel (default is -1).
            fold : int, optional
                The current fold number (default is 0).
        Returns:
        --------
            coefs : np.ndarray
                The coefficients of the fitted model.
            correlation_matrix : np.ndarray
                The correlation matrix of the predicted and actual EEG data.
            root_mean_square_error : np.ndarray
                The root mean square error of the predicted and actual EEG data.
        """
        # Define null model
        null_model = Receptive_field_adaptation(
                                                tmin=tmin, 
                                                tmax=tmax, 
                                                sample_rate=sr, 
                                                alpha=alpha, 
                                                relevant_indexes=np.array(relevant_indexes),
                                                train_indexes=train_indexes, 
                                                test_indexes=test_indexes, 
                                                stims_preprocess=stims_preprocess, 
                                                eeg_preprocess=eeg_preprocess,
                                                fit_intercept=False,
                                                n_jobs=n_jobs,
                                                shuffle=True, 
                                                estimator='time_delaying_ridge'
                                                )

        # The fit already already consider relevant indexes of train and test data and applies shuffle and standarization|normalization
        null_model.fit(stims, eeg)

        # Predict and save
        predicted, eeg_test = null_model.predict(stims)
        if (predicted==0).all():
            print(f'\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\t\tFold {fold+1}/{config.n_folds} prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
        # Calculates and saves correlation of each channel
        # warnings.filterwarnings("ignore", category=RuntimeWarning) # avoid runtime error dividing per zero, this is caught later
        try:
            correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
        except RuntimeWarning:
            correlation_matrix = np.zeros(eeg_test.shape[1])

        # Calculates and saves root mean square error of each channel
        root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))        
        return null_model.coefs, correlation_matrix, root_mean_square_error

def simulation_mtrf(iterations:int,
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
                    n_jobs:int=-1):
    """Perform mTRF simulation by running multiple iterations of the permutation test.

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
        n_jobs (int): Number of jobs to run in parallel. Default is -1.

    Returns:
        tuple: Updated null_weights, null_correlation, and null_errors.
    """
    # Define iterations
    iterations = np.arange(iterations)

    # if n_jobs!=1:
    #     with Pool(processes=cpu_count()) as pool:
    #         results = pool.starmap(permutations, zip(iterations, repeat(eeg), repeat(stims), repeat(tmin), repeat(tmax), repeat(sr), repeat(alpha),\
    #                   repeat(relevant_indexes), repeat(train_indexes), repeat(test_indexes), repeat(stims_preprocess), repeat(eeg_preprocess)))
    #     for i in iterations:
    #         null_weights[fold, i], null_correlation[fold, i], null_errors[fold, i], itera = results[i]
    # else:
    for i in iterations:
        null_weights[fold, i], null_correlation[fold, i], null_errors[fold, i] = permutations(
                                                                                            iteration=i, 
                                                                                            eeg=eeg, 
                                                                                            stims=stims, 
                                                                                            tmin=tmin, 
                                                                                            tmax=tmax, 
                                                                                            sr=sr, 
                                                                                            alpha=alpha, 
                                                                                            relevant_indexes=relevant_indexes, 
                                                                                            train_indexes=train_indexes, 
                                                                                            test_indexes=test_indexes, 
                                                                                            stims_preprocess=stims_preprocess, 
                                                                                            eeg_preprocess=eeg_preprocess,
                                                                                            n_jobs=n_jobs,
                                                                                            fold=fold
                                                                                            )
        if (len(iterations)>=10) and (i in iterations[::int(len(iterations)/10)]):
            print("\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
        elif len(iterations)<10:
            print("\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
    return null_weights, null_correlation, null_errors