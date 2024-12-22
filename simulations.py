# Standard libraries
import numpy as np

# Specific libraries
# from multiprocessing import cpu_count, Pool
# from itertools import repeat

# Modules
from mtrf_models import Receptive_field_adaptation
import config

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

# TODO CHECK DESCRIPTION
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
        if len(iterations)>=10:
            if i in iterations[::int(len(iterations)/10)]:
                print("\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
        else:
            print("\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
    return null_weights, null_correlation, null_errors