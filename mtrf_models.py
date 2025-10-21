#TODO implement solvers
# Standard libraries
import time
import numpy as np, mne
mne.set_log_level(verbose='WARNING')

# Specific libraries
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.linear_model import Ridge
from typing import Union
from tqdm import tqdm
import torch
import gc
# from torchaudio.functional import fftconvolve

# Modules
from utils.processing import Normalize, Standarize, shifted_matrix
# from utils.processing import band_freq, cheby2_bandpass_filter_torch
import config

class TorchMtrf:
    def __init__(
        self, 
        alpha:Union[float, np.ndarray], 
        relevant_indexes:np.ndarray, 
        train_indexes:np.ndarray, 
        test_indexes:np.ndarray, 
        stims_preprocess:str, 
        eeg_preprocess:str, 
        fit_intercept:bool=False, 
        shuffle:bool=False, 
        validation:bool=False,
        use_gpu:bool=True,
        solver:str='ridge'
    )->None:
        """
        Initialize the TorchMtrf model, a PyTorch implementation of the TimeDelayingRidge of stimulus to predict EEG.

        Parameters
        ----------
        alpha : float or np.ndarray, optional
            Regularization strength. If validation is True, this should be an array of alphas to be swept, by default None.
        relevant_indexes : np.ndarray
            Array of relevant indexes.
        train_indexes : np.ndarray
            Array of training indexes.
        test_indexes : np.ndarray
            Array of testing indexes.
        stims_preprocess : str
            Preprocessing solver for stimuli.
        eeg_preprocess : str
            Preprocessing solver for EEG data.
        fit_intercept : bool, optional
            Whether to fit the intercept, by default False.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        validation : bool, optional
            Whether to perform validation, by default False.
        use_gpu : bool, optional
            Whether to use the GPU (CUDA) for computation, by default True.
        solver : bool, optional
            Whether to apply Tikhonov regularization, by default False.

        Returns
        -------
        None
        
        Raises
        ------
        None
        """
        assert solver in ['ridge', 'ridge-laplacian', 'fourier-ridge'], f"solver {solver} is not supported. Use 'ridge', 'ridge-laplacian' or 'fourier-ridge'."
        self.solver = solver
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.alpha = alpha
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess    
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.validation = validation
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    def fit(
        self, 
        stims:np.ndarray, 
        eeg:np.ndarray
    )->None:
        """
        Fit the TorchMtrf model to the given stimuli and EEG data.

        This function constructs the design matrix from the stimuli, applies the relevant indexes,
        and separates the data into training and testing sets. It then standardizes and normalizes
        the data, and fits a Ridge regression model to the training data. If validation is enabled,
        it further splits the training data into training and validation sets and fits the model
        accordingly.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the input data shapes are not compatible with the model.
        """
        
        # Construct design matrix and transform for GPU computation
        X_train, X_pred = shifted_matrix(
            indices_to_keep=self.relevant_indexes,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True,
            delays=config.delays, 
            use_gpu=self.use_gpu,
            output_torch=True,
            features=stims
        )
        del stims
        
        if self.relevant_indexes is None:
            self.relevant_indexes = np.arange(X_train.shape[0]+X_pred.shape[0])
        
        # Remove rows with all zeros
        # from IPython import embed; embed()
        mask = ~(torch.all(X_train == 0, dim=1))
        X_train = X_train[mask]
        
        n_featuresbyn_delays = X_train.shape[1]
        n_features = n_featuresbyn_delays // len(config.delays)

        # Get relevant indexes and transform to device, if available. If not, transform to CPU
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_train = y_train[mask]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred =  X_pred.cpu()
            
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_train = y_train[mask.cpu()]
            y_test = y_temp[self.test_indexes]
        del y_temp
        
        if self.validation:
            # Delete held out
            del X_pred, y_test
            
            # Make split for validation: validation sets, fixing the train percent of data
            train_percent = .8
            self.train_cutoff = int(train_percent * len(self.train_indexes))
            try:
                X_train_for_val = X_train[:self.train_cutoff]
                X_val = X_train[self.train_cutoff:]
                del X_train
                y_train_for_val = y_train[:self.train_cutoff]
                y_val = y_train[self.train_cutoff:]
                del y_train
            except:
                X_train = X_train.cpu()
                y_train = y_train.cpu()
                
                X_train_for_val = X_train[:self.train_cutoff]
                X_val = X_train[self.train_cutoff:]
                del X_train
                y_train_for_val = y_train[:self.train_cutoff]
                y_val = y_train[self.train_cutoff:]
                del y_train
                        
            # Standarize and normalize 
            X_train_for_val, y_train_for_val, X_pred, y_val = self._standarize_normalize(
                X_train=X_train_for_val, 
                y_train=y_train_for_val, 
                X_pred=X_val, 
                y_test=y_val
            )
            correlations = torch.zeros(
                len(self.alpha), 
                device=self.device, 
                dtype=torch.float32
            )
            root_mean_square_error = torch.zeros(
                len(self.alpha), 
                device=self.device, 
                dtype=torch.float32
            )
            correlations_train = torch.zeros(
                len(self.alpha), 
                device=self.device, 
                dtype=torch.float32
            )
            root_mean_square_error_train = torch.zeros(
                len(self.alpha), 
                device=self.device, 
                dtype=torch.float32
            )
            trfs = torch.zeros(
                len(self.alpha), 
                len(config.delays), 
                device=self.device, 
                dtype=torch.float32
            )
            
            for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), desc='Sweeping progress', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):

                # Fit the model
                mtrfs = self._solver(
                    solver=self.solver,
                    alpha=alph, 
                    X_train=X_train_for_val, 
                    y_train=y_train_for_val
                )
                y_predicted = X_pred @ mtrfs
                y_predicted_train = X_train_for_val @ mtrfs
                
                # Compute correlation
                try:
                    correlations[i_alpha] = self._compute_correlation(
                        y_1=y_predicted,
                        y_2=y_val
                    ).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0
                try:
                    correlations_train[i_alpha] = self._compute_correlation(
                        y_1=y_predicted_train,
                        y_2=y_train_for_val
                    ).mean()
                except RuntimeWarning:
                    correlations_train[i_alpha] = 0
                
                root_mean_square_error[i_alpha] = torch.sqrt(torch.pow(y_predicted - y_val, 2).mean(dim=0)).mean(dim=0)
                root_mean_square_error_train[i_alpha] = torch.sqrt(torch.pow(y_predicted_train - y_train_for_val, 2).mean(dim=0)).mean(dim=0)
                trfs[i_alpha] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1).mean(dim=0).mean(dim=0) # shape n_chans, feats, delays --> delays
            del X_train_for_val, y_train_for_val, y_predicted, y_predicted_train, y_val, X_pred, mtrfs

            # Let GPU free memory
            gc.collect()
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(.1)
            return trfs.detach().cpu().numpy(), correlations.detach().cpu().numpy(), root_mean_square_error.detach().cpu().numpy(), correlations_train.detach().cpu().numpy(), root_mean_square_error_train.detach().cpu().numpy()
        else:
            if self.shuffle:
                iterations = np.arange(config.random_permutations)
                number_of_indices = X_train.shape[0]
                
                coefs = torch.zeros(
                    size=(config.random_permutations, config.info_mne['nchan'], n_features, len(config.delays)), 
                    device=self.device, 
                    dtype=torch.float32
                    )
                correlations = torch.zeros(
                    size=(config.random_permutations, config.info_mne['nchan']), 
                    device=self.device, 
                    dtype=torch.float32
                    )
                root_mean_square_error = torch.zeros(
                    size=(config.random_permutations, config.info_mne['nchan']),
                    device=self.device,
                    dtype=torch.float32
                    )
                
                X_train, y_train, X_pred, y_test = self.standarize_normalize(
                    X_train=X_train, 
                    X_pred=X_pred, 
                    y_train=y_train, 
                    y_test=y_test
                )
                # Shuffle the data, by requierment of random permutations
                for s in tqdm(iterations, desc='Performing permutations', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                    X_train_p = X_train[torch.randperm(number_of_indices)] # TODO Shufflear y en vez de X
                   
                    # Fit the model
                    mtrfs = self._solver(
                        solver=self.solver,
                        alpha=self.alpha, 
                        X_train=X_train_p, 
                        y_train=y_train
                    )
                    del X_train_p
                    y_predicted = X_pred @ mtrfs
                    coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                    del mtrfs
                    
                    try:
                        correlations[s] = self._compute_correlation(
                            y_1=y_predicted,
                            y_2=y_test
                        )
                    except RuntimeWarning:
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
                    root_mean_square_error[s] = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
                    
                del X_train, y_train, X_pred, y_test, y_predicted
                # Let GPU free memory   
                gc.collect()
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    time.sleep(.1)
                return coefs.cpu().numpy(), correlations.cpu().numpy(), root_mean_square_error.cpu().numpy()
            else:
                # Standarize and normalize
                X_train, y_train, X_pred, y_test = self._standarize_normalize(
                    X_train=X_train, 
                    X_pred=X_pred, 
                    y_train=y_train, 
                    y_test=y_test
                )

                # Fit the model 
                mtrfs = self._solver(
                    solver=self.solver,
                    alpha=self.alpha, 
                    X_train=X_train, 
                    y_train=y_train
                )
                del X_train, y_train
                
                # Perform predictions
                y_predicted = X_pred @ mtrfs
                del X_pred
                if torch.all(y_predicted==0):
                    print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.')
                
                # Store mtrfs
                mtrfs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1) # shape n_chans, feats, delays

                # Calculates and saves correlation of each channel # TODO HACER SOLO DE 0  EN ADELANTE
                try:
                    correlation_matrix = self._compute_correlation(
                        y_1=y_predicted,
                        y_2=y_test
                    )
                except RuntimeWarning:
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

                # Calculates and saves root mean square error of each channel
                root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
                # Let GPU free memory   
                gc.collect()
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    time.sleep(.1)
                return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()
    
    def _solver(
        self, 
        X_train:torch.Tensor, 
        y_train:torch.Tensor,
        alpha:Union[float, int]=1.0, 
        solver:str='ridge'
        )->torch.Tensor:
        """
        Solve the regression problem using the specified solver.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data, shape (n_samples, n_features).
        y_train : torch.Tensor
            Target values, shape (n_samples, n_channels).
        alpha : float or np.ndarray, optional
            Regularization strength, by default 1.0.
        solver : str, optional
            The solver to use for solving the regression problem, by default 'ridge'.

        Returns
        -------
        torch.Tensor
            The coefficients of the regression model.
        """
        if solver == 'ridge':
            XTX_reg = X_train.T @ X_train + torch.tensor(alpha, dtype=torch.float32) *  torch.eye(X_train.shape[1], device=self.device) # X^T * X + alpha*laplacian_matrix
            return torch.linalg.solve(XTX_reg, X_train.T @ y_train)

        elif solver == 'ridge-laplacian': # (X^T X + alpha * M) * mtrfs = X^T * y_train_p 
            n_features = X_train.shape[1]
            laplacian_matrix = torch.diag(torch.full((n_features,), 2.0, device=self.device, dtype=torch.float32))
            laplacian_matrix += torch.diag(torch.full((n_features-1,), -1.0, device=self.device, dtype=torch.float32), diagonal=1)
            laplacian_matrix += torch.diag(torch.full((n_features-1,), -1.0, device=self.device, dtype=torch.float32), diagonal=-1)
            laplacian_matrix[n_features-1, n_features-1] = 1.0
            laplacian_matrix[0, 0] = 1.0
            XTX_reg = X_train.T @ X_train + torch.tensor(alpha, dtype=torch.float32) *  laplacian_matrix # X^T * X + alpha*laplacian_matrix
            return torch.linalg.solve(XTX_reg, X_train.T @ y_train)
        
    def _compute_correlation(
        self,
        y_1:torch.Tensor, 
        y_2:torch.Tensor
        ):
        """
        Compute mean correlation between predicted and true values using centered Pearson correlation.

        Parameters
        ----------
        y_1 : torch.Tensor
            Predicted values, shape (n_samples, n_channels)
        y_2 : torch.Tensor
            True values, shape (n_samples, n_channels)

        Returns
        -------
        float
            Mean correlation across channels, or 0 if std is zero.
        """
        y_1_centered = y_1 - y_1.mean(dim=0, keepdim=True)
        y_2_centered = y_2 - y_2.mean(dim=0, keepdim=True)
        y_1_std = y_1_centered.std(dim=0, unbiased=True)
        y_2_std = y_2_centered.std(dim=0, unbiased=True)
        covariance = (y_1_centered*y_2_centered).mean(dim=0)

        if torch.all(y_2_std == 0) or torch.all(y_2_std == 0):
            print("\n Error: null standard deviation")
            return 0
        else:
            return (covariance / (y_1_std * y_2_std))
        
    def _standarize_normalize(
        self, 
        X_train:np.ndarray, 
        X_pred:np.ndarray, 
        y_train:np.ndarray=None, 
        y_test:np.ndarray=None
        ):
        """
        Standarize|Normalize training and test data.
        Parameters
        ----------
        X_train : np.ndarray
            Fatures to be normalized. Its dimensions should be samples x features 
        y_train : np.ndarray
            EEG samples to be normalized. Its dimensions should be samples x features

        Returns
        -------
        tuple
            A tuple containing the standardized/normalized training and test data: (X_train, y_train, X_pred, y_test).
        """
        # Instances of normalize and standarize
        normalization = Normalize(
            axis=0, 
            porcent=5, 
            by_gpu=self.use_gpu
        )
        standarization = Standarize(
            axis=0,
            by_gpu=self.use_gpu
        )

        # Iterates to normalize|standarize over features
        if self.stims_preprocess=='Standarize':
            X_train = standarization.fit_standarize_train(train_data=X_train) 
            X_pred = standarization.fit_standarize_test(test_data=X_pred)
            # for feat in range(X_train.shape[1]):
            #     X_train[:, feat] = standarization.fit_standarize_train(train_data=X_train[:, feat]) 
            #     X_pred[:, feat] = standarization.fit_standarize_test(test_data=X_pred[:, feat])
        if self.stims_preprocess=='Normalize':
            X_train = normalization.fit_normalize_train(train_data=X_train) 
            X_pred = normalization.fit_normalize_test(test_data=X_pred)
        if y_train is None or y_test is None:                
            return X_train, X_pred
        else:
            if self.eeg_preprocess=='Standarize':
                y_train=standarization.fit_standarize_train(train_data=y_train)
                y_test=standarization.fit_standarize_test(test_data=y_test)
            if self.eeg_preprocess=='Normalize':
                y_train=normalization.fit_normalize_percent(data=y_train)
                y_test=normalization.fit_normalize_test(test_data=y_test)
            return X_train, y_train, X_pred, y_test

class ReceptiveFieldAdaptation:
    def __init__(
        self, 
        tmin:float, 
        tmax:float, 
        sample_rate:int, 
        alpha:float,
        relevant_indexes:np.ndarray, 
        train_indexes:np.ndarray, 
        test_indexes:np.ndarray, 
        stims_preprocess:str, 
        eeg_preprocess:str, 
        estimator:str='time_delaying_ridge', 
        n_jobs:int=-1, 
        fit_intercept:bool=False, 
        shuffle:bool=False, 
        validation:bool=False
        ):
        """
        Initialize the ReceptiveFieldAdaptation model.

        Parameters
        ----------
        tmin : float
            The minimum time lag.
        tmax : float
            The maximum time lag.
        sample_rate : int
            The sampling frequency.
        alpha : float
            Regularization strength.
        relevant_indexes : np.ndarray
            Array of relevant indexes.
        train_indexes : np.ndarray
            Array of training indexes.
        test_indexes : np.ndarray
            Array of testing indexes.
        stims_preprocess : str
            Preprocessing method for stimuli.
        eeg_preprocess : str
            Preprocessing method for EEG data.
        estimator : str, optional
            The type of estimator to use, by default 'time_delaying_ridge'.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default -1.
        fit_intercept : bool, optional
            Whether to fit the intercept, by default False.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        validation : bool, optional
            Whether to perform validation, by default False.

        Returns
        -------
        None
        
        Raises
        ------
        SyntaxError
            If the estimator is not one of the allowed models.
        """
        allowed_models = ['ridge', 'time_delaying_ridge']
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.sample_rate = sample_rate
        if estimator not in allowed_models:
            raise SyntaxError(f"{estimator} is not an allowed situation. Allowed ones are: {allowed_models}")
        else:
            self.estimator = estimator

        if estimator =='time_delaying_ridge':
            self.rf = ReceptiveField(
                tmin=tmin,
                tmax=tmax, 
                sfreq=sample_rate,
                estimator=TimeDelayingRidgeRegression(
                    tmin=tmin, 
                    tmax=tmax, 
                    sfreq=sample_rate,
                    alpha=alpha,
                    relevant_indexes=relevant_indexes,
                    train_indexes=train_indexes,
                    test_indexes=test_indexes,
                    stims_preprocess=stims_preprocess, 
                    eeg_preprocess=eeg_preprocess,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                    shuffle=shuffle,
                    validation=validation
                    ),
                scoring='corrcoef'
                )
        else:
            self.rf = ReceptiveField(
                tmin=tmin, 
                tmax=tmax, 
                sfreq=sample_rate,
                estimator=RidgeRegression(
                    alpha=alpha,
                    relevant_indexes=relevant_indexes,
                    train_indexes=train_indexes,
                    test_indexes=test_indexes,
                    stims_preprocess=stims_preprocess, 
                    eeg_preprocess=eeg_preprocess,
                    fit_intercept=fit_intercept,
                    n_jobs=n_jobs,
                    shuffle=shuffle,
                    validation=validation
                    ),
                scoring='corrcoef'
                )
   
    def fit(
        self, 
        stims,
        eeg
        ):
        """
        Fit the ReceptiveField model to the given stimuli and EEG data.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features*n_delays). Mne should create the design matrix before performing this fit.
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the input data shapes are not compatible with the model.
        """
        self.rf.fit(stims, eeg)
        self.coefs = self.rf.coef_ # n_chanels, n_feats, n_delays

    def predict(
        self, 
        stims
        ):
        """
        Predict the EEG response for the given stimuli data.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).

        Returns
        -------
        tuple
            A tuple containing the predicted response and the test data.

        Raises
        ------
        ValueError
            If the input data shapes are not compatible with the model.
        """
        predicted = self.rf.predict(stims)
        if self.rf.estimator_.validation:
            test = self.rf.estimator_.y_val
            if self.estimator=='ridge':
                return predicted[self.train_indexes[self.rf.estimator_.train_cutoff:]], test
            else:
                return predicted[self.train_indexes[self.rf.estimator_.train_cutoff:]], test.reshape((test.shape[0], test.shape[2]))
        else:
            test = self.rf.estimator_.y_test
            if self.estimator=='ridge':
                return predicted[self.test_indexes], test
            else:
                return predicted[self.test_indexes], test.reshape((test.shape[0], test.shape[2]))

class RidgeRegression(Ridge):
    def __init__(
        self, 
        relevant_indexes:np.ndarray=None, 
        train_indexes:np.ndarray=None, 
        test_indexes:np.ndarray=None, 
        stims_preprocess:str='Normalize', 
        eeg_preprocess:str='Standarize', 
        alpha=1.0, 
        fit_intercept:bool=False, 
        shuffle:bool=False, 
        validation:bool=False, 
        n_jobs:int=-1
        ):
        """
        Initialize the RidgeRegression model.

        Parameters
        ----------
        relevant_indexes : np.ndarray, optional
            Array of relevant indexes.
        train_indexes : np.ndarray, optional
            Array of training indexes.
        test_indexes : np.ndarray, optional
            Array of testing indexes.
        stims_preprocess : str, optional
            Preprocessing method for stimuli, by default 'Normalize'.
        eeg_preprocess : str, optional
            Preprocessing method for EEG data, by default 'Standarize'.
        alpha : float, optional
            Regularization strength, by default 1.0.
        fit_intercept : bool, optional
            Whether to fit the intercept, by default False.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.

        Returns
        -------
        None
        """
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, solver='auto')
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess
        self.shuffle = shuffle
        self.validation = validation
        self.n_jobs = n_jobs

    def fit(
        self, 
        X, 
        y
        ):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features, n_delays).
        y : np.ndarray
            Target values, shape (n_samples, n_channels).

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If the input arrays have inconsistent numbers of samples.
        """        
        # Get relevant indexes
        X_r, y_r= X[self.relevant_indexes], y[self.relevant_indexes] 
        del X, y
                            
        if self.validation:
            # Make split
            X_train_val = X_r[self.train_indexes] 
            y_train_val = y_r[self.train_indexes]
            X_pred = X_r[self.test_indexes]
            y_test = y_r[self.test_indexes]
            del X_r, y_r
            
            # Separate training and validation sets, fixing the train percent of data
            train_percent = .8
            self.train_cutoff = int(train_percent * len(self.train_indexes))
            X_train = X_train_val[:self.train_cutoff]
            y_train = y_train_val[:self.train_cutoff]
            X_val = X_train_val[self.train_cutoff:]
            y_val = y_train_val[self.train_cutoff:]
            
            # When making random permutations, rearange delaying windows by shuffling
            if self.shuffle:
                np.random.shuffle(X_train)
                np.random.shuffle(X_pred)
                np.random.shuffle(y_train)
                np.random.shuffle(y_test)
            
            # Standarize and normalize
            X_train, y_train, self.X_pred, self.y_val = self.standarize_normalize(
                                                    X_train=X_train, 
                                                    X_pred=X_val, 
                                                    y_train=y_train, 
                                                    y_test=y_val
                                                    )
            return super().fit(X_train, y_train)
        else:
            # Make split
            X_train = X_r[self.train_indexes] 
            X_pred = X_r[self.test_indexes]
            y_train = y_r[self.train_indexes]
            y_test = y_r[self.test_indexes]
            del X_r, y_r
            
            # When making random permutations, rearange delaying windows by shuffling
            if self.shuffle:
                np.random.shuffle(X_train)
                np.random.shuffle(X_pred)
                np.random.shuffle(y_train)
                np.random.shuffle(y_test)
            
            # Standarize and normalize
            X_train, y_train, self.X_pred, self.y_test = self.standarize_normalize(
                                                    X_train=X_train, 
                                                    X_pred=X_pred, 
                                                    y_train=y_train, 
                                                    y_test=y_test
                                                    )
            return super().fit(X_train, y_train)

    def predict(
        self, 
        X
        ):
        """
        Predict the response for the given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features*delays).

        Returns
        -------
        np.ndarray
            Predicted response, shape (n_samples, n_channels).

        Raises
        ------
        ValueError
            If the input data shape is not compatible with the model.
        """
        n_samples = X.shape[0]
        y_restricted_prediction = super().predict(self.X_pred) # n_samples, n_channels
        
        # Padd with zeros to make it compatible with desired shape of mne.ReceptiveField.predict()
        y_pred_full = np.zeros(shape=(n_samples, y_restricted_prediction.shape[-1]))
        if self.validation:
            y_pred_full[self.train_indexes[self.train_cutoff:], :] = y_restricted_prediction # Notice that the filter is train_cutoff: because the following indexes are the one used for prediction
        else:
            y_pred_full[self.test_indexes] = y_restricted_prediction

        # When used relevant indexes must be filtered once again
        return y_pred_full
    
    def standarize_normalize(
        self,
        X_train:np.ndarray,
        X_pred:np.ndarray, 
        y_train:np.ndarray, 
        y_test:np.ndarray
        ):
        """Standarize|Normalize training and test data.
        Parameters
        ----------
        X_train : np.ndarray
            Fatures to be normalized. Its dimensions should be samples x features 
        y_train : np.ndarray
            EEG samples to be normalized. Its dimensions should be samples x features

        Returns
        -------
        tuple
            A tuple containing the standardized/normalized training and test data: (X_train, y_train, X_pred, y_test).
        """
        # Instances of normalize and standarize
        norm = Normalize(axis=0, porcent=5)
        estandar = Standarize(axis=0)
        
        # Iterates to normalize|standarize over features
        if self.stims_preprocess=='Standarize':
            for feat in range(X_train.shape[1]):
                X_train[:, feat] = estandar.fit_standarize_train(train_data=X_train[:, feat]) 
                X_pred[:, feat] = estandar.fit_standarize_test(test_data=X_pred[:, feat])
        if self.stims_preprocess=='Normalize':
            for feat in range(X_train.shape[1]):
                X_train[:, feat] = norm.fit_normalize_train(train_data=X_train[:, feat]) 
                X_pred[:, feat] = norm.fit_normalize_test(test_data=X_pred[:, feat])
        if self.eeg_preprocess=='Standarize':
            y_train=estandar.fit_standarize_train(train_data=y_train)
            y_test=estandar.fit_standarize_test(test_data=y_test)
        if self.eeg_preprocess=='Normalize':
            y_train=norm.fit_normalize_percent(data=y_train)
            y_test=norm.fit_normalize_test(test_data=y_test)
        return X_train, y_train, X_pred, y_test

#TODO: obsolete due to incorrect filtering implementation
class TimeDelayingRidgeRegression(TimeDelayingRidge):
    def __init__(
        self, 
        tmin:float,
        tmax:float, 
        sfreq:int, 
        relevant_indexes:np.ndarray=None, 
        train_indexes:np.ndarray=None, 
        test_indexes:np.ndarray=None, 
        stims_preprocess:str='Normalize',
        eeg_preprocess:str='Standarize', 
        alpha=1.0, 
        fit_intercept=False, 
        n_jobs:int=1,
        shuffle:bool=False,
        validation:bool=False
        ):
        """
        Initialize the TimeDelayingRidgeRegression model.

        Parameters
        ----------
        tmin : float
            The minimum time lag.
        tmax : float
            The maximum time lag.
        sfreq : int
            The sampling frequency.
        relevant_indexes : np.ndarray, optional
            Array of relevant indexes.
        train_indexes : np.ndarray, optional
            Array of training indexes.
        test_indexes : np.ndarray, optional
            Array of testing indexes.
        stims_preprocess : str, optional
            Preprocessing method for stimuli, by default 'Normalize'.
        eeg_preprocess : str, optional
            Preprocessing method for EEG data, by default 'Standarize'.
        alpha : float, optional
            Regularization strength, by default 1.0.
        fit_intercept : bool, optional
            Whether to fit the intercept, by default False.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        validation : bool, optional
            Whether to perform validation, by default False.

        Returns
        -------
        None
        """
        super().__init__(tmin=tmin, tmax=tmax, sfreq=sfreq, alpha=alpha, fit_intercept=fit_intercept, n_jobs=n_jobs)
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess
        self.shuffle = shuffle
        self.validation = validation

    def fit(
        self, 
        X, 
        y
        ):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        y : np.ndarray
            Target values, shape (n_samples, n_channels).

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If the input arrays have inconsistent numbers of samples.
        """
        # Get relevant indexes
        X_r, y_r = X[self.relevant_indexes], y[self.relevant_indexes] # relevant_samples relevant_samples, features*delays, antes relevant_samples, [epochs,features], delays
        del X, y

        if self.validation:
            # Make split
            X_train_val = X_r[self.train_indexes] 
            y_train_val = y_r[self.train_indexes]
            X_pred = X_r[self.test_indexes]
            y_test = y_r[self.test_indexes]
            del X_r, y_r
            
            # Separate training and validation sets, fixing the train percent of data
            train_percent = .8
            self.train_cutoff = int(train_percent * len(self.train_indexes))
            X_train = X_train_val[:self.train_cutoff]
            y_train = y_train_val[:self.train_cutoff]
            X_val = X_train_val[self.train_cutoff:]
            y_val = y_train_val[self.train_cutoff:]
            
            # Standarize and normalize
            X_train, y_train, self.X_pred, self.y_val = self.standarize_normalize(
                                                    X_train=X_train, 
                                                    X_pred=X_val, 
                                                    y_train=y_train, 
                                                    y_test=y_val
                                                    )
            return super().fit(X_train, y_train)
        else:
            # Make split
            X_train = X_r[self.train_indexes] 
            X_pred = X_r[self.test_indexes]
            y_train = y_r[self.train_indexes]
            y_test = y_r[self.test_indexes]
            del X_r, y_r
            
            # When making random permutations, rearange delaying windows by shuffling
            if self.shuffle:
                np.random.shuffle(X_train)
                np.random.shuffle(X_pred)
                np.random.shuffle(y_train)
                np.random.shuffle(y_test)
            
            # Standarize and normalize
            X_train, y_train, self.X_pred, self.y_test = self.standarize_normalize(
                                                    X_train=X_train, 
                                                    X_pred=X_pred, 
                                                    y_train=y_train, 
                                                    y_test=y_test
                                                    )
            return super().fit(X_train, y_train)
    
    def predict(
        self,
        X
        ):
        """
        Predict the response for the given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted response, shape (n_samples, n_channels).

        Raises
        ------
        ValueError
            If the input data shape is not compatible with the model.
        """
        n_samples = X.shape[0]
        y_restricted_prediction = super().predict(self.X_pred) # n_samples, n_channels
        
        # Padd with zeros to make it compatible with desired shape of mne.ReceptiveField.predict()
        y_pred_full = np.zeros(shape=(n_samples, 1, y_restricted_prediction.shape[-1]))
        if self.validation:
            y_pred_full[self.train_indexes[self.train_cutoff:], :, :] = y_restricted_prediction # Notice that the filter is train_cutoff: because the following indexes are the one used for prediction
        else:
            y_pred_full[self.test_indexes, :, :] = y_restricted_prediction
        # When used relevant indexes must be filtered once again
        return y_pred_full
    
    def standarize_normalize(
        self, 
        X_train:np.ndarray,
        X_pred:np.ndarray,
        y_train:np.ndarray, 
        y_test:np.ndarray
        ):
        """Standarize|Normalize training and test data.
        Parameters
        ----------
        X_train : np.ndarray
            Fatures to be normalized. Its dimensions should be samples x features 
        y_train : np.ndarray
            EEG samples to be normalized. Its dimensions should be samples x features

        Returns
        -------
        tuple
            A tuple containing the standardized/normalized training and test data: (X_train, y_train, X_pred, y_test).
        """
        # Instances of normalize and standarize
        norm = Normalize(axis=0, porcent=5)
        estandar = Standarize(axis=0)
        
        # Iterates to normalize|standarize over features
        if self.stims_preprocess=='Standarize':
            for feat in range(X_train.shape[1]):
                X_train[:, feat] = estandar.fit_standarize_train(train_data=X_train[:, feat]) 
                X_pred[:, feat] = estandar.fit_standarize_test(test_data=X_pred[:, feat])
        if self.stims_preprocess=='Normalize':
            for feat in range(X_train.shape[1]):
                X_train[:, feat] = norm.fit_normalize_train(train_data=X_train[:, feat]) 
                X_pred[:, feat] = norm.fit_normalize_test(test_data=X_pred[:, feat])
        if self.eeg_preprocess=='Standarize':
            y_train=estandar.fit_standarize_train(train_data=y_train)
            y_test=estandar.fit_standarize_test(test_data=y_test)
        if self.eeg_preprocess=='Normalize':
            y_train=norm.fit_normalize_percent(data=y_train)
            y_test=norm.fit_normalize_test(test_data=y_test)
        return X_train, y_train, X_pred, y_test

# from scipy.fft import fft, ifft
    # def fit2(
    #     self, 
    #     stims:np.ndarray, 
    #     eeg:np.ndarray
    #     )->None:
    #     """
    #     Fit the TorchMtrf model to the given stimuli and EEG data.

    #     This method constructs the design matrix from the stimuli, applies the relevant indexes,
    #     and separates the data into training and testing sets. It then standardizes and normalizes
    #     the data, and fits a Ridge regression model to the training data. If validation is enabled,
    #     it further splits the training data into training and validation sets and fits the model
    #     accordingly.

    #     Parameters
    #     ----------
    #     stims : np.ndarray
    #         The input stimuli data, shape (n_samples, n_features).
    #     eeg : np.ndarray
    #         The EEG response data, shape (n_samples, n_channels).

    #     Returns
    #     -------
    #     None

    #     Raises
    #     ------
    #     ValueError
    #         If the input data shapes are not compatible with the model.
    #     """
    #     samples_right = config.delays[-1]
    #     samples_left = -config.delays[0]
    #     window = samples_right + samples_left
        
    #     self.relevant_indexes = torch.tensor(self.relevant_indexes).to(torch.long).to(self.device)
    #     stims = torch.tensor(stims).to(torch.long).to(self.device)
    #     eeg  = torch.tensor(eeg).to(torch.long).to(self.device)
        
    #     # deltas = np.diff(self.relevant_indexes)
    #     deltas = self.relevant_indexes[1:] - self.relevant_indexes[:-1]
        
    #     # Get gaps greater than one step
    #     filter_deltas_considerable = (deltas>1).to(device=self.device)
    #     deltas_indices = torch.where(filter_deltas_considerable)[0]
    #     deltas_values = deltas[deltas_indices]
    #     to_add = []

    #     # Stick gaps smaller than TRF's window, else add left and right edges
    #     for i, gap in zip(deltas_indices, deltas_values):
    #         left_edge = self.relevant_indexes[i].item()
    #         right_edge = self.relevant_indexes[i + 1].item()
            
    #         # Stick the edges together
    #         if gap < window:
    #             to_add.append(torch.arange(left_edge + 1, right_edge, device=self.device))
            
    #         # Add left and right edges
    #         else:
    #             to_add.append(torch.arange(left_edge + 1, left_edge + samples_right + 1, device=self.device))
    #             to_add.append(torch.arange(right_edge - samples_left, right_edge, device=self.device))

    #     # concatenar y unificar
    #     self.relevant_indexes = torch.unique(torch.cat([self.relevant_indexes] + to_add)) #torch.tensor(to_add, device=self.device)
        
    #     X = stims[self.relevant_indexes]
    #     Y = eeg[self.relevant_indexes]
    #     del stims, eeg
        
    #     number_of_samples, number_of_feature_dimensions = X.shape
    #     trf_support = len(config.delays)

    #     # FFT a lo largo del tiempo (eje 0)
    #     X_f = torch.fft.fft(X, n=number_of_samples, dim=0)  # shape (number_of_samples, D)
    #     del X
    #     Y_f = torch.fft.fft(Y, n=number_of_samples, dim=0)  # shape (number_of_samples, C)
    #     del Y
    #     # Estimación de H_f (shape: number_of_samples x D x C) freqsxdimensionsxchanns
    #     # numerator = X_f[:, :, None].conj() * Y_f[:, None, :]        # (number_of_samples, D, C)
    #     # denominator = (np.abs(X_f)**2).sum(axis=1)[:, None] + self.alpha*number_of_samples  # (number_of_samples, 1) # TODO chequear la regularización
    #     # H_f = numerator / denominator[:, None, :]                   # (number_of_samples, D, C)
    #     numerator = X_f.unsqueeze(-1).conj() * Y_f.unsqueeze(1)
    #     # numerator = X_f.unsqueeze(-1) * Y_f.unsqueeze(1)
    #     denominator = (X_f.conj).sum(dim=1, keepdim=True) + self.alpha
    #     # del X_f, Y_f
    #     # H_f = numerator / denominator.unsqueeze(-1)
    #     # del numerator, denominator
        
    #     X_f = torch.fft.rfft(X, n=number_of_samples, dim=0)  # shape (number_of_samples, D)
    #     del X
    #     Y_f = torch.fft.rfft(Y, n=number_of_samples, dim=0)  # shape (number_of_samples, C)
    #     del Y
    #     # Estimación de H_f (shape: number_of_samples x D x C) freqsxdimensionsxchanns
    #     # numerator = X_f[:, :, None].conj() * Y_f[:, None, :]        # (number_of_samples, D, C)
    #     # denominator = (np.abs(X_f)**2).sum(axis=1)[:, None] + self.alpha*number_of_samples  # (number_of_samples, 1) # TODO chequear la regularización
    #     # H_f = numerator / denominator[:, None, :]                   # (number_of_samples, D, C)
    #     # numerator = X_f.unsqueeze(-1).conj() * Y_f.unsqueeze(1)
    #     numerator = X_f.T @ Y_f
    #     denominator = (X_f.T@X_f) + torch.tensor(self.alpha, dtype=torch.float32) *  torch.eye(X_f.shape[1], device=self.device)
    #     del X_f, Y_f
    #     H_f = torch.linalg.solve(denominator,numerator)
        
    #     del numerator, denominator

    #     # IFFT para recuperar TRF en el tiempo
    #     # h_full = np.fft.ifft(H_f, axis=0).real  # (number_of_samples, D, C)
    #     h_full = torch.fft.irfft(H_f, n=number_of_samples, dim=0).real

    #     # Alineación temporal: centramos la TRF en number_of_samples=0
    #     # h_full = np.roll(h_full, -number_of_samples // 2, axis=0)  # shift temporal
    #     # mtrfs = h_full[-number_of_samples // 2+config.delays[0]: -number_of_samples // 2+ config.delays[-1], :, :]  # (L, D, C)
    #     indices = (config.delays % number_of_samples)  
    #     mtrfs = h_full[indices, :]
        
    #     import matplotlib.pyplot as plt
    #     import mne
        
    #     plt.figure()
    #     plt.plot(mtrfs.detach().cpu().numpy().mean(axis=(1,2)))
    #     plt.show()
        # def fit(
    #     self, 
    #     stims:np.ndarray, 
    #     eeg:np.ndarray
    #     )->None:
    #     """
    #     Fit the TorchMtrf model to the given stimuli and EEG data.

    #     This method constructs the design matrix from the stimuli, applies the relevant indexes,
    #     and separates the data into training and testing sets. It then standardizes and normalizes
    #     the data, and fits a Ridge regression model to the training data. If validation is enabled,
    #     it further splits the training data into training and validation sets and fits the model
    #     accordingly.

    #     Parameters
    #     ----------
    #     stims : np.ndarray
    #         The input stimuli data, shape (n_samples, n_features).
    #     eeg : np.ndarray
    #         The EEG response data, shape (n_samples, n_channels).

    #     Returns
    #     -------
    #     None

    #     Raises
    #     ------
    #     ValueError
    #         If the input data shapes are not compatible with the model.
    #     """
    #     stims = torch.tensor(stims[self.relevant_indexes]).to(torch.float32).to(self.device)
    #     X_train = stims[self.train_indexes]
    #     X_pred = stims[self.test_indexes]
    #     del stims
    #     n_features = X_train.shape[1]

    #     # Get relevant indexes and transform to device, if available. If not, transform to CPU
    #     try:
    #         y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
    #         del eeg
    #         y_train = y_temp[self.train_indexes]
    #         y_test = y_temp[self.test_indexes]
    #     except:
    #         X_train = X_train.cpu()
    #         X_pred =  X_pred.cpu()
            
    #         y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
    #         del eeg            
    #         y_train = y_temp[self.train_indexes]
    #         y_test = y_temp[self.test_indexes]
    #     del y_temp
        
    #     if self.validation:
    #         del X_pred, y_test
            
    #         # Make split for validation: validation sets, fixing the train percent of data
    #         train_percent = .8
    #         self.train_cutoff = int(train_percent * len(self.train_indexes))
    #         try:
    #             X_train_for_val = X_train[:self.train_cutoff]
    #             X_val = X_train[self.train_cutoff:]
    #             del X_train
    #             y_train_for_val = y_train[:self.train_cutoff]
    #             y_val = y_train[self.train_cutoff:]
    #             del y_train
    #         except:
    #             X_train = X_train.cpu()
    #             y_train = y_train.cpu()
                
    #             X_train_for_val = X_train[:self.train_cutoff]
    #             X_val = X_train[self.train_cutoff:]
    #             del X_train
    #             y_train_for_val = y_train[:self.train_cutoff]
    #             y_val = y_train[self.train_cutoff:]
    #             del y_train
                        
    #         # Standarize and normalize 
    #         X_train_for_val, y_train_for_val, X_pred, y_val = self._standarize_normalize(
    #             X_train=X_train_for_val, 
    #             y_train=y_train_for_val, 
    #             X_pred=X_val, 
    #             y_test=y_val
    #         )
    #         correlations = torch.zeros(
    #             len(self.alpha), 
    #             device=self.device, 
    #             dtype=torch.float32
    #         )
    #         root_mean_square_error = torch.zeros(
    #             len(self.alpha), 
    #             device=self.device, 
    #             dtype=torch.float32
    #         )
    #         correlations_train = torch.zeros(
    #             len(self.alpha), 
    #             device=self.device, 
    #             dtype=torch.float32
    #         )
    #         root_mean_square_error_train = torch.zeros(
    #             len(self.alpha), 
    #             device=self.device, 
    #             dtype=torch.float32
    #         )
    #         trfs = torch.zeros(
    #             len(self.alpha), 
    #             len(config.delays), 
    #             device=self.device, 
    #             dtype=torch.float32
    #         )
            
    #         for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), desc='Sweeping progress', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):

    #             # Fit the model
    #             mtrfs = self._solver(
    #                 method=self.method,
    #                 alpha=alph, 
    #                 X_train=X_train_for_val, 
    #                 y_train=y_train_for_val
    #             )
    #             y_predicted = X_pred @ mtrfs
    #             y_predicted_train = X_train_for_val @ mtrfs
                
    #             # Compute correlation
    #             try:
    #                 correlations[i_alpha] = self._compute_correlation(
    #                     y_1=y_predicted,
    #                     y_2=y_val
    #                 )
    #             except RuntimeWarning:
    #                 correlations[i_alpha] = 0
    #             try:
    #                 correlations_train[i_alpha] = self._compute_correlation(
    #                     y_1=y_predicted_train,
    #                     y_2=y_train_for_val
    #                 )
    #             except RuntimeWarning:
    #                 correlations_train[i_alpha] = 0
                
    #             root_mean_square_error[i_alpha] = torch.sqrt(torch.pow(y_predicted - y_val, 2).mean(dim=0)).mean(dim=0)
    #             root_mean_square_error_train[i_alpha] = torch.sqrt(torch.pow(y_predicted_train - y_train_for_val, 2).mean(dim=0)).mean(dim=0)
    #             trfs[i_alpha] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1).mean(dim=0).mean(dim=0) # shape n_chans, feats, delays
    #         del X_train_for_val, y_train_for_val, y_predicted, y_val, X_pred
    #         return trfs.detach().cpu().numpy(), correlations.detach().cpu().numpy(), root_mean_square_error.detach().cpu().numpy(), correlations_train.detach().cpu().numpy(), root_mean_square_error_train.detach().cpu().numpy()
    #     else:
    #         if self.shuffle:
    #             iterations = np.arange(config.random_permutations)
    #             number_of_indices = X_train.shape[0]
                
    #             coefs = torch.zeros(
    #                 size=(config.random_permutations, config.info_mne['nchan'], n_features, len(config.delays)), 
    #                 device=self.device, 
    #                 dtype=torch.float32
    #                 )
    #             correlations = torch.zeros(
    #                 size=(config.random_permutations, config.info_mne['nchan']), 
    #                 device=self.device, 
    #                 dtype=torch.float32
    #                 )
    #             root_mean_square_error = torch.zeros(
    #                 size=(config.random_permutations, config.info_mne['nchan']),
    #                 device=self.device,
    #                 dtype=torch.float32
    #                 )
                
    #             X_train, y_train, X_pred, y_test = self.standarize_normalize(
    #                 X_train=X_train, 
    #                 X_pred=X_pred, 
    #                 y_train=y_train, 
    #                 y_test=y_test
    #             )
    #             # Shuffle the data, by requierment of random permutations
    #             for s in tqdm(iterations, desc='Performing permutations', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
    #                 X_train_p = X_train[torch.randperm(number_of_indices)] # TODO Shufflear y en vez de X
                   
    #                 # Fit the model
    #                 mtrfs = self._solver(
    #                     method=self.method,
    #                     alpha=self.alpha, 
    #                     X_train=X_train_p, 
    #                     y_train=y_train
    #                 )
    #                 del X_train_p
    #                 y_predicted = X_pred @ mtrfs
    #                 coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
    #                 del mtrfs
                    
    #                 try:
    #                     correlations[s] = self._compute_correlation(
    #                         y_1=y_predicted,
    #                         y_2=y_test
    #                     )
    #                 except RuntimeWarning:
    #                     correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
    #                 root_mean_square_error[s] = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
                    
    #             del X_train, y_train, X_pred, y_test, y_predicted
    #             return coefs.cpu().numpy(), correlations.cpu().numpy(), root_mean_square_error.cpu().numpy()
    #         else:
    #             # Standarize and normalize
    #             X_train, y_train, X_pred, y_test = self._standarize_normalize(
    #                 X_train=X_train, 
    #                 X_pred=X_pred, 
    #                 y_train=y_train, 
    #                 y_test=y_test
    #             )
        
    #             # Fit the model 
    #             mtrfs = self._solver( # delays, F, C
    #                 method=self.method,
    #                 alpha=self.alpha, 
    #                 X_train=X_train, 
    #                 y_train=y_train
    #             )
    #             del X_train, y_train
                
    #             L = X_pred.shape[0]
                
    #             mtrfs_padded = torch.zeros(
    #                 size=(L, mtrfs.shape[1], mtrfs.shape[2]), 
    #                 device=self.device, 
    #                 dtype=torch.float32
    #             )
               
    #             mtrfs_padded[
    #                 L//2+config.delays[0]-1:L//2+config.delays[-1]
    #             ] = mtrfs
                
    #             y_predicted = fftconvolve(
    #                 X_pred.T.unsqueeze(1), 
    #                 mtrfs_padded.permute(1, 2, 0), 
    #                 mode='same' 
    #             ).sum(dim=0).T
    #             y_test_filtered = cheby2_bandpass_filter_torch(
    #                 y_test, 
    #                 lowcut=1,
    #                 highcut=15, 
    #                 fs=config.sr,
    #                 order=4,
    #                 rs=20,
    #                 device=self.device
    #             )
    #             # from IPython import embed; embed()
    #             # import matplotlib.pyplot as plt
    #             # import matplotlib
    #             # matplotlib.use('TkAgg')
    #             # plt.figure(figsize=(10, 5))
    #             # plt.plot(y_predicted[:1000].cpu().numpy().mean(1), label='Predicted')
    #             # plt.plot(y_test_filtered[:1000].cpu().numpy().mean(1), label='True')
    #             # plt.legend()
    #             # plt.title('Predicted vs True')
    #             # plt.xlabel('Time')
    #             # plt.ylabel('Amplitude')
    #             # plt.show()
    #             del X_pred, y_test
    #             if torch.all(y_predicted==0):
    #                 print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.')
                
    #             # Store mtrfs
    #             mtrfs = mtrfs.permute(2, 1, 0) # shape n_chans, feats, delays

    #             # Calculates and saves correlation of each channel # TODO HACER SOLO DE 0  EN ADELANTE
    #             try:
    #                 correlation_matrix = self._compute_correlation(
    #                     y_1=y_predicted,
    #                     y_2=y_test_filtered
    #                 )
    #             except RuntimeWarning:
    #                 correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

    #             # Calculates and saves root mean square error of each channel
    #             root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test_filtered, 2).mean(dim=0))
    #             return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()

            # dtype = torch.complex64
            # N, F = X_train.shape
            # C = y_train.shape[1]

            # # FFT 
            # X_f = torch.fft.rfft(X_train, dim=0).to(dtype)     # (N, F)
            # Y_f = torch.fft.rfft(y_train, dim=0).to(dtype)     # (N, C)

            # # Frecuencias asociadas
            # freqs = torch.fft.rfftfreq(N, d=1/config.sr).to(self.device)   # (N,)
            # # fmin, fmax = band_freq(band='Theta')
            # fmin, fmax = 1, 15

            # # Frecuencias deseadas
            # mask = (freqs >= fmin) & (freqs <= fmax)
            # K = mask.sum().item()
            # freqs_band = freqs[mask]  # (K,)

            # TRF_f_band = torch.zeros((N//2+1, F, C), dtype=dtype, device=self.device)  # (K, F, C)
            # # for f_i in mask.nonzero(as_tuple=True)[0]:
            # #     X_f_i = X_f[f_i].unsqueeze(0)      # (1, F)
            # #     Y_f_i = Y_f[f_i].unsqueeze(0)      # (1, C)

            # #     # 
            # #     A = torch.matmul(
            # #         X_f_i.conj().transpose(0, 1), # (F, 1)
            # #         X_f_i # (1, F)
            # #     )  # (F, F)
            # #     A += alpha * torch.eye(F, dtype=dtype, device=self.device)  # (F, F)

            # #     B = torch.matmul(
            # #         X_f_i.conj().transpose(0,1), # (F, 1)
            # #         Y_f_i # (1, C)
            # #     )   # (F, C)

            # #     # Resolver para cada frecuencia
            # #     TRF_f_band[f_i] = torch.linalg.solve(A, B)   # (F, C)
            # # dtype = torch.complex64
            # # N, F = X_train.shape
            # # C = y_train.shape[1]

            # # # FFT 
            # # X_f = torch.fft.rfft(X_train, dim=0).to(dtype)     # (N, F)
            # # Y_f = torch.fft.rfft(y_train, dim=0).to(dtype)     # (N, C)

            # # # Frecuencias asociadas
            # # freqs = torch.fft.rfftfreq(N, d=1/config.sr).to(self.device)   # (N,)
            # # fmin, fmax = 1, 15

            # # # Frecuencias deseadas
            # # mask = (freqs >= fmin) & (freqs <= fmax)
            # # K = mask.sum().item()
            # # freqs_band = freqs[mask]  # (K,)

            # # TRF_f_band = torch.zeros((N//2+1, F, C), dtype=dtype, device=self.device)  # (K, F, C)
            # # for f_i in tqdm(mask.nonzero(as_tuple=True)[0], total=K):
            # #     X_f_i = X_f[f_i].unsqueeze(0)      # (1, F)
            # #     A = torch.matmul(
            # #         X_f_i.conj().transpose(0, 1), # (F, 1)
            # #         X_f_i # (1, F)
            # #     )  # (F, F)
            # #     A += alpha * torch.eye(F, dtype=dtype, device=self.device)  # (F, F)

            # #     for c in range(C):
            # #         Y_f_ic = Y_f[f_i, c].unsqueeze(0).unsqueeze(1)  # (1, 1)
            # #         B = torch.matmul(
            # #             X_f_i.conj().transpose(0,1), # (F, 1)
            # #             Y_f_ic # (1, 1)
            # #         ).squeeze(-1)   # (F,)
            # #         # Resolver para cada canal y frecuencia
            # #         TRF_f_band[f_i, :, c] = torch.linalg.solve(A, B)   # (F,)
            # # if F!=1:
            # #     X_f_band = X_f[mask].transpose(0,1).unsqueeze(0)      # (1, F, K)
            # #     Y_f_band = Y_f[mask].transpose(0,1).unsqueeze(0)      # (1, C, K)
            # #     A = torch.matmul(
            # #         X_f_band.conj().transpose(0, 1).permute(2,0,1), # (F, 1, K) -> (K, F, 1)
            # #         X_f_band.permute(2,0,1) # (1, F, K) -> (K, 1, F)
            # #     )  # (K, F, F) 
            # #     A += alpha * torch.eye(F, dtype=dtype, device=self.device).unsqueeze(-1)  # (F, F, K)+(F, F, 1)=(F, F, K)

            # #     B = torch.matmul(
            # #         X_f_band.conj().transpose(0,1).permute(2,0,1), # (F, 1, K) -> (K, F, 1)
            # #         Y_f_band.permute(2,0,1) # (1, C, K) -> (K, 1, C)
            # #     )   # (K, F, C) 

            # #     # Resolver para cada frecuencia
            # #     TRF_f_band[mask] = torch.linalg.solve(A, B)   # (K, F, C)
            # # else:
            # #     X_f_band = X_f[mask] # (k, 1)
            # #     Y_f_band = Y_f[mask] # (k, 1)
            # #     # A: (K, 1, 1), B: (K, 1, C)
            # #     A = (X_f_band.conj() * X_f_band).sum(dim=1, keepdim=True).unsqueeze(-1)  # (K, 1, 1)
            # #     A += alpha * torch.eye(1, dtype=dtype, device=self.device).unsqueeze(0)  # (K, 1, 1)

            # #     # B: (K, 1, C)
            # #     B = torch.matmul(
            # #         X_f_band.conj().unsqueeze(2),  # (K, 1, 1)
            # #         Y_f_band.unsqueeze(1)          # (K, 1, C)
            # #     )  # (K, 1, C)

            # #     TRF_f_band[mask] = torch.linalg.solve(A, B)  # (K, 1, C)
            # X_f_band = X_f[mask]  # (K, F)
            # Y_f_band = Y_f[mask]  # (K, C)

            # # A: (K, F, F)
            # A = torch.matmul(
            #     X_f_band.unsqueeze(2).conj(),  # (K, F, 1)
            #     X_f_band.unsqueeze(1)          # (K, 1, F)
            # )  # (K, F, F)
            # A += alpha * torch.eye(F, dtype=dtype, device=self.device).unsqueeze(0)  # (K, F, F)

            # # B: (K, F, C)
            # B = torch.matmul(
            #     X_f_band.unsqueeze(2).conj(),  # (K, F, 1)
            #     Y_f_band.unsqueeze(1)          # (K, 1, C)
            # )  # (K, F, C)

            # # Solve for each frequency (batched)
            # TRF_f_band[mask] = torch.linalg.solve(A, B)  # (K, F, C)

            # # Ventana de Hann
            # # win = torch.hann_window(K, periodic=True, device=self.device)  # (K,)
            # # TRF_f_band[mask] *= win[:, None, None]

            #             # Transform back to time domain
            # TRF_t_band = torch.fft.irfft(TRF_f_band, axis=0).real   # (N, F, C)
            # # from IPython import embed; embed()

            # return TRF_t_band[config.delays%(N-1)] # delays, F, C
                        
            # import mne
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use('TkAgg')
            # # # TRF_f_band[mask] *= win[:, None, None]
            # # # Ventana de Hann
            # # # win = torch.hann_window(K, periodic=True, device=self.device)  # (K,)
            # # # TRF_f_band[mask] *= win[:, None, None]
            # # TRF_t_band = torch.fft.irfft(TRF_f_band, axis=0).real    # (K, F, C)
            # delays = np.arange(int(np.round(-.2 * config.sr)), int(np.round(.6 * config.sr) + 1))
            # mtrfs = TRF_t_band.cpu().numpy().mean(axis=1) # shape n_chans, feats, delays
            # # mtrfs = TRF_t_band.cpu().numpy().mean(axis=2) # shape n_chans, feats, delays
            # evoked = mne.EvokedArray(
            #     data=mtrfs[delays%(N-1)].T, 
            #     info=config.info_mne
            # )     
            # evoked.shift_time(
            #     delays[0]/config.sr
            # )




            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # evoked_plot = evoked.plot(
            #     scalings={'eeg':1},
            #     zorder='std',
            #     time_unit='ms',
            #     show=False,
            #     spatial_colors=True,
            #     # unit=False,
            #     gfp=True,
            #     units='mTRFs (U.A)',
            #     axes=ax
            # )
            # ax.grid()
            # fig.show()
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # mtrfs = TRF_t_band.cpu().numpy().mean(axis=2) # shape n_chans, feats, delays
            # weights = mtrfs[delays%(N-1), :] # n_feats, n_delays

            # # Perform clustering
            # order = None
            # null_indexes = None

            # # Make color mesh
            # number_of_ticks = weights.shape[0]
            # # im = ax.pcolormesh(
            # #     delays/config.sr * 1000, 
            # #     np.arange(number_of_ticks), 
            # #     weights, 
            # #     cmap='RdBu_r', 
            # #     shading='auto',
            # #     vmin=-np.abs(weights).max(),
            # #     vmax=np.abs(weights).max()
            # #     )
            # im = ax.pcolormesh(
            #     delays/config.sr * 1000, 
            #     np.arange(weights.shape[1]),  # O el eje correcto para tu segundo eje
            #     weights.T,                    # Transponer si quieres (n_delays, n_feats)
            #     cmap='RdBu_r', 
            #     shading='auto',
            #     vmin=-np.abs(weights).max(),
            #     vmax=np.abs(weights).max()
            # )
            # fig.colorbar(
            #     im,
            #     ax=ax, 
            #     orientation='horizontal', 
            #     shrink=1, 
            #     label='Amplitude (a.u.)', 
            #     aspect=15
            #     )
            # fig.show()

#             Ventana de Hann
#             win = torch.hann_window(K, periodic=True, device=self.device)  # (K,)
#             TRF_f_band[mask] *= win[:, None, None]
#             return torch.fft.irfft(TRF_f_band, axis=0).real    # (T, F, C)
            


                            
#             from IPython import embed; embed()
#             import mne
#             import matplotlib.pyplot as plt
#             import matplotlib
#             matplotlib.use('TkAgg')
#             # TRF_f_band[mask] *= win[:, None, None]

#             TRF_t_band = torch.fft.irfft(TRF_f_band, axis=0).real    # (T, F, C)
#             delays = np.arange(int(np.round(-.4 * config.sr)), int(np.round(.8 * config.sr) + 1))
#             mtrfs = TRF_t_band.cpu().numpy().mean(axis=1) # shape n_chans, feats, delays
#             evoked = mne.EvokedArray(
#                     data=mtrfs[delays%(N-1)].T, 
#                     info=config.info_mne
#                 )     
#             evoked.shift_time(
#                 delays[0]/config.sr
#             )
#             fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#             evoked_plot = evoked.plot(
#                 scalings={'eeg':1},
#                 zorder='std',
#                 time_unit='ms',
#                 show=False,
#                 spatial_colors=True,
#                 # unit=False,
#                 gfp=True,
#                 units='mTRFs (U.A)',
#                 axes=ax
#             )
#             ax.grid()
#             fig.show()
#             mtrfs_f = h_full.cpu().numpy().real.mean(axis=1).mean(axis=1) # shape n_chans, feats, delays
#             mtrfs_f = mtrfs.cpu().numpy().real.mean(axis=1).mean(axis=1) # shape n_chans, feats, delays
#             center = mtrfs_f.shape[0] // 2
#             left = center - 20000
#             right = center + 20000

#             import matplotlib.pyplot as plt
#             import matplotlib
#             matplotlib.use('TkAgg')
#             plt.figure()
#             m=1000
#             plt.plot(
#                 delays*1e3/config.sr,
#                 # mtrfs[config.delays%(N-1)],  # mtrfs_f[config.delays%N],
#                 mtrfs[delays%(N-1)], 
#                 # mtrfs,
                
#                 label='Real part'
#             )
#             plt.title('MTRFs in time domain')
#             plt.xlabel('Time ms')    
#             plt.ylabel('Amplitude')
#             plt.legend()
#             plt.show()
#             IFFT → Temporal
#             TRF_t = torch.fft.ifft(TRF_f_full, dim=0).real    # (T, F, C)
#             return TRF_t
#         # Estimación de H_f (shape: number_of_samples x D x C) freqsxdimensionsxchanns
#             numerator = X_f[:, :, None].conj() * Y_f[:, None, :]        # (number_of_samples, D, C)
#             denominator = (X_f.real ** 2 + X_f.imag ** 2).sum(dim=1, keepdim=True) + self.alpha*N  # (number_of_samples, 1) # TODO chequear la regularización
#             H_f = numerator / denominator[:, None, :]                   # (number_of_samples, D, C)
#             # numerator = X_f.unsqueeze(-1).conj() * Y_f.unsqueeze(1)
#             del X_f, Y_f
#             # H_f = torch.linalg.solve(denominator,numerator)
            
#             del numerator, denominator

#             # IFFT para recuperar TRF en el tiempo
#             # h_full = np.fft.ifft(H_f, axis=0).real  # (number_of_samples, D, C)
#             h_full = torch.fft.irfft(H_f, n=N, dim=0).real

#             # Alineación temporal: centramos la TRF en number_of_samples=0
#             # h_full = np.roll(h_full, -number_of_samples // 2, axis=0)  # shift temporal
#             # mtrfs = h_full[-number_of_samples // 2+config.delays[0]: -number_of_samples // 2+ config.delays[-1], :, :]  # (L, D, C)
#             indices = (config.delays % N)  
#             mtrfs = h_full[indices, :]
            
#             X_f_band = X_f[mask]      # (K, F)
#             Y_f_band = Y_f[mask]      # (K, C)

#             # Producto externo para cada frecuencia
#             A = torch.matmul(
#                 X_f_band.unsqueeze(2).conj(), # (K, F, 1)
#                 X_f_band.unsqueeze(1) # (K, 1, F)
#             )  # (K, F, F)
#             A += alpha * torch.eye(F, dtype=dtype, device=self.device).unsqueeze(0)  # (K, F, F)

#             B = torch.matmul(
#                 X_f_band.unsqueeze(2).conj(), # (K, F, 1)
#                 Y_f_band.unsqueeze(1) # (K, 1, C)
#             )   # (K, F, C)

#             # Resolver para cada frecuencia
#             TRF_f_band = torch.linalg.solve(A, B)            # (K, F, C)

#             # Ventana de Hann
#             win = torch.hann_window(K, periodic=True, device=self.device)  # (K,)
#             TRF_f_band *= win[:, None, None]

#             # Reconstruir espectro completo
#             TRF_f_full = torch.zeros((N, F, C), dtype=dtype, device=self.device)
#             TRF_f_full[mask] = TRF_f_band
            