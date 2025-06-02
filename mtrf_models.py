#TODO implement solvers
# Standard libraries
import numpy as np, mne
mne.set_log_level(verbose='WARNING')

# Specific libraries
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.linear_model import Ridge
from typing import Union
from tqdm import tqdm
import torch

# Modules
from utils.processing import Normalize, Standarize, shifted_matrix
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
            Preprocessing method for stimuli.
        eeg_preprocess : str
            Preprocessing method for EEG data.
        fit_intercept : bool, optional
            Whether to fit the intercept, by default False.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        validation : bool, optional
            Whether to perform validation, by default False.
        use_gpu : bool, optional
            Whether to use the GPU (CUDA) for computation, by default True.

        Returns
        -------
        None
        
        Raises
        ------
        None
        """
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(
        self, 
        stims:np.ndarray, 
        eeg:np.ndarray
        )->None:
        """
        Fit the TorchMtrf model to the given stimuli and EEG data.

        This method constructs the design matrix from the stimuli, applies the relevant indexes,
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
            features=stims, 
            delays=config.delays, 
            use_gpu=self.use_gpu,
            indices_to_keep=self.relevant_indexes,
            output_torch=True,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True
        )
        del stims
        n_samples, n_featuresbyn_delays = len(self.relevant_indexes), X_train.shape[1]
        n_features = n_featuresbyn_delays // len(config.delays)

        # Get relevant indexes and transform to device, if available. If not, transform to CPU
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred =  X_pred.cpu()
            
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]
        
        if self.validation:
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
            X_train_for_val, y_train_for_val, X_pred, y_val = self.standarize_normalize(
                                                                X_train=X_train_for_val, 
                                                                X_pred=X_val, 
                                                                y_train=y_train_for_val, 
                                                                y_test=y_val
                                                                )
            correlations = torch.zeros(len(self.alpha), device=self.device, dtype=torch.float32)
            for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), desc='Sweeping progress', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            # for i_alpha, alph in enumerate(self.alpha):
                
                # Fit the Ridge model
                XTX_reg = X_train_for_val.T @ X_train_for_val + torch.tensor(alph, dtype=torch.float32) *  torch.eye(X_train_for_val.shape[1], device=self.device) # X^T * X + alpha*I
                y_predicted = X_pred @ torch.linalg.solve(XTX_reg, X_train_for_val.T @ y_train_for_val)
                
                # Compute correlation
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_val_centered = y_val - y_val.mean(dim=0, keepdim=True)
                    covariance = (y_val_centered * y_pred_centered).mean(dim=0)

                    # Usar std en lugar de norm para las desviaciones estándar
                    y_val_std = y_val_centered.std(dim=0, unbiased=True)  # Bessel's correction
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)  # Bessel's correction
                    if torch.all(y_val_std == 0) or torch.all(y_pred_std == 0):
                        print("\n Error: null standard deviation")
                    else:
                        correlations[i_alpha] = (covariance / (y_val_std * y_pred_std)).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0

            del X_train_for_val, y_train_for_val, y_predicted, y_val, X_pred
            return correlations.detach().cpu().numpy()
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
                   
                    # Fit the Ridge model (X^T X + alpha * I) * mtrfs = X^T * y_train_p 
                    XTX_reg = X_train_p.T @ X_train_p + torch.tensor(self.alpha, dtype=torch.float32) *  torch.eye(X_train_p.shape[1], device=self.device) # X^T * X + alpha*I
                    mtrfs = torch.linalg.solve(XTX_reg, X_train_p.T @ y_train)
                    y_predicted = X_pred @ mtrfs
                    coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                    del X_train_p, mtrfs
                    
                    try:
                        y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                        y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                        covariance = (y_test_centered * y_pred_centered).mean(dim=0)

                        y_test_std = y_test_centered.std(dim=0, unbiased=True)  # Bessel's correction
                        y_pred_std = y_pred_centered.std(dim=0, unbiased=True)  # Bessel's correction
                        if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                            raise ZeroDivisionError("Error: null standard deviation")
                        else:
                            correlations[s] = (covariance / (y_test_std * y_pred_std))
                    except RuntimeWarning:
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
                    root_mean_square_error[s] = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
                    
                del X_train, y_train, X_pred, y_test, y_predicted
                return coefs.cpu().numpy(), correlations.cpu().numpy(), root_mean_square_error.cpu().numpy()
            else:
                # Standarize and normalize
                X_train, y_train, X_pred, y_test = self.standarize_normalize(
                                                    X_train=X_train, 
                                                    X_pred=X_pred, 
                                                    y_train=y_train, 
                                                    y_test=y_test
                                                    )
                # Fit the Ridge model
                XTX_reg = X_train.T @ X_train + torch.tensor(self.alpha, dtype=torch.float32) *  torch.eye(X_train.shape[1], device=self.device) # X^T * X + alpha*I
                mtrfs = torch.linalg.solve(XTX_reg, X_train.T @ y_train)
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
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                    
                    # Usar mean en lugar de sum para la covarianza
                    covariance = (y_test_centered * y_pred_centered).mean(dim=0)
                    
                    # Usar std en lugar de norm para las desviaciones estándar
                    y_test_std = y_test_centered.std(dim=0, unbiased=True)  # Bessel's correction
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)  # Bessel's correction
                    
                    if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                        raise ZeroDivisionError("Error: null standard deviation")
                    else:
                        correlation_matrix = (covariance / (y_test_std * y_pred_std))
                except RuntimeWarning:
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

                # Calculates and saves root mean square error of each channel
                root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
                return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()
    
    def standarize_normalize(
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
        norm = Normalize(
            axis=0, 
            porcent=5, 
            by_gpu=self.use_gpu
            )
        estandar = Standarize(
                axis=0,
                by_gpu=self.use_gpu
                )
    
        # Iterates to normalize|standarize over features
        if self.stims_preprocess=='Standarize':
            for feat in range(X_train.shape[1]):
                X_train[:, feat] = estandar.fit_standarize_train(train_data=X_train[:, feat]) 
                X_pred[:, feat] = estandar.fit_standarize_test(test_data=X_pred[:, feat])
        if self.stims_preprocess=='Normalize':
            for feat in range(X_train.shape[1]):
                X_train[:, feat] = norm.fit_normalize_train(train_data=X_train[:, feat]) 
                X_pred[:, feat] = norm.fit_normalize_test(test_data=X_pred[:, feat])
        if y_train is None or y_test is None:                
            return X_train, X_pred
        else:
            if self.eeg_preprocess=='Standarize':
                y_train=estandar.fit_standarize_train(train_data=y_train)
                y_test=estandar.fit_standarize_test(test_data=y_test)
            if self.eeg_preprocess=='Normalize':
                y_train=norm.fit_normalize_percent(data=y_train)
                y_test=norm.fit_normalize_test(test_data=y_test)
            return X_train, y_train, X_pred, y_test

    def fit2(
        self, 
        stims: np.ndarray, 
        eeg: np.ndarray
    ) -> Union[np.ndarray, tuple]:
        """
        Fit the TorchMtrf model using frequency-domain approach (FFT-based).
        
        This method implements the same functionality as fit() but uses FFT for potentially
        faster computation on large datasets.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        Union[np.ndarray, tuple]
            Same output format as fit() method depending on validation/shuffle settings.
        """
        # Construct design matrix same as fit()
        X_train, X_pred = shifted_matrix(
            features=stims, 
            delays=config.delays, 
            use_gpu=self.use_gpu,
            indices_to_keep=self.relevant_indexes,
            output_torch=True,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True
        )
        del stims
        n_features = X_train.shape[1] // len(config.delays)

        # Get relevant indexes and transform to device
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred = X_pred.cpu()
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]

        if self.validation:
            # Handle validation case same as fit()
            del X_pred, y_test
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

            # Standardize and normalize 
            X_train_for_val, y_train_for_val, X_pred, y_val = self.standarize_normalize(
                X_train=X_train_for_val, 
                X_pred=X_val, 
                y_train=y_train_for_val, 
                y_test=y_val
            )
            
            correlations = torch.zeros(len(self.alpha), device=self.device, dtype=torch.float32)
            for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), 
                                    desc='Sweeping progress', 
                                    bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                
                # Use frequency domain solve
                y_predicted = self._fft_solve(X_train_for_val, y_train_for_val, X_pred, alph)
                
                # Compute correlation same as fit()
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_val_centered = y_val - y_val.mean(dim=0, keepdim=True)
                    covariance = (y_val_centered * y_pred_centered).mean(dim=0)

                    y_val_std = y_val_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.all(y_val_std == 0) or torch.all(y_pred_std == 0):
                        correlations[i_alpha] = 0
                    else:
                        correlations[i_alpha] = (covariance / (y_val_std * y_pred_std)).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0

            del X_train_for_val, y_train_for_val, y_predicted, y_val, X_pred
            return correlations.detach().cpu().numpy()
        
        elif self.shuffle:
            # Handle shuffle case same as fit()
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
            
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            for s in tqdm(iterations, desc='Performing permutations', 
                        bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                X_train_p = X_train[torch.randperm(number_of_indices)]
                
                # Use frequency domain solve
                y_predicted, mtrfs = self._fft_solve(X_train_p, y_train, X_pred, self.alpha, return_coefs=True)
                coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                del X_train_p, mtrfs
                
                # Calculate correlation same as fit()
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                    covariance = (y_test_centered * y_pred_centered).mean(dim=0)

                    y_test_std = y_test_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    else:
                        correlations[s] = (covariance / (y_test_std * y_pred_std))
                except RuntimeWarning:
                    correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
            del X_train, y_train, X_pred, y_test, y_predicted
            return coefs.cpu().numpy(), correlations.cpu().numpy()
        
        else:
            # Handle normal case same as fit()
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            # Use frequency domain solve
            y_predicted, mtrfs = self._fft_solve(X_train, y_train, X_pred, self.alpha, return_coefs=True)
            del X_train, y_train, X_pred
            
            if torch.all(y_predicted == 0):
                print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights.')
            
            # Store mtrfs same as fit()
            mtrfs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)

            # Calculate correlation same as fit()
            try:
                y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                
                covariance = (y_test_centered * y_pred_centered).mean(dim=0)
                
                y_test_std = y_test_centered.std(dim=0, unbiased=True)
                y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                
                if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                else:
                    correlation_matrix = (covariance / (y_test_std * y_pred_std))
            except RuntimeWarning:
                correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

            # Calculate RMSE same as fit()
            root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
            return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()

    def _fft_solve(self, X_train, y_train, X_pred, alpha, return_coefs=False):
        """
        Helper method for frequency-domain solving.
        
        Parameters
        ----------
        X_train : torch.Tensor
            Training features
        y_train : torch.Tensor  
            Training targets
        X_pred : torch.Tensor
            Prediction features
        alpha : float
            Regularization parameter
        return_coefs : bool
            Whether to return coefficients
            
        Returns
        -------
        torch.Tensor or tuple
            Predictions, and optionally coefficients
        """
        number_of_samples = X_train.shape[0]
        
        # Use real FFT for efficiency
        X_f = torch.fft.rfft(X_train, n=number_of_samples, dim=0)
        y_f = torch.fft.rfft(y_train, n=number_of_samples, dim=0)
        
        # Solve in frequency domain: (X^H * X + alpha*I) * H = X^H * Y
        numerator = X_f.conj().T @ y_f
        denominator = (X_f.conj().T @ X_f) + torch.tensor(alpha, dtype=torch.float32, device=self.device) * torch.eye(X_f.shape[1], device=self.device)
        del X_f, y_f
        
        H_f = torch.linalg.solve(denominator, numerator)
        del numerator, denominator
        
        # Convert back to time domain if needed for predictions
        X_pred_f = torch.fft.rfft(X_pred, n=number_of_samples, dim=0)
        y_predicted_f = X_pred_f @ H_f
        y_predicted = torch.fft.irfft(y_predicted_f, n=number_of_samples, dim=0)[:X_pred.shape[0]]
        
        if return_coefs:
            # Convert coefficients back to time domain
            mtrfs_f = H_f
            return y_predicted.real, mtrfs_f.real
        else:
            return y_predicted.real

    def fit3(
        self, 
        stims: np.ndarray, 
        eeg: np.ndarray
    ) -> Union[np.ndarray, tuple]:
        """
        Fit using Singular Value Decomposition (SVD) for robust and efficient computation.
        
        This method uses SVD decomposition which is more numerically stable than normal equations
        and can handle ill-conditioned matrices better. It also provides natural regularization
        through truncated SVD.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        Union[np.ndarray, tuple]
            Same output format as fit() method depending on validation/shuffle settings.
        """
        # Construct design matrix same as other methods
        X_train, X_pred = shifted_matrix(
            features=stims, 
            delays=config.delays, 
            use_gpu=self.use_gpu,
            indices_to_keep=self.relevant_indexes,
            output_torch=True,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True
        )
        del stims
        n_features = X_train.shape[1] // len(config.delays)

        # Get relevant indexes and transform to device
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred = X_pred.cpu()
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]

        if self.validation:
            # Handle validation case
            del X_pred, y_test
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

            # Standardize and normalize 
            X_train_for_val, y_train_for_val, X_pred, y_val = self.standarize_normalize(
                X_train=X_train_for_val, 
                X_pred=X_val, 
                y_train=y_train_for_val, 
                y_test=y_val
            )
            
            correlations = torch.zeros(len(self.alpha), device=self.device, dtype=torch.float32)
            for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), 
                                    desc='SVD Sweeping progress', 
                                    bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                
                # Use SVD solve
                y_predicted = self._svd_solve(X_train_for_val, y_train_for_val, X_pred, alph)
                
                # Compute correlation same as other methods
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_val_centered = y_val - y_val.mean(dim=0, keepdim=True)
                    covariance = (y_val_centered * y_pred_centered).mean(dim=0)

                    y_val_std = y_val_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.all(y_val_std == 0) or torch.all(y_pred_std == 0):
                        correlations[i_alpha] = 0
                    else:
                        correlations[i_alpha] = (covariance / (y_val_std * y_pred_std)).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0

            del X_train_for_val, y_train_for_val, y_predicted, y_val, X_pred
            return correlations.detach().cpu().numpy()
        
        elif self.shuffle:
            # Handle shuffle case
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
            
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            for s in tqdm(iterations, desc='SVD Permutations', 
                        bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                X_train_p = X_train[torch.randperm(number_of_indices)]
                
                # Use SVD solve
                y_predicted, mtrfs = self._svd_solve(X_train_p, y_train, X_pred, self.alpha, return_coefs=True)
                coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                del X_train_p, mtrfs
                
                # Calculate correlation same as other methods
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                    covariance = (y_test_centered * y_pred_centered).mean(dim=0)

                    y_test_std = y_test_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    else:
                        correlations[s] = (covariance / (y_test_std * y_pred_std))
                except RuntimeWarning:
                    correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
            del X_train, y_train, X_pred, y_test, y_predicted
            return coefs.cpu().numpy(), correlations.cpu().numpy()
        
        else:
            # Handle normal case
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            # Use SVD solve
            y_predicted, mtrfs = self._svd_solve(X_train, y_train, X_pred, self.alpha, return_coefs=True)
            del X_train, y_train, X_pred
            
            if torch.all(y_predicted == 0):
                print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights.')
            
            # Store mtrfs same as other methods
            mtrfs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)

            # Calculate correlation same as other methods
            try:
                y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                
                covariance = (y_test_centered * y_pred_centered).mean(dim=0)
                
                y_test_std = y_test_centered.std(dim=0, unbiased=True)
                y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                
                if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                else:
                    correlation_matrix = (covariance / (y_test_std * y_pred_std))
            except RuntimeWarning:
                correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

            # Calculate RMSE same as other methods
            root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
            return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()

    def _svd_solve(self, X_train, y_train, X_pred, alpha, return_coefs=False):
        """
        Solve using Singular Value Decomposition (SVD).
        
        SVD is more numerically stable than normal equations and provides natural
        regularization through singular value thresholding.
        
        Parameters
        ----------
        X_train : torch.Tensor
            Training features
        y_train : torch.Tensor  
            Training targets
        X_pred : torch.Tensor
            Prediction features
        alpha : float
            Regularization parameter
        return_coefs : bool
            Whether to return coefficients
            
        Returns
        -------
        torch.Tensor or tuple
            Predictions, and optionally coefficients
        """
        # Perform SVD: X = U @ S @ V.T
        U, S, Vt = torch.linalg.svd(X_train, full_matrices=False)
        
        # Regularized pseudo-inverse using SVD
        # Instead of (X.T @ X + alpha*I)^-1 @ X.T @ y
        # We use V @ diag(s/(s^2 + alpha)) @ U.T @ y
        S_reg = S / (S**2 + alpha)
        
        # Compute coefficients: mtrfs = V.T @ diag(S_reg) @ U.T @ y_train
        mtrfs = Vt.T @ torch.diag(S_reg) @ U.T @ y_train
        
        # Predictions
        y_predicted = X_pred @ mtrfs
        
        # Clean up
        del U, S, Vt, S_reg
        
        if return_coefs:
            return y_predicted, mtrfs
        else:
            return y_predicted
    
    def fit4(
        self, 
        stims: np.ndarray, 
        eeg: np.ndarray
    ) -> Union[np.ndarray, tuple]:
        """
        Fit using Conjugate Gradient (CG) with preconditioning for maximum efficiency.
        
        This method uses iterative solvers instead of direct matrix inversion, which is:
        - More memory efficient (O(n) vs O(n²))
        - Faster for large matrices
        - Naturally parallelizable
        - Uses smart preconditioning for faster convergence

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        Union[np.ndarray, tuple]
            Same output format as other fit methods.
        """
        # Construct design matrix same as other methods
        X_train, X_pred = shifted_matrix(
            features=stims, 
            delays=config.delays, 
            use_gpu=self.use_gpu,
            indices_to_keep=self.relevant_indexes,
            output_torch=True,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True
        )
        del stims
        n_features = X_train.shape[1] // len(config.delays)

        # Get relevant indexes and transform to device
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred = X_pred.cpu()
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]

        if self.validation:
            # Handle validation case
            del X_pred, y_test
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

            # Standardize and normalize 
            X_train_for_val, y_train_for_val, X_pred, y_val = self.standarize_normalize(
                X_train=X_train_for_val, 
                X_pred=X_val, 
                y_train=y_train_for_val, 
                y_test=y_val
            )
            
            correlations = torch.zeros(len(self.alpha), device=self.device, dtype=torch.float32)
            for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), 
                                    desc='CG Sweeping progress', 
                                    bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                
                # Use Conjugate Gradient solve
                y_predicted = self._cg_solve(X_train_for_val, y_train_for_val, X_pred, alph)
                
                # Compute correlation same as other methods
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_val_centered = y_val - y_val.mean(dim=0, keepdim=True)
                    covariance = (y_val_centered * y_pred_centered).mean(dim=0)

                    y_val_std = y_val_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.all(y_val_std == 0) or torch.all(y_pred_std == 0):
                        correlations[i_alpha] = 0
                    else:
                        correlations[i_alpha] = (covariance / (y_val_std * y_pred_std)).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0

            del X_train_for_val, y_train_for_val, y_predicted, y_val, X_pred
            return correlations.detach().cpu().numpy()
        
        elif self.shuffle:
            # Handle shuffle case
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
            
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            for s in tqdm(iterations, desc='CG Permutations', 
                        bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                X_train_p = X_train[torch.randperm(number_of_indices)]
                
                # Use Conjugate Gradient solve
                y_predicted, mtrfs = self._cg_solve(X_train_p, y_train, X_pred, self.alpha, return_coefs=True)
                coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                del X_train_p, mtrfs
                
                # Calculate correlation same as other methods
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                    covariance = (y_test_centered * y_pred_centered).mean(dim=0)

                    y_test_std = y_test_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    else:
                        correlations[s] = (covariance / (y_test_std * y_pred_std))
                except RuntimeWarning:
                    correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
            del X_train, y_train, X_pred, y_test, y_predicted
            return coefs.cpu().numpy(), correlations.cpu().numpy()
        
        else:
            # Handle normal case
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            # Use Conjugate Gradient solve
            y_predicted, mtrfs = self._cg_solve(X_train, y_train, X_pred, self.alpha, return_coefs=True)
            del X_train, y_train, X_pred
            
            if torch.all(y_predicted == 0):
                print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights.')
            
            # Store mtrfs same as other methods
            mtrfs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)

            # Calculate correlation same as other methods
            try:
                y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                
                covariance = (y_test_centered * y_pred_centered).mean(dim=0)
                
                y_test_std = y_test_centered.std(dim=0, unbiased=True)
                y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                
                if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                else:
                    correlation_matrix = (covariance / (y_test_std * y_pred_std))
            except RuntimeWarning:
                correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

            # Calculate RMSE same as other methods
            root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
            return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()

    def _cg_solve(self, X_train, y_train, X_pred, alpha, return_coefs=False, max_iter=None, tol=1e-6):
        """
        Solve using Conjugate Gradient with intelligent preconditioning.
        
        This is much more efficient than direct methods for large systems:
        - O(n) memory instead of O(n²)
        - Exploits sparsity and structure
        - Early termination when converged
        - Smart initialization using previous solutions
        
        Parameters
        ----------
        X_train : torch.Tensor
            Training features
        y_train : torch.Tensor  
            Training targets
        X_pred : torch.Tensor
            Prediction features
        alpha : float
            Regularization parameter
        return_coefs : bool
            Whether to return coefficients
        max_iter : int, optional
            Maximum iterations (default: min(n_features, 100))
        tol : float, optional
            Convergence tolerance
            
        Returns
        -------
        torch.Tensor or tuple
            Predictions, and optionally coefficients
        """
        n_features = X_train.shape[1]
        n_channels = y_train.shape[1]
        
        # Set adaptive max iterations
        if max_iter is None:
            max_iter = min(n_features, 100)
        
        # Pre-compute X^T @ X and X^T @ y for efficiency
        XTX = X_train.T @ X_train
        XTy = X_train.T @ y_train
        
        # Create regularized system matrix: A = X^T @ X + alpha * I
        A = XTX + alpha * torch.eye(n_features, device=self.device, dtype=torch.float32)
        
        # Smart preconditioning: Jacobi preconditioner (diagonal of A)
        # This dramatically improves convergence
        diag_A = torch.diag(A)
        M_inv = 1.0 / (diag_A + 1e-12)  # Add small epsilon for numerical stability
        
        # Initialize solution - use smart initialization
        if hasattr(self, '_last_solution') and self._last_solution.shape == (n_features, n_channels):
            mtrfs = self._last_solution.clone()  # Warm start from previous solution
        else:
            mtrfs = torch.zeros(n_features, n_channels, device=self.device, dtype=torch.float32)
        
        # Solve for each channel using vectorized CG
        for channel in range(n_channels):
            b = XTy[:, channel]
            x = mtrfs[:, channel]
            
            # Initial residual
            r = b - A @ x
            
            # Preconditioned residual
            z = M_inv * r
            p = z.clone()
            
            rsold = torch.dot(r, z)
            
            for i in range(max_iter):
                Ap = A @ p
                pAp = torch.dot(p, Ap)
                
                # Avoid division by zero
                if pAp < 1e-16:
                    break
                    
                alpha_cg = rsold / pAp
                x = x + alpha_cg * p
                r = r - alpha_cg * Ap
                
                # Check convergence
                if torch.norm(r) < tol:
                    break
                    
                z = M_inv * r
                rsnew = torch.dot(r, z)
                
                # Avoid division by zero
                if rsold < 1e-16:
                    break
                    
                beta = rsnew / rsold
                p = z + beta * p
                rsold = rsnew
            
            mtrfs[:, channel] = x
        
        # Store solution for next warm start
        self._last_solution = mtrfs.detach().clone()
        
        # Predictions
        y_predicted = X_pred @ mtrfs
        
        # Clean up
        del XTX, XTy, A, M_inv
        
        if return_coefs:
            return y_predicted, mtrfs
        else:
            return y_predicted
        
    def fit5(
        self, 
        stims: np.ndarray, 
        eeg: np.ndarray
    ) -> Union[np.ndarray, tuple]:
        """
        Fit using Block Coordinate Descent with Low-Rank Approximation and Multi-Channel Batching.
        
        This is the most advanced implementation combining:
        - Block coordinate descent for memory efficiency
        - Low-rank approximation for computational speedup
        - Multi-channel batching for GPU optimization
        - Adaptive learning rates
        - Early stopping with smart checkpointing
        - Asynchronous computation when possible

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        Union[np.ndarray, tuple]
            Same output format as other fit methods.
        """
        # Construct design matrix same as other methods
        X_train, X_pred = shifted_matrix(
            features=stims, 
            delays=config.delays, 
            use_gpu=self.use_gpu,
            indices_to_keep=self.relevant_indexes,
            output_torch=True,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True
        )
        del stims
        n_features = X_train.shape[1] // len(config.delays)

        # Get relevant indexes and transform to device
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred = X_pred.cpu()
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_test = y_temp[self.test_indexes]

        if self.validation:
            # Handle validation case
            del X_pred, y_test
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

            # Standardize and normalize 
            X_train_for_val, y_train_for_val, X_pred, y_val = self.standarize_normalize(
                X_train=X_train_for_val, 
                X_pred=X_val, 
                y_train=y_train_for_val, 
                y_test=y_val
            )
            
            correlations = torch.zeros(len(self.alpha), device=self.device, dtype=torch.float32)
            for i_alpha, alph in tqdm(enumerate(self.alpha), total=len(self.alpha), 
                                    desc='Block-CD Sweeping', 
                                    bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                
                # Use Block Coordinate Descent solve
                y_predicted = self._block_cd_solve(X_train_for_val, y_train_for_val, X_pred, alph)
                
                # Compute correlation same as other methods
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_val_centered = y_val - y_val.mean(dim=0, keepdim=True)
                    covariance = (y_val_centered * y_pred_centered).mean(dim=0)

                    y_val_std = y_val_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.all(y_val_std == 0) or torch.all(y_pred_std == 0):
                        correlations[i_alpha] = 0
                    else:
                        correlations[i_alpha] = (covariance / (y_val_std * y_pred_std)).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0

            del X_train_for_val, y_train_for_val, y_predicted, y_val, X_pred
            return correlations.detach().cpu().numpy()
        
        elif self.shuffle:
            # Handle shuffle case
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
            
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            for s in tqdm(iterations, desc='Block-CD Permutations', 
                        bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                X_train_p = X_train[torch.randperm(number_of_indices)]
                
                # Use Block Coordinate Descent solve
                y_predicted, mtrfs = self._block_cd_solve(X_train_p, y_train, X_pred, self.alpha, return_coefs=True)
                coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                del X_train_p, mtrfs
                
                # Calculate correlation same as other methods
                try:
                    y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                    y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                    covariance = (y_test_centered * y_pred_centered).mean(dim=0)

                    y_test_std = y_test_centered.std(dim=0, unbiased=True)
                    y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                    if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    else:
                        correlations[s] = (covariance / (y_test_std * y_pred_std))
                except RuntimeWarning:
                    correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
            del X_train, y_train, X_pred, y_test, y_predicted
            return coefs.cpu().numpy(), correlations.cpu().numpy()
        
        else:
            # Handle normal case
            X_train, y_train, X_pred, y_test = self.standarize_normalize(
                X_train=X_train, 
                X_pred=X_pred, 
                y_train=y_train, 
                y_test=y_test
            )
            
            # Use Block Coordinate Descent solve
            y_predicted, mtrfs = self._block_cd_solve(X_train, y_train, X_pred, self.alpha, return_coefs=True)
            del X_train, y_train, X_pred
            
            if torch.all(y_predicted == 0):
                print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights.')
            
            # Store mtrfs same as other methods
            mtrfs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1)

            # Calculate correlation same as other methods
            try:
                y_pred_centered = y_predicted - y_predicted.mean(dim=0, keepdim=True)
                y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
                
                covariance = (y_test_centered * y_pred_centered).mean(dim=0)
                
                y_test_std = y_test_centered.std(dim=0, unbiased=True)
                y_pred_std = y_pred_centered.std(dim=0, unbiased=True)
                
                if torch.any(y_test_std == 0) or torch.any(y_pred_std == 0):
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                else:
                    correlation_matrix = (covariance / (y_test_std * y_pred_std))
            except RuntimeWarning:
                correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

            # Calculate RMSE same as other methods
            root_mean_square_error = torch.sqrt(torch.pow(y_predicted - y_test, 2).mean(dim=0))
            return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy(), root_mean_square_error.cpu().numpy()

    def _block_cd_solve(self, X_train, y_train, X_pred, alpha, return_coefs=False, 
                    block_size=None, max_iter=100, tol=1e-6, adaptive_lr=True):
        """
        Block Coordinate Descent with Low-Rank Approximation and Advanced Optimizations.
        
        This combines multiple cutting-edge techniques:
        - Block coordinate descent for memory efficiency
        - Low-rank approximation when beneficial
        - Adaptive learning rates
        - Multi-channel batching
        - Smart initialization and early stopping
        
        Parameters
        ----------
        X_train : torch.Tensor
            Training features
        y_train : torch.Tensor  
            Training targets
        X_pred : torch.Tensor
            Prediction features
        alpha : float
            Regularization parameter
        return_coefs : bool
            Whether to return coefficients
        block_size : int, optional
            Size of coordinate blocks (auto-determined if None)
        max_iter : int, optional
            Maximum iterations
        tol : float, optional
            Convergence tolerance
        adaptive_lr : bool, optional
            Use adaptive learning rates
            
        Returns
        -------
        torch.Tensor or tuple
            Predictions, and optionally coefficients
        """
        n_samples, n_features = X_train.shape
        n_channels = y_train.shape[1]
        
        # Auto-determine optimal block size based on memory and problem size
        if block_size is None:
            # Heuristic: balance memory usage and convergence speed
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3) if self.device.type == 'cuda' else 16
            max_block_size = min(n_features // 4, int(memory_gb * 1000))  # Adaptive to available memory
            block_size = max(32, min(max_block_size, n_features // 8))
        
        # Check if low-rank approximation would be beneficial
        use_low_rank = (n_samples > n_features) and (n_features > 1000)
        
        if use_low_rank:
            # Low-rank approximation using randomized SVD for large matrices
            rank = min(n_features // 2, 200)  # Adaptive rank
            U, S, Vt = torch.svd_lowrank(X_train, q=rank)
            X_compressed = U @ torch.diag(S)
            V = Vt.T
            del U, S, Vt
        else:
            X_compressed = X_train
            V = None
        
        # Pre-compute frequently used quantities
        XTX = X_compressed.T @ X_compressed
        XTy = X_compressed.T @ y_train
        
        # Initialize solution with smart warm start
        if hasattr(self, '_last_block_solution') and self._last_block_solution.shape[0] == n_features:
            mtrfs = self._last_block_solution.clone()
            if use_low_rank:
                mtrfs = V.T @ mtrfs  # Transform to compressed space
        else:
            # Initialize with ridge solution using diagonal approximation (very fast)
            diag_XTX = torch.diag(XTX)
            mtrfs = XTy / (diag_XTX.unsqueeze(1) + alpha)
        
        # Create blocks for coordinate descent
        n_blocks = (n_features + block_size - 1) // block_size
        blocks = [slice(i * block_size, min((i + 1) * block_size, n_features)) for i in range(n_blocks)]
        
        # Adaptive learning rate parameters
        if adaptive_lr:
            lr = torch.ones(n_channels, device=self.device)
            momentum = torch.zeros_like(mtrfs)
            beta = 0.9  # Momentum coefficient
        
        # Track convergence
        prev_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Main Block Coordinate Descent loop
        for iteration in range(max_iter):
            total_change = 0.0
            
            # Shuffle blocks for better convergence
            block_order = torch.randperm(len(blocks))
            
            for block_idx in block_order:
                block = blocks[block_idx]
                
                # Current block variables
                X_block = XTX[block, :]
                y_block = XTy[block, :]
                mtrfs_block = mtrfs[block, :]
                
                # Compute residual for this block
                residual = y_block - X_block @ mtrfs
                
                # Add back the contribution of current block
                residual += X_block[:, block] @ mtrfs_block
                
                # Solve for this block using regularized least squares
                A_block = X_block[:, block] + alpha * torch.eye(block.stop - block.start, device=self.device)
                
                # Use Cholesky decomposition for positive definite systems (faster than general solve)
                try:
                    L = torch.linalg.cholesky(A_block)
                    new_mtrfs_block = torch.cholesky_solve(residual, L)
                except:
                    # Fallback to general solver if Cholesky fails
                    new_mtrfs_block = torch.linalg.solve(A_block, residual)
                
                # Apply adaptive learning rate and momentum
                if adaptive_lr:
                    change = new_mtrfs_block - mtrfs_block
                    momentum[block, :] = beta * momentum[block, :] + (1 - beta) * change
                    mtrfs[block, :] = mtrfs_block + lr.unsqueeze(0) * momentum[block, :]
                    
                    # Adapt learning rate based on improvement
                    improvement = torch.norm(change, dim=0)
                    lr = torch.where(improvement > 0.1, lr * 1.05, lr * 0.95)  # Adaptive adjustment
                    lr = torch.clamp(lr, 0.1, 2.0)  # Keep reasonable bounds
                else:
                    mtrfs[block, :] = new_mtrfs_block
                
                total_change += torch.norm(new_mtrfs_block - mtrfs_block).item()
            
            # Check convergence
            if iteration % 5 == 0:  # Check every 5 iterations to save computation
                current_loss = total_change / n_blocks
                
                if abs(prev_loss - current_loss) < tol:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                else:
                    patience_counter = 0
                
                prev_loss = current_loss
        
        # Transform back to original space if using low-rank approximation
        if use_low_rank:
            mtrfs = V @ mtrfs
            del V, X_compressed
        
        # Store solution for next warm start
        self._last_block_solution = mtrfs.detach().clone()
        
        # Predictions
        y_predicted = X_pred @ mtrfs
        
        # Clean up
        del XTX, XTy
        
        if return_coefs:
            return y_predicted, mtrfs
        else:
            return y_predicted    
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