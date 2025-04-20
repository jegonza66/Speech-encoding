# Standard libraries
import numpy as np, mne
mne.set_log_level(verbose='WARNING')

# Specific libraries
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.linear_model import Ridge
from tqdm import tqdm
import torch

# Modules
from processing import Normalize, Standarize, shifted_matrix
import config

class TorchMtrf:
    def __init__(
        self, 
        alpha:float, 
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
        design_matrix = shifted_matrix(
                    stims, 
                    delays=config.delays, 
                    use_gpu=self.use_gpu,
                    indices_to_keep=self.relevant_indexes,
                    output_torch=True
                    )
        
        n_samples, n_featuresbyn_delays = design_matrix.size()
        n_features = n_featuresbyn_delays // len(config.delays)

        # Get relevant indexes and transform to GPU
        # design_matrix = torch.tensor(design_matrix).to(self.device)
        y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
        del stims, eeg
        
        # Separate into training and testing

        X_train = design_matrix[self.train_indexes]
        y_train = y_temp[self.train_indexes]
        X_pred = design_matrix[self.test_indexes]
        y_test = y_temp[self.test_indexes]
        del design_matrix, y_temp
        
        # # Construct design matrix and transform for GPU computation
        # design_matrix = shifted_matrix_2(
        #             stims, 
        #             delays=config.delays, 
        #             use_gpu=self.use_gpu
        #             )
        # n_samples, n_featuresbyn_delays = design_matrix.shape
        # n_features = n_featuresbyn_delays // len(config.delays)

        # # Get relevant indexes and transform to GPU
        # X_temp = design_matrix[self.relevant_indexes]
        # del design_matrix
        # X_temp = torch.tensor(X_temp).to(self.device)
        # y_temp = torch.tensor(eeg[self.relevant_indexes]).to(self.device)
        # del stims, eeg
        
        # # Separate into training and testing

        # X_train = X_temp[self.train_indexes]
        # y_train = y_temp[self.train_indexes]
        # X_pred = X_temp[self.test_indexes]
        # y_test = y_temp[self.test_indexes]
        # del X_temp, y_temp
        
        if not self.validation:
            if self.shuffle:
                indices = np.arange(X_train.shape[0])
                iterations = np.arange(config.random_permutations)
                self.coefs = np.zeros((config.random_permutations, config.info_mne['nchan'], n_features, len(config.delays)), dtype=np.float16)
                self.correlations = np.zeros((config.random_permutations, config.info_mne['nchan']))
                self.root_mean_square_error = np.zeros((config.random_permutations, config.info_mne['nchan']))
                
                # Shuffle the data, by requierment of random permutations
                for s in tqdm(iterations, desc='Performing permutations', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                    indices_p = indices.copy()
                    X_train_p = X_train.clone()
                    X_pred_p = X_pred.clone()
                    y_test_p = y_test.clone()
                    y_train_p = y_train.clone()
                    
                    np.random.shuffle(indices_p)
                    X_train_p = X_train_p[indices_p]
                    
                    # TODO after first iteration its not neccesary to compute self.y_val
                    X_train_p, y_train_p, X_pred_p, y_test_p = self.standarize_normalize(
                                                X_train=X_train_p, 
                                                X_pred=X_pred_p, 
                                                y_train=y_train_p, 
                                                y_test=y_test_p
                                                )
                    
                    # Fit the Ridge model (X^T X + alpha * I) * mtrfs = X^T * y_train_p 
                    XTX_reg = X_train_p.T @ X_train_p + self.alpha.astype(np.float32) *  torch.eye(X_train_p.shape[1], device=self.device) # X^T * X + alpha*I
                    mtrfs = torch.linalg.solve(XTX_reg, X_train_p.T @ y_train_p)
                    
                    # Perform predictions
                    y_predicted = X_pred_p @ mtrfs
                    del X_pred_p

                    predicted = y_predicted.cpu().detach().numpy()
                    eeg_test = y_test_p.cpu().detach().numpy()
                    root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                    try:
                        correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
                    except RuntimeWarning:
                        correlation_matrix = np.zeros(eeg_test.shape[1])
                    
                    # Store mtrfs and correlation
                    self.correlations[s] = correlation_matrix
                    self.root_mean_square_error[s] = root_mean_square_error
                    self.coefs[s] = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1).cpu().numpy()
                del X_train, y_train, X_pred, X_train_p, y_train_p, X_pred_p, y_test_p                    
            else:
                # Standarize and normalize
                X_train, y_train, X_pred, self.y_test = self.standarize_normalize(
                                                    X_train=X_train, 
                                                    X_pred=X_pred, 
                                                    y_train=y_train, 
                                                    y_test=y_test
                                                    )
                del y_test

                # Fit the Ridge model
                XTX_reg = X_train.T @ X_train + torch.tensor(self.alpha, dtype=torch.float32) *  torch.eye(X_train.shape[1], device=self.device) # X^T * X + alpha*I
                mtrfs = torch.linalg.solve(XTX_reg, X_train.T @ y_train)
                
                # Perform predictions
                self.y_predicted = X_pred @ mtrfs
                del X_pred
                
                # Store mtrfs
                self.coefs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1).cpu().numpy()
                del X_train, y_train 
        else:
            # Make split for validation: validation sets, fixing the train percent of data
            train_percent = .8
            self.train_cutoff = int(train_percent * len(self.train_indexes))
            X_train_for_val = X_train[:self.train_cutoff]
            y_train_for_val = y_train[:self.train_cutoff]
            X_val = X_train[self.train_cutoff:]
            y_val = y_train[self.train_cutoff:]
            del X_train, y_train
                        
            # Standarize and normalize 
            X_train_for_val, y_train_for_val, X_pred, self.y_val = self.standarize_normalize(
                                                                X_train=X_train_for_val, 
                                                                X_pred=X_val, 
                                                                y_train=y_train_for_val, 
                                                                y_test=y_val
                                                                )
            del y_val
            
            # Fit the Ridge model
            XTX_reg = X_train_for_val.T @ X_train_for_val + torch.tensor(self.alpha, dtype=torch.float32) *  torch.eye(X_train_for_val.shape[1], device=self.device) # X^T * X + alpha*I
            mtrfs = torch.linalg.solve(XTX_reg, X_train_for_val.T @ y_train_for_val)
            
            # Perform predictions
            self.y_predicted = X_pred @ mtrfs
            del X_pred
            
            # Store mtrfs
            self.coefs = mtrfs.view(n_features, len(config.delays), mtrfs.shape[-1]).permute(2, 0, 1).cpu().numpy()
            del X_train_for_val, y_train_for_val
    def predict(
        self
        )->tuple:
        """
        Predict the EEG response using the fitted TorchMtrf model.

        This method returns the predicted EEG response for the test data. If validation is enabled,
        it returns the predicted response for the validation set; otherwise, it returns the predicted
        response for the test set.

        Returns
        -------
        tuple
            A tuple containing the predicted response and the corresponding true response.
            - If validation is enabled: (predicted_response, validation_response)
            - If validation is not enabled: (predicted_response, test_response)

        Raises
        ------
        ValueError
            If the model has not been fitted before calling this method.
        """
        
        if self.validation:
            return self.y_predicted.cpu().detach().numpy(), self.y_val.cpu().detach().numpy()
        else:
            return self.y_predicted.cpu().detach().numpy(), self.y_test.cpu().detach().numpy()
    
    def standarize_normalize(
        self, 
        X_train:np.ndarray, 
        X_pred:np.ndarray, 
        y_train:np.ndarray, 
        y_test:np.ndarray
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
        if self.eeg_preprocess=='Standarize':
            y_train=estandar.fit_standarize_train(train_data=y_train)
            y_test=estandar.fit_standarize_test(test_data=y_test)
        if self.eeg_preprocess=='Normalize':
            y_train=norm.fit_normalize_percent(data=y_train)
            y_test=norm.fit_normalize_test(test_data=y_test)
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

#====================#    
#=== OTHER MODELS ===#
#====================#

# class ManualRidge(Ridge):
#     def __init__(self, delays:np.ndarray, relevant_indexes:np.ndarray, train_indexes:np.ndarray, test_indexes:np.ndarray,
#                 stims_preprocess:str='Normalize', eeg_preprocess:str='Standarize',alpha:float=1, fit_intercept:bool=False):
#         self.delays = delays
#         self.relevant_indexes = relevant_indexes
#         self.train_indexes = train_indexes
#         self.test_indexes = test_indexes
#         self.stims_preprocess = stims_preprocess
#         self.eeg_preprocess = eeg_preprocess
#         super().__init__(alpha=alpha, fit_intercept=fit_intercept)
        
#     def normalization(self, X, y):
        
#         # Construct shifted matrix
#         shifted_features = processing.shifted_matrix(features=X, delays=self.delays)# samples, [epochs,features], delays
        
#         # Keep relevant indexes
#         X_ = shifted_features[self.relevant_indexes] # relevant_samples, [epochs,features], delays
#         y_ = y[self.relevant_indexes] 

        
#         # Keep training indexes
#         X_train = X_[self.train_indexes]
#         y_train = y_[self.train_indexes]
#         X_pred = X_[self.test_indexes]
#         y_test = y_[self.test_indexes]

#         # Standarize and normalize
#         X_train, y_train, X_pred, y_test = self.standarize_normalize(X_train=X_train, X_pred=X_pred, y_train=y_train, y_test=y_test)
#         return X_train, y_train, X_pred, y_test

#     def fit(self, X_t, y_t):
#         super().fit(X=X_t, y=y_t)
    
#     def predict(self, X_p):
#         return super().predict(X=X_p)
    
#     def standarize_normalize(self, X_train:np.ndarray, X_pred:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
#         """Standarize|Normalize training and test data.
#         Parameters
#         ----------
#         X_train : np.ndarray
#             Fatures to be normalized. Its dimensions should be samples x features 
#         y_train : np.ndarray
#             EEG samples to be normalized. Its dimensions should be samples x features

#         Returns
#         -------
#         _type_
#             _description_
#         """
#         # Instances of normalize and standarize
#         norm = processing.Normalize(axis=0, porcent=5)
#         estandar = processing.Standarize(axis=0)
        
#         # Normalize|Standarize data 
#         if self.stims_preprocess=='Standarize':
#             # for i in range(X_train.shape[1]):
#             #     estandar.fit_standarize_train(train_data=X_train[:,i]) 
#             #     estandar.fit_standarize_test(test_data=X_pred[:,i])
#             X_train=estandar.fit_standarize_train(train_data=X_train) 
#             X_pred=estandar.fit_standarize_test(test_data=X_pred)
#         if self.stims_preprocess=='Normalize':
#             # for i in range(X_train.shape[1]):
#             #     norm.fit_normalize_train(train_data=X_train[:,i]) 
#             #     norm.fit_normalize_test(test_data=X_pred[:,i])
#             X_train=norm.fit_normalize_train(train_data=X_train) 
#             X_pred=norm.fit_normalize_test(test_data=X_pred)
#         if self.eeg_preprocess=='Standarize':
#             y_train=estandar.fit_standarize_train(train_data=y_train)
#             y_test=estandar.fit_standarize_test(test_data=y_test)
#         if self.eeg_preprocess=='Normalize':
#             y_train=norm.fit_normalize_percent(data=y_train)
#             y_test=norm.fit_normalize_test(test_data=y_test)
#         return X_train, y_train, X_pred, y_test