# Standard libraries
import numpy as np, os

# Specific libraries
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from cuml.linear_model import Ridge as CumlRidge
from sklearn.multioutput import MultiOutputRegressor
from cuml.preprocessing import MinMaxScaler, StandardScaler
import cupy as cp

# Modules
from processing import shifted_matrix
import config
    
    
class CumlRidgeRegression:
    def __init__(
        self, 
        alpha: float, 
        relevant_indexes: np.ndarray, 
        train_indexes: np.ndarray, 
        test_indexes: np.ndarray, 
        stims_preprocess: str, 
        eeg_preprocess: str, 
        fit_intercept: bool=False, 
        shuffle: bool=False, 
        validation: bool=False
        ):
        """
        Initialize the CumlRidgeRegression model.

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

        Returns
        -------
        None
        """
        self.alpha = alpha
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.validation = validation

    def fit(self, stims: np.ndarray, eeg: np.ndarray):
        """
        Fit the CumlRidgeRegression model to the given stimuli and EEG data.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features*n_delays).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray
            Coefficients of the model
        """
        # Construct design matrix
        design_matrix = shifted_matrix(stims, delays=config.delays, use_gpu=True)
        n_samples, n_features, n_delays = design_matrix.shape
        design_matrix = design_matrix.reshape(n_samples, n_features*n_delays)

        # Get relevant indexes
        X_temp = design_matrix[self.relevant_indexes]
        del design_matrix
        y_temp = eeg[self.relevant_indexes]
        
        # Separate into training and testing
        X_train = X_temp[self.train_indexes]
        y_train = y_temp[self.train_indexes]
        X_pred = X_temp[self.test_indexes]
        y_test = y_temp[self.test_indexes]
        del X_temp, y_temp
        
        if not self.validation:
            # Shuffle the data if required for random permutations
            if self.shuffle:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train = X_train[indices]
            
            # Standarize and normalize
            X_train, y_train, self.X_pred, self.y_test = self.standarize_normalize(
                                                        X_train=X_train, 
                                                        X_pred=X_pred, 
                                                        y_train=y_train, 
                                                        y_test=y_test
                                                        )
            del X_pred, y_test
            
            # Transfer data to GPU
            X_train = cp.asarray(X_train)
            y_train = cp.asarray(y_train)

            # Fit the Ridge model
            self.model = MultiOutputRegressor(CumlRidge(alpha=self.alpha, fit_intercept=self.fit_intercept))
            self.model.fit(X_train, y_train)
            weights_unordered = cp.vstack([model.coef_ for model in self.model.estimators_])
            
            return cp.asnumpy(weights_unordered.reshape(weights_unordered.shape[0], n_features, n_delays))
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
            X_train_for_val, y_train_for_val, self.X_pred, self.y_val = self.standarize_normalize(
                                                                                X_train=X_train_for_val, 
                                                                                X_pred=X_val, 
                                                                                y_train=y_train_for_val, 
                                                                                y_test=y_val
                                                                                )
            del X_val, y_val
            
            # Transfer data to GPU
            X_train_for_val = cp.asarray(X_train_for_val)
            y_train_for_val = cp.asarray(y_train_for_val)

            # Fit the Ridge model
            self.model = MultiOutputRegressor(CumlRidge(alpha=self.alpha, fit_intercept=self.fit_intercept))
            self.model.fit(X_train_for_val, y_train_for_val)
            weights_unordered = cp.vstack([model.coef_ for model in self.model.estimators_])
            
            return cp.asnumpy(weights_unordered.reshape(weights_unordered.shape[0], n_features, n_delays))
        
    def predict(self) -> np.ndarray:
        """
        Predict the EEG response for the given stimuli data.

        Parameters
        ----------
            None
        Returns
        -------
        np.ndarray
            Predicted response, shape (n_samples, n_channels).
        """
        # Transfer data to GPU
        self.X_pred = cp.asarray(self.X_pred)
        
        if self.validation:
            return cp.asnumpy(self.model.predict(self.X_pred)), self.y_val
        else:
            return cp.asnumpy(self.model.predict(self.X_pred)), self.y_test
    
    def standarize_normalize(
        self, X_train:np.ndarray, X_pred:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
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
        # norm = Normalize(axis=0, porcent=5)
        # estandar = Standarize(axis=0)
        norm = MinMaxScaler()
        estandar = StandardScaler()
        
        # Iterates to normalize|standarize over features
        if self.stims_preprocess=='Standarize':
            _  = estandar.fit(X=X_train)
            X_train = estandar.transform(X=X_train)
            X_pred = estandar.transform(X=X_pred)
        if self.stims_preprocess=='Normalize':
            _  = norm.fit(X=X_train)
            X_train = norm.transform(X=X_train)
            X_pred = norm.transform(X=X_pred)
        if self.eeg_preprocess=='Standarize':
            _  = estandar.fit(X=y_train)
            y_train = estandar.transform(X=y_train)
            y_test = estandar.transform(X=y_test)
        if self.eeg_preprocess=='Normalize':
            _  = norm.fit(X=y_train)
            y_train = norm.transform(X=y_train)
            y_test = norm.transform(X=y_test)
        return X_train, y_train, X_pred, y_test