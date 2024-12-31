import numpy as np
from cuml.linear_model import Ridge as CumlRidge
from cuml.preprocessing import StandardScaler
    
class CumlRidgeRegression:
    def __init__(
        self, 
        tmin: float, 
        tmax: float, 
        sample_rate: int, 
        alpha: float, 
        relevant_indexes: np.ndarray, 
        train_indexes: np.ndarray, 
        test_indexes: np.ndarray, 
        stims_preprocess: str, 
        eeg_preprocess: str, 
        n_jobs: int=-1, 
        fit_intercept: bool=False, 
        shuffle: bool=False, 
        validation: bool=False
        ):
        """
        Initialize the CumlRidgeRegression model.

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
        """
        self.tmin = tmin
        self.tmax = tmax
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess
        self.n_jobs = n_jobs
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
        None
        """
        # Construct design matrix

        # Preprocess data
        X = stims[self.relevant_indexes][self.train_indexes]
        y = eeg[self.relevant_indexes][self.train_indexes]

        if self.shuffle:
            # Shuffle the data if required
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Fit the Ridge model
        self.model = CumlRidge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.model.fit(X, y)

    def predict(self, stims: np.ndarray) -> np.ndarray:
        """
        Predict the EEG response for the given stimuli data.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features*n_delays).

        Returns
        -------
        np.ndarray
            Predicted response, shape (n_samples, n_channels).
        """
        X_pred = stims[self.relevant_indexes][self.test_indexes]
        X_pred = StandardScaler().fit_transform(X_pred)  # Standardize the prediction data
        return self.model.predict(X_pred)