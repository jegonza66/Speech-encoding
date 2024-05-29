import numpy as np
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.linear_model import Ridge
import Processing

class ManualRidge(Ridge):
    def __init__(self, delays:np.ndarray, relevant_indexes:np.ndarray, train_indexes:np.ndarray, test_indexes:np.ndarray,
                stims_preprocess:str='Normalize', eeg_preprocess:str='Standarize',alpha:float=1, fit_intercept:bool=False):
        self.delays = delays
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)
        
    def normalization(self, X, y):
        
        # Construct shifted matrix
        shifted_features = Processing.shifted_matrix(features=X, delays=self.delays)# samples, [epochs,features], delays
        
        # Keep relevant indexes
        X_ = shifted_features[self.relevant_indexes] # relevant_samples, [epochs,features], delays
        y_ = y[self.relevant_indexes] 

        
        # Keep training indexes
        X_train = X_[self.train_indexes]
        y_train = y_[self.train_indexes]
        X_pred = X_[self.test_indexes]
        y_test = y_[self.test_indexes]

        # Standarize and normalize
        X_train, y_train, X_pred, y_test = self.standarize_normalize(X_train=X_train, X_pred=X_pred, y_train=y_train, y_test=y_test)
        return X_train, y_train, X_pred, y_test

    def fit(self, X_t, y_t):
        super().fit(X=X_t, y=y_t)
    
    def predict(self, X_p):
        return super().predict(X=X_p)
    
    def standarize_normalize(self, X_train:np.ndarray, X_pred:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
        """Standarize|Normalize training and test data.
        Parameters
        ----------
        X_train : np.ndarray
            Fatures to be normalized. Its dimensions should be samples x features #TODO rever
        y_train : np.ndarray
            EEG samples to be normalized. Its dimensions should be samples x features

        Returns
        -------
        _type_
            _description_
        """
        # Instances of normalize and standarize
        norm = Processing.Normalizar(axis=0, porcent=5)
        estandar = Processing.Estandarizar(axis=0)
        
        # Normalize|Standarize data # TODO DISYUNTIVA: normalizar cada feature x separado o todo junto? hasta ahora x columnas, incluso si dichas cols pertenecen al mismo feature. Tal vez cambiar dimension para diferenciar entre features y que no queden aglutinadas.
        if self.stims_preprocess=='Standarize':
            # for i in range(X_train.shape[1]):
            #     estandar.fit_standarize_train(train_data=X_train[:,i]) 
            #     estandar.fit_standarize_test(test_data=X_pred[:,i])
            X_train=estandar.fit_standarize_train(train_data=X_train) 
            X_pred=estandar.fit_standarize_test(test_data=X_pred)
        if self.stims_preprocess=='Normalize':
            # for i in range(X_train.shape[1]):
            #     norm.fit_normalize_train(train_data=X_train[:,i]) 
            #     norm.fit_normalize_test(test_data=X_pred[:,i])
            X_train=norm.fit_normalize_train(train_data=X_train) 
            X_pred=norm.fit_normalize_test(test_data=X_pred)
        if self.eeg_preprocess=='Standarize':
            y_train=estandar.fit_standarize_train(train_data=y_train)
            y_test=estandar.fit_standarize_test(test_data=y_test)
        if self.eeg_preprocess=='Normalize':
            y_train=norm.fit_normalize_percent(data=y_train)
            y_test=norm.fit_normalize_test(test_data=y_test)
        return X_train, y_train, X_pred, y_test

class RidgeRegression(Ridge):
    def __init__(self, relevant_indexes:np.ndarray=None, train_indexes:np.ndarray=None, test_indexes:np.ndarray=None, 
                stims_preprocess:str='Normalize', eeg_preprocess:str='Standarize', alpha=1.0, fit_intercept=False):
        """_summary_

        Parameters
        ----------
        relevant_indexes_train : np.ndarray, optional
            _description_, by default None
        relevant_indexes_test : np.ndarray, optional
            _description_, by default None
        stims_preprocess : str, optional
            _description_, by default 'Normalize'
        eeg_preprocess : str, optional
            _description_, by default 'Standarize'
        alpha : float, optional
            _description_, by default 1.0
        """
        # super().__init__(tmin=tmin, tmax=tmax, sfreq=sfreq, alpha=alpha, fit_intercept=fit_intercept)
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)

        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess = stims_preprocess
        self.eeg_preprocess = eeg_preprocess

    def fit(self, X, y):
        # Get relevant indexes
        X_, y_= X[self.relevant_indexes], y[self.relevant_indexes] # relevant_samples relevant_samples, features*delays, antes relevant_samples, [epochs,features], delays
        del X, y

        # Make split
        X_train = X_[self.train_indexes] 
        X_pred = X_[self.test_indexes]
        y_train = y_[self.train_indexes]
        y_test = y_[self.test_indexes]
        del X_, y_
        
        # Standarize and normalize
        X_train, y_train, self.X_pred, self.y_test = self.standarize_normalize(X_train=X_train, 
                                                                               X_pred=X_pred, 
                                                                               y_train=y_train, 
                                                                               y_test=y_test)

        return super().fit(X_train, y_train)
    
    def predict(self, X):
        y_pred = super().predict(self.X_pred) #samples,nchans
        n_samples = X.shape[0]
        
        # Padd with zeros to make it compatible with shape desired by mne.ReceptiveField.predict()
        y_pred_0 = np.zeros(shape = (n_samples, y_pred.shape[-1]))
        y_pred_0[self.test_indexes] = y_pred

        # When used relevant indexes must be filtered once again
        return y_pred_0
    
    def standarize_normalize(self, X_train:np.ndarray, X_pred:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
        """Standarize|Normalize training and test data.
        Parameters
        ----------
        X_train : np.ndarray
            Fatures to be normalized. Its dimensions should be samples x features #TODO rever
        y_train : np.ndarray
            EEG samples to be normalized. Its dimensions should be samples x features

        Returns
        -------
        _type_
            _description_
        """
        # Instances of normalize and standarize
        norm = Processing.Normalizar(axis=0, porcent=5)
        estandar = Processing.Estandarizar(axis=0)
        
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

class MNE_MTRF:
    def __init__(self, tmin:float, tmax:float, sample_rate:int, alpha:float, 
                 relevant_indexes:np.ndarray, train_indexes:np.ndarray, test_indexes:np.ndarray, 
                 stims_preprocess:str, eeg_preprocess:str, fit_intercept:bool=False):
        self.test_indexes = test_indexes
        self.sample_rate = sample_rate
        self.rf = ReceptiveField(
            tmin=tmin, 
            tmax=tmax, 
            sfreq=sample_rate,
            estimator=RidgeRegression(alpha=alpha,
                                      relevant_indexes=relevant_indexes,
                                      train_indexes=train_indexes,
                                      test_indexes=test_indexes,
                                      stims_preprocess=stims_preprocess, 
                                      eeg_preprocess=eeg_preprocess,
                                      fit_intercept=fit_intercept), 
            scoring='corrcoef', 
            verbose=False)
   
    def fit(self, stims, eeg):
        self.rf.fit(stims, eeg)
        self.coefs = self.rf.coef_ # n_chanels, n_feats, n_delays

    def predict(self, stims):
        predicted = self.rf.predict(stims)
        return predicted[self.test_indexes], self.rf.estimator_.y_test
    