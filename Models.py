import numpy as np
from mne.decoding import ReceptiveField
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
import Processing

class RidgeRegression(Ridge):
    def __init__(self, relevant_indexes_train:np.ndarray=None, relevant_indexes_test:np.ndarray=None, 
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
        super().__init__(alpha, fit_intercept=fit_intercept)
        self.relevant_indexes_train = relevant_indexes_train
        self.relevant_indexes_test = relevant_indexes_test
        self.stims_preprocess=stims_preprocess
        self.eeg_preprocess=eeg_preprocess

    def fit(self, X, y):
        # Keep just relevant indexes
        X_train=X[self.relevant_indexes_train]        
        y_train=y[self.relevant_indexes_train]   

        # Instances of normalize and standarize
        self.norm = Processing.Normalizar(axis=0, porcent=5)
        self.estandar = Processing.Estandarizar(axis=0)
        
        # Standarize and normalize
        X_train, y_train= self.standarize_normalize(X=X_train, y=y_train)

        return super().fit(X_train, y_train)
    
    def predict(self, X):
        total_samples = X.shape[0]
        X_test = X[self.relevant_indexes_test]
        X_test = self.standarize_normalize(X=X_test, train=False)
        y_pred = super().predict(X_test)

        # Padd with zeros to make it compatible with shape desired by mne.ReceptiveField.predict()
        y_pred_0 = np.zeros(shape=(total_samples, y_pred.shape[1]))
        y_pred_0[self.relevant_indexes_test] = y_pred

        # When used relevant indexes must be filtered once again
        return y_pred_0
    
    def standarize_normalize(self, X:np.ndarray, y:np.ndarray=None, train:bool=True):
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
        # Normalize|Standarize data # TODO DISYUNTIVA: normalizar cada feature x separado o todo junto? hasta ahora x columnas, incluso si dichas cols pertenecen al mismo feature. Tal vez cambiar dimension para diferenciar entre features y que no queden aglutinadas.
        if self.stims_preprocess=='Standarize':
            for i in range(X.shape[1]):
                self.estandar.fit_standarize_train(train_data=X[:,i])
        if self.stims_preprocess=='Normalize':
            for i in range(X.shape[1]):
                self.norm.fit_normalize_train(train_data=X[:,i])
        if train:
            if self.eeg_preprocess=='Standarize':
                self.estandar.fit_standarize_train(train_data=y)
            if self.eeg_preprocess=='Normalize':
                self.norm.fit_normalize_percent(data=y)
            return X, y
        else:
            return X

class MNE_MTRF:
    def __init__(self, tmin:float, tmax:float, sample_rate:int, alpha:float, 
                 relevant_indexes_train:np.ndarray, relevant_indexes_test:np.ndarray, 
                 stims_preprocess:str, eeg_preprocess:str, fit_intercept:bool=False):
        self.relevant_indexes_test = relevant_indexes_test
        self.sample_rate = sample_rate
        self.rf = ReceptiveField(
            tmin=tmin, 
            tmax=tmax, 
            sfreq=sample_rate,
            estimator=RidgeRegression(alpha=alpha,
                                      relevant_indexes_train=relevant_indexes_train,
                                      relevant_indexes_test=relevant_indexes_test,
                                      stims_preprocess=stims_preprocess, 
                                      eeg_preprocess=eeg_preprocess,
                                      fit_intercept=False), 
            scoring='corrcoef', 
            verbose=False)

    def fit(self, stims, eeg):
        self.rf.fit(stims, eeg)
        self.coefs = self.rf.coef_[:, 0, :]
        return self.coefs

    def predict(self, stims):
        predicted = self.rf.predict(stims)
        return predicted[self.relevant_indexes_test]


# class mne_mtrf_decoding:

#     def __init__(self, tmin, tmax, sr, info, alpha, t_lag):
#         self.sr = sr
#         self.t_lag = t_lag
#         self.rf = ReceptiveField(tmin, tmax, sr, feature_names=info.ch_names, estimator=alpha, scoring='corrcoef',
#                                  patterns=True, verbose=False)

#     def fit(self, eeg_train_val, dstims_train_val):
#         stim = dstims_train_val[:, self.t_lag]
#         stim = stim.reshape([stim.shape[0], 1])
#         self.rf.fit(eeg_train_val, stim)
#         self.coefs = self.rf.coef_[0, :, :]
#         self.patterns = self.rf.patterns_[0, :, :]

#     def predict(self, eeg_test):
#         predicted = self.rf.predict(eeg_test)
#         return predicted

#
# class mne_mtrf_decoding_inicial:
#
#     def __init__(self, tmin, tmax, sr, info, alpha):
#         self.sr = sr
#         self.rf = ReceptiveField(tmin, tmax, sr, feature_names=info.ch_names, estimator=alpha, scoring='corrcoef', patterns=True)
#
#     def fit(self, eeg_train_val, dstims_train_val):
#         stim = dstims_train_val[:, 0]
#         stim = stim.reshape([stim.shape[0], 1])
#         self.rf.fit(eeg_train_val, stim)
#         self.coefs = self.rf.coef_[0, :, 1:]
#         self.patterns = self.rf.patterns_[0, :, 1:]
#
#     def predict(self, eeg_test):
#         predicted = self.rf.predict(eeg_test)
#         return predicted

# class Ridge_model:
    
#     def __init__(self, alpha):
#         self.alpha = alpha
#         self.model = Ridge(self.alpha)
        
#     def fit(self, dstims_train_val, eeg_train_val):  
#         self.model.fit(dstims_train_val, eeg_train_val)
#         self.coefs = self.model.coef_
    
#     def predict(self, dstims_test):   
#         predicted = self.model.predict(dstims_test)
#         return predicted
