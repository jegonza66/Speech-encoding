import numpy as np
from mne.decoding import ReceptiveField
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
import Processing

class Ridge_model:
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.model = Ridge(self.alpha)
        
    def fit(self, dstims_train_val, eeg_train_val):  
        self.model.fit(dstims_train_val, eeg_train_val)
        self.coefs = self.model.coef_
    
    def predict(self, dstims_test):   
        predicted = self.model.predict(dstims_test)
        return predicted

class RidgeRegression(Ridge):
    def __init__(self, relevant_indexes_train:np.ndarray=None, relevant_indexes_test:np.ndarray=None, Stims_preprocess:str='Normalize', EEG_preprocess:str='Standarize', alpha=1.0):
        super().__init__(alpha)
        self.relevant_indexes_train = relevant_indexes_train
        self.relevant_indexes_test = relevant_indexes_test
        self.stims_preprocess=Stims_preprocess
        self.eeg_preprocess=EEG_preprocess

    def fit(self, X, y):
        # Keep just relevant indexes
        X_train=X[self.relevant_indexes_train]        
        y_train=y[self.relevant_indexes_train]     
        X_test=X[self.relevant_indexes_test]        
        y_test=y[self.relevant_indexes_test]     
        
        # Standarize and normalize
        eeg_train_val, eeg_test, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg_train_val=y_train,
                                        eeg_test=y_test,
                                        dstims_train_val=X_train,
                                        dstims_test=X_test,
                                        self.stims_preprocess,
                                        self.eeg_preprocess,
                                        axis=0,
                                        porcent=5)
        return super().fit(X, y)
    
    def predict(self, X):
        X=X[self.relevant_indexes_test]
        return super().predict(X)

class MNE_MTRF:
    def __init__(self, tmin, tmax, sr, alpha, relevant_indexes_train, relevant_indexes_test, Stims_preprocess, EEG_preprocess):
        self.sr = sr
        self.rf = ReceptiveField(tmin, tmax, sr,
            estimator=RidgeRegression(
                alpha=alpha,
                relevant_indexes_train=relevant_indexes_train,
                relevant_indexes_test=relevant_indexes_test,
                Stims_preprocess, 
                EEG_preprocess), 
            scoring='corrcoef', 
            verbose=False)

    def fit(self, stims, eeg):
        self.rf.fit(stims, eeg)
        self.coefs = self.rf.coef_[:, 0, :]

    def predict(self, dstims_test):
        predicted = self.rf.predict(dstims_test)
        return predicted


class mne_mtrf_decoding:

    def __init__(self, tmin, tmax, sr, info, alpha, t_lag):
        self.sr = sr
        self.t_lag = t_lag
        self.rf = ReceptiveField(tmin, tmax, sr, feature_names=info.ch_names, estimator=alpha, scoring='corrcoef',
                                 patterns=True, verbose=False)

    def fit(self, eeg_train_val, dstims_train_val):
        stim = dstims_train_val[:, self.t_lag]
        stim = stim.reshape([stim.shape[0], 1])
        self.rf.fit(eeg_train_val, stim)
        self.coefs = self.rf.coef_[0, :, :]
        self.patterns = self.rf.patterns_[0, :, :]

    def predict(self, eeg_test):
        predicted = self.rf.predict(eeg_test)
        return predicted

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
