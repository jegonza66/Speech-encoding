import numpy as np
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
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
        shifted_features = Processing.shifted_matrix(features=X, delays=self.delays)[:,0,:] # samples, [epochs,features], delays
        
        # Keep relevant indexes
        X_ = shifted_features[self.relevant_indexes]
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
            estandar.fit_standarize_train(train_data=X_train) 
            estandar.fit_standarize_test(test_data=X_pred)
        if self.stims_preprocess=='Normalize':
            # for i in range(X_train.shape[1]):
            #     norm.fit_normalize_train(train_data=X_train[:,i]) 
            #     norm.fit_normalize_test(test_data=X_pred[:,i])
            norm.fit_normalize_train(train_data=X_train) 
            norm.fit_normalize_test(test_data=X_pred)
        if self.eeg_preprocess=='Standarize':
            estandar.fit_standarize_train(train_data=y_train)
            estandar.fit_standarize_test(test_data=y_test)
        if self.eeg_preprocess=='Normalize':
            norm.fit_normalize_percent(data=y_train)
            norm.fit_normalize_test(test_data=y_test)
        return X_train, y_train, X_pred, y_test


class RidgeRegression(TimeDelayingRidge):
    def __init__(self, relevant_indexes:np.ndarray=None, train_indexes:np.ndarray=None, test_indexes:np.ndarray=None, 
                stims_preprocess:str='Normalize', eeg_preprocess:str='Standarize', tmin:float=-.2, tmax:float=.6, 
                sfreq:int=128, alpha=1.0, fit_intercept=False):
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
        super().__init__(tmin=tmin, tmax=tmax, sfreq=sfreq, alpha=alpha, fit_intercept=fit_intercept)
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.stims_preprocess=stims_preprocess
        self.eeg_preprocess=eeg_preprocess

    def fit(self, X, y):
        # Keep just relevant indexes. Epoch 0
        X_train=X[self.relevant_indexes, 0, :]        
        y_train=y[self.relevant_indexes, 0, :]   
        X_train=X_train[self.train_indexes]        
        y_train=y_train[self.train_indexes]   

        # Instances of normalize and standarize
        self.norm = Processing.Normalizar(axis=0, porcent=5)
        self.estandar = Processing.Estandarizar(axis=0)
        
        # Standarize and normalize
        X_train, y_train = self.standarize_normalize(X=X_train, y=y_train)

        # Reshsape in desire mne shape with middle dimenssion for epoch
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[-1])
        return super().fit(X_train, y_train)
    
    def predict(self, X):
        total_samples = X.shape[0]
        
        # Keep just relevant indexes. Epoch 0
        X_test = X[self.relevant_indexes, 0 , :] 
        X_test = X_test[self.test_indexes, :] 
        X_test = self.standarize_normalize(X=X_test, train=False)
        y_pred = super().predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[-1]))
        
        # Padd with zeros to make it compatible with shape desired by mne.ReceptiveField.predict()
        y_pred_0 = np.zeros(shape = (total_samples, 1, y_pred.shape[-1]))
        y_pred_0[self.test_indexes] = y_pred

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
                                      fit_intercept=False), 
            scoring='corrcoef', 
            verbose=False)

    def fit(self, stims, eeg):
        self.rf.fit(stims, eeg)
        self.coefs = self.rf.coef_[:, 0, :]
        # return self.coefs

    def predict(self, stims):
        predicted = self.rf.predict(stims)
        return predicted[self.test_indexes]
    


# class RidgeRegression(Ridge):
#     def __init__(self, relevant_indexes_train:np.ndarray=None, relevant_indexes_test:np.ndarray=None, 
#                  stims_preprocess:str='Normalize', eeg_preprocess:str='Standarize', alpha=1.0, fit_intercept=False):
#         """_summary_

#         Parameters
#         ----------
#         relevant_indexes_train : np.ndarray, optional
#             _description_, by default None
#         relevant_indexes_test : np.ndarray, optional
#             _description_, by default None
#         stims_preprocess : str, optional
#             _description_, by default 'Normalize'
#         eeg_preprocess : str, optional
#             _description_, by default 'Standarize'
#         alpha : float, optional
#             _description_, by default 1.0
#         """
#         super().__init__(alpha, fit_intercept=fit_intercept)
#         self.relevant_indexes_train = relevant_indexes_train
#         self.relevant_indexes_test = relevant_indexes_test
#         self.stims_preprocess=stims_preprocess
#         self.eeg_preprocess=eeg_preprocess

#     def fit(self, X, y):
#         # Keep just relevant indexes
#         X_train=X[self.relevant_indexes_train]        
#         y_train=y[self.relevant_indexes_train]   

#         # Instances of normalize and standarize
#         self.norm = Processing.Normalizar(axis=0, porcent=5)
#         self.estandar = Processing.Estandarizar(axis=0)
        
#         # Standarize and normalize
#         X_train, y_train= self.standarize_normalize(X=X_train, y=y_train)

#         return super().fit(X_train, y_train)
    
#     def predict(self, X):
#         total_samples = X.shape[0]
#         X_test = X[self.relevant_indexes_test]
#         X_test = self.standarize_normalize(X=X_test, train=False)
#         y_pred = super().predict(X_test)

#         # Padd with zeros to make it compatible with shape desired by mne.ReceptiveField.predict()
#         y_pred_0 = np.zeros(shape=(total_samples, y_pred.shape[1]))
#         y_pred_0[self.relevant_indexes_test] = y_pred

#         # When used relevant indexes must be filtered once again
#         return y_pred_0
    
#     def standarize_normalize(self, X:np.ndarray, y:np.ndarray=None, train:bool=True):
#         """Standarize|Normalize training and test data.
#         Parameters
#         ----------
#         X_train : np.ndarray
#             Fatures to be normalized. Its dimensions should be samples x features #TODO rever
#         y_train : np.ndarray
#             EEG samples to be normalized. Its dimensions should be samples x features

#         Returns
#         -------
#         _type_
#             _description_
#         """
#         # Normalize|Standarize data # TODO DISYUNTIVA: normalizar cada feature x separado o todo junto? hasta ahora x columnas, incluso si dichas cols pertenecen al mismo feature. Tal vez cambiar dimension para diferenciar entre features y que no queden aglutinadas.
#         if self.stims_preprocess=='Standarize':
#             for i in range(X.shape[1]):
#                 self.estandar.fit_standarize_train(train_data=X[:,i])
#         if self.stims_preprocess=='Normalize':
#             for i in range(X.shape[1]):
#                 self.norm.fit_normalize_train(train_data=X[:,i])
#         if train:
#             if self.eeg_preprocess=='Standarize':
#                 self.estandar.fit_standarize_train(train_data=y)
#             if self.eeg_preprocess=='Normalize':
#                 self.norm.fit_normalize_percent(data=y)
#             return X, y
#         else:
#             return X

# class RidgeRegression(TimeDelayingRidge):
#     def __init__(self, relevant_indexes_train:np.ndarray=None, relevant_indexes_test:np.ndarray=None, 
#                 stims_preprocess:str='Normalize', eeg_preprocess:str='Standarize', tmin:float=-.2, tmax:float=.6, sfreq:int=128, alpha=1.0, fit_intercept=False):
#         """_summary_

#         Parameters
#         ----------
#         relevant_indexes_train : np.ndarray, optional
#             _description_, by default None
#         relevant_indexes_test : np.ndarray, optional
#             _description_, by default None
#         stims_preprocess : str, optional
#             _description_, by default 'Normalize'
#         eeg_preprocess : str, optional
#             _description_, by default 'Standarize'
#         alpha : float, optional
#             _description_, by default 1.0
#         """
#         super().__init__(tmin, tmax, sfreq, alpha, fit_intercept=fit_intercept)
#         self.relevant_indexes_train = relevant_indexes_train
#         self.relevant_indexes_test = relevant_indexes_test
#         self.stims_preprocess=stims_preprocess
#         self.eeg_preprocess=eeg_preprocess

#     def fit(self, X, y):
#         # Keep just relevant indexes. Epoch 0
#         X_train=X[self.relevant_indexes_train, 0, :]        
#         y_train=y[self.relevant_indexes_train, 0, :]   

#         # Instances of normalize and standarize
#         self.norm = Processing.Normalizar(axis=0, porcent=5)
#         self.estandar = Processing.Estandarizar(axis=0)
        
#         # Standarize and normalize
#         X_train, y_train = self.standarize_normalize(X=X_train, y=y_train)

#         # Reshsape in desire mne shape with middle dimenssion for epoch
#         X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
#         y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[-1])
#         return super().fit(X_train, y_train)
    
#     def predict(self, X):
#         total_samples = X.shape[0]
        
#         # Keep just relevant indexes. Epoch 0
#         X_test = X[self.relevant_indexes_test, 0, :] 
#         X_test = self.standarize_normalize(X=X_test, train=False)
#         y_pred = super().predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[-1]))
        
#         # Padd with zeros to make it compatible with shape desired by mne.ReceptiveField.predict()
#         y_pred_0 = np.zeros(shape = (total_samples, 1, y_pred.shape[-1]))
#         y_pred_0[self.relevant_indexes_test] = y_pred

#         # When used relevant indexes must be filtered once again
#         return y_pred_0
    
#     def standarize_normalize(self, X:np.ndarray, y:np.ndarray=None, train:bool=True):
#         """Standarize|Normalize training and test data.
#         Parameters
#         ----------
#         X_train : np.ndarray
#             Fatures to be normalized. Its dimensions should be samples x features #TODO rever
#         y_train : np.ndarray
#             EEG samples to be normalized. Its dimensions should be samples x features

#         Returns
#         -------
#         _type_
#             _description_
#         """
#         # Normalize|Standarize data # TODO DISYUNTIVA: normalizar cada feature x separado o todo junto? hasta ahora x columnas, incluso si dichas cols pertenecen al mismo feature. Tal vez cambiar dimension para diferenciar entre features y que no queden aglutinadas.
#         if self.stims_preprocess=='Standarize':
#             for i in range(X.shape[1]):
#                 self.estandar.fit_standarize_train(train_data=X[:,i])
#         if self.stims_preprocess=='Normalize':
#             for i in range(X.shape[1]):
#                 self.norm.fit_normalize_train(train_data=X[:,i])
#         if train:
#             if self.eeg_preprocess=='Standarize':
#                 self.estandar.fit_standarize_train(train_data=y)
#             if self.eeg_preprocess=='Normalize':
#                 self.norm.fit_normalize_percent(data=y)
#             return X, y
#         else:
#             return X

# class MNE_MTRF:
#     def __init__(self, tmin:float, tmax:float, sample_rate:int, alpha:float, 
#                  relevant_indexes_train:np.ndarray, relevant_indexes_test:np.ndarray, 
#                  stims_preprocess:str, eeg_preprocess:str, fit_intercept:bool=False):
#         self.relevant_indexes_test = relevant_indexes_test
#         self.sample_rate = sample_rate
#         self.rf = ReceptiveField(
#             tmin=tmin, 
#             tmax=tmax, 
#             sfreq=sample_rate,
#             estimator=RidgeRegression(alpha=alpha,
#                                       relevant_indexes_train=relevant_indexes_train,
#                                       relevant_indexes_test=relevant_indexes_test,
#                                       stims_preprocess=stims_preprocess, 
#                                       eeg_preprocess=eeg_preprocess,
#                                       fit_intercept=False), 
#             scoring='corrcoef', 
#             verbose=False)

#     def fit(self, stims, eeg):
#         self.rf.fit(stims, eeg)
#         self.coefs = self.rf.coef_[:, 0, :]
#         # return self.coefs

#     def predict(self, stims):
#         predicted = self.rf.predict(stims)
#         return predicted[self.relevant_indexes_test]
