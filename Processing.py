import numpy as np, copy
from scipy import signal
from typing import Union

class Estandarizar():
    def __init__(self, axis:int=0):
        """Standarize train and test data to be used in a linear regressor model. 

        Parameters
        ----------
        axis : int, optional
            Axis to perform standrize, by default 0
        """
        self.axis = axis

    def fit_standarize_train(self, train_data:np.ndarray):
        """Standarize train data, also define mean and std to standarize future data.

        Parameters
        ----------
        train_data : np.ndarray
            Train data to be standarize
        """
        # Fix mean and standard deviation with train data
        self.mean = np.mean(train_data, axis=self.axis)
        self.std = np.std(train_data, axis=self.axis)

        # Standarize data
        train_data -= self.mean
        train_data /= self.std
        return train_data

    def fit_standarize_test(self, test_data:np.ndarray):
        """Standarize test data with mean and std of train data.

        Parameters
        ----------
        test_data : np.ndarray
            Test data to be standarize with mean and standard deviation of train data
        """
        # Standarize with mean and standard deviation of train
        test_data -= self.mean
        test_data /= self.std
        return test_data

    def standarize_data(self, data:np.ndarray):
        """Standarize data with own mean and standard deviation.

        Parameters
        ----------
        data : np.ndarray
            Data to be standarized
        """
        # Standarize data with own mean and standard deviation
        data -= np.mean(data, axis=self.axis)
        data /= np.std(data, axis=self.axis)
        return data

class Normalizar():
    def __init__(self, axis:int=0, porcent:float=5):
        """Normalize train and test data to be used in a linear regressor model.

        Parameters
        ----------
        axis : int, optional
            Axis to perform normalize, by default 0
        porcent : float, optional
            _description_, by default 5
        """
        self.axis = axis
        self.porcent = porcent

    def fit_normalize_train(self, train_data:np.ndarray):
        """Normalize train data, also define min and max to normalize future data.

        Parameters
        ----------
        train_data : np.ndarray
            Train data to be normalize by maximum and minimum (offset)
        """
        
        # Remove offset by minimum
        self.min = np.min(train_data, axis=self.axis)
        train_data -= self.min

        # Normalize by maximum
        self.max = np.max(train_data, axis=self.axis)
        return np.divide(train_data, self.max, out=np.zeros_like(train_data), where=self.max != 0)

    def fit_normalize_test(self, test_data:np.ndarray):
        """Normalize test data with min and max of train data.

        Parameters
        ----------
        test_data : np.ndarray
            Test data to be normalize with train data parameters
        """
        test_data -= self.min
        return np.divide(test_data, self.max, out=np.zeros_like(test_data), where=self.max != 0)

    def normalize_data(self, data:np.ndarray, kind:str="1"):
        """_summary_# TODO no queda claro para qué es la kind 2, creo que es para que esté centrada en 0

        Parameters
        ----------
        data : np.ndarray
            _description_
        kind : str, optional
            _description_, by default "1"
        """
        # Los estimulos los normalizo todos entre 0 y 1 estricto, la envolvente no tiene picos
        data -= np.min(data, axis=self.axis)
        data /= np.max(data, axis=self.axis)
        if kind=='2':
            data *= 2
            data -= 1
        return data

    def fit_normalize_percent(self, data:np.ndarray):
        """_summary_# TODO no queda claro qué es lo que sucede, creo que corta el 5 porciento de los datos hacia adelante y hacia atras y trabaja con los maximos alli descritos

        Parameters
        ----------
        data : np.ndarray
            Data to be normalize
        """
        # Find n 
        # n = int((self.porcent/100)*len(data)) 
        n = int((self.porcent * len(data) - 1) / 100) # TODO para mí va lo de arriba
        
        
        # Find the n-th minimum and offset that value
        sorted_data = copy.deepcopy(data)
        sorted_data.sort(self.axis)
        min_data_n = sorted_data[n]
        data -= min_data_n

        # Find the n-th maximum
        sorted_data = copy.deepcopy(data)
        sorted_data.sort(self.axis)
        max_data_n = sorted_data[-n]
        
        # Normalize data
        data = np.divide(data, self.max, out=np.zeros_like(data), where=max_data_n!=0)
        data /= max_data_n
        return data

def standarize_normalize(eeg_train_val, eeg_test, dstims_train_val, dstims_test, Stims_preprocess, EEG_preprocess, axis=0, porcent=5):
    norm = Normalizar(axis, porcent)
    estandar = Estandarizar(axis)

    if isinstance(dstims_train_val, list):
        if Stims_preprocess == 'Standarize':
            for i in range(len(dstims_train_val)):
                estandar.fit_standarize_train(train_data=dstims_train_val[i])
                estandar.fit_standarize_test(test_data=dstims_test[i])
            dstims_train_val = np.hstack([dstims_train_val[i] for i in range(len(dstims_train_val))])
            dstims_test = np.hstack([dstims_test[i] for i in range(len(dstims_test))])

        if Stims_preprocess == 'Normalize':
            for i in range(len(dstims_train_val)):
                norm.fit_normalize_train(train_data=dstims_train_val[i])
                norm.fit_normlize_test(test_data=dstims_test[i])
            dstims_train_val = np.hstack([dstims_train_val[i] for i in range(len(dstims_train_val))])
            dstims_test = np.hstack([dstims_test[i] for i in range(len(dstims_test))])
    else:
        if Stims_preprocess == 'Standarize':
            for i in range(dstims_train_val.shape[1]):
                estandar.fit_standarize_train(train_data=dstims_train_val[:,i])
                estandar.fit_standarize_test(test_data=dstims_test[:,i])
        if Stims_preprocess == 'Normalize':
            for i in range(dstims_train_val.shape[1]):
                norm.fit_normalize_train_data(dstims_train_val[:,i])
                norm.normlize_test_data(dstims_test[:,i])

    if EEG_preprocess == 'Standarize':
        estandar.fit_standarize_train(train_data=eeg_train_val)
        estandar.fit_standarize_test(test_data=eeg_test)
    if EEG_preprocess == 'Normalize':
        norm.fit_normalize_percent(data=eeg_train_val)
        norm.fit_normlize_test(test_data=eeg_test) # TODO OJO SE ESTA NORMALIZANDO CON EL LOS DATOS DE LOS FEATURES, EEG SOLO EN ESTE CASO

    return eeg_train_val, eeg_test, dstims_train_val, dstims_test

def shifted_matrix(features:np.ndarray, delays:np.ndarray):
    """Computes shifted matrix for a given array of delayes

    Parameters
    ----------
    features : array, shape (n_times[, n_epochs], n_features) or list of length n_times
        The time series to delay must be 2D or 3D if array.
    delays : np.ndarray
        Index delayes

    Returns
    -------
    np.ndarray
        Concatenated shifted matrix of shape samples, [epochs,features], delays
    """
    if isinstance(features, list):
        features = np.array(features)
    # Is asumed this is a one dimensional feature with number of samples as length
    shifted_matrix = np.zeros(features.shape + (len(delays),))

    for i, delay in enumerate(delays):
        # Put last elements at the begining. For ex.: feature, delay = [1,2,3,4].T, -1 --> [2,3,4,0].T
        if delay < 0:
            out = shifted_matrix[:delay, ..., i]
            use_X = features[-delay:]             
        # Put the first elements at the end. For ex.: feature, delay = [1,2,3,4].T, 1 --> [0,1,2,3].T
        elif delay > 0:
            out = shifted_matrix[delay:, ..., i]
            use_X = features[:-delay]
        # Leave it exactly the same
        else:
            out = shifted_matrix[..., i]
            use_X = features
        out[:] = use_X
    return shifted_matrix

def butter_filter(data, frecuencias, sampling_freq, btype, order, axis, ftype):
    if btype == 'lowpass' or btype == 'highpass':
        frecuencia = frecuencias / (sampling_freq / 2)
        b, a = signal.butter(order, frecuencia, btype=btype)
    elif btype == 'bandpass':
        frecuencias = [frecuencia / (sampling_freq / 2) for frecuencia in frecuencias]
        b, a = signal.butter(order, frecuencias, btype=btype)

    if ftype == 'Causal':
        y = signal.lfilter(b, a, data, axis=axis)
    elif ftype == 'NonCausal':
        y = signal.filtfilt(b, a, data, axis=axis, padlen=None)
    return y

def butter_bandpass_filter(data, frecuencia, sampling_freq, order, axis):
    frecuencia /= (sampling_freq / 2)
    b, a = signal.butter(order, frecuencia, btype='lowpass')
    y = signal.filtfilt(b, a, data, axis=axis, padlen=None)
    return y

def subsamplear(x, cada_cuanto):
    x = np.array(x)
    tomar = np.arange(0, len(x), int(cada_cuanto))
    return x[tomar]

def band_freq(band):
    if type(band) == str:

        if band == 'Delta':
            l_freq = 1
            h_freq = 4
        elif band == 'Theta':
            l_freq = 4
            h_freq = 8
        elif band == 'Alpha':
            l_freq = 8
            h_freq = 13
        elif band == 'Beta_1':
            l_freq = 13
            h_freq = 19
        elif band == 'Beta_2':
            l_freq = 19
            h_freq = 25
        elif band == 'All':
            l_freq = None
            h_freq = 40
        elif band == 'Delta_Theta':
            l_freq = 1
            h_freq = 8
        elif band == 'Alpha_Delta_Theta':
            l_freq = 1
            h_freq = 13

    elif type(band) == tuple:
        l_freq = band[0]
        h_freq = band[1]

    elif band == None:
        return None, None

    return l_freq, h_freq

