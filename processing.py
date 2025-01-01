# Standard libraries
import numpy as np, copy, mne
from datetime import datetime

# Specific libraries
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy import signal
import torch

class Standarize():
    def __init__(self, axis:int=0, by_gpu:bool=False):
        """Standarize train and test data to be used in a linear regressor model. 

        Parameters
        ----------
        axis : int, optional
            Axis to perform standrize, by default 0
        by_gpu : bool, optional
            Whether to perform computation on GPU, by default False
        """
        self.axis = axis
        self.by_gpu = by_gpu

    def _to_device(self, data: np.ndarray):
        """Move data to GPU if by_gpu is True."""
        if self.by_gpu:
            if isinstance(data, torch.Tensor):
                if data.is_cuda:
                    return data
                else:
                    return data.cuda()
            else:
                return torch.tensor(data).cuda()
        else:
            return torch.tensor(data)

    def fit_standarize_train(self, train_data: np.ndarray):
        """Standardize train data, also define mean and std to standardize future data."""
        train_data = self._to_device(train_data)
        
        # Fix mean and standard deviation with train data
        self.mean = train_data.mean(dim=self.axis)
        self.std = train_data.std(dim=self.axis)

        # Standardize data
        train_data -= self.mean
        train_data /= (self.std + 1e-8)  # Adding epsilon to avoid division by zero
        if self.by_gpu:
            return train_data.float()
        else:
            return train_data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

    def fit_standarize_test(self, test_data: np.ndarray):
        """Standardize test data with mean and std of train data."""
        test_data = self._to_device(test_data)
        
        # Standardize with mean and standard deviation of train
        test_data -= self.mean
        test_data /= (self.std + 1e-8)
        if self.by_gpu:
            return test_data.float()
        else:
            return test_data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

    def standarize_data(self, data: np.ndarray):
        """Standardize data with its own mean and standard deviation."""
        data = self._to_device(data)
        
        data -= data.mean(dim=self.axis)
        data /= data.std(dim=self.axis)
        if self.by_gpu:
            return data.float()
        else:
            return data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

class Normalize():
    def __init__(self, axis:int=0, porcent:float=5, by_gpu:bool=False):
        """Normalize train and test data to be used in a linear regressor model.

        Parameters
        ----------
        axis : int, optional
            Axis to perform normalize, by default 0
        porcent : float, optional
            Percentage for normalization, by default 5
        by_gpu : bool, optional
            Whether to perform computation on GPU, by default False
        """
        self.axis = axis
        self.porcent = porcent
        self.by_gpu = by_gpu

    def _to_device(self, data: np.ndarray):
        """Move data to GPU if by_gpu is True."""
        if self.by_gpu:
            if isinstance(data, torch.Tensor):
                if data.is_cuda:
                    return data.float()
                else:
                    return data.cuda()
            else:
                return torch.tensor(data).cuda()
        else:
            return torch.tensor(data)

    def fit_normalize_train(self, train_data: np.ndarray):
        """Normalize train data, also define min and max to normalize future data."""
        train_data = self._to_device(train_data)
        
        # Remove offset by minimum
        self.min = train_data.min(dim=self.axis)[0]
        train_data -= self.min

        # Normalize by maximum
        self.max = train_data.max(dim=self.axis)[0]
        train_data = train_data / (self.max + 1e-8)  # Adding epsilon to avoid division by zero

        if self.by_gpu:
            return train_data.float()
        else:
            return train_data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

    def fit_normalize_test(self, test_data: np.ndarray):
        """Normalize test data with min and max of train data."""
        test_data = self._to_device(test_data)
        
        test_data -= self.min
        test_data = test_data / (self.max + 1e-8)
        if self.by_gpu:
            return test_data.float()
        else:
            return test_data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

    def normalize_data(self, data: np.ndarray, kind:str="1"):
        """Normalize data."""
        data = self._to_device(data)
        
        data -= data.min(dim=self.axis)[0]
        data /= data.max(dim=self.axis)[0]
        
        if kind == '2':
            data *= 2
            data -= 1
            
        if self.by_gpu:
            return data.float()
        else:
            return data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

    def fit_normalize_percent(self, data: np.ndarray):
        """Normalize data using percentiles."""
        data = self._to_device(data)
        
        # Calculate n
        n = int((self.porcent * len(data) - 1) / 100) 
        
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
        data = data / (max_data_n + 1e-8)  # Adding epsilon to avoid division by zero
        
        if self.by_gpu:
            return data.float()
        else:
            return data.cpu().numpy().astype(np.float32)  # Return as numpy for compatibility

def shifted_matrix(
    features:np.ndarray, 
    delays:np.ndarray, 
    use_gpu:bool=True) -> np.ndarray:
    """
    Computes shifted matrix for a given array of delays
    
    Parameters
    ----------
    features : array, shape (n_times, n_features)
        The time series to delay must be 2D array.
    delays : np.ndarray
        Index delays
    use_gpu : bool, optional
        Whether to use GPU for computation (torch, CUDA), by default True

    Returns
    -------
    np.ndarray
        Concatenated shifted matrix of shape (samples, features*delays)
    """
    # Convert features to tensor and move to GPU if needed
    features_tensor = torch.tensor(features, dtype=torch.float32)
    if use_gpu:
        features_tensor = features_tensor.cuda()

    n_samples = features_tensor.shape[0]
    n_features = features_tensor.shape[-1]

    # Create an empty tensor for the shifted matrix
    shifted_matrix = torch.zeros((n_samples, n_features, len(delays)), dtype=torch.float32)

    for i, delay in enumerate(delays):
        if delay < 0:
            # For negative delays, slice the array and shift
            out = shifted_matrix[:delay, ..., i]
            use_X = features_tensor[-delay:]
        elif delay > 0:
            # For positive delays, slice the array and shift
            out = shifted_matrix[delay:, ..., i]
            use_X = features_tensor[:-delay]
        else:
            out = shifted_matrix[..., i]
            use_X = features_tensor
        
        # Assign shifted data
        out[:] = use_X
    
    # Return the shifted matrix reshaped for the final output
    # return shifted_matrix.cpu().numpy()
    return shifted_matrix.reshape(n_samples, n_features * len(delays)).cpu().numpy()
    

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

def subsamplear(x, cada_cuanto):
    if not isinstance(x, np.ndarray):
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
        elif band == 'Beta1':
            l_freq = 13
            h_freq = 19
        elif band == 'Beta2':
            l_freq = 19
            h_freq = 25
        elif band == 'All':
            l_freq = 1
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

# TODO CHECK DESCRIPTION
def tfce(average_weights_subjects:np.ndarray,
         stimulus:str, 
         n_jobs:int=-1, 
         sr:int=128,
         n_permutations:int=64, 
         threshold_tfce:dict=dict(start=0, step=0.2),
         verbose_tfce:bool=True):
    """_summary_

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        _description_
    n_permutations : int, optional
        _description_, by default 1024
    threshold_tfce : dict, optional
        _description_, by default dict(start=0, step=0.2)

    Returns
    -------
    _type_
        _description_
    """
    t_0 = datetime.now().replace(microsecond=0)

    # Get relevant parameters
    n_subjects, n_chan, total_number_features, n_delays  = average_weights_subjects.shape
    stimuli_correlated_by_frequency = ['Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 'Spectrogram']
    
    # Perform it across features, averaging first across channels
    if stimulus in stimuli_correlated_by_frequency:
        weights_subjects_mean_across_channels = average_weights_subjects.copy().mean(axis=1)
        weights = weights_subjects_mean_across_channels.swapaxes(1, 2) #---> n_sub, n_delays, n_feats for specific feat
        t_tfce, clusters, p_tfce, H0 = mne.stats.permutation_cluster_1samp_test(
                                                                                X=weights,
                                                                                adjacency=None,
                                                                                n_jobs=n_jobs,
                                                                                threshold=threshold_tfce,
                                                                                n_permutations=n_permutations,
                                                                                out_type="mask",
                                                                                verbose=verbose_tfce
                                                                                )
        p_tfce = p_tfce.reshape(t_tfce.shape)
        if verbose_tfce:
            t_f = datetime.now().replace(microsecond=0)-t_0
            print(f'Performed TFCE succesfully in {t_f}')

        # Return average across channels
        return t_tfce, p_tfce
    
    # Do it independently for each feature
    else:
        # Get adjacency matrix
        montage = mne.channels.make_standard_montage('biosemi128')
        info_mne = mne.create_info(ch_names=montage.ch_names[:], sfreq=sr, ch_types='eeg').set_montage(montage)
        adj_matrix, names = mne.channels.find_ch_adjacency(info=info_mne, ch_type='eeg')

        # Get average across subjects
        # weights_subjects_mean_across_subjects = average_weights_subjects.copy().mean(axis=0)
        t_tfce, p_tfce = [], []
        for feat in range(total_number_features):
            # It performs tfce on https://mne.tools/1.6/generated/mne.stats.permutation_cluster_test.html
            weights = average_weights_subjects.copy()[:, :, feat, :].swapaxes(1, 2) #---> n_sub, n_delay, n_chans for specific feat
            t_tfce_feat, clusters, p_tfce_feat, H0 = mne.stats.permutation_cluster_1samp_test(
                                                                                            X=weights,
                                                                                            adjacency=adj_matrix,
                                                                                            n_jobs=n_jobs,
                                                                                            threshold=threshold_tfce,
                                                                                            n_permutations=n_permutations,
                                                                                            out_type="mask",
                                                                                            verbose=verbose_tfce
                                                                                            )
            
            t_tfce.append(t_tfce_feat)
            p_tfce.append(p_tfce_feat.reshape(t_tfce_feat.shape))
            print(f'Feature {feat+1} out of {total_number_features}')
        t_tfce, p_tfce = np.stack(t_tfce, axis=0), np.stack(p_tfce, axis=0) 

        if verbose_tfce:
            t_f = datetime.now().replace(microsecond=0)-t_0
            print(f'Performed TFCE succesfully in {t_f}')

        # Return average across channels
        return t_tfce, p_tfce
    
def block_bootstrap(data:np.ndarray, block_size:int=104):
    """Bootstrap data with blocks of size block_size
    Parameters
    ----------
    data : np.ndarray
        Data to be bootstraped
    block_size : int
        Correlation length to have into account when making bootstrap

    Returns
    -------
    np.ndarray
        Resampled data
    """
    n_samples = len(data)
    n_blocks = n_samples // block_size
    indices = np.arange(n_samples)
    block_indices = np.random.choice(n_blocks, n_blocks, replace=True)
    resampled_indices = np.hstack([indices[i*block_size:(i+1)*block_size] for i in block_indices])
    return data[resampled_indices]

def clustering_by_correlation(weights:np.ndarray):
    """Cluster by correlation the weights

    Parameters
    ----------
    weights : np.ndarray
        Must be n_features X n_delays

    Returns
    -------
    list
        indexes in ordered of clustering
    """
    # Identify zero rows
    null_indexes = np.where(~weights.any(axis=1))[0]
    
    weights = weights[[i for i in np.arange(weights.shape[0]) if i not in null_indexes]]

    # Compute the correlation matrix 
    correlation_matrix = np.corrcoef(weights, rowvar=True)

    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - correlation_matrix

    # Ensure the diagonal of the distance matrix is zero and that the matrix is symmetric
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Use squareform to convert the distance matrix to a condensed form
    condensed_distance_matrix = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method='single')

    # Get the order of the variables
    ordered_indices = leaves_list(linkage_matrix)
    
    if null_indexes.shape[0]==0:
        null_indexes = None
    return ordered_indices, null_indexes



# ###############

# # def butter_bandpass_filter(data, frecuencia, sampling_freq, order, axis):
# #     frecuencia /= (sampling_freq / 2)
# #     b, a = signal.butter(order, frecuencia, btype='lowpass')
# #     y = signal.filtfilt(b, a, data, axis=axis, padlen=None)
# #     return y


# class Standarize():
#     def __init__(self, axis:int=0):
#         """Standarize train and test data to be used in a linear regressor model. 

#         Parameters
#         ----------
#         axis : int, optional
#             Axis to perform standrize, by default 0
#         """
#         self.axis = axis

#     def fit_standarize_train(self, train_data:np.ndarray):
#         """Standarize train data, also define mean and std to standarize future data.

#         Parameters
#         ----------
#         train_data : np.ndarray
#             Train data to be standarize
#         """
#         # Fix mean and standard deviation with train data
#         self.mean = np.mean(train_data, axis=self.axis)
#         self.std = np.std(train_data, axis=self.axis)

#         # Standarize data
#         train_data -= self.mean
#         train_data /= self.std
#         return train_data

#     def fit_standarize_test(self, test_data:np.ndarray):
#         """Standarize test data with mean and std of train data.

#         Parameters
#         ----------
#         test_data : np.ndarray
#             Test data to be standarize with mean and standard deviation of train data
#         """
#         # Standarize with mean and standard deviation of train
#         test_data -= self.mean
#         test_data /= self.std
#         return test_data

#     def standarize_data(self, data:np.ndarray):
#         """Standarize data with own mean and standard deviation.

#         Parameters
#         ----------
#         data : np.ndarray
#             Data to be standarized
#         """
#         # Standarize data with own mean and standard deviation
#         data -= np.mean(data, axis=self.axis)
#         data /= np.std(data, axis=self.axis)
#         return data

# class Normalize():
#     def __init__(self, axis:int=0, porcent:float=5):
#         """Normalize train and test data to be used in a linear regressor model.

#         Parameters
#         ----------
#         axis : int, optional
#             Axis to perform normalize, by default 0
#         porcent : float, optional
#             _description_, by default 5
#         """
#         self.axis = axis
#         self.porcent = porcent

#     def fit_normalize_train(self, train_data:np.ndarray):
#         """Normalize train data, also define min and max to normalize future data.

#         Parameters
#         ----------
#         train_data : np.ndarray
#             Train data to be normalize by maximum and minimum (offset)
#         """
        
#         # Remove offset by minimum
#         self.min = np.min(train_data, axis=self.axis)
#         train_data -= self.min

#         # Normalize by maximum
#         self.max = np.max(train_data, axis=self.axis)
#         return np.divide(train_data, self.max, out=np.zeros_like(train_data), where=self.max != 0)

#     def fit_normalize_test(self, test_data:np.ndarray):
#         """Normalize test data with min and max of train data.

#         Parameters
#         ----------
#         test_data : np.ndarray
#             Test data to be normalize with train data parameters
#         """
#         test_data -= self.min
#         return np.divide(test_data, self.max, out=np.zeros_like(test_data), where=self.max != 0)

#     def normalize_data(self, data:np.ndarray, kind:str="1"):
#         """_summary_# TODO no queda claro para qué es la kind 2, creo que es para que esté centrada en 0

#         Parameters
#         ----------
#         data : np.ndarray
#             _description_
#         kind : str, optional
#             _description_, by default "1"
#         """
#         # Los estimulos los normalizo todos entre 0 y 1 estricto, la envolvente no tiene picos
#         data -= np.min(data, axis=self.axis)
#         data /= np.max(data, axis=self.axis)
#         if kind=='2':
#             data *= 2
#             data -= 1
#         return data

#     def fit_normalize_percent(self, data:np.ndarray):
#         """_summary_# TODO no queda claro qué es lo que sucede, creo que corta el 5 porciento de los datos hacia adelante y hacia atras y trabaja con los maximos alli descritos

#         Parameters
#         ----------
#         data : np.ndarray
#             Data to be normalize
#         """
#         # Find n 
#         # n = int((self.porcent/100)*len(data)) 
#         n = int((self.porcent * len(data) - 1) / 100) # TODO para mí va lo de arriba
        
        
#         # Find the n-th minimum and offset that value
#         sorted_data = copy.deepcopy(data)
#         sorted_data.sort(self.axis)
#         min_data_n = sorted_data[n]
#         data -= min_data_n

#         # Find the n-th maximum
#         sorted_data = copy.deepcopy(data)
#         sorted_data.sort(self.axis)
#         max_data_n = sorted_data[-n]
        
#         # Normalize data
#         data = np.divide(data, self.max, out=np.zeros_like(data), where=max_data_n!=0)
#         data /= max_data_n
#         return data

# def standarize_normalize(eeg_train_val, eeg_test, dstims_train_val, dstims_test, Stims_preprocess, EEG_preprocess, axis=0, porcent=5):
#     norm = Normalize(axis, porcent)
#     estandar = Standarize(axis)

#     if isinstance(dstims_train_val, list):
#         if Stims_preprocess == 'Standarize':
#             for i in range(len(dstims_train_val)):
#                 estandar.fit_standarize_train(train_data=dstims_train_val[i])
#                 estandar.fit_standarize_test(test_data=dstims_test[i])
#             dstims_train_val = np.hstack([dstims_train_val[i] for i in range(len(dstims_train_val))])
#             dstims_test = np.hstack([dstims_test[i] for i in range(len(dstims_test))])

#         if Stims_preprocess == 'Normalize':
#             for i in range(len(dstims_train_val)):
#                 norm.fit_normalize_train(train_data=dstims_train_val[i])
#                 norm.fit_normlize_test(test_data=dstims_test[i])
#             dstims_train_val = np.hstack([dstims_train_val[i] for i in range(len(dstims_train_val))])
#             dstims_test = np.hstack([dstims_test[i] for i in range(len(dstims_test))])
#     else:
#         if Stims_preprocess == 'Standarize':
#             for i in range(dstims_train_val.shape[1]):
#                 estandar.fit_standarize_train(train_data=dstims_train_val[:,i])
#                 estandar.fit_standarize_test(test_data=dstims_test[:,i])
#         if Stims_preprocess == 'Normalize':
#             for i in range(dstims_train_val.shape[1]):
#                 norm.fit_normalize_train_data(dstims_train_val[:,i])
#                 norm.normlize_test_data(dstims_test[:,i])

#     if EEG_preprocess == 'Standarize':
#         estandar.fit_standarize_train(train_data=eeg_train_val)
#         estandar.fit_standarize_test(test_data=eeg_test)
#     if EEG_preprocess == 'Normalize':
#         norm.fit_normalize_percent(data=eeg_train_val)
#         norm.fit_normlize_test(test_data=eeg_test) # TODO OJO SE ESTA NORMALIZANDO CON EL LOS DATOS DE LOS FEATURES, EEG SOLO EN ESTE CASO

#     return eeg_train_val, eeg_test, dstims_train_val, dstims_test

