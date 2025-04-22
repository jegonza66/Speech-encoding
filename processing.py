# Standard libraries
import numpy as np, copy, mne
from datetime import datetime

# Specific libraries
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from typing import Optional, Sequence
from scipy import signal
import torch

# def shifted_matrix_old(
#     features: np.ndarray, 
#     delays: np.ndarray, 
#     use_gpu: bool = True
#     ) -> np.ndarray:
#     """
#     Computes shifted matrix for a given array of delays, optimized and aligned with original implementation.

#     Parameters
#     ----------
#     features : array, shape (n_times, n_features)
#         The time series to delay must be 2D array.
#     delays : np.ndarray
#         Index delays.
#     use_gpu : bool, optional
#         Whether to use GPU for computation (torch, CUDA), by default True.

#     Returns
#     -------
#     np.ndarray
#         Concatenated shifted matrix of shape (samples, features * len(delays)).
#     """
#     if features.ndim == 1:
#         features = features.reshape(-1, 1)

#     device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
#     try:
#         features_tensor = torch.tensor(features, dtype=torch.float32, device=device)

#         n_samples, n_features = features_tensor.shape
#         n_delays = len(delays)

#         # Create design matrix
#         shifted = torch.zeros((n_samples, n_delays, n_features), dtype=torch.float32, device=device)

#         for i, delay in enumerate(delays):
#             if delay < 0:
#                 # Shift up
#                 shifted[:delay, i, :] = features_tensor[-delay:, :]
#             elif delay > 0:
#                 # Shift dowm
#                 shifted[delay:, i, :] = features_tensor[:-delay, :]
#             else:
#                 # Doesnt shift
#                 shifted[:, i, :] = features_tensor

#         # Reshpe to match desire output
#         shifted_matrix = shifted.permute(0, 2, 1).reshape(n_samples, n_features * n_delays)
#         return shifted_matrix.cpu().numpy()
#     except Exception as e:
#         # Fallback to CPU computation in case of memory issues
#         print(f"CUDA out of memory: switching to CPU for computation.\n Following error occured: {e}.")
        
#         device = torch.device("cpu")
#         features_tensor = torch.tensor(features, dtype=torch.float32, device=device)

#         n_samples, n_features = features_tensor.shape
#         n_delays = len(delays)

#         # Create design matrix
#         shifted = torch.zeros((n_samples, n_delays, n_features), dtype=torch.float32, device=device)

#         for i, delay in enumerate(delays):
#             if delay < 0:
#                 # Shift up
#                 shifted[:delay, i, :] = features_tensor[-delay:, :]
#             elif delay > 0:
#                 # Shift dowm
#                 shifted[delay:, i, :] = features_tensor[:-delay, :]
#             else:
#                 # Doesnt shift
#                 shifted[:, i, :] = features_tensor

#         # Reshpe to match desire output
#         shifted_matrix = shifted.permute(0, 2, 1).reshape(n_samples, n_features * n_delays)
#         return shifted_matrix.cpu().numpy()

# def shifted_matrix_2(
#     features: np.ndarray,
#     delays: np.ndarray,
#     use_gpu: bool = True,
#     indices_to_keep: np.ndarray = None
# ) -> np.ndarray:
#     """
#     Computes shifted matrix for given delays, optionally only for specified indices.

#     Parameters
#     ----------
#     features : array, (n_times, n_features)
#         Time series data.
#     delays : array
#         Index delays to apply.
#     use_gpu : bool
#         Whether to use GPU.
#     indices_to_keep : array, optional
#         Row indices to compute, avoiding full matrix creation.

#     Returns
#     -------
#     array
#         Shifted matrix, possibly only for specified rows.
#     """
#     device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

#     if features.ndim == 1:
#         features = features.reshape(-1, 1)
#     features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
#     n_samples, n_features = features_tensor.shape
#     n_delays = len(delays)
#     try:
#         if indices_to_keep is not None:
#             kept_indices = torch.tensor(indices_to_keep, device=device, dtype=torch.long)
#             n_kept = kept_indices.size(0)
#             shifted = torch.zeros((n_kept, n_delays, n_features), dtype=torch.float32, device=device)

#             for i, delay in enumerate(delays):
#                 current_indices = kept_indices
#                 if delay > 0:
#                     mask = (current_indices >= delay) & (current_indices < n_samples)
#                     valid_kept = mask.nonzero().squeeze()
#                     if valid_kept.numel() > 0:
#                         feature_indices = current_indices[valid_kept] - delay
#                         shifted[valid_kept, i, :] = features_tensor[feature_indices, :]
#                 elif delay < 0:
#                     abs_delay = -delay
#                     max_valid = n_samples - abs_delay
#                     mask = current_indices < max_valid
#                     valid_kept = mask.nonzero().squeeze()
#                     if valid_kept.numel() > 0:
#                         feature_indices = current_indices[valid_kept] + abs_delay
#                         shifted[valid_kept, i, :] = features_tensor[feature_indices, :]
#                 else:
#                     mask = (current_indices >= 0) & (current_indices < n_samples)
#                     valid_kept = mask.nonzero().squeeze()
#                     if valid_kept.numel() > 0:
#                         shifted[valid_kept, i, :] = features_tensor[current_indices[valid_kept], :]
#         else:
#             shifted = torch.zeros((n_samples, n_delays, n_features), dtype=torch.float32, device=device)
#             for i, delay in enumerate(delays):
#                 if delay < 0:
#                     shifted[:delay, i, :] = features_tensor[-delay:, :]
#                 elif delay > 0:
#                     shifted[delay:, i, :] = features_tensor[:-delay, :]
#                 else:
#                     shifted[:, i, :] = features_tensor

#         shifted_matrix = shifted.permute(0, 2, 1).reshape(-1, n_features * n_delays)
#         return shifted_matrix.cpu().numpy()
#     except Exception as e:
#         print(f"CUDA out of memory: switching to CPU for computation.\nFollowing error occurred: {e}.")
#         device = torch.device("cpu")

#         if indices_to_keep is not None:
#             kept_indices = torch.tensor(indices_to_keep, dtype=torch.long, device=device)
#             n_kept = kept_indices.size(0)
#             shifted = torch.zeros((n_kept, n_delays, n_features), dtype=torch.float32, device=device)

#             for i, delay in enumerate(delays):
#                 current_indices = kept_indices
#                 if delay > 0:
#                     mask = (current_indices >= delay) & (current_indices < n_samples)
#                     valid_kept = mask.nonzero().squeeze()
#                     if valid_kept.numel() > 0:
#                         feature_indices = current_indices[valid_kept] - delay
#                         shifted[valid_kept, i, :] = features_tensor[feature_indices, :]
#                 elif delay < 0:
#                     abs_delay = -delay
#                     max_valid = n_samples - abs_delay
#                     mask = current_indices < max_valid
#                     valid_kept = mask.nonzero().squeeze()
#                     if valid_kept.numel() > 0:
#                         feature_indices = current_indices[valid_kept] + abs_delay
#                         shifted[valid_kept, i, :] = features_tensor[feature_indices, :]
#                 else:
#                     mask = (current_indices >= 0) & (current_indices < n_samples)
#                     valid_kept = mask.nonzero().squeeze()
#                     if valid_kept.numel() > 0:
#                         shifted[valid_kept, i, :] = features_tensor[current_indices[valid_kept], :]
#         else:
#             shifted = torch.zeros((n_samples, n_delays, n_features), dtype=torch.float32, device=device)
#             for i, delay in enumerate(delays):
#                 if delay < 0:
#                     shifted[:delay, i, :] = features_tensor[-delay:, :]
#                 elif delay > 0:
#                     shifted[delay:, i, :] = features_tensor[:-delay, :]
#                 else:
#                     shifted[:, i, :] = features_tensor

#         shifted_matrix = shifted.permute(0, 2, 1).reshape(-1, n_features * n_delays)
#         return shifted_matrix.cpu().numpy()
    
def _compute_shifted(
    feats_t: torch.Tensor,
    delays: Sequence[int],
    indices_to_keep: Optional[Sequence[int]]
) -> torch.Tensor:
    """
    Compute shifted matrix for given features and delays.

    Parameters
    ----------
    feats_t : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    delays : Sequence[int]
        Delays to apply to the features.
    indices_to_keep : Optional[Sequence[int]]
        Specific indices to compute the shifted matrix for.

    Returns
    -------
    torch.Tensor
        Shifted matrix of shape (n_rows, n_delays, n_features).
    """
    n_samples, n_features = feats_t.shape
    delays = torch.tensor(delays, device=feats_t.device, dtype=torch.int64)

    if indices_to_keep is not None:
        idx = torch.tensor(indices_to_keep, device=feats_t.device, dtype=torch.int64)
        idx_shifted = idx[:, None] - delays[None, :]  # Shape: (n_rows, n_delays)
    else:
        idx_shifted = torch.arange(n_samples, device=feats_t.device)[:, None] - delays[None, :]  # Shape: (n_samples, n_delays)

    # Mask for valid indices
    valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples) # Shape: (n_rows, n_delays)

    # Clamp indices to valid range (i.e: ensure values are between 0 and n_samples-1)
    idx_clipped = idx_shifted.clamp(0, n_samples - 1)
    
    # Gather features and apply the mask
    feats_exp = feats_t[idx_clipped]  # Shape: (n_rows, n_delays, n_features)
    
    # Broadcast mask to match feature dimensions (unsqueeze to add feature dimension)
    feats_exp *= valid_mask.unsqueeze(-1)  # Shape: (n_rows, n_delays, n_features)

    return feats_exp

def shifted_matrix(
    features: np.ndarray,
    delays: Sequence[int],
    use_gpu: bool = True,
    indices_to_keep: Optional[Sequence[int]] = None,
    output_torch: bool = False,
    train_indexes: np.ndarray = None,
    pred_indexes: np.ndarray = None,
    ) -> np.ndarray:
    """
    Build a time-shifted design matrix for given features and delays.

    This function stacks time-shifted versions of the input feature matrix along the second axis,
    optionally computing only for specified row indices to reduce memory.

    Parameters
    ----------
    features : np.ndarray, shape (n_times, n_features) or (n_times,)
        Input time series data. If 1D, it is treated as a single feature.
    delays : Sequence[int]
        Relative time shifts (in samples). Positive delays shift past values,
        negative delays shift future values, zero retains current.
    use_gpu : bool, default True
        Whether to attempt computation on CUDA device first. Falls back to CPU on OOM.
    indices_to_keep : Sequence[int], optional
        Specific time indices at which to compute rows of the shifted matrix.
        If None, computes all rows.
    output_torch : bool or float, default False
        If True, returns a PyTorch tensor instead of a NumPy array. 
        If False, returns a NumPy array.
    train_indexes : np.ndarray, optional
        Indices of training samples. If provided, only these indices are used for computation.
    pred_indexes : np.ndarray, optional
        Indices of prediction samples. If provided, only these indices are used for computation.

    Returns
    -------
    np.ndarray, shape (n_rows, n_features * n_delays)
        Design matrix where each row contains concatenated features for each delay.
    """
    # Determine device order: try GPU first, then CPU
    preferred = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    devices = [preferred]
    if preferred.type == "cuda":
        devices.append(torch.device("cpu"))

    # Ensure features is 2D
    feats = features.reshape(-1, 1) if features.ndim == 1 else features

    for dev in devices:
        try:
            # Move data onto device
            feats_t = torch.tensor(feats, dtype=torch.float32, device=dev)
            shifted = _compute_shifted(feats_t, delays, indices_to_keep)
            
            # Reshape: (n_rows, n_delays, n_features) -> (n_rows, n_features * n_delays)
            n_rows, n_delays, n_feat = shifted.shape
            mat = shifted.permute(0, 2, 1).reshape(n_rows, n_feat * n_delays)
            if train_indexes is not None and pred_indexes is not None:
                if output_torch:
                    return mat[train_indexes, :], mat[pred_indexes, :]
                else: 
                    return mat[train_indexes, :].cpu().numpy(), mat[pred_indexes, :].cpu().numpy()
            else:
                if output_torch:
                    return mat
                else: 
                    return mat.cpu().numpy()

        except RuntimeError as e:
            if dev.type == "cuda":
                print(f"CUDA OOM on device {dev}; retrying on CPU. Error: {e}")
                continue
            else:
                raise
    # If loop completes without return, something went wrong
    raise RuntimeError("shifted_matrix failed on all devices")
# features = torch.arange(10).reshape(-1,1)
# delays = [-1, 0, 1]
# indices_to_keep = [0, 1, 2, 3, 4]
# shifted = _compute_shifted(features, delays, indices_to_keep)

# n_rows, n_delays, n_feat = shifted.shape
# mat = shifted.permute(0, 2, 1).reshape(n_rows, n_feat * n_delays)

def butter_filter(
    data:np.ndarray, 
    frequencies:float, 
    sampling_freq:float, 
    btype:str='lowpass', 
    order:int=3, 
    axis:int=0, 
    ftype:str='Causal'
    )->np.ndarray:
    """
    Apply a Butterworth filter to the input data.
    
    Parameters
    ----------
    data : np.ndarray
        The input data to be filtered.
    frequencies : float or list
        The cutoff frequency (for 'lowpass' and 'highpass') or frequencies (for 'bandpass').
    sampling_freq : float
        The sampling frequency of the input data.
    btype : str, optional
        The type of filter to apply ('lowpass', 'highpass', 'bandpass'), by default 'lowpass'.
    order : int, optional
        The order of the filter, by default 3.
    axis : int, optional
        The axis along which to apply the filter, by default 0.
    ftype : str, optional
        The type of filtering ('Causal' or 'NonCausal'), by default 'Causal'.
    
    Returns
    -------
    np.ndarray
        The filtered data.
    
    Raises
    ------
    ValueError
        If an invalid filter type is provided.
    """
    if btype == 'lowpass' or btype == 'highpass':
        frequency = frequencies / (sampling_freq / 2)
        b, a = signal.butter(order, frequency, btype=btype)
    elif btype == 'bandpass':
        frequencies = [frequency / (sampling_freq / 2) for frequency in frequencies]
        b, a = signal.butter(order, frequencies, btype=btype)

    if ftype == 'Causal':
        y = signal.lfilter(b, a, data, axis=axis)
    elif ftype == 'NonCausal':
        y = signal.filtfilt(b, a, data, axis=axis, padlen=None)
    return y

def subsample(
    x:np.ndarray, 
    step:int
    )->np.ndarray:
    """
    Subsamples the input array by selecting every `step`-th element.
    
    Parameters
    ----------
    x : np.ndarray
        The input array to be subsampled.
    step : int
        The step size for subsampling.
    
    Returns
    -------
    np.ndarray
        The subsampled array.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    indices = np.arange(0, len(x), int(step))
    return x[indices]

def band_freq(
    band:str
    )->tuple:
    """
    Returns the frequency range for a given frequency EEG band.
    
    Parameters
    ----------
    band : str or tuple
        The EEG frequency band to return the frequency range for.
    
    Returns
    -------
    tuple
        The frequency range for the given EEG band.
    
    Raises
    ------
    Exception
        If an invalid band is provided.
    """
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

def tfce(
    average_weights_subjects:np.ndarray,
    stimulus:str, 
    n_jobs:int=-1, 
    sr:int=128,
    n_permutations:int=64, 
    threshold_tfce:dict=dict(start=0, step=0.2),
    verbose_tfce:bool=True
    )->tuple:
    """
    Perform Threshold-Free Cluster Enhancement (TFCE) on the input data. It performs TFCE according to
    https://mne.tools/1.6/generated/mne.stats.permutation_cluster_test.html
    
    Parameters
    ----------
    average_weights_subjects : np.ndarray
        The input data to perform TFCE on.
    stimulus : str
        The stimulus type to perform TFCE on.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default -1.
    sr : int, optional
        The sampling rate of the input data, by default 128.
    n_permutations : int, optional
        The number of permutations to perform, by default 64.
    threshold_tfce : dict, optional
        The threshold parameters for TFCE, by default dict(start=0, step=0.2).
    verbose_tfce : bool, optional
        Whether to print verbose output, by default True.
    
    Returns
    -------
    tuple
        The TFCE t-values and p-values
    
    Raises
    ------
    Exception
        If an invalid stimulus is provided.
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
            weights = average_weights_subjects.copy()[:, :, feat, :].swapaxes(1, 2) #---> n_sub, n_delay, n_chans for specific feat
            try:
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
            except Exception as err:
                print(f'Error in feature {feat+1} out of {total_number_features}. The error was: \n{err}')
                
                t_tfce_feat, p_tfce_feat = np.ones(weights.shape[1], weights.shape[2]), np.zeros(weights.shape[1], weights.shape[2])
            
            print(f'Feature {feat+1} out of {total_number_features}')
        t_tfce, p_tfce = np.stack(t_tfce, axis=0), np.stack(p_tfce, axis=0) 

        if verbose_tfce:
            t_f = datetime.now().replace(microsecond=0)-t_0
            print(f'Performed TFCE succesfully in {t_f}')

        # Return average across channels
        return t_tfce, p_tfce
   
def block_bootstrap(
    data:np.ndarray, 
    block_size:int=104
    )->np.ndarray:
    """
    Bootstrap data with blocks of size block_size
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

def clustering_by_correlation(
    weights:np.ndarray
    )->list:
    """
    Cluster by correlation the weights

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

class Standarize():
    def __init__(
        self, 
        axis:int=0, 
        by_gpu:bool=False
        )->None:
        """
        Standarize train and test data to be used in a linear regressor model. 

        Parameters
        ----------
        axis : int, optional
            Axis to perform standrize, by default 0
        by_gpu : bool, optional
            Whether to perform computation on GPU, by default False
        """
        self.axis = axis
        self.by_gpu = by_gpu
        self.device = torch.device("cuda" if by_gpu and torch.cuda.is_available() else "cpu")

    def _to_device(
        self, 
        data:np.ndarray
        )->torch.Tensor:
        """
        Move data to GPU if by_gpu is True, and ensure dtype is float32.
        
        Parameters
        ----------
        data : np.ndarray
            Data to be moved to GPU if by_gpu is True.
            
        Returns
        -------
        torch.Tensor
            Data moved to GPU if by_gpu is True, in float32.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            if (data.dtype != torch.float32):
                data = data.to(dtype=torch.float32)
            else:
                pass
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

        if self.by_gpu and torch.cuda.is_available():
            return data.to(self.device)
        else:
            return data

    def fit_standarize_train(
        self, 
        train_data:np.ndarray
        )->np.ndarray:
        """
        Standardize train data, also define mean and std to standardize future data.
        
        Parameters
        ----------
        train_data : np.ndarray
            Train data to be standardized.
            
        Returns
        -------
        np.ndarray
            Standardized train data.
        """
        train_data = self._to_device(train_data)
        
        # Fix mean and standard deviation with train data
        if isinstance(train_data, torch.Tensor):
            self.mean = train_data.mean(dim=self.axis)
            self.std = train_data.std(dim=self.axis, unbiased=False)  # Use biased std for consistency with numpy
        else:
            self.mean = train_data.mean(axis=self.axis)
            self.std = train_data.std(axis=self.axis)   

        # Standardize data
        train_data -= self.mean
        train_data /= (self.std + 1e-12)  # Adding epsilon to avoid division by zero
        
        return train_data

    def fit_standarize_test(
        self, 
        test_data:np.ndarray
        )->np.ndarray:
        """
        Standardize test data with mean and std of train data
        
        Parameters
        ----------
        test_data : np.ndarray

        Returns
        -------
        np.ndarray
            Standardized test data.
        """
        test_data = self._to_device(test_data)
        
        # Standardize with mean and standard deviation of train
        test_data -= self.mean
        test_data /= (self.std + 1e-12)
        
        return test_data

    def standarize_data(
        self,
        data:np.ndarray
        )->np.ndarray:
        """
        Standardize data with its own mean and standard deviation.
        
        Parameters
        ----------
        data : np.ndarray
            Data to be standardized.
        
        Returns
        -------
        np.ndarray
            Standardized data.
        """
        data = self._to_device(data)
        
        if isinstance(data, torch.Tensor):
            data -= data.mean(dim=self.axis)
            data /= data.std(dim=self.axis, unbiased=False)  # Use biased std for consistency with numpy
        else:
            data -= data.mean(axis=self.axis)
            data /= data.std(axis=self.axis)
            
        return data
    
class Normalize():
    def __init__(
        self, 
        axis:int=0, 
        porcent:float=5, 
        by_gpu:bool=False
        )->None:
        """
        Normalize train and test data to be used in a linear regressor model.

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
        self.device = torch.device("cuda" if by_gpu and torch.cuda.is_available() else "cpu")

    def _to_device(
        self, 
        data:np.ndarray
        )->torch.Tensor:
        """
        Move data to GPU if by_gpu is True, and ensure dtype is float32.
        
        Parameters
        ----------
        data : np.ndarray
            Data to be moved to GPU if by_gpu is True.
            
        Returns
        -------
        torch.Tensor
            Data moved to GPU if by_gpu is True, in float32.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            if (data.dtype != torch.float32):
                data = data.to(dtype=torch.float32)
            else:
                pass
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

        if self.by_gpu and torch.cuda.is_available():
            return data.to(self.device)
        else:
            return data

    def fit_normalize_train(
        self, 
        train_data:np.ndarray
        )->np.ndarray:
        """
        Normalize train data, also define min and max to normalize future data
        
        Parameters
        ----------
        train_data : np.ndarray
            Train data to be normalized.
            
        Returns
        -------
        np.ndarray
            Normalized train data.
        """
        train_data = self._to_device(train_data)
        
        # Remove offset by minimum
        if isinstance(train_data, torch.Tensor):
            self.min = train_data.min(dim=self.axis)[0]
        else:
            self.min = train_data.min(axis=self.axis)
        train_data -= self.min

        # Normalize by maximum
        if isinstance(train_data, torch.Tensor):
            self.max = train_data.max(dim=self.axis)[0]
        else:
            self.max = train_data.max(axis=self.axis)
            
        train_data = train_data / (self.max + 1e-12)  # Adding epsilon to avoid division by zero
        
        return train_data

    def fit_normalize_test(
        self, 
        test_data:np.ndarray
        )->np.ndarray:
        """
        Normalize test data with min and max of train data.
        
        Parameters
        ----------
        test_data : np.ndarray
            Test data to be normalized.
        
        Returns
        -------
        np.ndarray
            Normalized test data.
        """
        test_data = self._to_device(test_data)
        
        test_data -= self.min
        test_data = test_data / (self.max + 1e-12)
        
        return test_data

    def normalize_data(
        self, 
        data:np.ndarray, 
        kind:str="1"
        )->np.ndarray:
        """
        Normalize data
        
        Parameters
        ----------
        data : np.ndarray
            Data to be normalized.
        kind : str, optional
            Type of normalization, by default '1'
        
        Returns
        -------
        np.ndarray
            Normalized data.
        """
        data = self._to_device(data)
        
        if isinstance(data, torch.Tensor):
            data -= data.min(dim=self.axis)[0]
            data /= data.max(dim=self.axis)[0]
        else:
            data -= data.min(axis=self.axis)
            data /= data.max(axis=self.axis)
        
        if kind == '2':
            data *= 2
            data -= 1
        
        return data

    def fit_normalize_percent(
        self, 
        data:np.ndarray
        )->np.ndarray:
        """
        Normalize data using percentiles
        
        Parameters
        ----------
        data : np.ndarray
            Data to be normalized.
        
        Returns
        -------
        np.ndarray
            Normalized data.
        """
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
        data = data / (max_data_n + 1e-12)  # Adding epsilon to avoid division by zero
        
        return data
