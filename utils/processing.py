# Standard libraries
import numpy as np, copy, mne
from datetime import datetime
from typing import Tuple, Union

# Specific libraries
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy import signal

from typing import Optional, Sequence
import torch

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
    device = feats_t.device
    
    # Convert delays to tensor only once
    if not isinstance(delays, torch.Tensor):
        delays = torch.tensor(delays, device=device, dtype=torch.int64)

    if indices_to_keep is not None:
        # Convert indices to tensor only if not already a tensor
        if not isinstance(indices_to_keep, torch.Tensor):
            idx = torch.tensor(indices_to_keep, device=device, dtype=torch.int64)
        else:
            idx = indices_to_keep.to(device=device, dtype=torch.int64)
        n_rows = idx.shape[0]
        idx_shifted = idx.unsqueeze(1) - delays.unsqueeze(0)  # More explicit broadcasting
    else:
        n_rows = n_samples
        idx_shifted = torch.arange(n_samples, device=device, dtype=torch.int64).unsqueeze(1) - delays.unsqueeze(0)

    # Mask for valid indices - combine operations
    valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)

    # Pre-allocate output tensor with correct shape
    feats_exp = torch.zeros((n_rows, delays.shape[0], n_features), 
                           dtype=feats_t.dtype, device=device)
    
    # Only process valid indices to avoid unnecessary operations
    if valid_mask.any():
        # Clamp and gather only valid indices
        idx_clipped = idx_shifted.clamp(0, n_samples - 1)
        
        # Use advanced indexing more efficiently
        feats_gathered = feats_t[idx_clipped]  # Shape: (n_rows, n_delays, n_features)
        
        # Apply mask in-place to avoid extra memory allocation
        feats_gathered.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
        feats_exp = feats_gathered

    return feats_exp

# Optimized version specifically for medium-sized delay arrays (like 104)
def _compute_shifted_optimized(
    feats_t: torch.Tensor,
    delays: Sequence[int],
    indices_to_keep: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """
    Optimized for medium-sized delay arrays (~100 delays).
    Uses memory-efficient chunking with vectorized operations.
    """
    n_samples, n_features = feats_t.shape
    device = feats_t.device
    
    # Convert to tensor once
    if not isinstance(delays, torch.Tensor):
        delays = torch.tensor(delays, device=device, dtype=torch.int64)
    
    if indices_to_keep is not None:
        if not isinstance(indices_to_keep, torch.Tensor):
            idx = torch.tensor(indices_to_keep, device=device, dtype=torch.int64)
        else:
            idx = indices_to_keep.to(device=device, dtype=torch.int64)
        n_rows = idx.shape[0]
    else:
        idx = torch.arange(n_samples, device=device, dtype=torch.int64)
        n_rows = n_samples
    
    n_delays = delays.shape[0]
    
    # Pre-allocate output
    result = torch.zeros((n_rows, n_delays, n_features), 
                        dtype=feats_t.dtype, device=device)
    
    # Process in chunks to balance memory vs speed
    chunk_size = min(32, n_delays)  # Adjust based on your GPU memory
    
    for start_delay in range(0, n_delays, chunk_size):
        end_delay = min(start_delay + chunk_size, n_delays)
        delay_chunk = delays[start_delay:end_delay]
        
        # Vectorized computation for this chunk
        idx_shifted = idx.unsqueeze(1) - delay_chunk.unsqueeze(0)
        valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)
        
        # Only process if there are valid indices
        if valid_mask.any():
            idx_clipped = idx_shifted.clamp(0, n_samples - 1)
            chunk_result = feats_t[idx_clipped]
            chunk_result.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
            result[:, start_delay:end_delay, :] = chunk_result
    
    return result

def shifted_matrix(
    features: np.ndarray,
    delays: Sequence[int],
    use_gpu: bool = True,
    indices_to_keep: Optional[Sequence[int]] = None,
    output_torch: bool = False,
    train_indexes: np.ndarray = None,
    pred_indexes: np.ndarray = None,
    optimized_shifted: bool = False
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
            feats_t = torch.tensor(feats.astype(np.float64), dtype=torch.float32, device=dev)
            if optimized_shifted:
                shifted = _compute_shifted_optimized(feats_t, delays, indices_to_keep)
            else:
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

def cheby2_bandpass_filter_torch(
    y : torch.Tensor, 
    fs : float, 
    lowcut : int=1, 
    highcut : Union[float, int]=15, 
    order : int=4, 
    rs : float =20, 
    device: Union[str, torch.device] = 'cuda',
    axis : int = 0,
    channel_idx=None
)-> torch.Tensor:
    """
    Apply a Chebyshev Type II bandpass filter to the input tensor.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor to be filtered, typically of shape (n_samples, n_channels).
    fs : float
        Sampling frequency of the input data.
    lowcut : int, optional
        Lower cutoff frequency for the bandpass filter, by default 1 Hz.
    highcut : float or int, optional
        Upper cutoff frequency for the bandpass filter, by default 15 Hz.
    order : int, optional
        Order of the Chebyshev Type II filter, by default 4.
    rs : float, optional
        Ripple in the stop band, by default 20 dB.
    channel_idx : Optional[Union[int, Sequence[int]]], optional
        Index or indices of channels to filter. If None, all channels are filtered.
        If an int, filters only that channel. If a list, filters the specified channels.

    Returns
    -------
    torch.Tensor
        Filtered tensor with the same shape as the input tensor.
    """
    y_np = y.cpu().numpy()
    sos = signal.cheby2(order, rs, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    y_filt = np.copy(y_np)
    if channel_idx is None:
        # Filter all channels at once (axis=0 is time)
        y_filt = signal.sosfiltfilt(sos, y_np, axis=axis)
    else:
        # Filter only selected channel(s)
        if isinstance(channel_idx, int):
            channel_idx = [channel_idx]
        for ch in channel_idx:
            y_filt[:, ch] = signal.sosfiltfilt(sos, y_np[:, ch])
    return torch.from_numpy(y_filt.copy()).to(y.device, dtype=y.dtype)

def cheby2_bandpass_filter_np(
    y: np.ndarray,
    fs: float,
    lowcut: int = 1,
    highcut: float = 15,
    order: int = 4,
    rs: float = 20,
    axis: int = 0,
    channel_idx=None
) -> np.ndarray:
    """
    Apply a Chebyshev Type II bandpass filter to the input NumPy array.

    Parameters
    ----------
    y : np.ndarray
        Input array to be filtered, typically of shape (n_samples, n_channels).
    fs : float
        Sampling frequency of the input data.
    lowcut : int, optional
        Lower cutoff frequency for the bandpass filter, by default 1 Hz.
    highcut : float, optional
        Upper cutoff frequency for the bandpass filter, by default 15 Hz.
    order : int, optional
        Order of the Chebyshev Type II filter, by default 4.
    rs : float, optional
        Ripple in the stop band, by default 20 dB.
    axis : int, optional
        Axis along which to filter, by default 0 (time).
    channel_idx : Optional[Union[int, Sequence[int]]], optional
        Index or indices of channels to filter. If None, all channels are filtered.

    Returns
    -------
    np.ndarray
        Filtered array with the same shape as the input.
    """
    sos = signal.cheby2(order, rs, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    y_filt = np.copy(y)
    if channel_idx is None:
        # Filter all channels at once (axis=0 is time)
        y_filt = signal.sosfiltfilt(sos, y, axis=axis)
    else:
        # Filter only selected channel(s)
        if isinstance(channel_idx, int):
            channel_idx = [channel_idx]
        for ch in channel_idx:
            y_filt[:, ch] = signal.sosfiltfilt(sos, y[:, ch])
    return y_filt

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
        elif band == 'Beta':
            l_freq = 13
            h_freq = 25
        elif band == 'Beta1':
            l_freq = 13
            h_freq = 19
        elif band == 'Beta2':
            l_freq = 19
            h_freq = 25
        elif band == 'Broad':
            l_freq = 1
            h_freq = 15
        elif band == 'All':
            l_freq = 1
            h_freq = 40
        elif band == 'Delta_Theta':
            l_freq = 1
            h_freq = 8
        elif band == 'Alpha_Delta_Theta':
            l_freq = 1
            h_freq = 13
        elif band.startswith('Custom'):
            limits = band.split('Custom-')[1].split('#')
            l_freq = float(limits[0])
            h_freq = float(limits[1])
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

def calculate_partitions_2(
    A, B, AB_union
)-> np.ndarray:
    """
    Calculate the 3 partitions of 2 sets given their sizes and union sizes.
    
    Parameters
    ----------
    A : int
        Size of set A.
    B : int
        Size of set B.
    AB_union : int
        Size of the union of sets A and B.
    
    Returns
    -------
    np.ndarray
        An array containing the sizes of the 3 partitions: 
        [A only, B only, A∩B].
    """ 
    # Shared without each stimulus
    variance_shared_with_A = AB_union - B #10
    variance_shared_with_B = AB_union - A #01
    variance_int_complement_submodels = A + B - AB_union #11

    return np.array([
        variance_shared_with_A,
        variance_shared_with_B,
        variance_int_complement_submodels
    ])

def calculate_partitions_3(
    A, B, C, AB_union, AC_union, BC_union, ABC_union 
)-> np.ndarray:
    """
    Calculate the 7 partitions of 3 sets given their sizes and union sizes.
    
    Parameters
    ----------
    A : int
        Size of set A.
    B : int
        Size of set B.
    C : int
        Size of set C.
    AB_union : int
        Size of the union of sets A and B.
    AC_union : int
        Size of the union of sets A and C.
    BC_union : int
        Size of the union of sets B and C.
    ABC_union : int
        Size of the union of sets A, B, and C.
    
    Returns
    -------
    np.ndarray
        An array containing the sizes of the 7 partitions: 
        [A only, B only, A∩B only, C only, A∩C only, B∩C only, A∩B∩C].
    """ 
    # Shared without each stimulus
    variance_shared_with_A = ABC_union - BC_union #100
    variance_shared_with_B = ABC_union - AC_union #010
    variance_shared_with_C = ABC_union - AB_union #001

    # Explained by subshared, but not by all shared model
    variance_shared_with_AB = AC_union + BC_union - C - ABC_union #110
    variance_shared_with_AC = AB_union + BC_union - B - ABC_union #101
    variance_shared_with_BC = AB_union + AC_union - A - ABC_union #011

    # Explained by one, two, three and full shared model but not by subshared models
    variance_int_complement_submodels = ABC_union + A + B + C - AB_union - AC_union - BC_union #111

    return np.array([
        variance_shared_with_A,
        variance_shared_with_B,
        variance_shared_with_AB,
        variance_shared_with_C,
        variance_shared_with_AC,
        variance_shared_with_BC,
        variance_int_complement_submodels
    ])
    
def correct_pearson_square(
    values:np.ndarray,
)-> np.ndarray:
    """
    Correct the squared Pearson correlation values by applying a bias.

    Parameters
    ----------
    values : np.ndarray
        The original R^2 values to be corrected.

    Returns
    -------
    np.ndarray
        The corrected R^2 values.
    """

    # El optimizador intentará encontrar 7/3 valores de 'bias'
    
    # 1. Función objetivo: Minimizar la norma L2 de los sesgos 
    def objective_function(biases):
        return np.sum(np.square(biases))

    # 2. Restricciones (constraints): Las 7 particiones deben ser >= 0 
    def constraints_function(biases):
        # Aplica los sesgos a los R^2 medidos
        corrected_r2_values = values - biases
        
        # Calcula las particiones con los R^2 corregidos
        if len(values)==7:
            partitions = calculate_partitions_3(*corrected_r2_values)
        elif len(values)==3:
            partitions = calculate_partitions_2(*corrected_r2_values)
        else:
            raise ValueError("Length of values must be either 3 or 7.")
        
        # El optimizador requiere que todas las restricciones devuelvan >= 0
        return partitions

    # Configura las 7 restricciones (una para cada partición)
    # 'type': 'ineq' significa que la función de restricción debe ser >= 0
    constraints = [
        {
        'type': 'ineq', 
        'fun': lambda biases, i=i: constraints_function(biases)[i]
        } 
        for i in range(len(values))
    ]

    # 3. Ejecuta el optimizador
    # Inicia con sesgos de cero
    initial_biases = np.zeros(len(values))
    
    # 'args' pasa los valores medidos a nuestras funciones
    result = minimize(
        objective_function,
        initial_biases,
        method='SLSQP', # Un método bueno para problemas con restricciones
        constraints=constraints
    )

    if not result.success:
        print("Advertencia: La optimización de corrección de varianza falló.")
        # Decide cómo manejar el fallo (p.ej., usar los valores originales)
        return values 

    # 4. Devuelve los R^2 corregidos
    corrected_biases = result.x
    if len(values)==3:
        final_corrected_values = {
            'A' : values[0] - corrected_biases[0],
            'B' : values[1] - corrected_biases[1],
            'AB_union' : values[2] - corrected_biases[2]
        }
        return final_corrected_values
    elif len(values)==7:
        final_corrected_values = {
            'A' : values[0] - corrected_biases[0],
            'B' : values[1] - corrected_biases[1],
            'C' : values[2] - corrected_biases[2],
            'AB_union' : values[3] - corrected_biases[3],
            'AC_union' : values[4] - corrected_biases[4],
            'BC_union' : values[5] - corrected_biases[5],
            'ABC_union' : values[6] - corrected_biases[6]
        }
        return final_corrected_values

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
            self.mean = train_data.mean(dim=self.axis,  keepdim=True)
            self.std = train_data.std(dim=self.axis, unbiased=False,  keepdim=True)  # Use biased std for consistency with numpy
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
            data -= data.mean(dim=self.axis, keepdim=True)
            data /= data.std(dim=self.axis,  keepdim=True, unbiased=False)  # Use biased std for consistency with numpy
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


def subsampling_indexes_to_minimum(
    samples_info : dict,
    tollerance : float=0.1,
    kind : str='random_trials',
    seed : int = 42
    )-> Tuple[list, list, list, list]:
    """
    Subsampling indexes to minimum length of either design_matrix passed as input

    Parameters
    ----------
    samples_info : dict
        Dictionary containing the shifted indexes and trial lengths of leader and follower.
    tollerance : float
        Tollerance to downsample the indexes. Default is 0.1.
        If the relative difference to the minimum between the two indexes is less than this value, it will not be downsampled.
    kind : str
        Type of downsampling to be performed. Default is 'random_trials'.
        If 'random', random rows are removed. 
        Elif 'random_trials', random trials are removed, until difference is less than tollerance%
        Elif 'ordered_trials', bigger trials are removed first (enabling tradeoff to calculate rel. diff.), until difference is less than tollerance
        Else 'cutoff', select a contiguous segment of cutoff rows
    seed : int
        Seed to use in random algorithms

    Returns
    -------
    tuple
        Tuple containing the shifted indexes of the 1 and 2, respectively (same order as input)
    """
    kinds_of_sub = ['random', 'random_trials', 'ordered_trials', 'optimized_trials', 'cutoff']
    assert kind in kinds_of_sub, f'`{kind}` is not a valid kind of subsampling. Choose among {kinds_of_sub}'
    np.random.seed(seed=seed)
     
    results = {
        'shifted_indexes_leader1': None,
        'shifted_indexes_leader2': None,
        'shifted_indexes_follower1': None,
        'shifted_indexes_follower2': None
    }
    
    for subject in [1,2]:
        # Load trials
        shifted_indexes_follower = samples_info[f'keep_indexes_follower{subject}'].copy()
        shifted_indexes_leader = samples_info[f'keep_indexes_leader{subject}'].copy()
        
        trial_lengths_follower = samples_info[f'trial_lengths_follower{subject}'].copy()
        trial_lengths_leader = samples_info[f'trial_lengths_leader{subject}'].copy()
        
        # Compute trial difference
        minimum_length = min(len(shifted_indexes_leader), len(shifted_indexes_follower))
        relative_diff = (len(shifted_indexes_follower)-len(shifted_indexes_leader))/minimum_length
        
        if np.abs(relative_diff) >= tollerance:
            # Decide which gets cut
            if relative_diff > 0:
                exceded = 'follower'
                shift_exceded = shifted_indexes_follower
                shift_target = shifted_indexes_leader
                trial_lengths_exceded = trial_lengths_follower
            else:
                exceded = 'leader'
                shift_exceded = shifted_indexes_leader
                shift_target = shifted_indexes_follower
                trial_lengths_exceded = trial_lengths_leader
                tollerance *=-1
                                
            # Usefull variables
            number_of_indexes = len(shift_exceded)
            cutoff = len(shift_target)
            number_subsampled_indexes = number_of_indexes - cutoff
            trial_lengths_exceded_to_rem = trial_lengths_exceded.copy()

            # Remove indexes til tollerance is achieved
            while relative_diff > tollerance and len(trial_lengths_exceded_to_rem)!=1:
                
                # Remove a random trial
                if kind=='random_trials':
                    trial_to_remove = trial_lengths_exceded_to_rem.index(
                        np.random.choice(trial_lengths_exceded_to_rem[1:]) # 1: to avoid "trial 0"
                        )
                    _ = trial_lengths_exceded_to_rem.pop(trial_to_remove)
                    
                    lower_bound = sum(trial_lengths_exceded[:trial_to_remove])<np.array(shift_exceded)
                    upper_bound = np.array(shift_exceded)<sum(trial_lengths_exceded[:trial_to_remove]) + trial_lengths_exceded[trial_to_remove]
                    
                    shift_exceded = np.array(shift_exceded)[~(lower_bound&upper_bound)].tolist()
                    
                # Remove a ordered trial
                elif kind=='ordered_trials':
                    trial_lengths_exceded_to_rem = sorted(trial_lengths_exceded_to_rem)
                    trial_to_remove = trial_lengths_exceded.index(trial_lengths_exceded_to_rem[-1])
                    
                    _ = trial_lengths_exceded_to_rem.pop(len(trial_lengths_exceded_to_rem)-1)
                    
                    lower_bound = sum(trial_lengths_exceded[:trial_to_remove])<np.array(shift_exceded)
                    upper_bound = np.array(shift_exceded)<sum(trial_lengths_exceded[:trial_to_remove]) + trial_lengths_exceded[trial_to_remove]
                    
                    shift_exceded = np.array(shift_exceded)[~(lower_bound&upper_bound)].tolist()
                
                elif kind=='optimized_trials':
                    relative_differences = []
                    
                    # Calculate the relative diff for all trials
                    for trial in trial_lengths_exceded_to_rem:
                        trial_to_remove_ = trial_lengths_exceded.index(trial)
                        lower_bound_ = sum(trial_lengths_exceded[:trial_to_remove_])<np.array(shift_exceded)
                        upper_bound_ = np.array(shift_exceded)<sum(trial_lengths_exceded[:trial_to_remove_]) + trial_lengths_exceded[trial_to_remove_]
                        shift_exceded_ = np.array(shift_exceded)[~(lower_bound_&upper_bound_)].tolist()
                        relative_differences.append((len(shift_exceded_)-cutoff)/minimum_length)
                    
                    # Select the one that leave the rel. diff. closest to tollerance
                    trial_to_remove_rem = (np.abs(np.array(relative_differences))-np.abs(tollerance)).argmin()
                    trial_to_remove = trial_lengths_exceded.index(trial_lengths_exceded_to_rem[trial_to_remove_rem])
                    _ = trial_lengths_exceded_to_rem.pop(trial_to_remove_rem)
                    
                    lower_bound = sum(trial_lengths_exceded[:trial_to_remove])<np.array(shift_exceded)
                    upper_bound = np.array(shift_exceded)<sum(trial_lengths_exceded[:trial_to_remove]) + trial_lengths_exceded[trial_to_remove]
                    
                    shift_exceded = np.array(shift_exceded)[~(lower_bound&upper_bound)].tolist()
                
                # Remove samples at random
                elif kind=='random':
                    indices_to_remove = np.random.choice(
                                number_of_indexes,
                                size=number_subsampled_indexes,
                                replace=False
                                )
                    shift_exceded = list(np.delete(shift_exceded, indices_to_remove, axis=0))
                
                # Select a chunk of desire length
                else:
                    start = np.random.randint(0, number_subsampled_indexes)
                    shift_exceded = shift_exceded[start:start + cutoff]
                relative_diff = (len(shift_exceded)-cutoff)/minimum_length
        else:
            exceded = None
            
        if exceded=="follower":
            results[f'shifted_indexes_follower{subject}'] = shift_exceded
            results[f'shifted_indexes_leader{subject}'] = shift_target
        elif exceded=="leader":
            results[f'shifted_indexes_follower{subject}'] = shift_target
            results[f'shifted_indexes_leader{subject}'] = shift_exceded
        else:
            results[f'shifted_indexes_follower{subject}'] = shifted_indexes_follower
            results[f'shifted_indexes_leader{subject}'] = shifted_indexes_leader
        
    return results['shifted_indexes_leader1'], results['shifted_indexes_follower1'], results['shifted_indexes_leader2'], results['shifted_indexes_follower2']