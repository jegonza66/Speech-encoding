# Standard libraries
import numpy as np, pickle, os, sys, mne, csv#, warnings, pandas as pd, scipy
# from typing import Union

class Suppress_print:
    """
    A context manager to suppress the standard output (stdout).
    This class can be used to temporarily suppress the output of print statements
    or any other output to the standard output stream.
    
    Methods:
    --------
    __enter__():
        Redirects the standard output to os.devnull, effectively suppressing any output.
    __exit__(exc_type, exc_value, traceback):
        Restores the original standard output stream.
    """
    def __enter__(
        self
        )->None:
        """
        Redirect the standard output to os.devnull.
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(
        self, 
        exc_type, 
        exc_val, 
        exc_tb
        )->None:
        """
        Restore the original standard output stream.
        
        Parameters
        ----------
        exc_type : type
            The exception type.
        exc_val : Exception
            The exception value.
        exc_tb : traceback
            The traceback object.
            
        Returns
        -------
        None
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout

def all_possible_combinations(
    a:list
    )->list:
    """
    Generate all possible combinations of elements in the list `a`.
    
    Parameters
    ----------
    a : list
        List of elements to generate combinations from.
    
    Returns
    -------
    list
        A list of lists, where each sublist is a possible combination of elements from `a`.
    
    Raises
    ------
    TypeError
        If the input is not a list.
    """
    
    if not isinstance(a, list):
        raise TypeError("Input must be a list.")
    
    if len(a) == 0:
        return [[]]
    cs = []
    for c in all_possible_combinations(a[1:]):
        cs += [c, c + [a[0]]]
    return cs

def load_pickle(
    path:str
    )->object:
    """
    Load a pickle file from the specified path.
    
    Parameters
    ----------
    path : str
        Path to the pickle file.
    
    Returns
    -------
    object
        The object loaded from the pickle file.
    
    Raises
    ------
    Exception
        If the file does not exist or if there is an error loading the pickle file.
    """
    if os.path.isfile(path):
        try:
            with open(file = path, mode = "rb") as archive:
                data = pickle.load(file = archive)
            return data
        except:
            raise Exception("Something went wrong, check extension.")
    else:
        raise Exception(f"The file '{path}' doesn't exist.")
    
def dump_pickle(
    path:str, 
    obj, 
    rewrite:bool=False, 
    verbose:bool=False
    )->None:
    """
    Save an object to a pickle file at the specified path.
    
    Parameters
    ----------
    path : str
        Path to the pickle file.
    obj : object
        The object to be saved.
    rewrite : bool, optional
        If True, overwrite the file if it already exists. Default is False.
    verbose : bool, optional
        If True, print a message if the file is overwritten. Default is False.
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If the file already exists and rewrite is False.
    Exception
        If there is an error saving the object to the pickle file.
    """
    isfile = os.path.isfile(path)
    if isfile and not rewrite:
        raise Exception("This file already exists, change 'rewrite=True'.")
    try:
        with open(file = path, mode = "wb") as archive:
            pickle.dump(file = archive, obj=obj)
        if isfile and verbose:
            print(f'Atention: file overwritten in {path}')
    except:
        raise Exception("Something went wrong when saving")
    
def dict_to_csv(
    path:str, 
    obj:dict, 
    rewrite:bool=False, 
    verbose:bool=False
    )->None:
    """
    Save a dictionary to a CSV file at the specified path.
    
    Parameters
    ----------
    path : str
        Path to the CSV file.
    obj : dict
        The dictionary to be saved.
    rewrite : bool, optional
        If True, overwrite the file if it already exists. Default is False.
    verbose : bool, optional
        If True, print a message if the file is overwritten. Default is False.
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If the file already exists and rewrite is False.
    Exception
        If there is an error saving the dictionary to the CSV file.
    """
    isfile = os.path.isfile(path)
    if isfile and not rewrite:
        raise Exception("This file already exists, change 'rewrite=True'.")
    try:
        with open(path, 'w') as csv_file:  
            writer = csv.writer(csv_file, delimiter=':')
            for key, value in obj.items():
                writer.writerow([key, value])
        if isfile and verbose:
            print(f'Atention: file overwritten in {path}')
    except:
        raise Exception("Something went wrong when saving")

def iteration_percentage(
    txt:str, 
    i:int, 
    length_of_iterator:int
    )->None:
    """
    Display the iteration progress as a percentage bar.
    
    Parameters
    ----------
    txt : str
        Text to display before the percentage bar.
    i : int
        Current iteration index.
    length_of_iterator : int
        Total number of iterations.
    
    Returns
    -------
    None
    """
    l = int(50*(i+1)/length_of_iterator)
    if (i+1) == length_of_iterator:
        percentage_bar =  f"[{'*'*(l):50s}] {(l*2)/100:.0%}\n"
    else:
        percentage_bar =  f"[{'·'*(l):50s}] {(l*2)/100:.0%}\n"
    sys.stdout.write(txt+'\n'+percentage_bar)

def get_maximum_correlation_channels(
    average_correlation_across_subject:np.ndarray,
    number_of_lat_channels:int=12,
    lateralization:bool=False
    )->list:
    """
    Get the channels with the maximum correlation across subjects.

    Parameters
    ----------
    average_correlation_across_subject : np.ndarray
        Array containing the average correlation values across subjects for each channel.
    number_of_lat_channels : int, optional
        Number of lateralization channels to consider. Default is 12.
    lateralization : bool, optional
        If True, consider lateralization and return separate lists for left and right channels. Default is False.

    Returns
    -------
    list
        If lateralization is False, returns a list of booleans indicating the channels with the highest correlation.
    tuple
        If lateralization is True, returns two lists of booleans indicating the channels with the highest correlation for left and right channels respectively.

    Raises
    ------
    Exception
        If there is an error during the process.
    """
    # Get channels of headset
    montage = mne.channels.make_standard_montage('biosemi128')
    channel_names = montage.ch_names

    # List of all left and right channels
    all_channels_right = ['B27','B28','B29','B30','B31','B32','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16']
    all_channels_left = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','C24','C25','C26','C27','C28','C29','C30','C31','C32']
    
    if lateralization:
        # Get right and left channels that are used in this experiment
        ordered_chs_right = [ch for ch in channel_names if ch in all_channels_right]
        ordered_chs_left = [ch for ch in channel_names if ch in all_channels_left]

        # Filter coefficient to get respective correlations
        corr_right = average_correlation_across_subject[[ch in all_channels_right for ch in channel_names]]
        corr_left = average_correlation_across_subject[[ch in all_channels_left for ch in channel_names]]

        # Now get relevant indexes, sorted by correlation
        sorted_chs_right = [x for _, x in sorted(zip(corr_right, ordered_chs_right))]
        sorted_chs_left = [x for _, x in sorted(zip(corr_left, ordered_chs_left))]

        # Get most correlated channels for lateralization
        if number_of_lat_channels:
            corr_right = np.sort(corr_right)[-number_of_lat_channels:]
            corr_left = np.sort(corr_left)[-number_of_lat_channels:]
            sorted_chs_right = sorted_chs_right[-number_of_lat_channels:]
            sorted_chs_left = sorted_chs_left[-number_of_lat_channels:]
        return [ch in sorted_chs_left for ch in channel_names], [ch in sorted_chs_right for ch in channel_names]
    else:
        # List all channels
        all_channels = all_channels_left+all_channels_right

        # Get channels that are used in the experiment and corresponding correlations
        ordered_chs = [ch for ch in channel_names if ch in all_channels]
        corr = average_correlation_across_subject[[ch in all_channels for ch in channel_names]]

        # Now get relevant indexes, sorted by correlation
        sorted_chs = [x for _, x in sorted(zip(corr, ordered_chs))]

        # Get most correlated channels for lateralization
        if number_of_lat_channels:
            corr = np.sort(corr)[-number_of_lat_channels:]
            sorted_chs = sorted_chs[-number_of_lat_channels:]
        return [ch in sorted_chs for ch in channel_names]

def maximo_comun_divisor(
    a:int, 
    b:int
    )->int:
    """
    Calculate the greatest common divisor (GCD) of two integers using the Euclidean algorithm.
    
    Parameters
    ----------
    a : int
        The first integer.
    b : int
        The second integer.
    
    Returns
    -------
    int
        The greatest common divisor of the two integers.
    
    Raises
    ------
    ValueError
        If either of the inputs is not an integer.
    """
    temporal = 0
    while b != 0:
        temporal = b
        b = a % b
        a = temporal
    return a

def minimo_comun_multiplo(
    a:int, 
    b:int
    )->int:
    """
    Calculate the least common multiple (LCM) of two integers.
    
    Parameters
    ----------
    a : int
        The first integer.
    b : int
        The second integer.
    
    Returns
    -------
    int
        The least common multiple of the two integers.
    
    Raises
    ------
    ValueError
        If either of the inputs is not an integer.
    """
    return (a * b) / maximo_comun_divisor(a, b)

def cohen_d(
    x:np.ndarray, 
    y:np.ndarray
    )->float:
    """
    Calculate Cohen's d effect size between two samples.
    
    Parameters
    ----------
    x : array-like
        The first sample data.
    y : array-like
        The second sample data.
    
    Returns
    -------
    float
        The calculated Cohen's d value.
    
    Raises
    ------
    ValueError
        If the input arrays have different lengths.
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    cohen_d = abs((np.mean(x) - np.mean(y))) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return cohen_d

# def mne_to_numpy(obj:Union[mne.io.array.array.RawArray,mne.io.eeglab.eeglab.RawEEGLAB,list], verbose:bool=True):
#     """Transform mne arrays and Raw EEG objects to numpy ndarrays. If obj is 1D, returns a flatten array.

#     Parameters
#     ----------
#     obj : Union[mne.io.array.array.RawArray,mne.io.eeglab.eeglab.RawEEGLAB,lisr]
#         mne Array, RawEEGLAB or list of them.
#     verbose : bool
#         Wether to print warning that data already is np.ndarray.

#     Returns
#     -------
#     np.array
#         Array representation of object if it's not a list
#     list
#         A list of arrays representation of given list of objects
#     """
#     def to_numpy(obj_sub:Union[mne.io.array.array.RawArray,mne.io.eeglab.eeglab.RawEEGLAB]):
        
#         # Check is it's already a np.ndarray
#         if isinstance(obj_sub, np.ndarray):
#             if verbose:
#                 warnings.warn(f'The object passed already is a np.ndarray')
#             return obj_sub

#         # In general, mne objects are shaped as #chann X #samples, and usually we use #samples X #chann
#         data = obj_sub.get_data().T
#         # Assuming object doesn't have more than 2D. For 1D data, makes it flatten
#         if data.shape[1]==1:
#             return data.flatten()
#         else:
#             return data

#     if isinstance(obj, list):
#         output_list = []
#         for arr in obj:
#             output_list.append(to_numpy(obj_sub=arr))
#         return output_list
#     else:
#         return to_numpy(obj_sub=obj)
# def make_df(*args):
#     returns = []
#     for var in args:
#         returns.append(pd.DataFrame(var))
#     return tuple(returns)


# def correlacion(x, y, axis=0):
#     if (len(x) != len(y)):
#         print('Error: Vectores de diferente tamaño: {} y {}.'.format(len(x), len(y)))
#     else:
#         Correlaciones = []
#         for j in range(x.shape[axis]):
#             a, b = x[j], y[j]
#             corr = [1.]
#             for i in range(int(len(a) / 2)):
#                 corr.append(np.corrcoef(a[:-i - 1], b[i + 1:])[0, 1])
#             Correlaciones.append(corr)
#             print("\rProgress: {}%".format(int((j + 1) * 100 / x.shape[axis])), end='')
#     return np.array(Correlaciones)


# def decorrelation_time(Estimulos, sr, Autocorrelation_value = 0.1):
#     Autocorrelations = correlacion(Estimulos, Estimulos)
#     decorrelation_times = []

#     for Autocorr in Autocorrelations:
#         for i in range(len(Autocorr)):
#             if Autocorr[i] < Autocorrelation_value: break
#         dif_paso = Autocorr[i - 1] - Autocorr[i]
#         dif_01 = Autocorr[i - 1] - Autocorrelation_value
#         dif_time = dif_01 / sr / dif_paso
#         decorr_time = ((i - 1) / sr + dif_time) * 1000

#         if decorr_time > 0 and decorr_time < len(Autocorr)/sr*1000:
#             decorrelation_times.append(decorr_time)

#     return decorrelation_times

# def findFreeinterval(arr):
#     # If there are no set of interval
#     N = len(arr)
#     if N < 1:
#         return

#     # To store the set of free interval
#     P = []

#     # Sort the given interval according
#     # Starting time
#     arr.sort(key=lambda a: a[0])

#     # Iterate over all the interval
#     for i in range(1, N):

#         # Previous interval end
#         prevEnd = arr[i - 1][1]

#         # Current interval start
#         currStart = arr[i][0]

#         # If Previous Interval is less
#         # than current Interval then we
#         # store that answer
#         if prevEnd < currStart:
#             P.append([prevEnd, currStart])
#     return P


# def slope(x, y):
#     x = x[~np.isnan(y)]
#     y = y[~np.isnan(y)]
#     return scipy.stats.linregress(x, y)[0]

# def f_loss_Corr(x, stim, y, alpha):
#     """
#     Description
    
#     Parameters
#     ----------
#     param_name : type
#         Parameter description
    
#     Returns
#     -------
#     return_type
#         Return description
    
#     Raises
#     ------
#     Exception
#         Exception description
#     """
#     return -np.corrcoef(np.dot(stim,x), y)[0, 1] + alpha*sum(abs(x))


# def f_loss_Corr_ridge(x, stim, y):
#     """
#     Description
    
#     Parameters
#     ----------
#     param_name : type
#         Parameter description
    
#     Returns
#     -------
#     return_type
#         Return description
    
#     Raises
#     ------
#     Exception
#         Exception description
#     """
#     return -np.corrcoef(np.dot(stim,x), y)[0, 1]


# def consecutive(data, stepsize=1):
#     """
#     Description
    
#     Parameters
#     ----------
#     param_name : type
#         Parameter description
    
#     Returns
#     -------
#     return_type
#         Return description
    
#     Raises
#     ------
#     Exception
#         Exception description
#     """
#     return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


# def rename_paths(Stims_preprocess, EEG_preprocess, stim, Band, tmin, tmax, *paths):
#     returns = []
#     for path in paths:
#         path += 'Stim_{}_EEG_Band_{}/'.format(stim, Band)
#         returns.append(path)
#     return tuple(returns)


# def trunc(values, decs=0):
#     return np.trunc(values * 10 ** decs) / (10 ** decs)


# def flatten_list(t):
#     return [item for sublist in t for item in sublist]


# def make_array_dict(dict):
#     keys = list(dict.keys())
#     for key in keys:
#         dict[key] = dict[key].to_numpy()


# def make_array(*args):
#     returns = []
#     for var in args:
#         returns.append(np.array(var))
#     return tuple(returns)


# def make_df_dict(dict):
#     keys = list(dict.keys())
#     keys.remove('info')
#     if 'Phonemes' in keys:
#         keys.remove('Phonemes')
#     for key in keys:
#         dict[key] = pd.DataFrame(dict[key])





# def sliding_window(df, window_size=6, func='slope', step=1, min_points=6):
#     res = []
#     for i in range(0, len(df), step):
#         rows = df.iloc[i:i + window_size]
#         if func == "mean":
#             res_i = rows.apply(lambda y: np.nanmean(y) if sum(~np.isnan(y)) >= min_points else np.nan)
#         elif func == "slope":
#             x = rows.index
#             res_i = rows.apply(lambda y: slope(x, y) if sum(~np.isnan(y)) >= min_points else np.nan)
#         res.append(res_i)
#     res = np.array(res)
#     return res
