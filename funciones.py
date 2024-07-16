# Standard libraries
import numpy as np, pandas as pd, scipy, pickle, os, sys, mne, warnings, csv
from typing import Union

class Suppress_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def all_possible_combinations(a:list):
    """Make a list with all possible combinations of the elements of the list

    Parameters
    ----------
    a : list
        _description_

    Returns
    -------
    list
        list of lists with all possible combinations of the elements of the list
    """
    if len(a) == 0:
        return [[]]
    cs = []
    for c in all_possible_combinations(a[1:]):
        cs += [c, c+[a[0]]]
    return cs

def load_pickle(path:str):
    """Loads pickle file

    Parameters
    ----------
    path : str
        Path to pickle file

    Returns
    -------
    _type_
        Extracted file

    Raises
    ------
    Exception
        Something goes wrong, probably it's not a .pkl
    Exception
        The file doesn't exist
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
    
def dump_pickle(path:str, obj, rewrite:bool=False, verbose:bool=False):
    """Dumps object into pickle

    Parameters
    ----------
    path : str
        Path to pickle file
    obj : _type_
        Object to save as pickle. Almost anything
    rewrite : bool, optional
        If already exists a file named as path it rewrites it, by default False
    verbose : bool, optional
        Whether to print information about rewritting, by default False

    Raises
    ------
    Exception
        If the file already exists and rewrite wasnt called.
    Exception
        Something went wrong.
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
    
def dict_to_csv(path:str, obj:dict, rewrite:bool=False, verbose:bool=False):
    """Dumps dict into csv

    Parameters
    ----------
    path : str
        Path to pickle file
    obj : dict
        Dictionary to save as csv
    rewrite : bool, optional
        If already exists a file named as path it rewrites it, by default False
    verbose : bool, optional
        Whether to print information about rewritting, by default False

    Raises
    ------
    Exception
        If the file already exists and rewrite wasnt called.
    Exception
        Something went wrong.
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

def iteration_percentage(txt:str, i:int, length_of_iterator:int):
    """Adds a percentage bar at the bottom of a print in a loop.

    Parameters
    ----------
    txt : str
        Text wanted to print
    i : int
        i-th iteration of the loop
    length_of_iterator : int
        Length of the iterator
    """
    l = int(50*(i+1)/length_of_iterator)
    if (i+1) == length_of_iterator:
        percentage_bar =  f"[{'*'*(l):50s}] {(l*2)/100:.0%}\n"
    else:
        percentage_bar =  f"[{'·'*(l):50s}] {(l*2)/100:.0%}\n"
    sys.stdout.write(txt+'\n'+percentage_bar)

def mne_to_numpy(obj:Union[mne.io.array.array.RawArray,mne.io.eeglab.eeglab.RawEEGLAB,list], verbose:bool=True):
    """Transform mne arrays and Raw EEG objects to numpy ndarrays. If obj is 1D, returns a flatten array.

    Parameters
    ----------
    obj : Union[mne.io.array.array.RawArray,mne.io.eeglab.eeglab.RawEEGLAB,lisr]
        mne Array, RawEEGLAB or list of them.
    verbose : bool
        Wether to print warning that data already is np.ndarray.

    Returns
    -------
    np.array
        Array representation of object if it's not a list
    list
        A list of arrays representation of given list of objects
    """
    def to_numpy(obj_sub:Union[mne.io.array.array.RawArray,mne.io.eeglab.eeglab.RawEEGLAB]):
        
        # Check is it's already a np.ndarray
        if isinstance(obj_sub, np.ndarray):
            if verbose:
                warnings.warn(f'The object passed already is a np.ndarray')
            return obj_sub

        # In general, mne objects are shaped as #chann X #samples, and usually we use #samples X #chann
        data = obj_sub.get_data().T
        # Assuming object doesn't have more than 2D. For 1D data, makes it flatten
        if data.shape[1]==1:
            return data.flatten()
        else:
            return data

    if isinstance(obj, list):
        output_list = []
        for arr in obj:
            output_list.append(to_numpy(obj_sub=arr))
        return output_list
    else:
        return to_numpy(obj_sub=obj)
    
def match_lengths(dic, speaker_labels, minimum_length):
        """Match length of speaker labels and trial dictionary.

        Parameters
        ----------
        dic : dict
            Trial dictionary containing data of stimuli and EEG
        speaker_labels : np.ndarray
            Labels of current speaker.
        minimum_length : int, optional
            Length to match data length with. If not passed, takes the minimum length between dic and speaker_labels

        Returns
        -------
        tuple
            Updated dictionary, speaker_labels if minimum_length is passed. Elsewhise
            Updated dictionary, speaker_labels and minimum_length are returned.

        """
        # Get minimum array length (this includes features and EEG data)
        if minimum_length:
            minimum = minimum_length
        else:
            minimum = min([dic[key].get_data().T.shape[0] for key in dic] + [len(speaker_labels)])

        # Correct length and update mne array
        for key in dic:
            if key!= 'info' and key!='EEG':
                data = dic[key].get_data().T
                if data.shape[0] > minimum:
                    dic[key] = mne.io.RawArray(data=data[:minimum].T, info=dic[key].info, verbose=True)
            elif key=='EEG':
                if dic[key].get_data().shape[1]>minimum:
                    eeg_times = dic[key].times.tolist()
                    dic[key].crop(tmin=eeg_times[0], tmax=eeg_times[minimum], verbose=False)

        if len(speaker_labels) > minimum:
            speaker_labels = speaker_labels[:minimum]
        if minimum_length:
            print(dic, speaker_labels)
            return dic, speaker_labels
        else:
            print(dic, speaker_labels, minimum)
            return dic, speaker_labels, minimum


def maximo_comun_divisor(a, b):
    temporal = 0
    while b != 0:
        temporal = b
        b = a % b
        a = temporal
    return a


def minimo_comun_multiplo(a, b):
    return (a * b) / maximo_comun_divisor(a, b)


def f_loss_Corr(x, stim, y, alpha):
    return -np.corrcoef(np.dot(stim,x), y)[0, 1] + alpha*sum(abs(x))


def f_loss_Corr_ridge(x, stim, y):
    return -np.corrcoef(np.dot(stim,x), y)[0, 1]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


def rename_paths(Stims_preprocess, EEG_preprocess, stim, Band, tmin, tmax, *paths):
    returns = []
    for path in paths:
        path += 'Stim_{}_EEG_Band_{}/'.format(stim, Band)
        returns.append(path)
    return tuple(returns)


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def make_array_dict(dict):
    keys = list(dict.keys())
    for key in keys:
        dict[key] = dict[key].to_numpy()


def make_array(*args):
    returns = []
    for var in args:
        returns.append(np.array(var))
    return tuple(returns)


def make_df_dict(dict):
    keys = list(dict.keys())
    keys.remove('info')
    if 'Phonemes' in keys:
        keys.remove('Phonemes')
    for key in keys:
        dict[key] = pd.DataFrame(dict[key])


def make_df(*args):
    returns = []
    for var in args:
        returns.append(pd.DataFrame(var))
    return tuple(returns)


def correlacion(x, y, axis=0):
    if (len(x) != len(y)):
        print('Error: Vectores de diferente tamaño: {} y {}.'.format(len(x), len(y)))
    else:
        Correlaciones = []
        for j in range(x.shape[axis]):
            a, b = x[j], y[j]
            corr = [1.]
            for i in range(int(len(a) / 2)):
                corr.append(np.corrcoef(a[:-i - 1], b[i + 1:])[0, 1])
            Correlaciones.append(corr)
            print("\rProgress: {}%".format(int((j + 1) * 100 / x.shape[axis])), end='')
    return np.array(Correlaciones)


def decorrelation_time(Estimulos, sr, Autocorrelation_value = 0.1):
    Autocorrelations = correlacion(Estimulos, Estimulos)
    decorrelation_times = []

    for Autocorr in Autocorrelations:
        for i in range(len(Autocorr)):
            if Autocorr[i] < Autocorrelation_value: break
        dif_paso = Autocorr[i - 1] - Autocorr[i]
        dif_01 = Autocorr[i - 1] - Autocorrelation_value
        dif_time = dif_01 / sr / dif_paso
        decorr_time = ((i - 1) / sr + dif_time) * 1000

        if decorr_time > 0 and decorr_time < len(Autocorr)/sr*1000:
            decorrelation_times.append(decorr_time)

    return decorrelation_times

def findFreeinterval(arr):
    # If there are no set of interval
    N = len(arr)
    if N < 1:
        return

    # To store the set of free interval
    P = []

    # Sort the given interval according
    # Starting time
    arr.sort(key=lambda a: a[0])

    # Iterate over all the interval
    for i in range(1, N):

        # Previous interval end
        prevEnd = arr[i - 1][1]

        # Current interval start
        currStart = arr[i][0]

        # If Previous Interval is less
        # than current Interval then we
        # store that answer
        if prevEnd < currStart:
            P.append([prevEnd, currStart])
    return P


def slope(x, y):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return scipy.stats.linregress(x, y)[0]


def sliding_window(df, window_size=6, func='slope', step=1, min_points=6):
    res = []
    for i in range(0, len(df), step):
        rows = df.iloc[i:i + window_size]
        if func == "mean":
            res_i = rows.apply(lambda y: np.nanmean(y) if sum(~np.isnan(y)) >= min_points else np.nan)
        elif func == "slope":
            x = rows.index
            res_i = rows.apply(lambda y: slope(x, y) if sum(~np.isnan(y)) >= min_points else np.nan)
        res.append(res_i)
    res = np.array(res)
    return res

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    cohne_d = abs((np.mean(x) - np.mean(y))) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return cohne_d