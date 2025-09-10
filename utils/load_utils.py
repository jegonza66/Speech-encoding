from typing import Union
import pandas as pd
import numpy as np
import os

from utils.processing import shifted_matrix
import config

SEX_LIST = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']
ALLOWED_BANDS = [
    'Delta',
    'Theta',
    'Alpha',
    'Beta',
    'Beta1',
    'Beta2',
    'All',
    'Delta_Theta',
    'Alpha_Delta_Theta',
    'Broad',
    'Unfiltered'
]
ALLOWED_SITUATIONS = [
    'Internal',
    'External',
    'Internal_BS',
    'External_BS', 
    'Internal_All_Times',
    'External_All_Times',
    'All'
]
ALLOWED_STIMULI = [
    'Envelope', 'Envelope2', 'Phonological', 'Phonological1', 'Phonological2', 'Spectrogram', 
    'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 
    'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 
    'Phonemes', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 'Phonemes-Frequency', 
    'Phones', 'Phones-Envelope', 'Phones-Discrete',
    'Mistakes-Separated', 'Mistakes-Together', 'Control-Together', 'Control-Separated', 
    'Hearing-Turn',
    # 'Jitter', 'Shimmer'
]

def sort_stimuli_based_on_situation(
    subject_1:dict,
    subject_2:dict,
    situation:str = 'External'
)->tuple:
    """
    Sort stimuli based on situation

    Parameters
    ----------
    subject_1 : dict
        Dictionary with data of subject 1.
    subject_2 : dict
        Dictionary with data of subject 2.
    situation : str
        Situation considered when performing the analysis.  

    Returns
    -------
    tuple
        Updated dictionaries for subject 1 and subject 2.
    """
    sub1, sub2 = {}, {}
    if situation.startswith('External') or situation=='All':
        for key in subject_1:
            if key!='EEG':
               sub1[key] = subject_2[key]
               sub2[key] = subject_1[key]
            else:
                sub1[key] = subject_1[key]
                sub2[key] = subject_2[key]
    else:
        sub1 = subject_1
        sub2 = subject_2
    return sub1, sub2

def get_export_paths(
    preprocessed_data_path:str,
    band:str='Broad',
    )->dict:
    """
    Get export paths for each stimulus and EEG.

    Parameters
    ----------
    preprocessed_data_path : str
        Path directing to processed data.
    stimuli : str
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'.
    band : str
        Neural frequency band.
    
    Returns
    -------
    dict
        Dictionary with export paths for each stimulus and EEG.
    """
    export_paths = {}

    # Depending on filters the store path changes
    export_paths['Envelope'] = os.path.join(preprocessed_data_path, 'Envelope/')
    export_paths['EEG'] = os.path.join(preprocessed_data_path, f'EEG/{band}/')
    
    # The rest remain the same
    for stimulus in ALLOWED_STIMULI+config.stimuli:
        if stimulus in export_paths:
            continue
        else:
            export_paths[f'{stimulus}'] = os.path.join(preprocessed_data_path, f'{stimulus}/')
    return export_paths


def check_syntax(
    stimuli: Union[str, None]=None, 
    band: Union[str, None]=None, 
    situation: Union[str, None]=None, 
)->None:
    """
    Check if the syntax of the parameters is correct.

    Parameters
    ----------
    stimuli : Union[str, None], optional
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'.
    band : Union[str, None], optional
        Neural frequency band.
    situation : Union[str, None], optional
        Situation considered when performing the analysis.  
    Raises
    ------
    SyntaxError
        If 'stim' is not an allowed stimulus.
        If 'band' is not an allowed band frequency.
        If 'situation' is not an allowed situation.
    """
    
    # Check if band, stimuli and situation parameters where passed with the right syntax
    if stimuli is not None:
        for stimulus in stimuli.split('_'):
            if (stimulus not in ALLOWED_STIMULI):
                # Special dynamic case for DNNs name
                parts = stimulus.split('DNNs')
                layer, backbone = parts[-1].split('-')
                parts = [parts[0],layer]
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    continue
                else:
                    raise SyntaxError(f"{stimulus} is not an allowed stimulus. Allowed stimuli are: {ALLOWED_STIMULI}. If more than one stimulus is wanted, the separator should be '_'.")
    if band is not None:
        if not band.startswith('Custom-'):
            if band not in ALLOWED_BANDS:
                raise SyntaxError(f"{band} is not an allowed band frecuency. Allowed bands are: {ALLOWED_BANDS}")
    if situation is not None:
        if not (situation.split('_Silence')[0] in ALLOWED_SITUATIONS):
            raise SyntaxError(f"'{situation}' is not an allowed situation. Allowed ones are: {ALLOWED_SITUATIONS}")
    return None


def get_trials(
    session:int
)->list:
    """
    Get trials for a given session.

    Parameters
    ----------
    session : int
        Session number.

    Returns
    -------
    list
        List of trials for the given session.
    """
    phrases_path = f"data/phrases/S{session}/"
    trials = list(set([int(fname.split('.')[2]) for fname in os.listdir(phrases_path) if fname.endswith('phrases')]))
    trials.sort()
    return trials

def labeling(
    session:int,
    trial:int, 
    channel:int,
    sr:int
)->np.ndarray:
    """
    Gives an array with speaking channel: 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)

    Parameters
    ----------
    trial : int
        Number of trial
    channel : int
        Channel of audio signal

    Returns
    -------
    np.ndarray
        Speaking channels by sample, matching EEG sample rate and almost matching its length
    """
    
    # Read phrases into pandas.DataFrame
    speaker_path = os.path.join(
        f'data/phrases/S{session}/', 
        f's{session}.objects.{trial:02d}.channel{channel}.phrases'
    )
    speaker_table = pd.read_table(
        speaker_path, 
        header=None, 
        sep="\t"
    )

    # Replace and '#' by ''. And then all text by 1 and silences by 0 
    speaker_table.iloc[:, 2] = (
        speaker_table.iloc[:, 2].replace("#", "").apply(len) > 0
    ).apply(int)
    
    # Take difference in time and multiply it by sample rate in order to match envelope length (almost, miss by a sample or two)
    samples = np.round(
        (speaker_table[1] - speaker_table[0]).values * sr
    ).astype("int")
    
    # Repeat speaker labels by the number of samples in each phrase
    speaker = np.repeat(
        speaker_table.iloc[:, 2].values, 
        samples
    )
    
    # Same with listener
    listener_channel = (channel - 3) * -1
    listener_path = os.path.join(
        f'data/phrases/S{session}/', 
        f's{session}.objects.{trial:02d}.channel{listener_channel}.phrases'
    )
    listener_table = pd.read_table(
        listener_path, 
        header=None, 
        sep="\t"
    )

    # Replace and '#' by ''. And then all text by 1 and silences by 0
    listener_table.iloc[:, 2] = (
        listener_table.iloc[:, 2].replace("#", "").apply(len) > 0
    ).apply(int)
    
    # Take difference in time and multiply it by sample rate in order to match envelope length (almost, miss by a sample or two)
    samples = np.round(
        (listener_table[1] - listener_table[0]).values * sr
    ).astype("int")
    
    # Repeat speaker labels by the number of samples in each phrase
    listener = np.repeat(
        listener_table.iloc[:, 2].values,
        samples
    )

    # If there are differences in length, corrects them with 0-padding
    diff = len(speaker) - len(listener)
    if diff > 0:
        listener = np.concatenate(
            [listener, np.repeat(0, diff)]
        )
    elif diff < 0:
        speaker = np.concatenate(
            [speaker, np.repeat(0, np.abs(diff))]
        )

    # Return an array with envelope length, having values 3 if both participants are speaking; 2, if just locutor; 1, interlocutor and 0, silence
    return speaker + listener * 2

def match_lengths(
    dic:dict, 
    speaker_labels:np.ndarray,
    minimum:int = None
    )->tuple:
    """
    Match length of speaker labels and trial dictionary. It takes the minimum length between dic and speaker_labels (EEG)

    Parameters
    ----------
    dic : dict
        Trial dictionary containing data of stimuli and EEG
    speaker_labels : np.ndarray
        Labels of current speaker: 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)

    Returns
    -------
    tuple
        Updated dictionary, speaker_labels if minimum_length is passed. Elsewhise
        Updated dictionary, speaker_labels and minimum_length are returned.

    """
    # Get minimum between Speaker labels, EEG and envelope to make cutoff
    if minimum is None:
        minimum = min(
            [dic['Envelope'].shape[0]] + [dic['EEG'].shape[0]] + [len(speaker_labels)]
        )
    # Correct length 
    for key, data in dic.items():
        if isinstance(data, np.ndarray) and (data.shape[0] > minimum):
            dic[key] = data[:minimum]
    if len(speaker_labels) > minimum:
        speaker_labels = speaker_labels[:minimum]
    return dic, speaker_labels, minimum

def shifted_indexes_to_keep(
    speaker_labels:np.ndarray,
    situation:str = 'External'
)->np.ndarray:
    """
    Obtain shifted matrix indexes that match situation

    Parameters
    ----------
    speaker_labels : np.ndarray
        Labels of type of speaking for given sample

    Returns
    -------
    np.ndarray
        Indexes to keep for the analysis
    """
    # Change 0 with 4s, because shifted matrix pad zeros that could be mistaken with situation 0    
    speaker_labels = np.array(speaker_labels)
    speaker_labels = np.where(speaker_labels==0, 4, speaker_labels)

    # Computes shifted matrix
    shifted_matrix_speaker_labels = shifted_matrix(
        features=speaker_labels, 
        use_gpu=config.use_gpu,
        delays=config.delays 
    ).astype(float)
            
    if 'Silence' in situation and any(char.isdigit() for char in situation):
        percentage = int(situation.split('Silence_')[1])
        
        # Filter silence plus condition, plus padding
        filter_silence_external = ((shifted_matrix_speaker_labels==0)|(shifted_matrix_speaker_labels==4)|(shifted_matrix_speaker_labels==1)).all(axis=1)
        
        # Just windows with x percent of silence condition
        filter_silence_x_percent = (shifted_matrix_speaker_labels==4).sum(axis=1)<=int(percentage*len(config.delays)/100)
    
        return (filter_silence_external & filter_silence_x_percent).nonzero()[0]
    
    # Make the appropiate label
    if situation == 'All':
        return np.arange(len(shifted_matrix_speaker_labels))
    elif situation.endswith('BS'):
        situation_label = 3
    elif situation.startswith('External'):
        situation_label = 1
    elif situation.startswith('Internal'):
        situation_label = 2
    else: # Silence
        situation_label = 4

    # Shifted matrix index where the given situation is ocurring in all row (number of samples dimension)
    filter_indexes = (shifted_matrix_speaker_labels==situation_label) | (shifted_matrix_speaker_labels==0)
    return (filter_indexes).all(axis=1).nonzero()[0]
