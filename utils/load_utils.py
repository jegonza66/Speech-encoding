from fractions import Fraction
from typing import Union
import pandas as pd
import numpy as np
import mne
import os

# Set TensorFlow environment variables BEFORE any TensorFlow imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors only

from sklearn.linear_model import LinearRegression
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal as sgn
from scipy.io import wavfile
import torch

import utils.general_functions as general_functions
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
    'Hearing-Turn', 'Audio-Resampled',
    'EEG-feature',
    'ROIs'
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
            if key!='EEG' and key!='ROI':
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
    stimuli,
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
    for stimulus in ALLOWED_STIMULI+config.stimuli+stimuli.split('_'):
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
            # Special dynamic case for DNNs and spectrograms name
            if (stimulus not in ALLOWED_STIMULI):
                if stimulus.startswith('Spectrogram'):
                    n_mels = int(stimulus.split('-')[-1])
                    if isinstance(n_mels, int) and (n_mels > 0):
                        continue
                    else:
                        raise SyntaxError(f"{stimulus} is not an allowed stimulus. Allowed stimuli are: {ALLOWED_STIMULI}. If more than one stimulus is wanted, the separator should be '_'.")
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

def get_dnn_reduced_representation(
    wavfile:str,
    backbone:str,
    encoder_layer:int,
    n_components:int,
    full_layer_path:str
)-> PCA:
    """
    Fits PCA to the concatenated hidden states of a DNN model and returns 
    the reduction matrix to apply the transformation.
    """
    # data\wavs\S21\s21.objects.01.channel1.wav
    session = int(wavfile.split('S')[1][:2])
    trial_ = int(wavfile.split('objects.')[1][:2])
    channel = int(wavfile.split('channel')[1].split('.')[0])
    matrix_path = os.path.normpath(
        f"data/DNNs_cache/matrix_reduction/{backbone}-{n_components}/layer_{encoder_layer}/session_{session}_channel_{channel}.npy"
    )
    scaler_path = matrix_path.replace('.npy', '_scaler.npy')
    if os.path.exists(matrix_path) and os.path.exists(scaler_path):
        return np.load(scaler_path, allow_pickle=True).item(), np.load(matrix_path, allow_pickle=True).item()

    dnn_representations = []
    for trial in get_trials(session):
        layer_trial = full_layer_path.replace(f'trial_{trial_:02d}', f'trial_{trial:02d}')
        dnn_representations.append(general_functions.load_pickle(path=layer_trial))
    full_dnn_representation = np.concatenate(dnn_representations, axis=0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    full_dnn_representation = scaler.fit_transform(full_dnn_representation)
    pca = PCA(
        n_components=min(n_components, full_dnn_representation.shape[1])
    )
    pca.fit(full_dnn_representation)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    np.save(scaler_path, scaler)
    np.save(matrix_path, pca)
    os.makedirs(os.path.dirname(matrix_path.replace('.npy', '.txt')), exist_ok=True)
    with open(matrix_path.replace('.npy', '.txt'), 'w') as f:
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Encoder layer: {encoder_layer}\n")
        f.write(f"Number of components: {n_components}\n")
        f.write(f"Full representation shape: {full_dnn_representation.shape}\n")
        f.write(f"Explained variance ratio: {pca.explained_variance_ratio_}\n")
        f.write(f"Explained variance (cumulative): {pca.explained_variance_ratio_.sum()}\n")
    return scaler, pca


def get_reduced_audio_representation(
    wav_path_interlocutor:str
)-> np.ndarray:
    
    ORIGINAL_SAMPLE_RATE = pad_width = 1024
    DESIRED_SAMPLE_RATE = 512
    SESSION = int(wav_path_interlocutor.split('S')[1][:2])
    TRIAL = int(wav_path_interlocutor.split('objects.')[1][:2])
    CHANNEL_INTERLOCUTOR = int(wav_path_interlocutor.split('channel')[1].split('.')[0])

    audio_resampled_path = os.path.normpath(
        f"data/resampled_audio_cache/session_{SESSION}_channel_{CHANNEL_INTERLOCUTOR}.npy"
    )

    if os.path.exists(audio_resampled_path):
       final_wav=np.load(audio_resampled_path, allow_pickle=True)
    else:
        os.makedirs(os.path.dirname(audio_resampled_path), exist_ok=True)
        audio_total = []
        for trial in get_trials(SESSION):
            wav_path_trial = wav_path_interlocutor.replace(f'objects.{TRIAL:02d}', f'objects.{trial:02d}')
            sr_wav, wav = wavfile.read(wav_path_trial)
            wav = wav.astype(np.float32)
            audio_total.append(wav)
        wav = np.concatenate(audio_total, axis=0)
        wav = wav / np.max(np.abs(wav))

        # Resample to match EEG original sampling rate
        ratio = Fraction(ORIGINAL_SAMPLE_RATE, sr_wav)
        up, down = ratio.numerator, ratio.denominator
        wav = sgn.resample_poly(wav, up, down, padtype='reflect')
        
        # A. High-pass (0.1 Hz, Order 16896)
        b_hp = sgn.firwin(
            numtaps=16896 + 1, 
            cutoff=0.1, 
            fs=ORIGINAL_SAMPLE_RATE, 
            pass_zero=False, 
            window='hamming'
        )
        # B. Low-pass (100 Hz, Order 100)
        b_lp = sgn.firwin(
            numtaps=100 + 1, 
            cutoff=100, 
            fs=ORIGINAL_SAMPLE_RATE, 
            pass_zero=True, 
            window='hamming'
        )
        # C. Notch (49-51 Hz, Order 3380) -> Band-stop
        b_notch = sgn.firwin(
            numtaps=3380 + 1, 
            cutoff=[49, 51], 
            fs=ORIGINAL_SAMPLE_RATE, 
            pass_zero=True, # Band-stop (pasa extremos, corta centro)
            window='hamming'
        )
        wav = sgn.filtfilt(b_hp, 1.0, wav)
        wav = sgn.filtfilt(b_lp, 1.0, wav)
        wav = sgn.filtfilt(b_notch, 1.0, wav)
        
        # Resample to desired sample rate
        ratio = Fraction(DESIRED_SAMPLE_RATE, ORIGINAL_SAMPLE_RATE)
        up, down = ratio.numerator, ratio.denominator
        final_wav = sgn.resample_poly(wav, up, down, padtype='reflect')
        np.save(audio_resampled_path, final_wav)
    
    start_sample_target = 0
    end_sample_target = 0
    current_sample = 0
    for trial in get_trials(SESSION):
        wav_path_trial = wav_path_interlocutor.replace(f'objects.{TRIAL:02d}', f'objects.{trial:02d}')
        sr_wav, wav_temp = wavfile.read(wav_path_trial, mmap=True) 
        trial_time = wav_temp.shape[0]/sr_wav
        len_target = int(round(trial_time * DESIRED_SAMPLE_RATE))
        if trial == TRIAL:
            start_sample_target = current_sample
            end_sample_target = current_sample + len_target
            break # Ya lo encontramos
            
        current_sample += len_target
    return final_wav[int(start_sample_target):int(end_sample_target)]
# # PARA AGILIZAR EL PROCESO DE LIMPIEZA DE CROSSTALK EN TODOS LOS AUDIOS CONVIENE CORRER ESTE CHUNK ANTES DEL LOAD.PY
# import config 
# from tqdm import tqdm
# for session in tqdm(config.sessions, total=len(config.sessions)):
#     for channel in [1, 2]:
#         wav_path_interlocutor = f"data/wavs/S{session}/s{session}.objects.01.channel{channel}.wav"
#         get_reduced_audio_representation(
#             wav_path_interlocutor=wav_path_interlocutor
#         )
def clean_crosstalk(
    audio_data:np.ndarray,
    eeg_data:np.ndarray,
) -> np.ndarray:
    """
    Elimina el crosstalk de audio del EEG mediante regresión lineal.
    
    Parameters:
        audio_data:
        np.ndarray: (n_samples, ) o (n_samples, 1) Señal de audio.
        eeg_data:
        np.ndarray: (n_samples, n_channels) Señal EEG.

    Returns:
        eeg_clean: (n_samples, n_channels) Señal limpia.
    """
    audio_data = audio_data.reshape(-1, 1) if audio_data.ndim == 1 else audio_data
    minimum_length = min(eeg_data.shape[0], audio_data.shape[0])
    eeg_data = eeg_data[:minimum_length]
    audio_data = audio_data[:minimum_length]
    
    reg = LinearRegression(fit_intercept=True)
    reg.fit(audio_data, eeg_data)
    noise_prediction = reg.predict(audio_data)
    eeg_clean = eeg_data - noise_prediction
    # max_beta = np.max(np.abs(reg.coef_))
    
    # print(f"\nMax beta for crosstalk removal: {max_beta}")
    # print(f"\nExplained variance for crosstalk removal: {reg.score(audio_data, eeg_data)}")
    # print(f"\nAverage noise prediction power: {np.mean(noise_prediction)}\n\n")
    return eeg_clean

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path")
    parser.add_argument("--model_id")
    parser.add_argument("--encoder_layer", type=int)
    parser.add_argument("--output_path")
    parser.add_argument("--half_precision", type=bool, default=False)
    args = parser.parse_args()

    # Load audio
    import numpy as np
    import torch
    import sys
    import scipy.io.wavfile as wavfile
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    import utils.general_functions as general_functions
    sr, wav = wavfile.read(args.wav_path)    
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32) / max(1, np.iinfo(wav.dtype).max)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_id)
    model = WavLMModel.from_pretrained(args.model_id).to("cuda").eval()
    model_sr = getattr(processor, "sampling_rate", 16000)
    if sr != model_sr:
        from scipy import signal as sgn
        wav_model = sgn.resample_poly(wav, up=model_sr, down=sr)
    else:
        wav_model = wav

    with torch.no_grad():
        inputs = processor(
            wav_model,
            sampling_rate=model_sr,
            return_tensors="pt",
            padding=False
        )
        if args.half_precision:
            input_tensor = inputs.input_values.half().to("cuda")
            model = model.half()
        else:
            input_tensor = inputs.input_values.to("cuda")
        try:
            outputs = model(
                input_values=input_tensor,
                output_hidden_states=True,
                return_dict=True
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("CUDA out of memory", file=sys.stderr)
                sys.exit(1) 
            else:
                raise
        
        hs = outputs.hidden_states
        if len(hs) == model.config.num_hidden_layers + 1:
            hs = hs[1:]
        idx = args.encoder_layer if args.encoder_layer >= 0 else len(hs) + args.encoder_layer
        H = hs[idx].squeeze(0).cpu().numpy()
    general_functions.dump_pickle(
        path=args.output_path, obj=H, rewrite=True, verbose=True
    )