# Standard libraries
from typing import Union
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import json
import os

# Specific libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from praatio import pitch_and_intensity 
from scipy.interpolate import interp1d
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
import textgrids
import librosa
import torch
import mne

# Set TensorFlow environment variables BEFORE any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TF from allocating all GPU memory
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use specific GPU if available

from phonet.phonet import Phonet 

from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor,
    WhisperProcessor, WhisperModel,
    HubertModel,
    WavLMModel
)

# Modules
from utils.load_utils import shifted_indexes_to_keep, match_lengths, check_syntax, labeling, get_trials, get_export_paths, SEX_LIST, sort_stimuli_based_on_situation
from utils.phoneme_implementation_from_phonet import compute_phones
import utils.general_functions as general_functions
import utils.processing as processing
import config

# Logging
from utils.logs import setup_logger

# Initialize logger
logger = setup_logger(
    name='load',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR if config.LOG_TO_FILE else None,
    level=config.LOG_LEVEL
)

try:
    import tensorflow as tf
    
    # Set TensorFlow logging level to suppress info messages and progress bars
    tf.get_logger().setLevel('ERROR')
    
    # Disable progress bars globally
    tf.keras.utils.disable_interactive_logging()
    
    # Configure TensorFlow for better performance
    tf.config.threading.set_inter_op_parallelism_threads(6)
except Exception as e:
    logger.warning(f"Could not configure TensorFlow optimally: {e}")

# Review this If we want to update packages
mne.set_log_level(verbose='CRITICAL')

# Extra parameters
TRANSFORMER_MODEL = "openai/whisper-base"
# TRANSFORMER_MODEL = "openai/whisper-tiny"
# TRANSFORMER_MODEL = "facebook/wav2vec2-large-xlsr-53-distilled"
# TRANSFORMER_MODEL = "facebook/wav2vec2-base"


# Global cache for HF models to avoid re-loading on every call
_PHONET_CACHE = {}
def get_phonet_instance():
    """
    Get a cached Phonet instance to avoid reloading models
    """
    global _PHONET_CACHE
    
    if 'phonet' not in _PHONET_CACHE:
        logger.debug("Loading Phonet model (first time only)...")
        _PHONET_CACHE['phonet'] = Phonet(["all"])
        logger.debug("Phonet model loaded and cached")
    
    return _PHONET_CACHE['phonet']
_DNN_MODEL_CACHE = {}
def _get_dnn_model(backbone: str, model_id: str, device: str):
    bl = backbone.lower()
    key = (bl, model_id, device)
    if key in _DNN_MODEL_CACHE:
        return _DNN_MODEL_CACHE[key]

    if bl == "whisper":
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperModel.from_pretrained(model_id).to(device).eval()
    elif bl == "wav2vec2":
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2Model.from_pretrained(model_id).to(device).eval()
    elif bl == "hubert":
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)  # Use only feature extractor for Hubert
        model = HubertModel.from_pretrained(model_id).to(device).eval()
    elif bl == "wavlm":#FIXME: dado que utiliza más checkpoints y cosas intermedias no entra en gpu, habría que guardar cada layer por separado 
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = WavLMModel.from_pretrained(model_id).to(device).eval()
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    _DNN_MODEL_CACHE[key] = (processor, model)
    return processor, model

class GetTrialData:
    def __init__(
        self, 
        band:str='Theta', 
        session:int=21, 
        channel:int=1, 
        trial:int=1, 
        stimuli_length:int=None
    )->None: 
        """
        Initializes the TrialChannelData class with the given parameters.

        Parameters
        ----------
        session : int, optional
            Session number, by default 21
        trial : int, optional
            Trial number, by default 1
        channel : int, optional
            Channel number used to record the audio (it can be from subject 1 or subject 2)
        band : str, optional
            EEG frequency band, by default 'All'.
        stimuli_length : int, optional
            Length of the stimuli to be used, by default None. If None, the original length of the stimuli will be used.

        Returns
        -------
        None

        Raises
        ------
        SyntaxError
            If band is not allowed.
        """
        check_syntax(band=band)
        self.stimuli_length = stimuli_length
        self.session = session
        self.channel = channel
        self.trial = trial
        self.band = band

        # Participants sex, ordered by session
        self.sex = SEX_LIST[(session - 21) * 2 + channel - 1] 

        # Minimum and maximum frequency allowed within specified band
        if self.band != 'Unfiltered':
            self.l_freq_eeg, self.h_freq_eeg = processing.band_freq(self.band)
        else:
            self.l_freq_eeg, self.h_freq_eeg = None, None
        
        # Silence threshold for pitch processing
        self.silence_threshold = 0.03
        self.audio_sr = 16000
        
        # EEG sampling rate
        self.sr = config.sr
        
        # Relevant paths
        self.praat_executable_path = config.praat_executable_path
        self.mistakes_control_path = os.path.normpath(f"data/mistakes_control/filtered_session{session}_trial{trial:02d}_channel{channel}.TextGrid")
        self.mistakes_path = os.path.normpath(f"data/mistakes_corrected/filtered_session{session}_trial{trial:02d}_channel{channel}.TextGrid")
        self.eeg_fname = os.path.normpath(f"data/EEG/S{session}/s{session}-{channel}-Trial{trial}-Deci-Filter-Trim-ICA-Pruned.set")
        self.phrases_fname = os.path.normpath(f"data/phrases/S{session}/s{session}.objects.{trial:02d}.channel{channel}.phrases")
        self.wav_fname = os.path.normpath(f"data/wavs/S{session}/s{session}.objects.{trial:02d}.channel{channel}.wav")
        turn_channel_logic = 2 if channel == 1 else 1 # Turns are from the interlocutor perspective (hearing) --> cada uno se guarda su propia transición (funciona para que quede bien en external)
        self.turn_fname = os.path.normpath(f"data/turns/switches_external/sess_{session}_trial_{trial:02d}_ch_{turn_channel_logic}.json")
        self.pitch_fname = os.path.normpath(f"S{session}/s{session}.objects.{trial:02d}.channel{channel}.txt")
        # self.turn_fname = os.path.normpath(f"data/turns/holds_external/sess_{session}_trial_{trial:02d}_ch_{channel}.json")
        
    def extract_eeg(
        self
    )->np.ndarray:
        """
        Reads the EEG data from a .set file, applies a filter based on the specified band, and downsamples the data.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The EEG data as a numpy array with dimensions (samples, channels).
        """
        # Read the .set file. warning of annotations and 'boundry' events -data discontinuities-.
        eeg = mne.io.read_raw_eeglab(
            input_fname=self.eeg_fname, 
            preload=True,
            verbose="CRITICAL"
        )

        # Apply a lowpass filter
        if self.band != 'Unfiltered':
            iir_params = {
                "ftype": "cheby2",       # Filter type: Chebyshev Type II
                "order": 4,              # Filter order
                "rs": 20,                # Stopband attenuation (dB)
            }
            # iir_params = { #Nuevos exp de dili
            #     "ftype": "butter",
            #     "order": 2,
            # }
            eeg = eeg.filter(
                l_freq=self.l_freq_eeg,
                h_freq=self.h_freq_eeg,
                method="iir",
                iir_params=iir_params
            )
            
        # Get mne representation 
        eeg = eeg.resample(
            sfreq=self.sr, 
            npad=0, 
            window='hamming', 
            method='fft'
        )
        eeg = eeg.get_data().T*1e6 
        return eeg
    
    def extract_envelope(
        self,
        kind:str='Envelope'
        )->np.ndarray: 
        """
        Takes the low pass filtered -butterworth-, downsample and smoothened envelope of .wav file. Then matches in length to the EEG

        Parameters
        ----------
        kind : str, optional
            Kind of envelope to use, by default 'Envelope'. Available kinds are:
            ['Envelope', 'Envelope2']. Envelope2 is the same as Envelope but it has an extra dimension with frequency shift
        Returns
        -------
        np.ndarray
            Envelope of wav signal with desire dimensions, using Hilbert transform
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")

        # Calculate envelope
        analytic_signal = sgn.hilbert(wav)
        envelope = np.abs(analytic_signal)
        
        # Resample 
        # window_size, stride = int(self.audio_sr/self.sr), int(self.audio_sr/self.sr)
        # envelope = np.array([#TODO REVISAR USAR SCIPY DECIMATE (TRANSFORMADA HAMMINH)
        #     np.mean(envelope[i:i+window_size]) \
        #     for i in range(0, len(envelope), stride)\
        #     if i+window_size<=len(envelope)
        #     ]
        # )
        downsampling_factor = int(self.audio_sr/self.sr)
        filter_coeffs = sgn.firwin(
            numtaps=3000, 
            cutoff=(self.sr / 2.0)  *.99,  # Nyquist frequency of target sampling rate
            fs=self.audio_sr, 
            pass_zero='lowpass'
        )

        # Filter 
        envelope = sgn.filtfilt(
            b=filter_coeffs, 
            a=np.array([1.0]), 
            x=envelope,
            axis=0
        )

        # Downsample 
        envelope = envelope[::downsampling_factor]
        
        if kind == 'Envelope2':
            window_size, stride = int(self.audio_sr/self.sr), int(self.audio_sr/self.sr)
            instantaneous_phase = np.unwrap(
                np.angle(analytic_signal)
            )
            instantaneous_frequency = np.gradient(instantaneous_phase) * self.audio_sr / (2 * np.pi)
            
            # Resample
            instantaneous_frequency = np.array([
                np.mean(instantaneous_frequency[i:i+window_size]) \
                for i in range(0, len(instantaneous_frequency), stride)\
                if i+window_size<=len(instantaneous_frequency)
                ]
            )
            total_envelope = np.hstack(
                (envelope.reshape(-1,1), instantaneous_frequency.reshape(-1,1))
                )
            
            return total_envelope
        else:
            return envelope.reshape(-1, 1)

    def extract_spectrogram(
        self,
        kind:str
        )->np.ndarray:
        """
        Calculates spectrogram of .wav file between -## Mel frequencies

        Parameters
        ----------
        kind : str
            Kind of spectrogram to compute, by default 'Spectrogram-16'.

        Returns
        -------
        np.ndarray
            Matrix with sprectrogram in given mel frequncies of dimension (Samples X Mel)
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")
        
        # Get sample window size to match the sampling rate of the EEG
        sample_window = int(self.audio_sr/self.sr)
        
        # Calculates the mel frequencies spectrogram giving the desire sampling (match the EEG)
        S = librosa.feature.melspectrogram(
            hop_length=sample_window, 
            n_fft=sample_window, 
            sr=self.audio_sr, 
            n_mels=int(kind.split('-')[-1]),
            y=wav
        )
        # Transform to dB using normalization to 1
        S_DB = librosa.power_to_db(
            ref=np.max,
            S=S
        ).T

        return S_DB
        
    def extract_mfccs(
        self,
        number_of_mels:int=16,
        kind:str='Mfccs'
        )->np.ndarray:
        """
        Calculates mel frequency clepstral coefficients from .wav.

        Parameters
        ----------
        kind : str, optional
           Kind of pitch use, by default 'Log-Raw'. Available kinds are:
            ['Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas']

        Returns
        -------
        np.ndarray
            Matrix with shape samples x coefficients

        Raises
        ------
        SyntaxError
            Whether the input value of 'kind' is passed correctly. It must be a string among ['Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas'].
        """

        # Check if given kind is a permited input value
        allowed_kind = ['Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of mfccs. Allowed kinds are: {allowed_kind}")
        
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")
        
        # Get sample window size to match the sampling rate of the EEG
        sample_window = int(self.audio_sr/self.sr)
        
        # Calculate matrix of mfccs
        mfccs = librosa.feature.mfcc(
            hop_length=sample_window,
            n_mfcc=number_of_mels, 
            n_mels=number_of_mels, 
            n_fft=sample_window, 
            sr=self.audio_sr, 
            y=wav
        )
        
        # Append deltas and deltas deltas
        if kind.startswith('Mfccs'):
            if kind.endswith('Deltas'):
                delta_mfccs = librosa.feature.delta(mfccs)
                mfccs_features = np.concatenate(
                    (mfccs, delta_mfccs)
                )
                if kind.endswith('Deltas-Deltas'):
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    mfccs_features = np.concatenate(
                        (mfccs, delta_mfccs, delta2_mfccs)
                    )
                    return mfccs_features.T
                else:
                    return mfccs_features.T
            else:
                return mfccs.T
        else:
            if kind.endswith('Deltas'):
                delta_mfccs = librosa.feature.delta(mfccs)
                if kind.endswith('Deltas-Deltas'):
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    deltas = np.concatenate(
                        (delta_mfccs, delta2_mfccs)
                    )
                    return deltas.T
                else:
                    return delta_mfccs.T
   
    def extract_pitch(
        self, 
        kind:str
        )->np.ndarray: 
        """
        Loads the pitch of the speaker, after calculating it from .wav file, using Praat.

        Parameters
        ----------
        kind : str
            Kind of pitch use, by default 'Log-Raw'. Available kinds are:
            'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes'
        Returns
        -------
        np.ndarray
            One-dimensional array with pitch values

        Raises
        ------
        SyntaxError
            Whether the input value of 'kind' is passed correctly. It must be one of:
            ['Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes']
        """

        # Check if given kind is a permited input value
        allowed_kind = ['Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of pitch. Allowed phonemes are: {allowed_kind}")
        
        # Makes path for storing data
        output_folder = os.path.normpath(f'data/pitch/{kind}_threshold_{self.silence_threshold}/')
        os.makedirs(output_folder, exist_ok=True)
        self.pitch_fname = os.path.join(output_folder, self.pitch_fname)
        
        # Distinguish subject
        if self.sex == 'M':
            minPitch = 50
            maxPitch = 300
        elif self.sex == 'F':
            minPitch = 75
            maxPitch = 500
        
        # Define sample step and calculate pitch
        sampleStep = 1/self.sr # .01
        if 'Quad' in kind:
            pitch_and_intensity.extractPI(
                outputFN=os.path.abspath(self.pitch_fname), 
                inputFN=os.path.abspath(self.wav_fname), 
                silenceThreshold=self.silence_threshold,
                praatEXE=self.praat_executable_path, 
                sampleStep=sampleStep, 
                pitchQuadInterp=True,
                minPitch=minPitch,
                maxPitch=maxPitch
            )
            # Loads data
            data = np.genfromtxt(
                os.path.abspath(self.pitch_fname), 
                missing_values='--undefined--', 
                filling_values=np.inf,
                dtype=float, 
                delimiter=','
            )
            time, pitch = data[:, 0], data[:, 1]

            # Get defined indexes
            defined_indexes = np.where(pitch!=np.inf)[0]
            
            # Log transformation
            logpitch = np.log(pitch)
            
            # Set left values to zero
            logpitch[logpitch==np.inf]=0
            return logpitch.reshape(-1, 1)
        else:
            pitch_and_intensity.extractPI(
                outputFN=os.path.abspath(self.pitch_fname), 
                silenceThreshold=self.silence_threshold,
                inputFN=os.path.abspath(self.wav_fname), 
                praatEXE=self.praat_executable_path, 
                sampleStep=sampleStep, 
                minPitch=minPitch,
                maxPitch=maxPitch
            )
            
            # forceRegenerate - if running this function for the same file, if False
            #                 just read in the existing pitch file
            # undefinedValue - if None remove from the dataset, otherset set to
            #                 undefinedValue
            # pitchQuadInterp - if True, quadratically interpolate pitch
                
        # Loads data
        data = np.genfromtxt(
            os.path.abspath(self.pitch_fname), 
            missing_values='--undefined--', 
            filling_values=np.inf,
            delimiter=',', 
            dtype=float
            )
        time, pitch = data[:, 0], data[:, 1]

        # Get defined indexes
        defined_indexes = np.where(pitch!=np.inf)[0]
        
        # Approximated window size
        window_size = 100e-3
        n_steps_in_window = np.ceil(window_size/sampleStep)
        window_size = n_steps_in_window*sampleStep

        if kind.endswith('Manual'):
            if 'Log' in kind:
                # Log transformation
                logpitch = np.log(pitch)

                # Interpolate relevant moments of silence
                for i in range(len(defined_indexes)):
                    if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
                        logpitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(
                            x=time[defined_indexes[i-1]+1:defined_indexes[i]], 
                            fp=logpitch[defined_indexes],
                            xp=time[defined_indexes]
                        )
            
                # Set left values to zero
                logpitch[logpitch==np.inf] = 0
                return logpitch.reshape(-1, 1)
            else: 
                # Interpolate relevant moments of silence
                for i in range(len(defined_indexes)):
                    if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
                        pitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(
                            xp=time[defined_indexes], fp=pitch[defined_indexes],
                            x=time[defined_indexes[i-1]+1:defined_indexes[i]]
                        )
                
                # Set left values to zero
                pitch[pitch==np.inf] = 0
                return pitch.reshape(-1,1)
                   
        elif kind.endswith('Raw'):
            if 'Log' in kind:
                # Log transformation
                logpitch = np.log(pitch)
        
                # Set left values to zero
                logpitch[logpitch==np.inf]=0
                return logpitch.reshape(-1, 1)
            else: 
                # Set left values to zero
                pitch[pitch==np.inf]=0
                return pitch.reshape(-1,1)

    def extract_phonemes( 
        self, 
        kind:str='Phonemes-Discrete'
        )->np.ndarray:
        """
        It makes a time-match matrix between the phonemes and the envelope using Phonet implementation. 
        The values and shape of given matrix depend on kind.

        Parameters
        ----------
        kind : str, optional
        Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
            ['Phonemes', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 'Phonemes-Frequency', 'Phonemes-Frequency']

        Returns
        -------
        np.ndarray
            if kind.startswith('Phonemes-Envelope'):
                Matrix with envelope amplitude at given sample. The matrix dimension is SamplesXPhonemes_labels(in order)
            elif kind.startswith('Phonemes-Discrete'):
                Also a matrix but it has 1s and 0s instead of envelope amplitude.
            elif kind.startswith('Phonemes-Onset'):
                In this case the value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
        """
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")
        if kind=='Phonemes':
            labels_phonemes = config.exp_info.phonemes.copy()
            labels_phones = config.exp_info.phones.copy()
            posterior_prob = compute_phones(
                phonet_obj=get_phonet_instance(), 
                audio_file=self.wav_fname,
                PLLR=True
            )
            posterior_prob = np.clip(
                posterior_prob, 1e-6, 1-1e-6
            )

            # Repeat last sample (probably silence)
            difference = len(posterior_prob) - self.stimuli_length
            if difference > 0:
                posterior_prob = posterior_prob[:-difference]
            elif difference < 0:
                for i in range(np.abs(difference)):
                    aux = posterior_prob[-1].copy() 
                    posterior_prob = np.vstack((posterior_prob, aux.reshape(-1,1).T))
            
            # Map phones to phonemes, making the sum
            posterior_prob_phonemes = np.zeros(
                shape=(posterior_prob.shape[0], len(labels_phonemes))
            )
            for h, phone in enumerate(labels_phones):
                phoneme_index = labels_phonemes.index(
                    config.exp_info.phones_to_phonemes[phone]
                )
                posterior_prob_phonemes[:, phoneme_index] += posterior_prob[:, h]
                        
            # Calculate posterior llr
            pllr = np.zeros(shape=posterior_prob_phonemes.shape)
            number_of_phonemes = posterior_prob_phonemes.shape[1]
            for ph in range(number_of_phonemes):
                pllr[:, ph] = np.log10(
                    posterior_prob_phonemes[:, ph]/(1-posterior_prob_phonemes[:, ph]+ 1e-8)
                )

            # Centralizamos 
            pllr = np.nan_to_num(pllr, nan=0.0, posinf=0.0, neginf=0.0)
            pllr = pllr - np.mean(pllr, axis=1, keepdims=True)  
            
            # Removemos silencios
            pllr_without_silence = pllr[:, np.arange(number_of_phonemes) != labels_phonemes.index('/sil/')]
            return pllr_without_silence
        else:
            sec_phones = compute_phones(
                phonet_obj=get_phonet_instance(), 
                audio_file=self.wav_fname
            )
        
        # Remove silences, since it won't be used in prediction (when silence occurs, all phoneme are 0)
        labels = config.exp_info.phonemes.copy()
        labels.remove('/sil/')
            
        # Match features length
        difference = len(sec_phones) - self.stimuli_length
        if difference > 0:
            sec_phones = sec_phones[:-difference]
        elif difference < 0:
            # In this case, silences are append
            for i in range(np.abs(difference)):
                sec_phones.append('<p:>')
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape=(len(sec_phones), len(labels)))
        
        # Match phoneme with kind
        # if kind.startswith('Phonemes-Envelope'):
        #     for i, tagg in enumerate(sec_phones):
        #         if (tagg!='<p:>') and (tagg!='sil'):
        #             phonemes[i, labels.index(config.exp_info.phones_to_phonemes[tagg])] = envelope[i]
        if kind.startswith('Phonemes-Discrete'):
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phonemes[i, labels.index(config.exp_info.phones_to_phonemes[tagg])] = 1
        elif kind.startswith('Phonemes-Frequency'):
            try:
                freq = general_functions.load_pickle('data/phon_frequency_dict/frequency_dict.pkl')
            except:
                logger.warning("Frequency dictionary isn't Load. \n ---> loading it now...")
                os.makedirs('data/phon_frequency_dict', exist_ok=True)
                freq = general_functions.load_phon_frequency_dict(
                    phonet_obj=get_phonet_instance(),
                    save_path='data/phon_frequency_dict',
                    plot_freq=True,
                )
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phonemes[i, labels.index(config.exp_info.phones_to_phonemes[tagg])] = 1/freq[config.exp_info.phones_to_phonemes[tagg]]
        elif kind.startswith('Phonemes-Onset'):
            # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
            phonemes_onset = [sec_phones[0]]
            for i in range(1, len(sec_phones)):
                if sec_phones[i] == sec_phones[i-1]:
                    phonemes_onset.append(0)
                else:
                    phonemes_onset.append(sec_phones[i])
            # Match phoneme with envelope
            for i, tagg in enumerate(phonemes_onset):
                if (tagg!='<p:>') and (tagg!='sil') and (tagg!=0):
                    phonemes[i, labels.index(config.exp_info.phones_to_phonemes[tagg])] = 1
        return phonemes
    
    def extract_phones( 
        self, 
        kind:str='Phones-Discrete'
        )->np.ndarray:
        """
        It makes a time-match matrix between the phones and the envelope using Phonet implementation. The values and shape of given matrix depend on kind.

        Parameters
        ----------
        kind : str, optional
           Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
            ['Phones', 'Phones-Envelope', 'Phones-Discrete', 'Phones-Onset']

        Returns
        -------
        np.ndarray
            if kind.startswith('Phones-Envelope'):
                Matrix with envelope amplitude at given sample. The matrix dimension is SamplesXPhones_labels(in order)
            elif kind.startswith('Phones-Discrete'):
                Also a matrix but it has 1s and 0s instead of envelope amplitude.
            elif kind.startswith('Phones-Onset'):
                In this case the value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
            
        Raises
        ------
        SyntaxError
            Whether the input value of 'kind' is passed correctly. It must be a one of:
            ['Phones', 'Phones-Envelope', 'Phones-Discrete', 'Phones-Onset'].
        """
        # Check if given kind is a permited input value
        allowed_kind = ['Phones', 'Phones-Envelope', 'Phones-Discrete', 'Phones-Onset']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phones are: {allowed_kind}")

        if kind=='Phones':
            # Extract phonemes using Phonet protocol
            labels_phones = config.exp_info.phones.copy()
            
            wav = wavfile.read(self.wav_fname)[1]
            wav = wav.astype("float")
            posterior_prob = compute_phones(
                phonet_obj=get_phonet_instance(), 
                audio_file=self.wav_fname,
                PLLR=True
            )

            # posterior_prob: (num_frames, num_phones), original_fs ≈ 100 Hz # TODO: falta tomar los mismos recaudos que se toman en phonemes para no dividir por cero
            num_target_frames = int((wav.shape[0] / self.audio_sr) * self.sr)
            posterior_prob = sgn.resample(
                posterior_prob, 
                num_target_frames, 
                axis=0
            )
            
            # Match features length
            difference = len(posterior_prob) - self.stimuli_length

            if difference > 0:
                posterior_prob = posterior_prob[:-difference]
            elif difference < 0:
                # Repeat last sample (probably silence)
                for i in range(np.abs(difference)):
                    aux = posterior_prob[-1].copy() 
                    posterior_prob = np.vstack((posterior_prob, aux.reshape(-1,1).T))
            
            # Calculate posterior llr
            pllr = np.zeros(shape=posterior_prob.shape)
            number_of_phones = posterior_prob.shape[1]
            for ph in range(number_of_phones):
                pllr[:, ph] = np.log10(posterior_prob[:, ph]/(1-posterior_prob[:, ph]))
            
            # Centralizamos 
            pllr = pllr - np.mean(pllr, axis=1, keepdims=True)  
            
            # Removemos silencios
            pllr_without_silence = pllr[:, (np.arange(number_of_phones) != labels_phones.index('sil'))&(np.arange(number_of_phones) != labels_phones.index('<p:>'))]
            return pllr_without_silence
        else:
            # Extract phones
            phonet = get_phonet_instance()
            sec_phones = compute_phones(
                phonet_obj=phonet, 
                audio_file=self.wav_fname
            )
        
        # Get phonet phones labels
        labels = config.exp_info.phones.copy()
        labels.remove('<p:>')
        labels.remove('sil')
        
        # Resample sec_phones to match stimuli length
        if len(sec_phones) != self.stimuli_length:
            # Convert to array of strings for resample
            sec_phones = np.array(sec_phones)
            
            # Map phones to integers for resampling
            phone_to_int = {phone: i for i, phone in enumerate(labels + ['<p:>', 'sil'])}
            int_to_phone = {i: phone for phone, i in phone_to_int.items()}
            sec_phones_int = np.array([phone_to_int.get(p, phone_to_int['<p:>']) for p in sec_phones])
            sec_phones_resampled = np.round(
                sgn.resample(sec_phones_int.astype(float), self.stimuli_length)
            ).astype(int)
            sec_phones = [int_to_phone[i] for i in sec_phones_resampled]

        # Match features length
        difference = len(sec_phones) - self.stimuli_length
        if difference > 0:
            sec_phones = sec_phones[:-difference]
        elif difference < 0:
            # In this case, silences are appended
            for i in range(np.abs(difference)):
                sec_phones.append('<p:>')
        
        # Make empty array of phones
        phones = np.zeros(shape=(len(sec_phones), len(labels)))
        
        # Match phoneme with kind
        # if kind.startswith('Phones-Envelope'):
        #     for i, tagg in enumerate(sec_phones):
        #         if (tagg!='<p:>') and (tagg!='sil'):
        #             phones[i, labels.index(tagg)] = envelope[i]
        if kind.startswith('Phones-Discrete'):
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phones[i, labels.index(tagg)] = 1
        elif kind.startswith('Phones-Onset'):
            # Makes a list giving only first ocurrences of phones (also ordered by sample) 
            phones_onset = [sec_phones[0]]
            for i in range(1, len(sec_phones)):
                if sec_phones[i] == sec_phones[i-1]:
                    phones_onset.append(0)
                else:
                    phones_onset.append(sec_phones[i])
            # Match phoneme with envelope
            for i, tagg in enumerate(phones_onset):
                if (tagg!='<p:>') and (tagg!='sil') and (tagg!=0):
                    phones[i, labels.index(tagg)] = 1
        return phones

    def extract_phonological(
        self, 
        kind: Union[str, None] = 'Phonological'
        )->np.ndarray:
        """
        Retrive phonological features as a matrix matching envelope length, using Phonet implementation.

        Parameters
        ----------
        kind: str, optional
            Whether to extract phonological features of first or second groups. Available kinds are:
            ['Phonological1', 'Phonological2']. If None, it will extract all phonological features.

        Returns
        -------
        np.ndarray
            Matrix with phonological features with shape SAMPLES X FEATURES
        """
        # Use cached Phonet instance instead of creating new one
        phonet = get_phonet_instance()
        phon_features = phonet.get_PLLR(
            audio_file=self.wav_fname, 
            plot_flag=False
        )
        
        # Interpole data in desire times
        desired_time = np.linspace(
            0, self.stimuli_length/self.sr + 1/self.sr, self.stimuli_length
        )

        # Get feature names
        if kind!='Phonological':
            if kind == 'Phonological1':
                phon_features_names = [
                    feat for feat in phon_features.columns if feat not in ['time', 'trill', 'pause'] + config.exp_info.phonological_labels2 
                ]
            elif kind == 'Phonological2':
                phon_features_names = [
                    feat for feat in phon_features.columns if feat not in ['time', 'trill', 'pause'] + config.exp_info.phonological_labels1
                ]
        else:
            phon_features_names = [
                feat for feat in phon_features.columns if feat not in ['time', 'trill', 'pause']
            ]
        
        phonological_features = []
        
        # Interpolate each phonological feature
        for phon_feat in phon_features_names:
            phonological_features.append(
                np.interp(
                    x=desired_time, 
                    xp=phon_features['time'].values, 
                    fp=phon_features[f'{phon_feat}'].values
                )
            )

        # Return data in desired shape
        return np.stack(phonological_features, axis=0).T

    def extract_DNNs(
        self,
        kind: str = '18DNNs1-hubert',
        model_id: str = None                    # None -> use defaults below
    ) -> np.ndarray:
        """
        Build a n_components representation of the audio using a DNN encoder, then:
          1) Standardize + PCA along feature dim
          2) Interpolate over time to match envelope length
        Returns array with shape (n_components, len(envelope)).

        Notes:
        - Uses Whisper by default (log-mel -> encoder). Disables 30 s padding.
        - Time resampling uses linear interpolation in normalized time [0..1].
        - For better stability across files, consider fitting PCA on a corpus.
        """
        # Stimuli should be call ##DNNs##, where first ## implies number of components, and second ## the layer
        n_components, encoder_layer_backbone = [part for part in kind.split("DNNs")]
        encoder_layer, backbone = encoder_layer_backbone.split('-')# "whisper", "wav2vec2", "hubert" or "wavlm"
        n_components, encoder_layer = int(n_components), int(encoder_layer)
        
        # Path for caching full layer representation
        layer_path = os.path.normpath(f"data/DNNs_cache/{backbone}/layer_{encoder_layer}/session_{self.session}_trial{self.trial:02}_channel_{self.channel}.pkl")
        backbone_lower = backbone.lower()
        
        # Try to load preprocessed layer, to reduce later the dimension
        try:
            H = general_functions.load_pickle(
                path=layer_path
            )
        # Run the whole pipeline, storing full layer for future use
        except:
            logger.warning(f"Cached DNN layer not found: {layer_path}. \n ---> extracting and caching it now...")
            # Create the folder if it doesn't exist
            os.makedirs(os.path.dirname(layer_path), exist_ok=True)
            
            # === 1 = LOAD AUDIO ====
            sr, wav = wavfile.read(self.wav_fname)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            
            # Normalize to [-1, 1]
            if np.issubdtype(wav.dtype, np.integer):
                wav = wav.astype(np.float32) / max(1, np.iinfo(wav.dtype).max)
            else:
                wav = wav.astype(np.float32)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Defaults if not provided
            if model_id is None:
                bl = backbone.lower()
                if  bl == "whisper":
                    model_id = "openai/whisper-tiny"#TODO updt
                elif bl == "wav2vec2":
                    model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
                elif bl == "hubert":
                    model_id = "facebook/hubert-large-ll60k"  
                elif bl == "wavlm":
                    model_id = "microsoft/wavlm-large"
                    # model_id = "microsoft/wavlm-base-plus"
                else:
                    raise ValueError(f"Unknown backbone: {backbone}")
            
            # Use cached model/processor
            processor, model = _get_dnn_model(backbone, model_id, device)
            
            # Resample to model SR
            if backbone_lower in ["hubert", "wavlm"]:
                model_sr = getattr(processor, "sampling_rate", 16000)
            else:
                model_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
            if sr != model_sr:
                wav_model = sgn.resample_poly(wav, up=model_sr, down=sr)
            else:
                wav_model = wav
                
            # === 2 = GET ENCODER SEQUENCE ====
            if backbone_lower == "whisper":
                with torch.no_grad():
                    inputs = processor(
                        audio=wav_model,
                        sampling_rate=model_sr,
                        return_tensors="pt",
                        # padding=False  # avoid 30s padding
                        padding="max_length",          # pad a 30s -> 3000 frames
                        return_attention_mask=True  
                    )
                    # Whisper uses input_features (log-mel)
                    use_amp = (device == "cuda")
                    with torch.amp.autocast(device, enabled=use_amp):
                        outputs = model.encoder(
                            input_features=inputs.input_features.to(device),
                            attention_mask=inputs.attention_mask.to(device),
                            output_hidden_states=True,
                            return_dict=True
                        )
                    
                    # Tomamos las hidden states del encoder
                    hs = outputs.hidden_states
                    # Quitar el embedding inicial si viene incluido (len = layers + 1)
                    if len(hs) == model.config.encoder_layers + 1:
                        hs = hs[1:]
                    # Soportar índices negativos (e.g., -1 = última)
                    idx = encoder_layer if encoder_layer >= 0 else len(hs) + encoder_layer
                    if idx < 0 or idx >= len(hs):
                        raise ValueError(f"encoder_layer={encoder_layer} fuera de rango (0..{len(hs)-1})")
                    # [B, T, D] -> [T, D]
                    H = hs[idx].squeeze(0).cpu().numpy()

                    # Save full layer for future use
                    general_functions.dump_pickle(
                        path=layer_path, obj=H, rewrite=True, verbose=True
                    )
            elif backbone_lower == "wav2vec2":
                with torch.no_grad():
                    inputs = processor(
                        wav_model,
                        sampling_rate=model_sr,
                        return_tensors="pt",
                        padding=False
                    )
                    outputs = model(
                        input_values=inputs.input_values.to(device),
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hs = outputs.hidden_states
                    # Quitar el embedding inicial si viene incluido (len = layers + 1)
                    if len(hs) == model.config.num_hidden_layers + 1:
                        hs = hs[1:]
                    idx = encoder_layer if encoder_layer >= 0 else len(hs) + encoder_layer
                    if idx < 0 or idx >= len(hs):
                        raise ValueError(f"encoder_layer={encoder_layer} fuera de rango (0..{len(hs)-1})")
                    # [B, T, D] -> [T, D]
                    H = hs[idx].squeeze(0).cpu().numpy()
                    
                    # Save full layer for future use
                    general_functions.dump_pickle(
                        path=layer_path, obj=H, rewrite=True, verbose=True
                    )
            elif backbone_lower == "hubert":
                with torch.no_grad():
                    # Use feature extractor directly
                    inputs = processor(
                        wav_model,
                        sampling_rate=model_sr,
                        return_tensors="pt",
                        padding=False
                    )
                    outputs = model(
                        input_values=inputs.input_values.to(device),
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hs = outputs.hidden_states
                    # Remove initial embedding if present
                    if len(hs) == model.config.num_hidden_layers + 1:
                        hs = hs[1:]
                    idx = encoder_layer if encoder_layer >= 0 else len(hs) + encoder_layer
                    if idx < 0 or idx >= len(hs):
                        raise ValueError(f"encoder_layer={encoder_layer} fuera de rango (0..{len(hs)-1})")
                    # [B, T, D] -> [T, D]
                    H = hs[idx].squeeze(0).cpu().numpy()
                    
                    # Save full layer for future use
                    general_functions.dump_pickle(
                        path=layer_path, obj=H, rewrite=True, verbose=True
                    )
            elif backbone_lower == "wavlm":
                try:
                    subprocess.run([
                        "python", "-m", "utils.load_utils",
                        "--wav_path", self.wav_fname,
                        "--model_id", model_id,
                        "--encoder_layer", str(encoder_layer),
                        "--output_path", layer_path
                    ], check=True, capture_output=True, text=True)
                
                except subprocess.CalledProcessError as e:
                    if e.stderr and 'CUDA out of memory' in e.stderr:
                        logger.error(f"CUDA out of memory with model {model_id} and input length {len(wav_model)/model_sr:.1f}s. Try a smaller model or use CPU.")
                        logger.error(f"Trying with half precision...")
                        subprocess.run([
                                "python", "-m", "utils.load_utils",
                                "--wav_path", self.wav_fname,
                                "--model_id", model_id,
                                "--encoder_layer", str(encoder_layer),
                                "--output_path", layer_path,
                                "--half_precision", "True"
                            ], check=True, capture_output=True, text=True)
                    else:
                        logger.error(f"Error running subprocess to extract WavLM features: {e.stderr}")
                        raise e
                try:
                    H = general_functions.load_pickle(
                        path=layer_path
                    )
                except:
                    raise ValueError(f"Error loading cached WavLM layer from {layer_path}")
                # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                #     future = executor.submit(
                #         run_dnn_in_subprocess,
                #         wav_model, model_sr, None, None, encoder_layer, model.config, device, model_id
                #     )
                #     H = future.result()
                # with torch.no_grad():
                #     inputs = processor(
                #         wav_model,
                #         sampling_rate=model_sr,
                #         return_tensors="pt",
                #         padding=False
                #     )
                #     try:
                #         input_tensor = inputs.input_values.to(device)
                #         outputs = model(
                #             input_values=input_tensor,
                #             output_hidden_states=True,
                #             return_dict=True
                #         )
                #     except RuntimeError as e:
                #         if 'CUDA out of memory' in str(e):
                #             logger.error(f"CUDA out of memory with model {model_id} and input length {len(wav_model)/model_sr:.1f}s. Try a smaller model or use CPU.")
                #             logger.error(f"Trying with half precision...")
                #             input_tensor = inputs.input_values.half().to(device)
                #             model = model.half()
                #             outputs = model(
                #                 input_values=input_tensor,
                #                 output_hidden_states=True,
                #                 return_dict=True
                #             )
                #     hs = outputs.hidden_states
                #     # Remove initial embedding if present
                #     if len(hs) == model.config.num_hidden_layers + 1:
                #         hs = hs[1:]
                #     idx = encoder_layer if encoder_layer >= 0 else len(hs) + encoder_layer
                #     if idx < 0 or idx >= len(hs):
                #         raise ValueError(f"encoder_layer={encoder_layer} fuera de rango (0..{len(hs)-1})")
                #     # [B, T, D] -> [T, D]
                #     H = hs[idx].squeeze(0).cpu().numpy()
                    
                #     # Free GPU memory
                #     if device == "cuda":
                #         torch.cuda.empty_cache()
                #         torch.cuda.ipc_collect()
            else:
                raise ValueError(f"Unknown backbone: {backbone}. Use 'whisper', 'wav2vec2', 'hubert', or 'wavlm'.")
        
        # Guard: very short inputs
        if H.ndim != 2 or H.shape[0] < 2:
            # Return zeros if we cannot form a sequence
            logger.warning("Input audio is too short to extract features.")
            return np.zeros((n_components, self.stimuli_length), dtype=np.float32)

        # === 3 = REDUCE DIMENSION ====
        scaler = StandardScaler(with_mean=True, with_std=True)
        Hs = scaler.fit_transform(H)                 # (T, D)
        pca = PCA(n_components=min(n_components, Hs.shape[1]))
        Z = pca.fit_transform(Hs)                    # (T, n_components_effective)

        # If model produced fewer dims than requested, pad with zeros
        if Z.shape[1] < n_components:
            pad = np.zeros((Z.shape[0], n_components - Z.shape[1]), dtype=Z.dtype)
            Z = np.hstack([Z, pad])

        # === 4 = RESAMPLE TO DESIRED STIMULI LENGTH ====
        T_src = Z.shape[0]
        T_tgt = self.stimuli_length

        if T_src == T_tgt:
            Z_t = Z
        else:
            # Normalize time to [0,1] to avoid relying on sample rates
            x_src = np.linspace(0.0, 1.0, T_src, endpoint=False)
            x_tgt = np.linspace(0.0, 1.0, T_tgt, endpoint=False)
            interp = interp1d(x_src, Z, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
            Z_t = interp(x_tgt)                      # (T_tgt, n_components)

        # Return as (T_tgt, n_components) to match typical (features, time)
        dnn_features = Z_t.astype(np.float32)
        return dnn_features

    def extract_mistakes(
        self, 
        kind:str='Mistakes-Separated'
        )->np.ndarray:
        """
        Calculates mistakes (lexical, articulatory, discursive) signal from annotated data

        Returns
        -------
        np.ndarray
            stimuli_lengthX3 binary array if separated else stimuli_lengthX1 binary array
            stimuli_lengthX3 binary array if separated else stimuli_lengthX1
        """
        # Define kind
        separated=True if kind.endswith('Separated') else False

        # Read phrases to identify time of error inside phrases time
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        start_time, end_time = phrases[0].iloc[0], phrases[1].iloc[-1]
        phrases_time = np.arange(start_time, end_time, 1/self.sr)

        # Identify start and end of error within mistake
        mistake_code = {
                        'A':0, # articulatorio
                        'L':1, # léxico
                        'D':2 # discursivo
                        } if separated else {'A':0, 'L':0, 'D':0}
        mistake_signal = np.zeros(shape=(len(phrases_time), 3)) if separated else np.zeros(shape=(len(phrases_time), 1))
        
        if os.path.isfile(self.mistakes_path):
            # Read textgrid        
            grid = textgrids.TextGrid(self.mistakes_path)[f"canal {int(self.mistakes_path.split('channel')[1][0])}"]
            
            # Identify onset, offset and type of mistake
            mistake_taggs, mistake_count = np.unique([el.text.split('Palabra del error: ')[1] for el in grid], return_counts=True)
            mistakes = {mistake:{'start':None, 'end':None, 'type':None} for mistake in mistake_taggs}
            mistake_taggs = np.repeat(mistake_taggs, mistake_count)

            for item, mistake in zip(grid, mistake_taggs):
                # Identify time_intervals and mistake type
                mistakes[mistake]['type'] = mistake_code[item.text.split('Etiqueta: ')[1][0]]
                if int(item.text[0])==1:
                    mistakes[mistake]['start'] = item.xpos
                elif int(item.text[0])==2:
                    mistakes[mistake]['end'] = item.xpos
                # else:
                #     nextword_start.append(item.xpos)

            # Fill mistake_signal
            for mistake in mistakes:
                onset_filter = mistakes[mistake]['start']<=phrases_time
                offset_filter = phrases_time<=mistakes[mistake]['end']
                
                mistake_signal[onset_filter&offset_filter, mistakes[mistake]['type']] = -np.ones(shape=np.sum(onset_filter&offset_filter))

        # Match length of mistake signal with desired stimuli length
        difference = len(mistake_signal)-self.stimuli_length
        if difference>0:
            mistake_signal = mistake_signal[:-difference]
        elif difference<0:
            mistake_signal = np.concatenate((mistake_signal, np.zeros(shape=(np.abs(difference), 3)))) if separated else np.concatenate((mistake_signal, np.zeros(shape=(np.abs(difference), 1))))

        return mistake_signal
    
    def extract_mistakes_control(
        self, 
        kind:str='Control-Separated'
        )->np.ndarray:
        """
        Calculates mistakes control (lexical, articulatory, discursive) signal from annotated data

        Returns
        -------
        np.ndarray
            stimuli_length X3 binary array if separated else stimuli_length X1
        """
        # Define kind
        separated=True if kind.endswith('Separated') else False

        # Read phrases to identify time of error inside phrases time
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        start_time, end_time = phrases[0].iloc[0], phrases[1].iloc[-1]
        phrases_time = np.arange(start_time, end_time, 1/self.sr)
        
        # Identify start and end of error within mistake
        control_code = {
                        'A':0, # articulatorio
                        'L':1, # léxico
                        'D':2 # discursivo
                        } if separated else {'A':0, 'L':0, 'D':0}
        control_signal = np.zeros(shape=(len(phrases_time), 3)) if separated else np.zeros(shape=(len(phrases_time), 1))
        
        if os.path.isfile(self.mistakes_control_path):
            # Read textgrid        
            grid = textgrids.TextGrid(self.mistakes_control_path)[f"canal {int(self.mistakes_control_path.split('channel')[1][0])}"]
            
            # Identify onset, offset and type of mistake
            control_taggs, control_count = np.unique([el.text.split('Palabra del error: ')[1] for el in grid], return_counts=True)
            controls = {control:{'start':None, 'end':None, 'type':None} for control in control_taggs}
            control_taggs = np.repeat(control_taggs, control_count)

            for item, control in zip(grid, control_taggs):
                # Identify time_intervals and control type
                controls[control]['type'] = control_code[item.text.split('Etiqueta: ')[1][0]]
                controls[control]['score'] = float(item.text.split('normalizado: ')[1].split(',')[0])
                
                if int(item.text[0])==1:
                    controls[control]['start'] = item.xpos
                elif int(item.text[0])==2:
                    controls[control]['end'] = item.xpos

            # Fill control_signal
            for control in controls:
                onset_filter = controls[control]['start']<=phrases_time
                offset_filter = phrases_time<=controls[control]['end']
                control_signal[onset_filter&offset_filter, controls[control]['type']] = np.ones(shape=np.sum(onset_filter&offset_filter))#*controls[control]['score']
        
        # Match length of mistake signal with desired stimuli length
        difference = len(control_signal)-self.stimuli_length         
        if difference>0:
            control_signal = control_signal[:-difference]
        elif difference<0:
            control_signal = np.concatenate((control_signal, np.zeros(shape=(np.abs(difference), 3)))) if separated else np.concatenate((control_signal, np.zeros(shape=(np.abs(difference), 1))))
        return control_signal
    
    def extract_turn_taking(
        self
    ):
        """
        Extracts turn-taking features from the audio envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        np.ndarray
            Array with turn-taking features.
        """
        # Read json file
        try: 
            with open(self.turn_fname, 'r') as json_file:
                turn_data_list = json.load(json_file)
        except Exception as e:
            logger.error(f"Error reading turn-taking file: {e}")
            return np.zeros(shape=(self.stimuli_length, 1))

        turn_feature = np.zeros(shape=(self.stimuli_length, 1))
        for turn_data in turn_data_list:
            start_sample = int(turn_data['ipu1_start_time'] * self.sr)
            end_sample = int(turn_data['ipu1_end_time'] * self.sr)
            turn_feature[start_sample:end_sample] = np.linspace(
                0, 1, end_sample - start_sample
            ).reshape(-1, 1)
            
        # Verify length
        if turn_feature.shape[0] != self.stimuli_length:
            raise ValueError(f"Turn feature length {turn_feature.shape[0]} does not match desired stimuli length {self.stimuli_length}")
        return turn_feature

    def extract_offset(
        self,
    )-> np.ndarray:
        """
        Gives an array of ones to create offset

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        np.ndarray
            
        """
        # Create an array with the same length as the envelope filled with the offset value
        return np.ones(shape=(self.stimuli_length, 1), dtype=np.float32)    
    
    def load_trial(
        self, 
        stimuli:list,
        eeg_exists:bool=False,
        )->dict: 
        """Extract EEG and calculates specified stimuli.
        Parameters
        ----------
        stimuli : list
            A list containing possible stimuli.

        Returns
        -------
        dict
            Dictionary with EEG, info and specified stimuli as mne objects
        """
        channel = {}
        if not eeg_exists:
            channel['EEG'] = self.extract_eeg()

        for stimulus in stimuli:
            if stimulus.startswith('Envelope'):
                channel['Envelope'] = self.extract_envelope(
                    kind=stimulus
                )
            if stimulus.startswith('Mfccs') or stimulus.startswith('Deltas'):
                channel[stimulus] = self.extract_mfccs( 
                    kind=stimulus
                )
            if stimulus.startswith('Pitch'):
                channel[stimulus] = self.extract_pitch(
                    kind=stimulus
                )
            if stimulus.startswith('Phonological'):
                channel[stimulus] = self.extract_phonological(
                    kind=stimulus
                )
            if stimulus.startswith('Mistakes'):
                channel[stimulus] = self.extract_mistakes(
                    kind=stimulus
                )
            if stimulus.startswith('Control'):
                channel[stimulus] = self.extract_mistakes_control(
                    kind=stimulus
                )
            if 'DNNs' in stimulus:
                channel[stimulus] = self.extract_DNNs(
                    kind=stimulus
                )
            if stimulus.startswith('Spectrogram'):
                channel[stimulus] = self.extract_spectrogram(
                    kind=stimulus
                )
            if stimulus.startswith('Phonemes'):
                channel[stimulus] = self.extract_phonemes(
                    kind=stimulus
                )
            if stimulus.startswith('Phones'):
                channel[stimulus] = self.extract_phones(
                    kind=stimulus
                )
            if stimulus == 'Offset':
                channel[stimulus] = self.extract_offset(
                )
            if stimulus == 'Hearing-Turn':
                channel[stimulus] = self.extract_turn_taking(
                )

            # Resample to desire length
            if self.stimuli_length is not None:
                channel[stimulus] = self.resampling_to_desire_length(
                    X=channel[stimulus]
                )
        return channel

    def resampling_to_desire_length(
        self,
        X: np.ndarray
    )-> np.ndarray:
        """
        Resamples the input array X to match the desired length using linear interpolation.

        Parameters
        ----------
        X : np.ndarray
            Input array to be resampled.

        Returns
        -------
        np.ndarray
            Resampled array with the desired length.
        """
        # Resample to desire length
        T_src = X.shape[0]

        if T_src == self.stimuli_length:
            return X
        else:
            # Normalize time to [0,1) to avoid relying on sample rates
            x_src = np.linspace(0.0, 1.0, T_src, endpoint=False)
            x_tgt = np.linspace(0.0, 1.0, self.stimuli_length, endpoint=False)
            interp = interp1d(x_src, X, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
            return interp(x_tgt)    # (stim_length, n_components)

def load_samples_info(
    preprocessed_data_path:str,
    situation:str = 'External',
    session:int = 21,
    save_results:bool = True,
    overwrite:bool = False
)->dict:
    """
    Loads samples info dictionary for a given session and situation.
    
    Parameters
    ----------
    preprocessed_data_path : str
        Path directing to processed data.
    situation : str, optional
        Situation considered when performing the analysis, by default 'External'.
    session : int, optional
        Session number, by default 21.
    save_results : bool, optional
        Whether to save the results, by default True.
    overwrite : bool, optional
        Whether to overwrite existing results, by default False.
    
    Returns
    -------
    dict
        Information about the samples.
    """
    samples_info_path = os.path.join(preprocessed_data_path, f'samples_info/{situation}/')
    if not overwrite:
        try:
            samples_info = general_functions.load_pickle(
                path=os.path.join(
                    samples_info_path, f'samples_info_{session}.pkl'
                )
            )
            return samples_info
        except Exception as e:
            logger.warning(f"Couldn't load samples info for session {session}.")
            logger.debug(f"Error: {e}")

    samples_info = {
        'trial_lengths1': [0],
        'trial_lengths2': [0],
        'keep_indexes1':[],
        'keep_indexes2':[]
    }
    trials = get_trials(
        session=session
    )
    subject_1, subject_2 = {}, {}
    for p, trial in enumerate(tqdm(
        trials, 
        desc=f'Loading samples info of session {session}', 
        bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )):
        # Load envelope and EEG
        channel_1 = GetTrialData(
            session=session, 
            trial=trial,
            band='Broad', 
            channel=1,
            stimuli_length=None
        )
        channel_2 = GetTrialData(
            session=session,
            trial=trial,
            band='Broad',
            channel=2,
            stimuli_length=None
        )
        trial_channel_1 = channel_1.load_trial(
            stimuli=['Envelope'], 
            eeg_exists=False
        )
        trial_channel_2 = channel_2.load_trial(
            stimuli=['Envelope'], 
            eeg_exists=False
        )
        # When using Internal, EEG prediction is based on own stimuli. 
        if situation.startswith('Internal'):
            trial_subject_1 = trial_channel_1.copy()
            trial_subject_2 = trial_channel_2.copy()
        # When using External, EEG prediction is based on interlocutor stimuli
        else:
            trial_subject_1 = {key: trial_channel_2[key] for key in trial_channel_2 if key!='EEG'} 
            trial_subject_2 = {key: trial_channel_1[key] for key in trial_channel_1 if key!='EEG'}
            trial_subject_1['EEG'], trial_subject_2['EEG'] = trial_channel_1['EEG'], trial_channel_2['EEG']
        
        # Labeling of current speaker. {3:both_speaking, 2:speaker_talks, 1:interlocutor_talks, 0:silence}.
        # The difference between _1 and _2 is that values 1 and 2 are swapped (changes the perspective of who is the speaker and who is the interlocutor)
        current_speaker_1 = labeling(
            session=session,
            trial=trial, 
            channel=2, 
            sr=config.sr
        )
        current_speaker_2 = current_speaker_1.copy()
        current_speaker_2[current_speaker_1 == 1] = 2
        current_speaker_2[current_speaker_1 == 2] = 1

        # Match length of speaker labels and trials with the info of its lengths
        trial_subject_1, current_speaker_1, minimum1 = match_lengths(
            speaker_labels=current_speaker_1,
            dic=trial_subject_1 
        )
        trial_subject_2, current_speaker_2, minimum2 = match_lengths(
            speaker_labels=current_speaker_2,
            dic=trial_subject_2 
        )
        samples_info['trial_lengths1'].append(minimum1)
        samples_info['trial_lengths2'].append(minimum2)

        # Preprocessing: calaculates the relevant indexes for the apropiate analysis. Add sum of all previous trials length. This is because at the end, all trials previous to the actual will be concatenated
        shifted_1 = shifted_indexes_to_keep(speaker_labels=current_speaker_1, situation=situation)
        shifted_2 = shifted_indexes_to_keep(speaker_labels=current_speaker_2, situation=situation)
        samples_info['keep_indexes1'] += (shifted_1 + np.sum(samples_info['trial_lengths1'][:-1])).tolist()
        samples_info['keep_indexes2'] += (shifted_2 + np.sum(samples_info['trial_lengths2'][:-1])).tolist()

        # Store data as Internal, for consistency (have in mind that both channels have exactly the same length, so it does not matter which minimum we use to match lengths)
        assert minimum1 == minimum2, "The minimum lengths of both channels should be the same."
        trial_channel_1, current_speaker_1, _ = match_lengths(
            speaker_labels=current_speaker_1,
            dic=trial_channel_1,
            minimum=minimum1 
        )
        trial_channel_2, current_speaker_2, _ = match_lengths(
            speaker_labels=current_speaker_2,
            dic=trial_channel_2,
            minimum=minimum2 
        )
        
        # Each subject with it's own data
        for key in trial_channel_1:
            if key not in subject_1:
                subject_1[key] = trial_channel_1[key]
                subject_2[key] = trial_channel_2[key]
            else:
                subject_1[key] = np.concatenate(
                    (subject_1[key], trial_channel_1[key]), axis=0
                )
                subject_2[key] = np.concatenate(
                    (subject_2[key], trial_channel_2[key]), axis=0
                )
    
    # Saves modified relevant indexes
    export_paths = get_export_paths(
        preprocessed_data_path=preprocessed_data_path,
        band="Broad"
    )
    if save_results: 
        os.makedirs(
            samples_info_path, 
            exist_ok=True
        )
        general_functions.dump_pickle(
            path=os.path.join(samples_info_path, f'samples_info_{session}.pkl'), 
            obj=samples_info, 
            rewrite=True
        )
        for key in subject_1:
            os.makedirs(
                export_paths[key], 
                exist_ok=True
            )
            general_functions.dump_pickle(
                path=os.path.join(export_paths[key], f'Sesion{session}.pkl'), 
                obj=[subject_1[key], subject_2[key]], 
                rewrite=True
            )
    return samples_info

def load_stimuli(
    preprocessed_data_path:str,
    situation:str,
    samples_info:dict,
    stimuli:str = 'Envelope',
    band:str = 'Broad',
    session:int = 21,
    save_results:bool = True,
    overwrite:bool = False
)->tuple:
    """
    Loads and processes EEG and stimuli data for a given session.

    Parameters
    ----------
    preprocessed_data_path : str, optional
        Path directing to processed data, by default preprocessed_data_path.
    stimuli : str, optional
        Stimuli to use in the analysis, by default Envelope. 
    band : str, optional
        Neural frequency band, by default 'Broad'.
    session : int, optional
        Session number, by default session.
    save_results : bool, optional
        If True, saves the results in the preprocessed_data_path. Default is True.
    overwrite : bool, optional
        If True, overwrites the existing preprocessed data, forcing load_raw. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: session of 1st subjects.
        - dict: session of 2nd subjects.
    """
    export_paths = get_export_paths(
        preprocessed_data_path=preprocessed_data_path,
        stimuli=stimuli,
        band=band
    )
    subject_1, subject_2 = {}, {}
    if not overwrite:
        stimuli_to_be_computed = []
        for stimulus in stimuli.split('_')+['EEG']:
            stim_path = export_paths[stimulus]
            try:
                subject_1[stimulus], subject_2[stimulus] = general_functions.load_pickle(
                    path=os.path.join(
                        stim_path, f'Sesion{session}.pkl'
                        )
                )
            except Exception as e:
                stimuli_to_be_computed.append(stimulus)
                logger.warning(f"Couldn't load {stimulus} for session {session}.")
        if stimuli_to_be_computed==[]:
            sub1, sub2 = sort_stimuli_based_on_situation(
                subject_1=subject_1,
                subject_2=subject_2,
                situation=situation
            )

            return sub1, sub2
    else:
        stimuli_to_be_computed = stimuli.split('_')+['EEG']

    stimuli = '_'.join(stimuli_to_be_computed)
    stimuli = stimuli.replace('_EEG', '')
    eeg_exists = 'EEG' not in stimuli_to_be_computed
    logger.info(f"Stimuli to be computed: {stimuli}")
    
    trials = get_trials(
        session=session
    )
    for p, trial in enumerate(tqdm(
        trials, 
        desc=f'Loading stimuli of session {session}', 
        bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )):
        # Load envelope and EEG
        channel_1 = GetTrialData(
            session=session, 
            trial=trial, 
            channel=1,
            stimuli_length=samples_info['trial_lengths1'][p+1]
        )
        channel_2 = GetTrialData(
            session=session,
            trial=trial,
            channel=2,
            stimuli_length=samples_info['trial_lengths2'][p+1]
        )
        trial_channel_1 = channel_1.load_trial(
            stimuli=stimuli.split('_'),
            eeg_exists=eeg_exists
        )
        trial_channel_2 = channel_2.load_trial(
            stimuli=stimuli.split('_'),
            eeg_exists=eeg_exists
        )
        # When using Internal, EEG prediction is based on own stimuli. 
        if situation.startswith('Internal'):
            trial_subject_1 = trial_channel_1.copy()
            trial_subject_2 = trial_channel_2.copy()            
        # When using External, EEG prediction is based on interlocutor stimuli
        else:
            trial_subject_1 = {key: trial_channel_2[key] for key in trial_channel_2 if key!='EEG'} 
            trial_subject_2 = {key: trial_channel_1[key] for key in trial_channel_1 if key!='EEG'}
            if not eeg_exists:
                trial_subject_1['EEG'], trial_subject_2['EEG'] = trial_channel_1['EEG'], trial_channel_2['EEG']
        current_speaker_1 = labeling(
            session=session,
            trial=trial, 
            channel=2, 
            sr=config.sr
        )
        current_speaker_2 = current_speaker_1.copy()
        current_speaker_2[current_speaker_1 == 1] = 2
        current_speaker_2[current_speaker_1 == 2] = 1
        
        # Store data as Internal, for consistency (have in mind that both channels have exactly the same length, so it does not matter which minimum we use to match lengths)
        assert samples_info['trial_lengths1'][p+1] == samples_info['trial_lengths2'][p+1], "The minimum lengths of both channels should be the same."
        trial_channel_1, current_speaker_1, _ = match_lengths(
            speaker_labels=current_speaker_1,
            dic=trial_channel_1,
            minimum=samples_info['trial_lengths1'][p+1]
        )
        trial_channel_2, current_speaker_2, _ = match_lengths(
            speaker_labels=current_speaker_2,
            dic=trial_channel_2,
            minimum=samples_info['trial_lengths2'][p+1]
        )
        
        # Each subject with it's own data
        for key in trial_channel_1:
            if key not in subject_1:
                subject_1[key] = trial_channel_1[key]
                subject_2[key] = trial_channel_2[key]
            else:
                subject_1[key] = np.concatenate(
                    (subject_1[key], trial_channel_1[key]), axis=0
                )
                subject_2[key] = np.concatenate(
                    (subject_2[key], trial_channel_2[key]), axis=0
                )

    # Saves stimuli
    if save_results:
        for key in subject_1:
            os.makedirs(
                export_paths[key], 
                exist_ok=True
            )
            general_functions.dump_pickle(
                path=os.path.join(export_paths[key], f'Sesion{session}.pkl'), 
                obj=[subject_1[key], subject_2[key]], 
                rewrite=True
            )
    sub1, sub2 = sort_stimuli_based_on_situation(
        subject_1=subject_1,
        subject_2=subject_2,
        situation=situation
    )
    return sub1, sub2

def load_data(
    session:int, 
    stimuli:str, 
    band:str,
    preprocessed_data_path:str, 
    situation:str='External',
    save_results:bool=True,
    overwrite: bool=False
)->tuple:
    """
    Loads and processes EEG and stimuli data for a given session.

    Parameters
    ----------
    session : int
        Session number.
    stimuli : str
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'.
        Allowed stimuli are: 
    preprocessed_data_path : str
        Path directing to processed data.
    situation : str, optional
        Situation considered when performing the analysis, by default 'External'. Allowed situations are: 
        ['Internal','Internal_BS','External', 'External_BS', 'All'].
        Also any of the above options concatenated by '_Silence_x', where x is an integer that represents 
        the percentage of samples with silence within a row of the design matrix.
    save_results : bool, optional
        If True, saves the results in the preprocessed_data_path. Default is False.
    overwrite : bool, optional
        If True, overwrites the existing preprocessed data, forcing load_raw. Default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: sessions of both subjects.
        - dict: information about the samples.
    """
    check_syntax(stimuli=stimuli, band=band, situation=situation)
                
    # Re-order stimuli and band to create just one file for each case: 'Phonemes_Envelope' --> 'Envelope_Phonemes'
    ordered_stimuli = sorted(stimuli.split('_'))

    # Try to load procesed data, if it fails it loads raw data
    logger.info('Loading preprocessed stimuli data\n')
    
    # This function already compute envelope if it doesn't exist
    samples_info = load_samples_info(
        preprocessed_data_path=preprocessed_data_path,
        situation=situation,
        session=session,
        save_results=save_results,
        overwrite=overwrite
    )
    session_1, session_2 = load_stimuli(
        preprocessed_data_path=preprocessed_data_path,
        samples_info=samples_info,
        situation=situation,
        stimuli='_'.join(ordered_stimuli),
        band=band,
        session=session,
        overwrite=overwrite, 
        save_results=save_results,
    )    
    
    return session_1, session_2, samples_info

import multiprocessing as mp
import concurrent.futures
from utils.from_commands import create_dynamic_parser, apply_args_to_config
config.LOG_LEVEL = 'WARNING'

parser = create_dynamic_parser()
args = parser.parse_args()
apply_args_to_config(args)


# =====================
# SIMPLE EXECUTION CODE
def main_one_process(
    situations = config.situations,
    sessions = config.sessions,
    stimuli = config.stimuli,
    bands = config.bands,
    saves_dir = config.saves_dir,
    tmin = config.tmin,
    tmax = config.tmax,
    save_results = False
):
    total_results = {
        situation: {
            band: {
                stim: {
                    session: None for session in sessions
                } for stim in stimuli
            } for band in bands
        } for situation in situations
    }
    for situation in situations:
        preprocessed_data_path_main = f'{saves_dir}/preprocessed_data/tmin{tmin}_tmax{tmax}/'
        for band in bands:
            for stimulus in stimuli:
                sorted_stimuli, sorted_bands = sorted(stimulus.split('_')), sorted(band.split('_'))
                stimulus, band = '_'.join(sorted_stimuli), '_'.join(sorted_bands)

                # Update
                logger.info(
                    '\n===========================\n'
                    '\tPARAMETERS\n\n'
                    f'Model: {config.model}\n'
                    f'Band: {band}\n'
                    f'Stimulus: {stimulus}\n'
                    f'Condition: {situation}\n'
                    f'Time interval: ({tmin},{tmax})s\n'
                    '\n===========================\n'
                )
                for session in sessions:
                    print(f'\n-------> Start of session {session}\n')
                    
                    subject_1, subject_2, samples_info = load_data(
                        preprocessed_data_path=preprocessed_data_path_main,
                        situation=situation,
                        stimuli=stimulus,
                        session=session,
                        band=band,
                        save_results=save_results
                    )
                    
                    # Print the progress of the iteration
                    general_functions.iteration_percentage(
                        txt=f'\n-------> End of session {session}\n', 
                        i=sessions.index(session), 
                        length_of_iterator=len(sessions),
                        logger=logger
                                       )
                    total_results[situation][band][stimulus] = (subject_1, subject_2, samples_info)

# =======================
# PARALLEL EXECUTION CODE
def main_parallel(
    situations = config.situations,
    sessions = config.sessions,
    stimuli = config.stimuli,
    bands = config.bands,
    saves_dir = config.saves_dir,
    tmin = config.tmin,
    tmax = config.tmax,
    number_of_workers = config.number_of_workers,
    save_results = True
):
    total_results = {
        situation: {
            band: {
                stim: {
                    session: None for session in sessions
                } for stim in stimuli
            } for band in bands
        } for situation in situations
    }

    # Crear todas las combinaciones de parámetros
    param_combinations = []
    for situation in situations:
        preprocessed_data_path_main = f'{saves_dir}/preprocessed_data/tmin{tmin}_tmax{tmax}/'
        for band in bands:
            for stimuli_ in stimuli:
                sorted_stimuli, sorted_bands = sorted(stimuli_.split('_')), sorted(band.split('_'))
                stimuli_, band = '_'.join(sorted_stimuli), '_'.join(sorted_bands)
                
                for session in sessions:
                    param_combinations.append({
                        'situation': situation,
                        'preprocessed_data_path': preprocessed_data_path_main,
                        'band': band,
                        'stimuli': stimuli_,
                        'session': session,
                    })
    
    def process_single_session(params):
        """Procesa una sesión individual"""
        try:
            logger.info(
                f"Processing: {params['stimuli']} | {params['band']} | "
                f"{params['situation']} | Session {params['session']}"
            )
            
            subject_1, subject_2, samples_info = load_data(
                preprocessed_data_path=params['preprocessed_data_path'],
                situation=params['situation'],
                stimuli=params['stimuli'],
                session=params['session'],
                band=params['band'],
                save_results=save_results
            )
            
            return {
                'session': params['session'],
                'status': 'success',
                'params': params
            }
            
        except Exception as e:
            logger.error(f"Error processing session {params['session']}: {e}")
            return {
                'session': params['session'],
                'status': 'error',
                'error': str(e),
                'params': params
            }
    
    
    # Paralelizar el procesamiento
    max_workers = min(mp.cpu_count() - 1, number_of_workers)  # Usar máximo 8 workers para evitar sobrecarga
    logger.info(f"Starting parallel processing with {max_workers} workers")
    logger.info(f"Total combinations to process: {len(param_combinations)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todos los trabajos
        futures = [executor.submit(process_single_session, params) for params in param_combinations]
        
        # Procesar resultados conforme se completan
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            
            # Fill total_results with the result (success or error info)
            params = result['params']
            total_results[params['situation']][params['band']][params['stimuli']][params['session']] = result
            
            if result['status'] == 'success':
                logger.info(f"✓ Completed session {result['session']} ({i+1}/{len(futures)})")
            else:
                logger.error(f"✗ Failed session {result['session']}: {result['error']}")
            
            # Mostrar progreso
            general_functions.iteration_percentage(
                txt=f"Overall progress: {i+1}/{len(futures)} completed",
                i=i,
                length_of_iterator=len(futures)
            )
    return total_results

if __name__ == "__main__":
    if config.parallel_load:
        # Parallel execution
        results = main_parallel(
            
        )
    else:
        # Parallel execution
        results = main_one_process(
        )