# Standard libraries
import numpy as np, pandas as pd, os, warnings, time
from tqdm import tqdm

# Specific libraries
import torch, mne, librosa, opensmile, textgrids
from praatio import pitch_and_intensity
from phonet.phonet import Phonet

from transformers import WhisperProcessor, WhisperModel # from transformers import Wav2Vec2Model, Wav2Vec2Processor

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from scipy.interpolate import interp1d
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
import resampy

# Modules
import processing, funciones, config
from phoneme_implementation_from_phonet import Phones

# Review this If we want to update packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(verbose='CRITICAL')
exp_info = config.Exp_info()

TRANSFORMER_MODEL = "openai/whisper-base"
# TRANSFORMER_MODEL = "openai/whisper-tiny"
# TRANSFORMER_MODEL = "facebook/wav2vec2-large-xlsr-53-distilled"
# TRANSFORMER_MODEL = "facebook/wav2vec2-base"

class Trial_channel:
    def __init__(
        self, 
        s:int=21, 
        trial:int=1, 
        channel:int=1, 
        band:str='All', 
        sr:float=128, 
        causal_filter_eeg:bool=True, 
        envelope_filter:bool=False, 
        silence_threshold:float=0.03,
        situation:str='External',
        praat_executable_path:str=r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe"
        )->None: 
        """
        Initializes the Trial_channel class with the given parameters.

        Parameters
        ----------
        s : int, optional
            Session number, by default 21
        trial : int, optional
            Trial number, by default 1
        channel : int, optional
            Channel number used to record the audio (it can be from subject 1 or subject 2), by default 1
        band : str, optional
            EEG frequency band, by default 'All'. It could be one of:
            ['Delta','Theta',Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
        sr : float, optional
            Sampling rate, by default 128
        causal_filter_eeg : bool, optional
            Whether to use a causal filter for EEG, by default True
        envelope_filter : bool, optional
            Whether to use an envelope filter, by default False
        silence_threshold : float, optional
            Silence threshold of the dialogue, by default 0.03
        situation : str, optional
            Situation considered when performing the analysis, by default 'External'. Allowed situations are:
            ['Internal','Internal_BS','External', 'External_BS', 'Internal_All_Times', 'External_All_Times']
        praat_executable_path : str, optional
            Path to Praat executable, by default r'C:\\Users\\User\\Downloads\\programas_descargados_por_octavio\\Praat.exe'

        Returns
        -------
        None

        Raises
        ------
        SyntaxError
            If the band is not in the allowed_band_frequencies list. It must be one of:
            ['Delta','Theta','Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
        """
        
        # Participants sex, ordered by session
        sex_list = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']
        allowed_band_frequencies = ['Delta','Theta','Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
        if band in allowed_band_frequencies:
            self.band= band
        else:
            raise SyntaxError(f"{band} is not an allowed band frecuency. Allowed bands are: {allowed_band_frequencies}")

        # Minimum and maximum frequency allowed within specified band
        self.l_freq_eeg, self.h_freq_eeg = processing.band_freq(self.band)
        self.sr = sr
        self.silence_threshold = silence_threshold
        self.audio_sr = 16000
        self.sex = sex_list[(s - 21) * 2 + channel - 1]
        self.causal_filter_eeg = causal_filter_eeg
        self.envelope_filter = envelope_filter
        self.situation = situation
        self.session = s
        self.trial = trial
        self.channel = channel
        
        # To be filled with loaded data
        self.eeg = None

        # Relevant paths
        self.praat_executable_path = praat_executable_path
        self.eeg_fname = os.path.normpath(f"Datos/EEG/S{s}/s{s}-{channel}-Trial{trial}-Deci-Filter-Trim-ICA-Pruned.set")
        self.wav_fname = os.path.normpath(f"Datos/wavs/S{s}/s{s}.objects.{trial:02d}.channel{channel}.wav")
        self.pitch_fname = os.path.normpath(f"S{s}/s{s}.objects.{trial:02d}.channel{channel}.txt")
        self.phn_fname = os.path.normpath(f"Datos/phonemes/S{s}/s{s}.objects.{trial:02d}.channel{channel}.aligned_fa.TextGrid")
        self.phn_fname_manual = os.path.normpath(f"Datos/phonemes/S{s}/manual/s{s}_objects_{trial:02d}_channel{channel}_aligned_faTAMARA.TextGrid")
        self.phrases_fname = os.path.normpath(f"Datos/phrases/S{s}/s{s}.objects.{trial:02d}.channel{channel}.phrases")
        # self.mistakes_path = os.path.normpath(f"Datos/mistakes/filtered_session{s}_trial{trial:02d}_channel{channel}.TextGrid")
        self.mistakes_path = os.path.normpath(f"Datos/mistakes_corrected/filtered_session{s}_trial{trial:02d}_channel{channel}.TextGrid")
        self.mistakes_control_path = os.path.normpath(f"Datos/mistakes_control/filtered_session{s}_trial{trial:02d}_channel{channel}.TextGrid")
        
    def f_eeg(
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
        eeg = mne.io.read_raw_eeglab(input_fname=self.eeg_fname, preload=True) 
        # eeg = mne.io.read_raw_eeglab(input_fname=r'Datos\EEG\S21\s21-1-Trial1-Deci-Filter-Trim-ICA-Pruned.set', preload=True) 
        
        # Apply a lowpass filter
        if self.band:
            if self.causal_filter_eeg:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, phase='minimum')
                # iir_params = {
                # "ftype": "cheby2",       # Filter type: Chebyshev Type II
                # "order": 4,              # Filter order
                # "rs": 20,                # Stopband attenuation (dB)
                # }
                
                # eeg = eeg.filter(
                #                 l_freq=self.l_freq_eeg,
                #                 h_freq=self.h_freq_eeg,
                #                 method="iir",
                #                 iir_params=iir_params
                #                 )
            else:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg)
                # eeg = eeg.filter(l_freq=4, h_freq=8)
        # # Store dimension mne.raw
        # eeg.resample(sfreq=self.sr)

        # # Return mne representation Times x nchannels
        # self.eeg = eeg.copy()
        # return self.eeg.get_data().T*1e6
        
        # Get mne representation Times x nchannels
        self.eeg = eeg.copy()
        eeg = self.eeg.get_data().T*1e6  # paso a array y tiro la primer columna de tiempo
        # eeg = eeg.get_data().T*1e6  # paso a array y tiro la primer columna de tiempo
        

        # Downsample
        eeg = processing.subsample(
            x=eeg, 
            step=int(self.eeg.info.get("sfreq")/ self.sr)
            )
        # eeg = processing.subsample(
        #     x=eeg, 
        #     step=4
        #     )
        return eeg

    def f_info(
        self
        )->mne.Info:
        """
        A montage is define as a descriptor for the set up: EEG channel names and relative positions of sensors on the scalp. 
        A montage can also contain locations for HPI points, fiducial points, or extra head shape points.In this case, 'BioSemi
        cap with 128 electrodes (128+3 locations)'.

        Returns
        -------
        mne.Info
            Montage of the measurment.
        """
        # Define montage and info object
        montage = mne.channels.make_standard_montage('biosemi128')
        channel_names = montage.ch_names
        return mne.create_info(ch_names=channel_names[:], sfreq=self.sr, ch_types='eeg').set_montage(montage)

    def f_mistakes(
        self, 
        envelope:np.ndarray, 
        kind:str='Mistakes-Separated'
        )->np.ndarray:
        """
        Calculates mistakes (lexical, articulatory, discursive) signal from annotated data

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform

        Returns
        -------
        np.ndarray
            len(envelope)X3 binary array if separated else len(envelope)X1 binary array
            len(envelope)X3 binary array if separated else len(envelope)X1
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
                
        # Match length of mistake signal with envelope
        difference = len(mistake_signal)-len(envelope)
        if difference>0:
            mistake_signal = mistake_signal[:-difference]
        elif difference<0:
            mistake_signal = np.concatenate((mistake_signal, np.zeros(shape=(np.abs(difference), 3)))) if separated else np.concatenate((mistake_signal, np.zeros(shape=(np.abs(difference), 1))))

        return mistake_signal
    
    def f_mistakes_control(
        self, 
        envelope:np.ndarray, 
        kind:str='Control-Separated'
        )->np.ndarray:
        """
        Calculates mistakes control (lexical, articulatory, discursive) signal from annotated data

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform

        Returns
        -------
        np.ndarray
            len(envelope)X3 binary array if separated else len(envelope)X1
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
        
        # Match length of mistake signal with envelope
        difference = len(control_signal)-len(envelope)                
        if difference>0:
            control_signal = control_signal[:-difference]
        elif difference<0:
            control_signal = np.concatenate((control_signal, np.zeros(shape=(np.abs(difference), 3)))) if separated else np.concatenate((control_signal, np.zeros(shape=(np.abs(difference), 1))))
        return control_signal

    def f_envelope(
        self
        )->np.ndarray: 
        """
        Takes the low pass filtered -butterworth-, downsample and smoothened envelope of .wav file. Then matches in length to the EEG

        Returns
        -------
        np.ndarray
            Envelope of wav signal with desire dimensions, using Hilbert transform
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        # wav = wavfile.read(r'Datos\wavs\S21\s21.objects.01.channel1.wav')[1]
        
        wav = wav.astype("float")

        # Calculate envelope
        envelope = np.abs(sgn.hilbert(wav))
        
        # Apply lowpass butterworth filter
        if self.envelope_filter == 'Causal':# TODO can it be replaced for a mne filter?
            envelope = processing.butter_filter(
                    data=envelope, 
                    frequencies=25, #frequencies=25 creo que es el cutoff
                    sampling_freq=self.audio_sr,  
                    btype='lowpass', 
                    order=3, 
                    axis=0, 
                    ftype='Causal'
                    ).reshape(-1,1)
        elif self.envelope_filter == 'NonCausal':
            envelope = processing.butter_filter(
                    data=envelope, 
                    frequencies=25, 
                    sampling_freq=self.audio_sr,
                    btype='lowpass', 
                    order=3, 
                    axis=0, 
                    ftype='NonCausal'
                    ).reshape(-1,1)
        
        # Resample # TODO padear un cero en el envelope
        window_size, stride = int(self.audio_sr/self.sr), int(self.audio_sr/self.sr)
        # window_size, stride = 125, 125
        envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
        envelope = envelope.reshape(-1, 1)
        return envelope
        # # Creates mne raw array
        # info_envelope = mne.create_info(ch_names=['Envelope'], sfreq=self.audio_sr, ch_types='misc')
        # envelope_mne_array = mne.io.RawArray(data=envelope.T, info=info_envelope)

        # # Resample to match EEG data
        # envelope_mne_array.resample(sfreq=self.sr)
        # return envelope_mne_array.get_data().T

    def f_spectrogram(
        self
        )->np.ndarray:
        """
        Calculates spectrogram of .wav file between 16 Mel frequencies

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform

        Returns
        -------
        np.ndarray
            Matrix with sprectrogram in given mel frequncies of dimension (Samples X Mel)
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")
        
        # Calculates the mel frequencies spectrogram giving the desire sampling (match the EEG)
        sample_window = int(self.audio_sr/self.sr)
        S = librosa.feature.melspectrogram(
            y=wav,
            sr=self.audio_sr, 
            n_fft=sample_window, 
            hop_length=sample_window, 
            n_mels=16
            )
        # Transform to dB using normalization to 1
        S_DB = librosa.power_to_db(S=S, ref=np.max)
        
        return S_DB.T

    def f_wav2vec2(
        self,
        envelope:np.ndarray,
        eeg:np.ndarray,
        n_components:int=16
        )->np.ndarray:
        """
        Extracts features from an audio file using the Wav2Vec2 model, applies PCA for dimensionality reduction, and returns the reduced features.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.
        eeg : np.ndarray
            Filtered signal of EEG
        n_components : int
            Number of principal components to retain after applying PCA.

        Returns
        -------
        np.ndarray
            Reduced hidden states from the Wav2Vec2 model after applying PCA.
        """
        print('WARNING: TO RUN THIS FEATURE YOU NEED TO HAVE THE TRANSFORMER MODEL DOWNLOADED AND HAVE THE ENVELOPE MODEL ALREADY RUN FOR THE SITUATION OF INTEREST.')
        
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")
        
        # Identify which moments of the given condition are present in the audio file
        samples_info = funciones.load_pickle(path=f'saves/preprocessed_data/{self.situation}/tmin-0.2_tmax0.6/samples_info/samples_info_{self.session}.pkl')
        keepindexes = samples_info[f'keep_indexes{self.channel}']
        len_trial = samples_info[f'trial_lengths{self.channel}'][self.trial]
        keepindexes = [keep for keep in keepindexes if keep<=len_trial]
        
        filter_index_used_in_trial = []
        for i in range(len(envelope)):
            try:
                filter_index_used_in_trial.append(keepindexes.index(i))
            except Exception as err:
                continue
        
        indexes_128Hz = np.array(keepindexes)[filter_index_used_in_trial]
        
        # Now, as the audio is sampled at 16e3 Hz instead of 128 Hz we have to consider more indexes (since in one sample step at 128 Hz has 8 indexes at 16e3 Hz)
        indexes_16kHz = indexes_128Hz*(16e3/128)
        dense_indexes = []
        
        # Complete consecutive samples
        for i in range(len(indexes_128Hz)-1):
            start = indexes_16kHz[i]
            end = indexes_16kHz[i+1]
            if indexes_128Hz[i+1]-indexes_128Hz[i]==1:
                dense_indexes.extend(np.arange(start, end + 1))
            else:
                dense_indexes.append(start)
        dense_indexes.append(indexes_16kHz[-1])
        dense_indexes = list(map(int,dense_indexes))
        
        # Redefine .wav to make 0 elements outside dense_indexes
        mask = np.ones_like(wav, dtype=bool)
        mask[dense_indexes] = False
        wav[mask] = 0

        # Get name of folder
        modelfname = f'wav2vec2_weights_{TRANSFORMER_MODEL.split("wav2vec2-")[1]}' if 'wav2vec2' in TRANSFORMER_MODEL else f'whisper_weights_{TRANSFORMER_MODEL.split("whisper-")[1]}'

        # Loads model and proccesor
        ini = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Passing `gradient_checkpointing` to a config initialization is deprecated")
            processor = WhisperProcessor.from_pretrained(TRANSFORMER_MODEL, cache_dir=f'saves/preprocessed_data/{modelfname}')
            model = WhisperModel.from_pretrained(TRANSFORMER_MODEL, cache_dir=f'saves/preprocessed_data/{modelfname}')

            # processor = Wav2Vec2Processor.from_pretrained(wac2vec2model, cache_dir=f'saves/preprocessed_data/{modelfname}')
            # model = Wav2Vec2Model.from_pretrained(wac2vec2model, cache_dir=f'saves/preprocessed_data/{modelfname}')
        
        # Preprocessing: break .wav in consecutive windows to feed the model and then concatenate the results
        # sample_windows = np.arange(0, wav.shape[0], 800e-3*self.audio_sr, dtype=int)
        # sample_windows = np.concatenate((sample_windows, np.array([wav.shape[0]])))
        # sample_windows = []
        # current_window = [dense_indexes[0]]
        # for i in range(1, len(dense_indexes)):
        #     if dense_indexes[i]==(dense_indexes[i-1]+1):
        #         current_window.append(dense_indexes[i])
        #     else:
        #         sample_windows.append(current_window)
        #         current_window = [dense_indexes[i]]
        # Preprocessing: break .wav in windows of 800ms to feed the model and then concatenate the results
        sample_windows = np.arange(0, wav.shape[0], 800e-3*self.audio_sr, dtype=int)
        sample_windows = np.concatenate((sample_windows, np.array([wav.shape[0]])))
        
        full_hidden_states = []
        for sample_window_d, sample_window in tqdm(zip(np.roll(sample_windows, shift=1)[1:], sample_windows[1:]), total=sample_windows.shape[0]):
            input_values = processor(
                        wav[sample_window_d:sample_window],
                        sampling_rate=self.audio_sr, 
                        return_tensors="pt"
                        ).input_features
            # Get model's output
            outputs = model.encoder(
                    input_values, 
                    output_hidden_states=True
                    )
        
            # Get last hidden layer
            hidden_states = outputs.hidden_states[-1]  #(batch_size, sequence_length, hidden_size)
            
            # Adjust dimensions (take out batch dimension and resample sequence length to match envelope)
            hidden_states = hidden_states.squeeze(0).detach().numpy()  
            
            full_hidden_states.append(hidden_states)
            
        # Apply PCA to find principal components 
        pca = PCA(n_components=n_components)
        pca.fit_transform(np.concatenate(full_hidden_states, axis=0))
        print(f'Portion of variance of whole hidden layer explained by {n_components} components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%')

        hidden_state_final = []
        # for hidden_states, sample_window, in zip(full_hidden_states, sample_windows):
        for hidden_states, sample_window_d, sample_window in zip(full_hidden_states, np.roll(sample_windows, shift=1)[1:], sample_windows[1:]):
            window_time = (sample_window-sample_window_d)/self.audio_sr
            sample_freq_w = int(np.round(hidden_states.shape[0] / window_time, 0))
            
            hidden_states_reduced = pca.transform(hidden_states)
            target_hidden_states = resampy.resample(
                hidden_states_reduced, 
                sample_freq_w, 
                config.sr, 
                axis=0
                )
            hidden_state_final.append(target_hidden_states)
        
        hidden_state_final = np.concatenate(hidden_state_final, axis=0)
        
        print(f'The model took {(time.time()-ini)/60:.2f} minutes')
        
        # Cutoff to minimum length
        min_length = min(eeg.shape[0], hidden_state_final.shape[0])
        return hidden_state_final[:min_length]
        
        # reduced_hidden_states = pca.fit_transform(hidden_states.detach().numpy())
        # reduced_hidden_states.shape

        # # reduced_hidden_states = gaussian_filter1d(hidden_states_resampled, sigma=15, axis=0)
        # # reduced_hidden_states = smooth_with_spline(reduced_hidden_states, smoothing_factor=15)
        
        # # Create an interpolation function for each hidden state
        # interp_funcs = [interp1d(np.arange(reduced_hidden_states.shape[0]), reduced_hidden_states[:, i], kind='linear') for i in range(reduced_hidden_states.shape[1])]
        # new_time_points = np.linspace(0, reduced_hidden_states.shape[0] - 1, envelope.shape[0])

        # # Apply interpolation to each hidden state
        # hidden_states_resampled = np.array([interp_func(new_time_points) for interp_func in interp_funcs]).T  # Shape (envelope_length, hidden_size)
        
        # end = time.time()
        # print(f'The model took {(end-ini)/60:.2f} minutes')
        # return hidden_states_resampled
        
        # import matplotlib.pyplot as plt
        # # plt.figure()
        # # plt.plot(reduced_hidden_states[:,0])
        # # plt.show(block=False)
        # plt.figure()
        # # hidden_states_filtered = lowpass_filter(hidden_states_resampled, cutoff=.1, fs=128)
        # # plt.plot(hidden_states_filtered[:,0])
        # plt.plot(hidden_states_resampled[:,0])
        # plt.show(block=False)
        
        

        # # Suponiendo que 'hidden_states_resampled' es de forma (n_samples, hidden_size)
        # # y querés suavizar a lo largo del tiempo (axis=0)
        # hidden_states_smoothed = gaussian_filter1d(hidden_states_resampled, sigma=15, axis=0)
        
        # plt.figure()
        # # plt.plot(gaussian_filter1d(reduced_hidden_states[:,1], sigma=15, axis=0))
        # plt.plot(reduced_hidden_states[:,1])
        
        # plt.show(block=False)
       
        # # # Apply PCA to find principal components that maximize correlation between EEG and audio
        # cca = CCA(n_components=16)
        # # if eeg.shape[0] < hidden_states_resampled.shape[0]:
        # #     eeg = np.repeat(eeg, hidden_states_resampled.shape[0]//eeg.shape[0], axis=0)

        # if eeg.shape[0]!=hidden_states_resampled.shape[0]:
        #     min_length = min(eeg.shape[0], hidden_states_resampled.shape[0])
        #     eeg = eeg[:min_length]
        #     hidden_states_resampled = hidden_states_resampled[:min_length]
        # ini = time.time()
        # scaler_X = StandardScaler()
        # scaler_Y = StandardScaler()
        # X_scaled = scaler_X.fit_transform(hidden_states_resampled)
        # Y_scaled = scaler_Y.fit_transform(eeg)
        # cca = CCA(n_components=n_components, max_iter=1000)
        # X_canonical, _ = cca.fit_transform(X_scaled, Y_scaled)
        # end = time.time()
        # print(f'The CCA took {(end-ini)/60:.2f} minutes')
        # return X_canonical #- #np.mean(X_canonical, axis=1, keepdims=True)  
        
    def f_mfccs(
        self, 
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

        # # DIFIEREN EN UN FACTOR DE ESCALA WAV=32768*SIGNAL; siendo la normalización de punto flotante nbits=16, 2 ** (nbits - 1)=32768
        # signal, audio_sr = librosa.load(self.wav_fname, sr=None) 
        
        # Calculate matrix of mfccs
        sample_window = int(self.audio_sr/self.sr)
        mfccs = librosa.feature.mfcc(y=wav, n_mfcc=16, n_mels=16, sr=self.audio_sr, n_fft=sample_window, hop_length=sample_window)
        
        # Append deltas and deltas deltas
        if kind.startswith('Mfccs'):
            if kind.endswith('Deltas'):
                delta_mfccs = librosa.feature.delta(mfccs)
                mfccs_features = np.concatenate((mfccs, delta_mfccs))
                if kind.endswith('Deltas-Deltas'):
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
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
                    deltas = np.concatenate((delta_mfccs, delta2_mfccs))
                    return deltas.T
                else:
                    return delta_mfccs.T

        # import librosa.display
        # librosa.display.specshow(mfccs, 
        #                         x_axis="time", 
        #                         sr=sr)
        # plt.colorbar(format="%+2.f")
   
    def f_jitter_shimmer(
        self, 
        envelope:np.ndarray
        )->tuple: # NEVER USED
        """
        Gives the jitter and shimmer matching the size of the envelope

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform

        Returns
        -------
        tuple
            jitter and shimmer arrays with length smaller or equal to envelope length.
        """
        # Processing object to extract audio features
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

        # Creates a pd.DataFrame to store audio features
        y = smile.process_file(self.wav_fname)
        
        # Removes file index of multindex, leaving just start and end times as index
        y.index = y.index.droplevel(0)

        # Transform to single index with elapsed time in seconds
        y.index = y.index.map(lambda x: x[0].total_seconds())

        # Extract series with specific features
        jitter = y['jitterLocal_sma3nz']
        shimmer = y['shimmerLocaldB_sma3nz']
        
        # Calculate the least common multiple between envelope and jitter lengths (jimmer length is the same as jitter)
        mcm = funciones.minimo_comun_multiplo(len(jitter), len(envelope))
        
        # Repeat each value the number of times it takes the length of jitter to achive the mcm. The result is that jitter length matches mcm
        jitter = np.repeat(jitter, mcm / len(jitter))
        shimmer = np.repeat(shimmer, mcm / len(shimmer))

        # Subsample by the number of times it takes the length of the envelope to achive the mcm. Now it has exactly the same size as envelope
        jitter = processing.subsample(
            x=jitter, 
            step=mcm/len(envelope)
            )
        shimmer = processing.subsample(
            x=shimmer, 
            step=mcm/len(envelope)
            )

        # Reassurance that the count is correct
        jitter = jitter[:min(len(jitter), len(envelope))].reshape(-1,1)
        shimmer = shimmer[:min(len(shimmer), len(envelope))].reshape(-1,1)
        return jitter, shimmer
    
    def f_phonemes_phonet( 
        self, 
        envelope:np.ndarray, 
        kind:str='Phonemes-Discrete-Phonet'
        )->np.ndarray:
        """
        It makes a time-match matrix between the phonemes and the envelope using Phonet implementation. The values and shape of given matrix depend on kind.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform
        kind : str, optional
           Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
            ['Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet']

        Returns
        -------
        np.ndarray
            if kind.startswith('Phonemes-Envelope'):
                Matrix with envelope amplitude at given sample. The matrix dimension is SamplesXPhonemes_labels(in order)
            elif kind.startswith('Phonemes-Discrete'):
                Also a matrix but it has 1s and 0s instead of envelope amplitude.
            elif kind.startswith('Phonemes-Onset'):
                In this case the value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
            
        Raises
        ------
        SyntaxError
            Whether the input value of 'kind' is passed correctly. It must be a one of:
            ['Phonemes-Phonet','Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet'].
        """
        # Check if given kind is a permited input value
        allowed_kind = ['Phonemes-Phonet','Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phonemes are: {allowed_kind}")
        
        # Extract phonemes
        if kind=='Phonemes-Phonet':
            phonet_labels_phonemes = exp_info.phonemes_phonet.copy()
            phonet_labels_phones = exp_info.ph_labels_phonet.copy()
            
            phones_obj = Phones(audio_file=self.wav_fname)
            posterior_prob = phones_obj.compute_phones(PLLR=True) #9167
            
            
            # Match features length
            difference = len(posterior_prob) - len(envelope)

            if difference > 0:
                posterior_prob = posterior_prob[:-difference]
            elif difference < 0:
                # Repeat last sample (probably silence)
                for i in range(np.abs(difference)):
                    aux = posterior_prob[-1].copy() 
                    posterior_prob = np.vstack((posterior_prob, aux.reshape(-1,1).T))
            
            # Map phones to phonemes, making the sum
            posterior_prob_phonemes = np.zeros(shape=(posterior_prob.shape[0], len(phonet_labels_phonemes)))

            for h, phone in enumerate(phonet_labels_phones):
                phoneme_index = phonet_labels_phonemes.index(exp_info.phones_to_phonemes[phone])
                posterior_prob_phonemes[:, phoneme_index] += posterior_prob[:, h]
            
            # Calculate posterior llr
            pllr = np.zeros(shape=posterior_prob_phonemes.shape)
            number_of_phonemes = posterior_prob_phonemes.shape[1]
            for ph in range(number_of_phonemes):
                pllr[:, ph] = np.log10(posterior_prob_phonemes[:, ph]/(1-posterior_prob_phonemes[:, ph]))
            
            # Centralizamos 
            pllr = pllr - np.mean(pllr, axis=1, keepdims=True)  
            
            # Removemos silencios
            pllr_without_silence = pllr[:, np.arange(number_of_phonemes) != phonet_labels_phonemes.index('/sil/')]
            return pllr_without_silence
        else:
            phones_obj = Phones(audio_file=self.wav_fname)
            time,  sec_phones = phones_obj.compute_phones() #9167
        
        # Remove silences, since it won't be used in prediction (when silence occurs, all phoneme are 0)
        phonet_labels = exp_info.phonemes_phonet.copy()
        phonet_labels.remove('/sil/')
            
        # Match features length
        difference = len(sec_phones) - len(envelope)

        if difference > 0:
            sec_phones = sec_phones[:-difference]
        elif difference < 0:
            # In this case, silences are append
            for i in range(np.abs(difference)):
                sec_phones.append('<p:>')
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape=(len(sec_phones), len(phonet_labels)))
        
        # Match phoneme with kind
        if kind.startswith('Phonemes-Envelope'):
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phonemes[i, phonet_labels.index(exp_info.phones_to_phonemes[tagg])] = envelope[i]
        elif kind.startswith('Phonemes-Discrete'):
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phonemes[i, phonet_labels.index(exp_info.phones_to_phonemes[tagg])] = 1
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
                    phonemes[i, phonet_labels.index(exp_info.phones_to_phonemes[tagg])] = 1
        return phonemes
    
    def f_phones_phonet( # TODO CAMBAIR EN TODOS LADOS ESTO SON FONOS NO FONEMAS
        self, 
        envelope:np.ndarray, 
        kind:str='Phones-Discrete-Phonet'
        )->np.ndarray:
        """
        It makes a time-match matrix between the phones and the envelope using Phonet implementation. The values and shape of given matrix depend on kind.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform
        kind : str, optional
           Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
            ['Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet', 'Phones-Onset-Phonet']

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
            ['Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet', 'Phones-Onset-Phonet'].
        """
        # Check if given kind is a permited input value
        allowed_kind = ['Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet', 'Phones-Onset-Phonet']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phones are: {allowed_kind}")

        # Extract phonemes
        if kind=='Phones-Phonet':
            phonet_labels_phones = exp_info.ph_labels_phonet.copy()
            
            phones_obj = Phones(audio_file=self.wav_fname)
            posterior_prob = phones_obj.compute_phones(PLLR=True) #9167
            
            # Match features length
            difference = len(posterior_prob) - len(envelope)

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
            pllr_without_silence = pllr[:, (np.arange(number_of_phones) != phonet_labels_phones.index('sil'))&(np.arange(number_of_phones) != phonet_labels_phones.index('<p:>'))]
            return pllr_without_silence
        else:
            # Extract phones
            phones_obj = Phones(audio_file=self.wav_fname)
            _,  sec_phones = phones_obj.compute_phones() 
        
        # Get phonet phoneme labels
        phonet_labels = exp_info.ph_labels_phonet.copy()
        phonet_labels.remove('<p:>')
        phonet_labels.remove('sil')
        
        # Match features length
        difference = len(sec_phones) - len(envelope)

        if difference > 0:
            sec_phones = sec_phones[:-difference]
        elif difference < 0:
            # In this case, silences are appended
            for i in range(np.abs(difference)):
                sec_phones.append('<p:>')
        
        # Make empty array of phones
        phones = np.zeros(shape=(len(sec_phones), len(phonet_labels)))
        
        # Match phoneme with kind
        if kind.startswith('Phones-Envelope'):
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phones[i, phonet_labels.index(tagg)] = envelope[i]
        elif kind.startswith('Phones-Discrete'):
            for i, tagg in enumerate(sec_phones):
                if (tagg!='<p:>') and (tagg!='sil'):
                    phones[i, phonet_labels.index(tagg)] = 1
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
                    phones[i, phonet_labels.index(tagg)] = 1
        return phones

    def f_phonemes(
        self, 
        envelope:np.ndarray, 
        kind:str='Phonemes-Envelope-Manual'
        )->np.ndarray:
        """
        It makes a time-match matrix between the phonemes and the envelope. The values and shape of given matrix depend on kind.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform
        kind : str, optional
           Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
            ['Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete', 'Phonemes-Discrete-Manual', 'Phonemes-Onset', 'Phonemes-Onset-Manual']

        Returns
        -------
        np.ndarray
            if kind.startswith('Phonemes-Envelope'):
                Matrix with envelope amplitude at given sample. The matrix dimension is SamplesXPhonemes_labels(in order)
            elif kind.startswith('Phonemes-Discrete'):
                Also a matrix but it has 1s and 0s instead of envelope amplitude.
            elif kind.startswith('Phonemes-Onset'):
                In this case the value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
            
        Raises
        ------
        SyntaxError
            Whether the input value of 'kind' is passed correctly. It must be one of:
            ['Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete', 'Phonemes-Discrete-Manual', 'Phonemes-Onset', 'Phonemes-Onset-Manual'].
        """
        if kind.endswith('anual'):
            exp_info_labels = exp_info.ph_labels_man
        else: 
            exp_info_labels = exp_info.ph_labels            

        # Check if given kind is a permited input value
        allowed_kind = ['Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete', 'Phonemes-Discrete-Manual', 'Phonemes-Onset', 'Phonemes-Onset-Manual']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phonemes are: {allowed_kind}")

        # Get trial total time length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        # phrases = pd.read_table(r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\phrases\S21\s21.objects.01.channel1.phrases', header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname)
        # grid = textgrids.TextGrid(r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\phonemes\S21\s21.objects.01.channel1.aligned_fa.TextGrid')

        # Get phonemes
        phonemes_grid = grid['transcription : phones']

        # Extend first silence time to trial start time
        phonemes_grid[0].xmin = 0.

        # Parse for labels, times and number of samples within each phoneme
        labels = []
        times = []
        samples = []
        
        for ph in phonemes_grid:
            label = ph.text.transcode()
            label = label.replace(' ', '')
            label = label.replace('º', '')
            label = label.replace('-', '')

            # Rename silences
            if label in ['sil','sp','sile','silsil','SP','s¡p','sils']:
                label = ""
            
            # Check if the phoneme is in the list
            if not(label in exp_info_labels or label==""):
                print(f'"{label}" is not in not a recognized phoneme. Will be added as silence.')
                label = ""
            labels.append(label)
            times.append((ph.xmin, ph.xmax))
            samples.append(np.round((ph.xmax - ph.xmin) * self.sr).astype("int"))

        # Extend on more phoneme of silence till end of trial 
        labels.append("")
        times.append((ph.xmin, trial_tmax))
        samples.append(np.round((trial_tmax - ph.xmax) * self.sr).astype("int"))

        # If use envelope amplitude to make continuous stimuli: the total number of samples must match the samples use for stimuli
        diferencia = np.sum(samples) - len(envelope)

        if diferencia > 0:
            # Making the way back checking when does the number of samples of the ith phoneme exceed diferencia
            for ith_phoneme in [-i-1 for i in range(len(samples))]:
                if diferencia > samples[ith_phoneme]:
                    diferencia -= samples[ith_phoneme]
                    samples[ith_phoneme] = 0
                # When samples is greater than the difference, takes the remaining samples to match the envelope
                else:
                    samples[ith_phoneme] -= diferencia
                    break
        elif diferencia < 0:
            # In this case, the last silence is prolonged
            samples[-1] -= diferencia
        
        # Make a list with phoneme labels tha already are in the known set
        updated_taggs = exp_info_labels + [ph for ph in np.unique(labels) if ph not in exp_info_labels]

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
        # Match phoneme with kind
        if kind.startswith('Phonemes-Envelope'):
            for i, tagg in enumerate(phonemes_tgrid):
                phonemes[i, updated_taggs.index(tagg)] = envelope[i]
        elif kind.startswith('Phonemes-Discrete'):
            for i, tagg in enumerate(phonemes_tgrid):
                phonemes[i, updated_taggs.index(tagg)] = 1
        elif kind.startswith('Phonemes-Onset'):
            # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
            phonemes_onset = [phonemes_tgrid[0]]
            for i in range(1, len(phonemes_tgrid)):
                if phonemes_tgrid[i] == phonemes_tgrid[i-1]:
                    phonemes_onset.append(0)
                else:
                    phonemes_onset.append(phonemes_tgrid[i])
            # Match phoneme with envelope
            for i, tagg in enumerate(phonemes_onset):
                if tagg!=0:
                    phonemes[i, updated_taggs.index(tagg)] = 1
        return phonemes

    def f_phonological_features(
        self, 
        envelope:np.ndarray
        )->np.ndarray:
        """
        Retrive phonological features as a matrix matching envelope length, using Phonet implementation.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform

        Returns
        -------
        np.ndarray
            Matrix with phonological features with shape SAMPLES X FEATURES
        """
        # Define phonological instance and phonological features
        phon_features = Phonet(["all"]).get_PLLR(audio_file=self.wav_fname, plot_flag=False)
        # phon_features = Phonet(['all']).get_PLLR(audio_file=r'Datos/wavs/S21/s21.objects.01.channel1.wav', plot_flag=False)
        
        # Interpole data in desire times
        desire_time = np.linspace(0, envelope.shape[0]/self.sr + 1/self.sr , envelope.shape[0])
        # desire_time = np.linspace(0, envelope.shape[0]/128 + 1/128 , envelope.shape[0])

        # Get feature names
        phon_features_names = [feat for feat in phon_features.columns if feat not in ['time', 'trill', 'pause']]
        
        phonological_features = []
        for phon_feat in phon_features_names:
            phonological_features.append(np.interp(
                                                desire_time, 
                                                phon_features['time'].values, 
                                                phon_features[f'{phon_feat}'].values
                                                )
                                        )
        
        # Return data in desired shape
        return np.stack(phonological_features, axis=0).T

    def f_pitch(
        self, 
        envelope:np.ndarray, 
        kind:str
        )->np.ndarray: 
        """
        Loads the pitch of the speaker, after calculating it from .wav file, using Praat.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform
        kind : str
            Kind of pitch use, by default 'Log-Raw'. Available kinds are:
            'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes'
        Returns
        -------
        np.ndarray
            One-dimensional array with pitch values

        Raises
        ------
        SyntaxError
            Whether the input value of 'kind' is passed correctly. It must be one of:
            ['Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes']
        """

        # Check if given kind is a permited input value
        allowed_kind = ['Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes']
        if kind not in allowed_kind:
            raise SyntaxError(f"{kind} is not an allowed kind of pitch. Allowed phonemes are: {allowed_kind}")
        
        # Makes path for storing data
        output_folder = os.path.normpath(f'Datos/{kind}_threshold_{self.silence_threshold}/')
        
        # Create paths and distinguish subject
        os.makedirs(output_folder, exist_ok=True)
        self.pitch_fname = os.path.join(output_folder, self.pitch_fname)
        if self.sex == 'M':
            minPitch = 50
            maxPitch = 300
        elif self.sex == 'F':
            minPitch = 75
            maxPitch = 500
        
        # Define sample step and calculate pitch
        self.sampleStep = 1/self.sr # .01
        if 'Quad' in kind:
            pitch_and_intensity.extractPI(
                                inputFN=os.path.abspath(self.wav_fname), 
                                outputFN=os.path.abspath(self.pitch_fname), 
                                praatEXE=self.praat_executable_path, 
                                minPitch=minPitch,
                                maxPitch=maxPitch, 
                                sampleStep=self.sampleStep, 
                                silenceThreshold=self.silence_threshold,
                                pitchQuadInterp=True
                                )
            # Loads data
            data = np.genfromtxt(os.path.abspath(self.pitch_fname), dtype=np.float, delimiter=',', missing_values='--undefined--', filling_values=np.inf)
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
                                inputFN=os.path.abspath(self.wav_fname), 
                                outputFN=os.path.abspath(self.pitch_fname), 
                                praatEXE=self.praat_executable_path, 
                                minPitch=minPitch,
                                maxPitch=maxPitch, 
                                sampleStep=self.sampleStep, 
                                silenceThreshold=self.silence_threshold
                                )
            # sampleStep - the frequency to sample pitch at
            # silenceThreshold - segments with lower intensity won't be analyzed
            #                 for pitch
            # forceRegenerate - if running this function for the same file, if False
            #                 just read in the existing pitch file
            # undefinedValue - if None remove from the dataset, otherset set to
            #                 undefinedValue
            # pitchQuadInterp - if True, quadratically interpolate pitch
                
        # Loads data
        data = np.genfromtxt(os.path.abspath(self.pitch_fname), dtype=np.float, delimiter=',', missing_values='--undefined--', filling_values=np.inf)
        time, pitch = data[:, 0], data[:, 1]

        # Get defined indexes
        defined_indexes = np.where(pitch!=np.inf)[0]
        
        # Approximated window size
        window_size = 100e-3
        n_steps_in_window = np.ceil(window_size/self.sampleStep)
        window_size = n_steps_in_window*self.sampleStep

        if kind.endswith('Manual'):
            # Log transformation
            if 'Log' in kind:
                logpitch = np.log(pitch)

                # Interpolate relevant moments of silence
                for i in range(len(defined_indexes)):
                    if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
                        logpitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(x=time[defined_indexes[i-1]+1:defined_indexes[i]], xp=time[defined_indexes], fp=logpitch[defined_indexes])
            
                # Set left values to zero
                logpitch[logpitch==np.inf] = 0
                return logpitch.reshape(-1, 1)
            else: 
                # Interpolate relevant moments of silence
                for i in range(len(defined_indexes)):
                    if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
                        pitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(x=time[defined_indexes[i-1]+1:defined_indexes[i]], xp=time[defined_indexes], fp=pitch[defined_indexes])
                
                # Set left values to zero
                pitch[pitch==np.inf] = 0
                return pitch.reshape(-1,1)

        elif kind.endswith('Phonemes'):
            # Load phonemes matrix, excluding '' label
            phonemes = self.f_phonemes(envelope=envelope, kind='Phonemes-Discrete-Manual')
            phonemes = np.delete(arr=phonemes, obj=-1, axis=1)

            # Given that spacing between samples is 1/self.sr
            sound_indexes = np.where(phonemes.any(axis=1))[0]

            # Log transformation
            if 'Log' in kind:
                logpitch = np.log(pitch)

                # Within window of window_size of said phoneme if there is a silence it gets interpoled
                for i in range(len(sound_indexes)):
                    if 1<(sound_indexes[i]-sound_indexes[i-1])<=n_steps_in_window:
                        logpitch[sound_indexes[i-1]+1:sound_indexes[i]] = np.interp(x=time[sound_indexes[i-1]+1:sound_indexes[i]], xp=time[defined_indexes], fp=logpitch[defined_indexes])
        
                # Set left values to zero
                logpitch[logpitch==np.inf] = 0
                return logpitch.reshape(-1, 1)
            else: 
                # Within window of window_size of said phoneme if there is a silence it gets interpoled
                for i in range(len(sound_indexes)):
                    if 1<(sound_indexes[i]-sound_indexes[i-1])<=n_steps_in_window:
                        pitch[sound_indexes[i-1]+1:sound_indexes[i]] = np.interp(x=time[sound_indexes[i-1]+1:sound_indexes[i]], xp=time[defined_indexes], fp=pitch[defined_indexes])

                # Set left values to zero
                pitch[pitch==np.inf] = 0
                return pitch.reshape(-1,1)
        
        elif kind.endswith('Raw'):
            # Log transformation
            if 'Log' in kind:
                logpitch = np.log(pitch)
        
                # Set left values to zero
                logpitch[logpitch==np.inf]=0
                return logpitch.reshape(-1, 1)
            else: 
                # Set left values to zero
                pitch[pitch==np.inf]=0
                return pitch.reshape(-1,1)
            
    def load_trial(
        self, 
        stims:list
        )->dict: 
        """Extract EEG and calculates specified stimuli.
        Parameters
        ----------
        stims : list
            A list containing possible stimuli. Possible input values are: 
            ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 
            'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 
            'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet',
            'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 'Control-Together', 'Control-Separated', 'Wav2vec2']

        Returns
        -------
        dict
            Dictionary with EEG, info and specified stimuli as mne objects
        """
        channel = {}
        channel['EEG'] = self.f_eeg()
        channel['info'] = self.f_info()
        channel['Envelope'] = self.f_envelope()

        for stim in stims:
            if stim.startswith('Mfccs') or stim.startswith('Deltas'):
                channel[stim] = self.f_mfccs(kind=stim)
            if stim.startswith('Pitch'):
                channel[stim] = self.f_pitch(envelope=channel['Envelope'], kind=stim)
            if stim=='Phonological':
                channel[stim] = self.f_phonological_features(envelope=channel['Envelope'])
            if stim.startswith('Mistakes'):
                channel[stim] = self.f_mistakes(envelope=channel['Envelope'], kind=stim)
            if stim.startswith('Control'):
                channel[stim] = self.f_mistakes_control(envelope=channel['Envelope'], kind=stim)
            if stim=='Wav2vec2':
                channel[stim] = self.f_wav2vec2(envelope=channel['Envelope'], eeg=channel['EEG'])
            if stim=='Spectrogram':
                channel['Spectrogram'] = self.f_spectrogram()
            if stim.startswith('Phonemes'):
                if stim.endswith('Phonet'):
                    channel[stim] = self.f_phonemes_phonet(envelope=channel['Envelope'], kind=stim)
                else:
                    channel[stim] = self.f_phonemes(envelope=channel['Envelope'], kind=stim)
            if stim.startswith('Phones'):
                channel[stim] = self.f_phones_phonet(envelope=channel['Envelope'], kind=stim)
        return channel

class Sesion_class: 
    def __init__(
        self, 
        sesion:int=21, 
        stim:str='Envelope', 
        band:str='All', 
        sr:float=128, 
        causal_filter_eeg:bool=True, 
        envelope_filter:bool=False, 
        situation:str='External', 
        silence_threshold:float=0.03,
        delays:np.ndarray=None,
        preprocessed_data_path:str=os.path.normpath(f'saves/preprocessed_data/tmin{-0.6}_tmax{-.002}/'),
        praat_executable_path:str=r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe"
        )->None:
        """
        This class handles the loading (concatenating trials) and processing of EEG and stimuli data for a given session. 
        It supports both raw and preprocessed data, and can extract various features such as envelope, MFCCs, pitch, phonemes, and more.
        
        Parameters
        ----------
        sesion : int
            Session number, by default 21
        stim : str
            Stimuli to use in the analysis, by default 'Envelope'. If more than one stimulus is wanted, the separator should be '_'. Allowed stimuli are:
            ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 
            'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 
            'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet', 
            'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 'Control-Together', 'Control-Separated', 'Wav2vec2','Phones-Onset-Manual', 'Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet']
        band : str
            Neural frequency band. It could be one of:
            ['Delta','Theta', 'Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
        sr : float
            Sample rate in Hz of the EEG
        causal_filter_eeg : bool, optional
            Whether to use or not a causal filter for the EEG, by default True
        envelope_filter : bool, optional
            Whether to use or not an envelope filter, by default False
        situation : str, optional
            Situation considered when performing the analysis, by default 'External'. Allowed situations are:
            ['Internal','Internal_BS','External', 'External_BS', 'External_All_Times', 'Internal_All_Times']
        silence_threshold : float, optional
            Silence threshold of the dialogue, by default 0.03
        delays : np.ndarray, optional
            Delay array to construct shifted matrix, by default None
        preprocessed_data_path : str
            Path directing to processed data
        praat_executable_path : str
            Path directing to Praat executable
        
        Returns
        -------
        None
        
        Raises
        ------
        SyntaxError
            If 'stim' is not an allowed stimulus. Allowed stimuli are:
            ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 
            'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 
            'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet', 
            'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 'Control-Together', 'Control-Separated', 'Wav2vec2','Phones-Onset-Manual', 'Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet']
            If 'band' is not an allowed band frequency. Allowed frequencies are:
            ['Delta','Theta', 'Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
            If 'situation' is not an allowed situation. Allowed situations are:
            ['Internal','Internal_BS','External', 'External_BS', 'External_All_Times', 'Internal_All_Times']
        """
        # Check if band, stim and situation parameters where passed with the right syntax
        allowed_stims = ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', \
                        'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', \
                        'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet', 'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 'Control-Together', 'Control-Separated', 'Wav2vec2','Phones-Onset-Manual', 'Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet']
        allowed_band_frequencies = ['Delta','Theta','Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
        allowed_situations = ['Internal','Internal_BS','External', 'External_BS', 'Internal_All_Times', 'External_All_Times']
        for st in stim.split('_'):
            if st in allowed_stims:
                pass
            else:
                raise SyntaxError(f"{st} is not an allowed stimulus. Allowed stimuli are: {allowed_stims}. If more than one stimulus is wanted, the separator should be '_'.")
        self.stim = stim
        if band in allowed_band_frequencies:
            self.band = band
        else:
            raise SyntaxError(f"{band} is not an allowed band frecuency. Allowed bands are: {allowed_band_frequencies}")
        if situation in allowed_situations:
            self.situation = situation
        else:
            raise SyntaxError(f"{situation} is not an allowed situation. Allowed situations are: {allowed_situations}")
        
        # Define parameters
        self.sesion = sesion
        self.l_freq_eeg, self.h_freq_eeg = processing.band_freq(band)
        self.sr = sr
        self.delays = delays
        self.causal_filter_eeg = causal_filter_eeg
        self.envelope_filter = envelope_filter
        self.silence_threshold = silence_threshold

        # Relevant paths
        self.praat_executable_path = praat_executable_path
        self.preprocessed_data_path = preprocessed_data_path
        self.samples_info_path = os.path.join(self.preprocessed_data_path, f'samples_info/')
        self.phn_path = f"Datos/phonemes/S{self.sesion}/"
        self.phrases_path = f"Datos/phrases/S{self.sesion}/"

        # Define paths to export data
        self.export_paths = {}
        if self.causal_filter_eeg:
            self.export_paths['EEG'] = os.path.join(self.preprocessed_data_path, f'EEG/{band}/Causal/')
        else:
            self.export_paths['EEG'] = os.path.join(self.preprocessed_data_path, f'EEG/{band}/')
        if self.envelope_filter:
            self.export_paths['Envelope'] = os.path.join(self.preprocessed_data_path, f'Envelope/{self.envelope_filter}/')
        else:
            self.export_paths['Envelope'] = os.path.join(self.preprocessed_data_path, 'Envelope/')
                
        self.export_paths['Mfccs'] = os.path.join(self.preprocessed_data_path, 'Mfccs/')
        self.export_paths['Mfccs-Deltas'] = os.path.join(self.preprocessed_data_path, 'Mfccs-Deltas/')
        self.export_paths['Mfccs-Deltas-Deltas'] = os.path.join(self.preprocessed_data_path, 'Mfccs-Deltas-Deltas/')
        self.export_paths['Deltas'] = os.path.join(self.preprocessed_data_path, 'Deltas/')
        self.export_paths['Deltas-Deltas'] = os.path.join(self.preprocessed_data_path, 'Deltas-Deltas/')
        self.export_paths['Pitch-Log-Quad'] = os.path.join(self.preprocessed_data_path, f'Pitch-Log-Quad_threshold_{self.silence_threshold}/')
        self.export_paths['Pitch-Raw'] = os.path.join(self.preprocessed_data_path, f'Pitch-Raw_threshold_{self.silence_threshold}/')
        self.export_paths['Pitch-Log-Raw'] = os.path.join(self.preprocessed_data_path, f'Pitch-Log-Raw_threshold_{self.silence_threshold}/')
        self.export_paths['Pitch-Manual'] = os.path.join(self.preprocessed_data_path, f'Pitch-Manual_threshold_{self.silence_threshold}/')
        self.export_paths['Pitch-Log-Manual'] = os.path.join(self.preprocessed_data_path, f'Pitch-Log-Manual_threshold_{self.silence_threshold}/')
        self.export_paths['Pitch-Phonemes'] = os.path.join(self.preprocessed_data_path, f'Pitch-Phonemes_threshold_{self.silence_threshold}/')
        self.export_paths['Pitch-Log-Phonemes'] = os.path.join(self.preprocessed_data_path, f'Pitch-Log-Phonemes_threshold_{self.silence_threshold}/')
        self.export_paths['Spectrogram'] = os.path.join(self.preprocessed_data_path, 'Spectrogram/')
        self.export_paths['Phonemes-Envelope'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Envelope/')
        self.export_paths['Phonemes-Envelope-Manual'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Envelope-Manual/')
        self.export_paths['Phonemes-Discrete'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Discrete/')
        self.export_paths['Phonemes-Discrete-Manual'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Discrete-Manual/')
        self.export_paths['Phonemes-Onset'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Onset/')
        self.export_paths['Phonemes-Onset-Manual'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Onset-Manual/')
        self.export_paths['Phonemes-Envelope-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Envelope-Phonet/')
        self.export_paths['Phonemes-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Phonet/')
        self.export_paths['Phonemes-Discrete-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Discrete-Phonet/')
        self.export_paths['Phonemes-Onset-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phonemes-Onset-Phonet/')
        self.export_paths['Phonological'] = os.path.join(self.preprocessed_data_path, 'Phonological/')
        self.export_paths['Mistakes-Separated'] = os.path.join(self.preprocessed_data_path, 'Mistakes-Separated/')
        self.export_paths['Mistakes-Together'] = os.path.join(self.preprocessed_data_path, 'Mistakes-Together/')
        self.export_paths['Control-Separated'] = os.path.join(self.preprocessed_data_path, 'Control-Separated/')
        self.export_paths['Control-Together'] = os.path.join(self.preprocessed_data_path, 'Control-Together/')
        self.export_paths['Wav2vec2'] = os.path.join(self.preprocessed_data_path, 'Wav2vec2/')
        self.export_paths['Phones-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phones-Phonet/')
        self.export_paths['Phones-Envelope-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phones-Envelope-Phonet/')
        self.export_paths['Phones-Discrete-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phones-Discrete-Phonet/')
        self.export_paths['Phones-Onset-Phonet'] = os.path.join(self.preprocessed_data_path, 'Phones-Onset-Phonet/')
        
    def load_from_raw(
        self
        )->dict:
        """
        Loads raw data, this includes EEG, info and stimuli.

        Returns
        -------
        dict
            Sessions of both subjects
        """
        
        # Subjects dictionaries, stores their data
        sujeto_1 = {}
        sujeto_2 = {}

        # Retrive number of files, i.e: trials. This is done this way because there are missing phonemes values
        trials = [int(fname.split('.')[2]) for fname in os.listdir(self.phn_path) if fname.endswith('TextGrid')]
        trials = list(set([tr for tr in trials if trials.count(tr) > 1]))

        # Try to open preprocessed info of samples, if not crates raw. This dictionary contains data of trial lengths and indexes to keep up to given trial
        try:
            self.samples_info = funciones.load_pickle(path=os.path.join(self.samples_info_path, f'samples_info_{self.sesion}.pkl'))
            loaded_samples_info = True
        except:
            loaded_samples_info = False
            self.samples_info = {
                                'trial_lengths1': [0],
                                'trial_lengths2': [0],
                                'keep_indexes1':[],
                                'keep_indexes2':[]
                                }

        # Retrive and concatenate data of all trials
        for p, trial in enumerate(trials):

            # Update on number of trials
            Sesion_class.print_trials(p, trial, trials)

            # Create trial for both channels in order to extract features and EEG signal
            try:
                channel_1 = Trial_channel(
                        s=self.sesion, 
                        trial=trial, 
                        channel=1,
                        band=self.band, 
                        sr=self.sr,
                        causal_filter_eeg=self.causal_filter_eeg,
                        envelope_filter=self.envelope_filter,
                        silence_threshold=self.silence_threshold,
                        praat_executable_path=self.praat_executable_path,
                        situation=self.situation,
                        )
                channel_2 = Trial_channel(
                        s=self.sesion,
                        trial=trial,
                        channel=2,
                        band=self.band,
                        sr=self.sr,
                        causal_filter_eeg=self.causal_filter_eeg,
                        envelope_filter=self.envelope_filter,
                        silence_threshold=self.silence_threshold,
                        praat_executable_path=self.praat_executable_path,
                        situation=self.situation,
                        )

                # Extract dictionaries with the data
                trial_channel_1 = channel_1.load_trial(stims=self.stim.split('_'))
                trial_channel_2 = channel_2.load_trial(stims=self.stim.split('_'))
    
                # Load data to dictionary taking own stimuli and eeg signal. I.e: each subject predicts its own EEG with its own stimuli
                if self.situation.startswith('Internal'):
                    trial_sujeto_1 = {key: trial_channel_1[key] for key in trial_channel_1.keys()}
                    trial_sujeto_2 = {key: trial_channel_2[key] for key in trial_channel_2.keys()}
                # TODO para predecir en 'Internal_BS' no habría que armar los estímulos con alguna especie de suma entre las señales de ambos participantes?                
                
                # Load data to dictionary taking own eeg signal and interlocutors stimuli. I.e: predicts own EEG using stimuli from interlocutor
                else:
                    trial_sujeto_1 = {key: trial_channel_2[key] for key in trial_channel_2.keys() if key!='EEG'} 
                    trial_sujeto_2 = {key: trial_channel_1[key] for key in trial_channel_1.keys() if key!='EEG'}
                    trial_sujeto_1['EEG'], trial_sujeto_2['EEG'] = trial_channel_1['EEG'], trial_channel_2['EEG']

                # Labeling of current speaker. {3:both_speaking,2:speaks_locutor,1:speaks_interlocutor,0:silence}. La diferencia entre _1 y _2 ese que se permutan los valores 1 y 2 (cambia la perspectiva de quién es locutor e interlocutor)
                current_speaker_1 = self.labeling(trial=trial, channel=2) # len matching eeg
                current_speaker_2 = self.labeling(trial=trial, channel=1)

                # Match length of speaker labels and trials with the info of its lengths
                trial_sujeto_1, current_speaker_1, minimum1 = self.match_lengths(dic=trial_sujeto_1, speaker_labels=current_speaker_1)
                trial_sujeto_2, current_speaker_2, minimum2 = self.match_lengths(dic=trial_sujeto_2, speaker_labels=current_speaker_2)

                # Define/Re-define samples_info trial length
                if not loaded_samples_info:
                    self.samples_info['trial_lengths1'].append(minimum1)
                    self.samples_info['trial_lengths2'].append(minimum2)

                    # Preprocessing: calaculates the relevant indexes for the apropiate analysis. Add sum of all previous trials length. This is because at the end, all trials previous to the actual will be concatenated
                    self.samples_info['keep_indexes1'] += (self.shifted_indexes_to_keep(speaker_labels=current_speaker_1) + np.sum(self.samples_info['trial_lengths1'][:-1])).tolist()
                    self.samples_info['keep_indexes2'] += (self.shifted_indexes_to_keep(speaker_labels=current_speaker_2) + np.sum(self.samples_info['trial_lengths2'][:-1])).tolist()
                
                # Concatenates data of each subject 
                for key in trial_sujeto_1:
                    if key != 'info':
                        if key not in sujeto_1:
                            sujeto_1[key] = trial_sujeto_1[key]
                        else:
                            sujeto_1[key] = np.concatenate((sujeto_1[key], trial_sujeto_1[key]), axis=0)
                for key in trial_sujeto_2:
                    if key != 'info':
                        if key not in sujeto_2:
                            sujeto_2[key] = trial_sujeto_2[key]
                        else:
                            sujeto_2[key] = np.concatenate((sujeto_2[key], trial_sujeto_2[key]), axis=0)

            # Empty trial
            except:
                print(f"Trial {trial} of session {self.sesion} couldn't be loaded.")
                self.samples_info['trial_lengths1'][p] = 0
                self.samples_info['trial_lengths2'][p] = 0

        # Get info of the setup that was exluded in the previous iteration
        info = trial_channel_1['info']

        # Saves modified relevant indexes 
        os.makedirs(self.samples_info_path, exist_ok=True)
        funciones.dump_pickle(path=os.path.join(self.samples_info_path, f'samples_info_{self.sesion}.pkl'), obj=self.samples_info, rewrite=True)

        # Save results
        for key in sujeto_1:
            # # Drops silences phoneme column
            # if key.startswith('Phonemes'):
            #     # Remove silence column, the last one by construction
            #     sujeto_1[key] = np.delete(arr=sujeto_1[key], obj=-1, axis=1)
            #     sujeto_2[key] = np.delete(arr=sujeto_2[key], obj=-1, axis=1)

            # Save preprocesed data
            os.makedirs(self.export_paths[key], exist_ok=True)
            funciones.dump_pickle(path=os.path.join(self.export_paths[key], f'Sesion{self.sesion}.pkl'), obj=[sujeto_1[key], sujeto_2[key]], rewrite=True)

        # Saves info of the setup                    
        funciones.dump_pickle(path=os.path.join(self.preprocessed_data_path, 'EEG/info.pkl'), obj=info, rewrite=True)

        # Redefine subjects dictionaries to return only used stimuli
        sujeto_1_return = {key: sujeto_1[key] for key in self.stim.split('_') + ['EEG']}
        sujeto_2_return = {key: sujeto_2[key] for key in self.stim.split('_') + ['EEG']}
        sujeto_1_return['info'] = info
        sujeto_2_return['info'] = info

        return {'Sujeto_1': sujeto_1_return, 'Sujeto_2': sujeto_2_return}, self.samples_info
    
    def load_procesed(
        self
        )->dict:
        """
        Loads procesed data, this includes EEG, info and stimuli.

        Returns
        -------
        dict
            Sessions of both subjects.
        """
        # Load EEGs and procesed data
        eeg_sujeto_1, eeg_sujeto_2 = funciones.load_pickle(path=os.path.join(self.export_paths['EEG'], f'Sesion{self.sesion}.pkl'))
        info = funciones.load_pickle(path=os.path.join(self.preprocessed_data_path, f'EEG/info.pkl'))
        samples_info = funciones.load_pickle(path=os.path.join(self.samples_info_path, f'samples_info_{self.sesion}.pkl'))
        sujeto_1 = {'EEG': eeg_sujeto_1, 'info': info}
        sujeto_2 = {'EEG': eeg_sujeto_2, 'info': info}
        
        # Loads stimuli to each subject
        for stimulus in self.stim.split('_'):
            sujeto_1[stimulus], sujeto_2[stimulus] = funciones.load_pickle(path=os.path.join(self.export_paths[stimulus], f'Sesion{self.sesion}.pkl'))
        return {'Sujeto_1': sujeto_1, 'Sujeto_2': sujeto_2}, samples_info
    
    def labeling(
        self, 
        trial:int, 
        channel:int
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
        ubi_speaker = os.path.join(self.phrases_path, f's{self.sesion}.objects.{trial:02d}.channel{channel}.phrases')
        
        h1t = pd.read_table(ubi_speaker, header=None, sep="\t")

        # Replace and '#' by ''. And then all text by 1 and silences by 0 
        h1t.iloc[:, 2] = (h1t.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
        
        # Take difference in time and multiply it by sample rate in order to match envelope length (almost, miss by a sample or two)
        samples = np.round((h1t[1] - h1t[0]) * self.sr).astype("int")
        speaker = np.repeat(h1t.iloc[:, 2], samples).ravel()
        
        # Same with listener
        listener_channel = (channel - 3) * -1
        ubi_listener = os.path.join(self.phrases_path, f's{self.sesion}.objects.{trial:02d}.channel{listener_channel}.phrases')
        h2t = pd.read_table(ubi_listener, header=None, sep="\t")

        # Replace and '#' by ''. And then all text by 1 and silences by 0
        h2t.iloc[:, 2] = (h2t.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
        samples = np.round((h2t[1] - h2t[0]) * self.sr).astype("int")
        listener = np.repeat(h2t.iloc[:, 2], samples).ravel()

        # If there are differences in length, corrects them with 0-padding
        diff = len(speaker) - len(listener)
        if diff > 0:
            listener = np.concatenate([listener, np.repeat(0, diff)])
        elif diff < 0:
            speaker = np.concatenate([speaker, np.repeat(0, np.abs(diff))])

        # Return an array with envelope length, having values 3 if both participants are speaking; 2, if just locutor; 1, interlocutor and 0, silence
        return speaker + listener * 2
    
    def shifted_indexes_to_keep(
        self,
        speaker_labels:np.ndarray
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
        if 'All_Times' in self.situation:
            return np.arange(len(speaker_labels))
        
        # Change 0 with 4s, because shifted matrix pad zeros that could be mistaken with situation 0    
        speaker_labels = np.array(speaker_labels)
        speaker_labels = np.where(speaker_labels==0, 4, speaker_labels)        

        # Computes shifted matrix
        shifted_matrix_speaker_labels = processing.shifted_matrix_2(features=speaker_labels, delays=self.delays, use_gpu=config.use_gpu).astype(float)

        # Make the appropiate label
        if self.situation.endswith('BS'):
            situation_label = 3
        elif self.situation.startswith('External'):
            situation_label = 1
        elif self.situation.startswith('Internal'):
            situation_label = 2
        else: # Silence
            situation_label = 4

        # Shifted matrix index where the given situation is ocurring in all row (number of samples dimension) # TODO: discutir si dejar 0 (silencios) o no.
        # return ((shifted_matrix_speaker_labels==situation_label) | (shifted_matrix_speaker_labels==0)).all(axis=1).nonzero()[0]
        return ((shifted_matrix_speaker_labels==situation_label)).all(axis=1).nonzero()[0]
    
    @staticmethod
    def print_trials(
        p:int,
        trial:int,
        trials:list
        )->None:
        """
        Make print for trial update

        Parameters
        ----------
        p : int
            index of given trial inside trials
        trial : int
            given trial
        trials : list
            list of trials
        """
        if (trials[p-1]+1!=trial) and p!=0:
            missing_trials = []
            t = trial
            while trials[p-1]+1!=t:
                missing_trials.append(t-1)
                t-=1
            missing_trials.sort()
            if len(missing_trials)>1:
                print(f'Trial {trial} of {trials[-1]}. Missing trials {", ".join(str(i) for i in missing_trials)}.')
            else:
                print(f'Trial {trial} of {trials[-1]}. Missing trial {", ".join(str(i) for i in missing_trials)}.')
        elif (p==0) and (trials[0]!=1):
            print(f'Trial {trial} of {trials[-1]}. Missing trial 1.')
        else:
            print(f'Trial {trial} of {trials[-1]}.')

    def match_lengths(
        self, 
        dic:dict, 
        speaker_labels:np.ndarray
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
        # Get minimum between EEG and envelope to make cutoff
        minimum = min([dic['Envelope'].shape[0]] +[dic['EEG'].shape[0]]+ [len(speaker_labels)])
        
        # Correct length 
        for key in dic:
            if key != 'info':
                data = dic[key]
                if data.shape[0] > minimum:
                    dic[key] = data[:minimum]

        if len(speaker_labels) > minimum:
            speaker_labels = speaker_labels[:minimum]
        return dic, speaker_labels, minimum
    
def load_data(
    sesion:int, 
    stim:str, 
    band:str,
    sr:float,
    preprocessed_data_path:str, 
    praat_executable_path:str,
    situation:str='External', 
    causal_filter_eeg:bool=True, 
    envelope_filter:bool=False, 
    silence_threshold:float=0.03, 
    delays:np.ndarray=None
    )->tuple:
    """
    Loads and processes EEG and stimuli data for a given session.

    Parameters
    ----------
    sesion : int
        Session number.
    stim : str
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'.
        Allowed stimuli are: ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 
        'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 
        'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 
        'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 
        'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet', 'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 
        'Control-Together', 'Control-Separated', 'Wav2vec2','Phones-Onset-Manual', 'Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet']
    band : str
        Neural frequency band. It could be one of: ['Delta','Theta','Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta'].
    sr : float
        Sample rate in Hz of the EEG.
    preprocessed_data_path : str
        Path directing to processed data.
    praat_executable_path : str
        Path directing to Praat executable.
    situation : str, optional
        Situation considered when performing the analysis, by default 'External'. Allowed situations are: 
        ['Internal','Internal_BS','External', 'External_BS', 'Internal_All_Times', 'External_All_Times'].
    causal_filter_eeg : bool, optional
        Whether to use or not a causal filter for the EEG, by default True.
    envelope_filter : bool, optional
        Whether to use or not an envelope filter, by default False.
    silence_threshold : float, optional
        Silence threshold of the dialogue, by default 0.03.
    delays : np.ndarray, optional
        Delay array to construct shifted matrix, by default None.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: Sessions of both subjects.
        - dict: Information about the samples.

    Raises
    ------
    SyntaxError
        If 'stim' is not an allowed stimulus. Allowed ones are:
        ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 
        'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 
        'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 
        'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 
        'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet', 'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 
        'Control-Together', 'Control-Separated', 'Wav2vec2','Phones-Onset-Manual', 'Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet']
        If 'band' is not an allowed band frequency. Allowed ones are:
        ['Delta','Theta','Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']
        If 'situation' is not an allowed situation. Allowed ones are:
        ['Internal','Internal_BS','External', 'External_BS', 'External_All_Times', 'Internal_All_Times']
    """
    # Define allowed stimuli
    allowed_stims = ['Envelope', 'Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Deltas', 'Deltas-Deltas', 'Pitch-Log-Quad', 'Pitch-Raw', 'Pitch-Manual', 'Pitch-Phonemes',\
                    'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset',\
                    'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Phonet', 'Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet', 'Phonological', 'Mistakes-Separated', 'Mistakes-Together', 'Control-Together', 'Control-Separated', 'Wav2vec2','Phones-Onset-Manual', 'Phones-Phonet', 'Phones-Envelope-Phonet', 'Phones-Discrete-Phonet']
    allowed_situations = ['Internal','Internal_BS','External', 'External_BS', 'Internal_All_Times', 'External_All_Times']
    allowed_bands = ['Delta','Theta','Alpha','Beta1','Beta2','All','Delta_Theta','Alpha_Delta_Theta']

    # And conditions
    condition_1 = all(stimulus in allowed_stims for stimulus in stim.split('_'))
    condition_2 = band in allowed_bands
    condition_3 = situation in allowed_situations

    if condition_1:
        if condition_2:
            if condition_3:
                
                # Re-order stim and band to create just one file for each case: 'Phonemes_Envelope' --> 'Envelope_Phonemes'
                ordered_stims = sorted(stim.split('_'))
                ordered_band = sorted(band.split('_'))
                sesion_obj = Sesion_class(sesion=sesion, 
                                        stim='_'.join(ordered_stims), 
                                        band='_'.join(ordered_band), 
                                        sr=sr,
                                        causal_filter_eeg=causal_filter_eeg,
                                        envelope_filter=envelope_filter, 
                                        situation=situation,
                                        silence_threshold=silence_threshold, 
                                        preprocessed_data_path=preprocessed_data_path, 
                                        praat_executable_path=praat_executable_path,
                                        delays=delays)

                # Try to load procesed data, if it fails it loads raw data
                try:
                    print('Loading preprocesed data\n')
                    Sesion, samples_info = sesion_obj.load_procesed()
                    print('Data loaded succesfully\n')
                except:
                    print("Couldn't load data, compute it from raw\n")
                    Sesion, samples_info = sesion_obj.load_from_raw()
                return Sesion['Sujeto_1'], Sesion['Sujeto_2'], samples_info
            else:
                raise SyntaxError(f"{situation} is not an allowed situation. Allowed ones are: {allowed_situations}")
        else:
            raise SyntaxError(f"{band} is not an allowed band frequency. Allowed bands are: {allowed_bands}")
    else:
        raise SyntaxError(f"{stim} is not an allowed stimulus. Allowed stimuli are: {allowed_stims}. If more than one stimulus is wanted, the separator should be '_'.")