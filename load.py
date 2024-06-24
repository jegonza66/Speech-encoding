# Standard libraries
import numpy as np, pandas as pd, os, warnings
import mne, librosa, platform, opensmile, textgrids

# Specific libraries
import scipy.io.wavfile as wavfile#, parselmouth
from scipy import signal as sgn
from praatio import pitch_and_intensity

# Modules
import processing, funciones, setup

# Review this If we want to update packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(verbose='CRITICAL')
exp_info = setup.exp_info()

class Trial_channel:
    def __init__(self, s:int=21, trial:int=1, channel:int=1, band:str='All', sr:float=128, 
                 causal_filter_eeg:bool=True, envelope_filter:bool=False, silence_threshold:float=0.03,
                 praat_executable_path:str= r"C:\Program Files\Praat\Praat.exe"):
        """Extract transcriptions, audio signal and EEG signal of given session and channel to calculate specific features.

        Parameters
        ----------
        s : int, optional
            Session number, by default 21
        trial : int, optional
            Number of trial, by default 1
        channel : int, optional
            Channel used to record the audio (it can be from subject 1 and 2), by default 1
        band : str, optional
            Neural frequency band, by default 'All'. It could be one of:
            ['Delta','Theta',Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
        sr : float, optional
            Sample rate in Hz of the EEG, by default 128
        delays : np.ndarray, optional
            Delay array to construct shifted matrix, by default np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
        causal_filter_eeg : bool, optional
            Whether to use or not a cusal filter, by default True
        envelope_filter : bool, optional
            Whether to use or not an envelope filter, by default False
        silence_threshold : float, optional
            Silence threshold of the dialogue, by default 0.03
        praat_executable_path : str
            Path directing to Praat executable
            

        Raises
        ------
        SyntaxError
            If 'band' is not an allowed band frecuency. Allowed bands are:
            ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
        """
        # Participants sex, ordered by session
        sex_list = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']
        allowed_band_frequencies = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
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
        
        # To be filled with loaded data
        self.eeg = None

        # Relevant paths
        self.praat_executable_path = praat_executable_path
        self.eeg_fname = f"Datos/EEG/S{s}/s{s}-{channel}-Trial{trial}-Deci-Filter-Trim-ICA-Pruned.set"
        self.wav_fname = f"Datos/wavs/S{s}/s{s}.objects.{trial:02d}.channel{channel}.wav"
        self.pitch_fname = f"Datos/pitch_threshold_{silence_threshold}/S{s}/s{s}.objects.{trial:02d}.channel{channel}.txt"
        self.phn_fname = f"Datos/phonemes/S{s}/s{s}.objects.{trial:02d}.channel{channel}.aligned_fa.TextGrid"
        self.phn_fname_manual = f"Datos/phonemes/S{s}/manual/s{s}_objects_{trial:02d}_channel{channel}_aligned_faTAMARA.TextGrid"
        self.phrases_fname = f"Datos/phrases/S{s}/s{s}.objects.{trial:02d}.channel{channel}.phrases"
        
    def f_eeg(self):
        """Extract eeg file downsample it to get the same rate as self.sr and stores its data inside the class instance.

        Returns
        -------
        np.ndarray
            Matrix representation of EEG per channel
        """
        # Read the .set file. warning of annotations and 'boundry' events -data discontinuities-.
        eeg = mne.io.read_raw_eeglab(input_fname=self.eeg_fname) 
        eeg.load_data()
        
        # Apply a lowpass filter
        if self.band:
            if self.causal_filter_eeg:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, phase='minimum')
            else:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg)
        
        # # Store dimension mne.raw
        # eeg.resample(sfreq=self.sr)

        # # Return mne representation Times x nchannels
        # self.eeg = eeg.copy()
        # return self.eeg.get_data().T*1e6
        
        # Get mne representation Times x nchannels
        self.eeg = eeg.copy()
        eeg = self.eeg.get_data().T*1e6  # paso a array y tiro la primer columna de tiempo

        # Downsample
        eeg = processing.subsamplear(eeg, int(self.eeg.info.get("sfreq")/ self.sr))
        return eeg

    def f_info(self):
        """A montage is define as a descriptor for the set up: EEG channel names and relative positions of sensors on the scalp. 
        A montage can also contain locations for HPI points, fiducial points, or extra head shape points.In this case, 'BioSemi
        cap with 128 electrodes (128+3 locations)'.

        Returns
        -------
        mne.io.meas_info.Info
            Montage of the measurment.
        """
        # Define montage and info object
        montage = mne.channels.make_standard_montage('biosemi128')
        channel_names = montage.ch_names
        return mne.create_info(ch_names=channel_names[:], sfreq=self.sr, ch_types='eeg').set_montage(montage)

    def f_envelope(self): 
        """Takes the low pass filtered -butterworth-, downsample and smoothened envelope of .wav file. Then matches its length to the EEG.

        Returns
        -------
        np.ndarray
            Envelope of wav signal with desire dimensions
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")

        # Calculate envelope
        envelope = np.abs(sgn.hilbert(wav))
        
        # Apply lowpass butterworth filter
        if self.envelope_filter == 'Causal':# TODO can it be replaced for a mne filter?
            envelope = processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='Causal').reshape(-1,1)
        elif self.envelope_filter == 'NonCausal':
            envelope = processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='NonCausal').reshape(-1,1)
        
        # Resample
        window_size, stride = int(self.audio_sr/self.sr), int(self.audio_sr/self.sr)
        envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
        return envelope.reshape(-1, 1)
        # else:
            # envelope = envelope.reshape(-1,1)
            
        # # Creates mne raw array
        # info_envelope = mne.create_info(ch_names=['Envelope'], sfreq=self.audio_sr, ch_types='misc')
        # envelope_mne_array = mne.io.RawArray(data=envelope.T, info=info_envelope)

        # # Resample to match EEG data
        # envelope_mne_array.resample(sfreq=self.sr)
        # return envelope_mne_array.get_data().T

    def f_spectrogram(self):
        """Calculates spectrogram of .wav file between 16 Mel frequencies.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

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
        S = librosa.feature.melspectrogram(y=wav,
                                           sr=self.audio_sr, 
                                           n_fft=sample_window, 
                                           hop_length=sample_window, 
                                           n_mels=16)
        # Transform to dB using normalization to 1
        S_DB = librosa.power_to_db(S=S, ref=np.max)
        
        return S_DB.T
    
    def f_jitter_shimmer(self, envelope:np.ndarray): # NEVER USED
        """Gives the jitter and shimmer matching the size of the envelope

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
        jitter = processing.subsamplear(jitter, mcm / len(envelope))
        shimmer = processing.subsamplear(shimmer, mcm / len(envelope))

        # Reassurance that the count is correct
        jitter = jitter[:min(len(jitter), len(envelope))].reshape(-1,1)
        shimmer = shimmer[:min(len(shimmer), len(envelope))].reshape(-1,1)
        return jitter, shimmer
    
    def f_phonemes(self, envelope:np.ndarray, kind:str='Phonemes-Envelope-Manual'):
        """It makes a time-match matrix between the phonemes and the envelope. The values and shape of given matrix depend on kind.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.
        kind : str, optional
           Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
            'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete', 'Phonemes-Discrete-Manual', 'Phonemes-Onset', 'Phonemes-Onset-Manual'

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
            Whether the input value of 'kind' is passed correctly. It must be a string among ['Envelope', 'Discrete', 'Onset'].
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
            # samples.append(np.round((ph.xmax - ph.xmin) * self.sr).astype("int"))
            samples.append(np.round((ph.xmax - ph.xmin) * 128).astype("int"))


        # Extend on more phoneme of silence till end of trial 
        labels.append("")
        times.append((ph.xmin, trial_tmax))
        # samples.append(np.round((trial_tmax - ph.xmax) * self.sr).astype("int"))
        samples.append(np.round((trial_tmax - ph.xmax) *128).astype("int"))


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
      
    def f_pitch(self, envelope:np.ndarray, kind:str): 
        """Loads the pitch of the speaker, after calculating it from .wav file and stores it.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        np.ndarray
            Array containing pitch of the audio signal
        """
        # Makes path for storing data
        if platform.system() == 'Linux':
            output_folder = f'Datos/{kind}_threshold_{self.silence_threshold}'
        else:
            output_folder = f"Datos/{kind}_threshold_{self.silence_threshold}"
        
        # Create paths and distinguish subject
        os.makedirs(output_folder, exist_ok=True)
        if self.sex == 'M':
            minPitch = 50
            maxPitch = 300
        elif self.sex == 'F':
            minPitch = 75
            maxPitch = 500
        
        # Define sample step and calculate pitch
        self.sampleStep = 1/self.sr # .01
        with funciones.Suppress_print():
            pitch_and_intensity.extractPI(inputFN=os.path.abspath(self.wav_fname), 
                                        outputFN=os.path.abspath(self.pitch_fname), 
                                        praatEXE=self.praat_executable_path, 
                                        minPitch=minPitch,
                                        maxPitch=maxPitch, 
                                        sampleStep=self.sampleStep, 
                                        silenceThreshold=self.silence_threshold)
        # silence_threshold = .03
        # minPitch = 50
        # maxPitch = 300
        # output_folder = f"Datos/pitch_threshold_{silence_threshold}"
        # sampleStep = 1/128
        # praat_executable_path = 'C:/Program Files/Praat/Praat.exe'
        # pitch_fname = f"Datos/pitch_threshold_{.03}/S{21}/s{21}.objects.{1:02d}.channel{1}.txt"
        # wav_fname = r"C:\repos\Speech-encoding\repo_speech_encoding\Datos\wavs\S21\s21.objects.01.channel1.wav"
        # pi = pitch_and_intensity.extractPI(inputFN=os.path.abspath(wav_fname), 
        #                                 outputFN=os.path.abspath(pitch_fname), 
        #                                 praatEXE=praat_executable_path, 
        #                                 minPitch=minPitch,
        #                                 maxPitch=maxPitch, 
        #                                 sampleStep=sampleStep, 
        #                                 silenceThreshold=silence_threshold)
        
        # Loads data
        data = np.genfromtxt(os.path.abspath(self.pitch_fname), dtype=np.float, delimiter=',', missing_values='--undefined--', filling_values=np.inf)
        time, pitch = data[:, 0], data[:, 1]

        # Get defined indexes
        defined_indexes = np.where(pitch!=np.inf)[0]
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # aux_p = np.log(pitch)
        # aux_p[aux_p==np.inf] = 0
        # plt.scatter(time, aux_p, s=5, label='data')
        
        # Approximated window size
        window_size = 300e-3
        n_steps_in_window = np.ceil(window_size/self.sampleStep)
        window_size = n_steps_in_window*self.sampleStep

        if kind.endswith('Manual'):
            # Interpolate relevant moments of silence
            # lista_aux = []
            for i in range(len(defined_indexes)):
                if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
                    pitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(x=time[defined_indexes[i-1]+1:defined_indexes[i]], xp=time[defined_indexes], fp=pitch[defined_indexes])
                    # lista_aux+=list(np.arange(defined_indexes[i-1]+1,defined_indexes[i],1))
        else:
            # Load phonemes matrix, excluding '' label
            phonemes = self.f_phonemes(envelope=envelope, kind='Phonemes-Discrete-Manual')
            phonemes = np.delete(arr=phonemes, obj=-1, axis=1)

            # Given that spacing between samples is 1/self.sr
            sound_indexes = np.where(phonemes.any(axis=1))[0]

            # Within window of window_size of said phoneme if there is a silence it gets interpoled
            # lista_aux = []
            for i in range(len(sound_indexes)):
                if 1<(sound_indexes[i]-sound_indexes[i-1])<=n_steps_in_window:
                    pitch[sound_indexes[i-1]+1:sound_indexes[i]] = np.interp(x=time[sound_indexes[i-1]+1:sound_indexes[i]], xp=time[sound_indexes], fp=pitch[sound_indexes])
                    # lista_aux+=list(np.arange(sound_indexes[i-1]+1,sound_indexes[i],1))

        # Log transformation
        logpitch = np.log(pitch).reshape(-1, 1)
        
        # Set left values to zero
        logpitch[logpitch==np.inf]=0
        # plt.scatter(time[lista_aux], logpitch[lista_aux].reshape(-1), s=5, label ='interpoled')
        # plt.xlabel('Time (s)')
        # plt.ylabel('logpitch')
        # plt.legend()
        # plt.grid(visible=True)
        # plt.show(block=True)
        return logpitch


        # import matplotlib.pyplot as plt
        # plt.figure()
        # # plt.scatter(time[defined_indexes], pitch[defined_indexes], s=3)
        # # plt.scatter(time[undefined_indexes], interpoled_data, s=3)
        # plt.scatter(time, pitch, s=3)
        # for i, m in enumerate(moments_of_silence):
        #     plt.scatter(time[m[0]:m[1]], interpolado[i])
        # # plt.scatter(time, logpitch, s=3)
        # # plt.yticks([])
        # plt.show()
        
        # # Match length with envelope
        # # Repeat each value of the pitch to make audio_sr*sampleStep times (in particular, a multiple of audio_sr)
        # pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=np.float32)
        
        # pitch = processing.subsamplear(pitch, int(self.audio_sr/self.sr))
        # pitch = pitch[:min(len(pitch), len(envelope))].reshape(-1,1)
        # return pitch

    def load_trial(self, stims:list): 
        """Extract EEG and calculates specified stimuli.
        Parameters
        ----------
        stims : list
            A list containing possible stimuli. Possible input values are: 
            ['Envelope', 'Pitch', 'Pitch-Manual', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Discrete', 'Phonemes-Onset', 
            'Phonemes-Envelope-Manual', 'Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual']

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
            if stim.startswith('Pitch'):
                channel[stim] = self.f_pitch(envelope=channel['Envelope'], kind=stim)
            if stim=='Spectrogram':
                channel['Spectrogram'] = self.f_spectrogram()
            elif stim.startswith('Phonemes'):
                channel[stim] = self.f_phonemes(envelope=channel['Envelope'], kind=stim)
        return channel

class Sesion_class: 
    def __init__(self, sesion:int=21, stim:str='Envelope', band:str='All', sr:float=128, 
                 causal_filter_eeg:bool=True, envelope_filter:bool=False, situation:str='Escucha', 
                 silence_threshold:float=0.03, delays:np.ndarray=None,
                 procesed_data_path:str=f'saves/Preprocesed_Data/tmin{-0.6}_tmax{-.002}/',
                 praat_executable_path:str=r"C:\Program Files\Praat\Praat.exe"
                 ):
        """Construct an object for the given session containing all concerning data.

        Parameters
        ----------
        sesion : int, optional
            Session number, by default 21
        stim : str, optional
            Stimuli to use in the analysis, by default 'Envelope'. If more than one stimulus is wanted, the separator should be '_'. Allowed stimuli are:
            ['Envelope', 'Pitch', 'Pitch-Manual', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete','Phonemes-Discrete-Manual', 'Phonemes-Onset','Phonemes-Onset-Manual']
        band : str, optional
            Neural frequency band, by default 'All'. It could be one of:
            ['Delta','Theta',Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
        sr : float, optional
            Sample rate in Hz of the EEG, by default 128
        causal_filter_eeg : bool, optional
            Whether to use or not a cusal filter, by default True
        envelope_filter : bool, optional
            Whether to use or not an envelope filter, by default False
        situation : str, optional
            Situation considerer when performing the analysis, by default 'Escucha'. Allowed sitations are:
            ['Habla_Propia','Ambos_Habla','Escucha']
        silence_threshold : float, optional
            Silence threshold of the dialogue, by default 0.03
        delays : np.ndarray, optional
            Delay array to construct shifted matrix, by default np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
        procesed_data_path : str, optional
            Path directing to procesed data, by default f'saves/Preprocesed_Data/tmin{-0.6}_tmax{-0.002}/'
        praat_executable_path : str
            Path directing to Praat executable

        Raises
        ------
        SyntaxError
            If 'stim' is not an allowed stimulus. Allowed stimuli are:
            ['Envelope', 'Pitch', 'Pitch-Manual', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete','Phonemes-Discrete-Manual', 'Phonemes-Onset','Phonemes-Onset-Manual']. 
            If more than one stimulus is wanted, the separator should be '_'.
        SyntaxError
            If 'band' is not an allowed band frecuency. Allowed bands are:
            ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
        SyntaxError
            If situation is not an allowed situation. Allowed situations are: 
            ['Habla_Propia','Ambos_Habla','Escucha']
        """
       
        # Check if band, stim and situation parameters where passed with the right syntax
        allowed_stims = ['Envelope', 'Pitch', 'Pitch-Manual', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete','Phonemes-Discrete-Manual', 'Phonemes-Onset','Phonemes-Onset-Manual']
        allowed_band_frequencies = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
        allowed_situationes = ['Habla_Propia','Ambos_Habla','Escucha']
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
        if situation in allowed_situationes:
            self.situation = situation
        else:
            raise SyntaxError(f"{situation} is not an allowed situation. Allowed situations are: {allowed_situationes}")
        
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
        self.procesed_data_path = procesed_data_path
        self.samples_info_path = self.procesed_data_path + f'samples_info/Sit_{self.situation}/'
        self.phn_path = f"Datos/phonemes/S{self.sesion}/"
        self.phrases_path = f"Datos/phrases/S{self.sesion}/"

        # Define paths to export data
        self.export_paths = {}
        if self.causal_filter_eeg:
            self.export_paths['EEG'] = self.procesed_data_path + f'EEG/Causal_Sit_{self.situation}_band_{self.band}/'
        else:
            self.export_paths['EEG'] = self.procesed_data_path + f'EEG/Sit_{self.situation}_band_{self.band}/'
        if self.envelope_filter:
            self.export_paths['Envelope'] = self.procesed_data_path + f'Envelope/{self.envelope_filter}_Sit_{self.situation}/'
        else:
            self.export_paths['Envelope'] = self.procesed_data_path + f'Envelope/Sit_{self.situation}/'
        self.export_paths['Pitch'] = self.procesed_data_path + f'Pitch_threshold_{self.silence_threshold}/Sit_{self.situation}_Faltantes/'
        self.export_paths['Pitch-Manual'] = self.procesed_data_path + f'Pitch-Manual_threshold_{self.silence_threshold}/Sit_{self.situation}_Faltantes/'
        self.export_paths['Spectrogram'] = self.procesed_data_path + f'Spectrogram/Sit_{self.situation}/'
        self.export_paths['Phonemes-Envelope'] = self.procesed_data_path + f'Phonemes-Envelope/Sit_{self.situation}/'
        self.export_paths['Phonemes-Envelope-Manual'] = self.procesed_data_path + f'Phonemes-Manual/Sit_{self.situation}/'
        self.export_paths['Phonemes-Discrete'] = self.procesed_data_path + f'Phonemes-Discrete/Sit_{self.situation}/'
        self.export_paths['Phonemes-Discrete-Manual'] = self.procesed_data_path + f'Phonemes-Discrete-Manual/Sit_{self.situation}/'
        self.export_paths['Phonemes-Onset'] = self.procesed_data_path + f'Phonemes-Onset/Sit_{self.situation}/'
        self.export_paths['Phonemes-Onset-Manual'] = self.procesed_data_path + f'Phonemes-Onset-Manual/Sit_{self.situation}/'
        
    def load_from_raw(self):
        """Loads raw data, this includes EEG, info and stimuli.

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
            samples_info = funciones.load_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl')
            loaded_samples_info = True
        except:
            loaded_samples_info = False
            samples_info = {'trial_lengths1': [0],
                            'trial_lengths2': [0],
                            'keep_indexes1':[],
                            'keep_indexes2':[]}

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
                    praat_executable_path=self.praat_executable_path)
                channel_2 = Trial_channel(
                    s=self.sesion,
                    trial=trial,
                    channel=2,
                    band=self.band,
                    sr=self.sr,
                    causal_filter_eeg=self.causal_filter_eeg,
                    envelope_filter=self.envelope_filter,
                    silence_threshold=self.silence_threshold,
                    praat_executable_path=self.praat_executable_path)

                # Extract dictionaries with the data
                trial_channel_1 = channel_1.load_trial(stims=self.stim.split('_'))
                trial_channel_2 = channel_2.load_trial(stims=self.stim.split('_'))
    
                # Load data to dictionary taking stimuli and eeg from speaker. I.e: each subject predicts its own EEG
                if self.situation == 'Habla_Propia' or self.situation == 'Ambos_Habla':
                    trial_sujeto_1 = {key: trial_channel_1[key] for key in trial_channel_1.keys()}
                    trial_sujeto_2 = {key: trial_channel_2[key] for key in trial_channel_2.keys()}
                
                # Load data to dictionary taking stimuli from speaker and eeg from listener. I.e: predicts own EEG using stimuli from interlocutor
                else:
                    trial_sujeto_1 = {key: trial_channel_2[key] for key in trial_channel_2.keys() if key!='EEG'} 
                    trial_sujeto_2 = {key: trial_channel_1[key] for key in trial_channel_1.keys() if key!='EEG'}
                    trial_sujeto_1['EEG'], trial_sujeto_2['EEG'] = trial_channel_1['EEG'], trial_channel_2['EEG']

                # Labeling of current speaker. {3:both_speaking,2:speaks_listener,3:speaks_interlocutor,0:silence} 
                current_speaker_1 = self.labeling(trial=trial, channel=2)
                current_speaker_2 = self.labeling(trial=trial, channel=1)

                # Match length of speaker labels and trials with the info of its lengths
                if loaded_samples_info:
                    trial_sujeto_1, current_speaker_1 = Sesion_class.match_lengths(dic=trial_sujeto_1, speaker_labels=current_speaker_1, minimum_length=samples_info['trial_lengths1'][p+1])
                    trial_sujeto_2, current_speaker_2 = Sesion_class.match_lengths(dic=trial_sujeto_2, speaker_labels=current_speaker_2, minimum_length=samples_info['trial_lengths2'][p+1])

                else:
                    # If there isn't any data matches lengths of both variables comparing every key and speaker labels length (the Trial gets modify inside the function)
                    trial_sujeto_1, current_speaker_1, minimo_largo1 = Sesion_class.match_lengths(dic=trial_sujeto_1, speaker_labels=current_speaker_1)
                    trial_sujeto_2, current_speaker_2, minimo_largo2 = Sesion_class.match_lengths(dic=trial_sujeto_2, speaker_labels=current_speaker_2)
                    samples_info['trial_lengths1'].append(minimo_largo1)
                    samples_info['trial_lengths2'].append(minimo_largo2)

                    # Preprocessing: calaculates the relevant indexes for the apropiate analysis. Add sum of all previous trials length. This is because at the end, all trials previous to the actual will be concatenated
                    samples_info['keep_indexes1'] += (self.shifted_indexes_to_keep(speaker_labels=current_speaker_1) + np.sum(samples_info['trial_lengths1'][:-1])).tolist()
                    samples_info['keep_indexes2'] += (self.shifted_indexes_to_keep(speaker_labels=current_speaker_2) + np.sum(samples_info['trial_lengths2'][:-1])).tolist()


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
                samples_info['trial_lengths1'][p] = 0
                samples_info['trial_lengths2'][p] = 0

        # Get info of the setup that was exluded in the previous iteration
        info = trial_channel_1['info']

        # Saves relevant indexes 
        if not loaded_samples_info:
            os.makedirs(self.samples_info_path, exist_ok=True)
            funciones.dump_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl', obj=samples_info, rewrite=True)

        # Save results
        for key in sujeto_1:
            # Drops silences phoneme column
            if key.startswith('Phonemes'):
                # Remove silence column, the last one by construction
                sujeto_1[key] = np.delete(arr=sujeto_1[key], obj=-1, axis=1)
                sujeto_2[key] = np.delete(arr=sujeto_2[key], obj=-1, axis=1)

            # Save preprocesed data
            os.makedirs(self.export_paths[key], exist_ok=True)
            funciones.dump_pickle(path=self.export_paths[key] + f'Sesion{self.sesion}.pkl', obj=[sujeto_1[key], sujeto_2[key]], rewrite=True)

        # Saves info of the setup                    
        funciones.dump_pickle(path=self.procesed_data_path + 'EEG/info.pkl', obj=info, rewrite=True)

        # Redefine subjects dictionaries to return only used stimuli
        sujeto_1_return = {key: sujeto_1[key] for key in self.stim.split('_') + ['EEG']}
        sujeto_2_return = {key: sujeto_2[key] for key in self.stim.split('_') + ['EEG']}
        sujeto_1_return['info'] = info
        sujeto_2_return['info'] = info

        return {'Sujeto_1': sujeto_1_return, 'Sujeto_2': sujeto_2_return}, samples_info
    
    def load_procesed(self):
        """Loads procesed data, this includes EEG, info and stimuli.

        Returns
        -------
        dict
            Sessions of both subjects.
        """
        # Load EEGs and procesed data
        eeg_sujeto_1, eeg_sujeto_2 = funciones.load_pickle(path=self.export_paths['EEG']+ f'Sesion{self.sesion}.pkl')
        info = funciones.load_pickle(path=self.procesed_data_path + 'EEG/info.pkl')
        samples_info = funciones.load_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl')
        sujeto_1 = {'EEG': eeg_sujeto_1, 'info': info}
        sujeto_2 = {'EEG': eeg_sujeto_2, 'info': info}
        
        # Loads stimuli to each subject
        for stimulus in self.stim.split('_'):
            sujeto_1[stimulus], sujeto_2[stimulus] = funciones.load_pickle(path=self.export_paths[stimulus]+f'Sesion{self.sesion}.pkl')
            # if stimulus == 'Pitch':
            #     # Remove missing values
            #     if self.valores_faltantes == None: #TODO CREO QUE ACA VA 0 TODAVIA NO VIMOS ESTOS DATOS
            #         sujeto_1[stimulus], sujeto_2[stimulus] = sujeto_1[stimulus][sujeto_1[stimulus]!=0], sujeto_2[stimulus][sujeto_2[stimulus]!=0]
            #     elif self.valores_faltantes:
            #         sujeto_1[stimulus], sujeto_2[stimulus] = sujeto_1[stimulus][sujeto_1[stimulus]==0], sujeto_2[stimulus][sujeto_2[stimulus]==0]
        return {'Sujeto_1': sujeto_1, 'Sujeto_2': sujeto_2}, samples_info
    
    def labeling(self, trial:int, channel:int):
        """Gives an array with speaking channel: 
            3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)

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
        ubi_speaker = self.phrases_path + f'/s{self.sesion}.objects.{trial:02d}.channel{channel}.phrases'
        h1t = pd.read_table(ubi_speaker, header=None, sep="\t")

        # Replace text by 1, silence by 0 and '#' by ''
        h1t.iloc[:, 2] = (h1t.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
        
        # Take difference in time and multiply it by sample rate in order to match envelope length (almost, miss by a sample or two)
        samples = np.round((h1t[1] - h1t[0]) * self.sr).astype("int")
        speaker = np.repeat(h1t.iloc[:, 2], samples).ravel()

        # Same with listener
        listener_channel = (channel - 3) * -1
        ubi_listener = self.phrases_path + f'/s{self.sesion}.objects.{trial:02d}.channel{listener_channel}.phrases'
        h2t = pd.read_table(ubi_listener, header=None, sep="\t")

        # Replace text by 1, silence by 0 and '#' by ''
        h2t.iloc[:, 2] = (h2t.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
        samples = np.round((h2t[1] - h2t[0]) * self.sr).astype("int")
        listener = np.repeat(h2t.iloc[:, 2], samples).ravel()

        # If there are difference in length, corrects with 0-padding
        diff = len(speaker) - len(listener)
        if diff > 0:
            listener = np.concatenate([listener, np.repeat(0, diff)])
        elif diff < 0:
            speaker = np.concatenate([speaker, np.repeat(0, np.abs(diff))])

        # Return an array with envelope length, with values 3 if both speak; 2, just listener; 1, speaker and 0 silence
        return speaker + listener * 2
    
    def shifted_indexes_to_keep(self, speaker_labels:np.ndarray):
        """Obtain shifted matrix indexes that match situation

        Parameters
        ----------
        speaker_labels : np.ndarray
            Labels of type of speaking for given sample

        Returns
        -------
        np.ndarray
            Indexes to keep for the analysis
        """
        if self.situation == 'Todo':
            return np.arange(len(speaker_labels))
        
        # Change 0 with 4s, because shifted matrix pad zeros that could be mistaken with situation 0    
        speaker_labels = np.array(speaker_labels)
        speaker_labels = np.where(speaker_labels==0, 4, speaker_labels)        

        # Computes shifted matrix
        shifted_matrix_speaker_labels = processing.shifted_matrix(features=speaker_labels, delays=self.delays).astype(float)

        # Make the appropiate label
        if self.situation == 'Silencio':
            situation_label = 4
        elif self.situation == 'Escucha':
            situation_label = 1
        elif self.situation == 'Habla' or self.situation == 'Habla_Propia':
            situation_label = 2
        elif self.situation == 'Ambos' or self.situation == 'Ambos_Habla':
            situation_label = 3
        
        # Shifted matrix index where the given situation is ocurring in all row (number of samples dimension) # TODO: discutir si dejar 0 o no.
        # return ((shifted_matrix_speaker_labels==situation_label) | (shifted_matrix_speaker_labels==0)).all(axis=1).nonzero()[0]
        return ((shifted_matrix_speaker_labels==situation_label)).all(axis=1).nonzero()[0]
    
    @staticmethod
    def print_trials(p:int, trial:int, trials:list):
        """Make print for trial update

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


    @staticmethod
    def match_lengths(dic:dict, speaker_labels:np.ndarray, minimum_length:int=None):
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
            minimum = min([dic[key].shape[0] for key in dic if key!='info'] + [len(speaker_labels)])

        # Correct length 
        for key in dic:
            if key != 'info':
                data = dic[key]
                if data.shape[0] > minimum:
                    dic[key] = data[:minimum]

        if len(speaker_labels) > minimum:
            speaker_labels = speaker_labels[:minimum]
            
        if minimum_length:
            return dic, speaker_labels
        else:
            return dic, speaker_labels, minimum

def load_data(sesion:int, stim:str, band:str, sr:float, procesed_data_path:str, 
              praat_executable_path:str, situation:str='Escucha', 
              causal_filter_eeg:bool=True, envelope_filter:bool=False, 
              silence_threshold:float=0.03, delays:np.ndarray=None):
    """Loads sessions of both subjects

    Parameters
    ----------
    sesion : int
        Session number
    stim : str
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'. Allowed stimuli are:
            ['Envelope', 'Pitch', 'Pitch-Manual', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete','Phonemes-Discrete-Manual', 'Phonemes-Onset','Phonemes-Onset-Manual'].
    band : str
        Neural frequency band. It could be one of:
            ['Delta','Theta', 'Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']
    sr : float
        Sample rate in Hz of the EEG
    tmin : float
        Minimimum window time
    tmax : float
        Maximum window time
    praat_executable_path : str
        Path directing to Praat executable
    procesed_data_path : str
        Path directing to procesed data
    situation : str, optional
        Situation considerer when performing the analysis, by default 'Escucha'. Allowed sitations are:
            ['Habla_Propia','Ambos_Habla','Escucha']
    causal_filter_eeg : bool, optional
        Whether to use or not a cusal filter, by default True
    envelope_filter : bool, optional
        Whether to use or not an envelope filter, by default False
    silence_threshold : float, optional
        Silence threshold of the dialogue, by default 0.03
    delays : np.ndarray, optional
            Delay array to construct shifted matrix, by default np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
    

    Returns
    -------
    tuple
        Dictionaries containing data from both subjects

    Raises
    ------
    SyntaxError
        If 'stimulus' is not an allowed stimulus. Allowed stimuli are:
            ['Envelope', 'Pitch', 'Pitch-Manual', 'Spectrogram', 'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete','Phonemes-Discrete-Manual', 'Phonemes-Onset','Phonemes-Onset-Manual'].
        If more than one stimulus is wanted, the separator should be '_'.
    """

    # Define allowed stimuli
    allowed_stims = ['Envelope', 'Pitch', 'Pitch-Manual','Spectrogram', 'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete','Phonemes-Discrete-Manual', 'Phonemes-Onset','Phonemes-Onset-Manual']
    allowed_situations = ['Habla_Propia','Ambos_Habla','Escucha']
    allowed_bands = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Alpha_Delta_Theta']

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
                                        procesed_data_path=procesed_data_path, 
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