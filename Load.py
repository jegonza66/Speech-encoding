# Standard libraries
import numpy as np, pandas as pd, os, warnings, time
import mne, librosa, platform, opensmile, textgrids

# Specific libraries
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
from praatio import pitch_and_intensity

# Modules
import Processing, Funciones, setup

# Review this If we want to update packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(verbose='CRITICAL')
exp_info = setup.exp_info()

class Trial_channel:
    def __init__(self, s:int=21, trial:int=1, channel:int=1, Band:str='All', sr:float=128, tmin:float=-0.6,
                 tmax:float=-0.003, valores_faltantes:int=0, Causal_filter_EEG:bool=True, 
                 Env_Filter:bool=False, SilenceThreshold:float=0.03):
        """Extract transcriptions, audio signal and EEG signal of given session and channel to calculate specific features.

        Parameters
        ----------
        s : int, optional
            Session number, by default 21
        trial : int, optional
            Number of trial, by default 1
        channel : int, optional
            Channel used to record the audio (it can be from subject 1 and 2), by default 1
        Band : str, optional
            Neural frequency band, by default 'All'. It could be one of:
            ['Delta','Theta',Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        sr : float, optional
            Sample rate in Hz of the EEG, by default 128
        tmin : float, optional
            Minimimum window time, by default -0.6
        tmax : float, optional
            Maximum window time, by default -0.003
        valores_faltantes : int, optional
            Number to replace the missing values (nans), by default 0
        Causal_filter_EEG : bool, optional
            Whether to use or not a cusal filter, by default True
        Env_Filter : bool, optional
            Whether to use or not an envelope filter, by default False
        SilenceThreshold : float, optional
            Silence threshold of the dialogue, by default 0.03

        Raises
        ------
        SyntaxError
            If 'Band' is not an allowed band frecuency. Allowed bands are:
            ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        """
        # Participants sex, ordered by session
        sex_list = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']
        allowed_band_frequencies = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        if Band in allowed_band_frequencies:
            self.Band= Band
        else:
            raise SyntaxError(f"{Band} is not an allowed band frecuency. Allowed bands are: {allowed_band_frequencies}")

        # Minimum and maximum frequency allowed within specified band
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(self.Band)
        self.sr = sr
        self.sampleStep = 0.01
        self.SilenceThreshold = SilenceThreshold
        self.audio_sr = 16000
        self.tmin, self.tmax = tmin, tmax #TODO: referenced but never used
        self.delays = - np.arange(np.floor(tmin * self.sr), np.ceil(tmax * self.sr), dtype=int)
        self.valores_faltantes = valores_faltantes
        self.sex = sex_list[(s - 21) * 2 + channel - 1]
        self.Causal_filter_EEG = Causal_filter_EEG
        self.Env_Filter = Env_Filter
        
        # To be filled with loaded data
        self.eeg = None

        # Relevant paths
        self.eeg_fname = f"Datos/EEG/S{s}/s{s}-{channel}-Trial{trial}-Deci-Filter-Trim-ICA-Pruned.set"
        self.wav_fname = f"Datos/wavs/S{s}/s{s}.objects.{trial:02d}.channel{channel}.wav"
        self.pitch_fname = f"Datos/Pitch_threshold_{SilenceThreshold}/S{s}/s{s}.objects.{trial:02d}.channel{channel}.txt"
        self.phn_fname = f"Datos/phonemes/S{s}/s{s}.objects.{trial:02d}.channel{channel}.aligned_fa.TextGrid"
        self.phn_fname_manual = f"Datos/phonemes/S{s}/manual/s{s}_objects_{trial:02d}_channel{channel}_aligned_faTAMARA.TextGrid"
        self.phrases_fname = f"Datos/phrases/S{s}/s{s}.objects.{trial:02d}.channel{channel}.phrases"
        
    def f_eeg(self):
        """Extract eeg file downsample it to get the same rate as self.sr and stores its data inside the class instance.

        Returns
        -------
        mne.io.eeglab.eeglab.RawEEGLAB
            EEG mne representation
        """
        # Read the .set file
        # TODO: warning of annotations and 'boundry' events -data discontinuities-.
        eeg = mne.io.read_raw_eeglab(input_fname=self.eeg_fname) 
        eeg.load_data()
        
        # Apply a lowpass filter
        if self.Band:
            if self.Causal_filter_EEG:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, phase='minimum') # TODO: preguntar la diferencia causal_filter False, tambien quite el verbose, preguntar si esta ok
            else:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg)
        
        # Store dimension mne.raw
        eeg.resample(sfreq=self.sr)

        # Return mne representation Times x nchannels
        self.eeg = eeg.copy()
        return self.eeg

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

    def f_envelope(self): #TODO check self.envelope assignation
        """Takes the low pass filtered -butterworth-, downsample and smoothened envelope of .wav file. Then matches its length to the EEG.

        Returns
        -------
        mne.io.array.array.RawArray
            Envelope of wav signal with desire dimensions
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")

        # Calculate envelope
        envelope = np.abs(sgn.hilbert(wav))
        
        # Apply lowpass butterworth filter
        if self.Env_Filter == 'Causal':# TODO can it be replaced for a mne filter?
            envelope = Processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='Causal').reshape(-1,1)
        elif self.Env_Filter == 'NonCausal':
            envelope = Processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='NonCausal').reshape(-1,1)
        else:
            envelope = envelope.reshape(-1,1)
            
        # Creates mne raw array
        info_envelope = mne.create_info(ch_names=['Envelope'], sfreq=self.audio_sr, ch_types='misc')
        envelope_mne_array = mne.io.RawArray(data=envelope.T, info=info_envelope)

        # Resample to match EEG data
        envelope_mne_array.resample(sfreq=self.sr)
        return envelope_mne_array

    def f_spectrogram(self, envelope:np.ndarray):
        """Calculates spectrogram of .wav file between 16 Mel frequencies.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        mne.io.array.array.RawArray
            Matrix with sprectrogram in given mel frequncies of dimension (Mel X Samples)
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

        # Creates mne raw array. Note that frequency is sr now
        info_spectrogram = mne.create_info(ch_names=[f'mel_{i}' for i in range(1,17)], sfreq=self.sr, ch_types='misc')
        
        return mne.io.RawArray(data=S_DB, info=info_spectrogram)

    def f_jitter_shimmer(self, envelope:np.ndarray): # NEVER USED
        """Gives the jitter and shimmer matching the size of the envelope

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform

        Returns
        -------
        tuple
            jitter and shimmer mne arrays with length smaller or equal to envelope length.
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
        mcm = Funciones.minimo_comun_multiplo(len(jitter), len(envelope))
        
        # Repeat each value the number of times it takes the length of jitter to achive the mcm. The result is that jitter length matches mcm
        jitter = np.repeat(jitter, mcm / len(jitter))
        shimmer = np.repeat(shimmer, mcm / len(shimmer))

        # Subsample by the number of times it takes the length of the envelope to achive the mcm. Now it has exactly the same size as envelope
        jitter = Processing.subsamplear(jitter, mcm / len(envelope))
        shimmer = Processing.subsamplear(shimmer, mcm / len(envelope))

        # Reassurance that the count is correct
        jitter = jitter[:min(len(jitter), len(envelope))].reshape(-1,1)
        shimmer = shimmer[:min(len(shimmer), len(envelope))].reshape(-1,1)

        # Creates mne raw arrays
        info_jitter = mne.create_info(ch_names=['Jitter'], sfreq=int(1/(y.index[1]-y.index[0])), ch_types='misc')
        info_shimmer = mne.create_info(ch_names=['Shimmer'], sfreq=int(1/(y.index[1]-y.index[0])), ch_types='misc')
        jitter_mne_array = mne.io.RawArray(data=jitter.T, info=info_jitter)
        shimmer_mne_array = mne.io.RawArray(data=shimmer.T, info=info_shimmer)

        return jitter_mne_array, shimmer_mne_array
    
    def f_phonemes(self, envelope:np.ndarray):
        """It makes a time-match between the phonemes and the envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        # pandas.DataFrame
        #     Columns given by phonemes and index given by samples. The values are the amplitude of the envolpe at the given sample.
        mne.io.array.array.RawArray
            Matrix with envelope amplitude at given sample. The matrix dimension is Phonemes_labels(in order)XSamples.
        """

        # Get trial total time length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname)

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
            
            # Rename silences
            if label == 'sil' or label == 'sp':
                label = ""
            
            # Check if the phoneme is in the list
            if not(label in exp_info.ph_labels or label==""):
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
        updated_taggs = exp_info.ph_labels + [ph for ph in np.unique(labels) if ph not in exp_info.ph_labels]

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
        # Match phoneme with envelope
        for i, tagg in enumerate(phonemes_tgrid):
            phonemes[i, updated_taggs.index(tagg)] = envelope[i]

        # Creates mne raw array
        info_phonemes = mne.create_info(ch_names=updated_taggs, sfreq=self.sr, ch_types='misc')

        return mne.io.RawArray(data=phonemes.T, info=info_phonemes)

    def f_phonemes_discrete(self, envelope:np.ndarray): #TODO: es exactamente igual a f_phonemes, salvo las últimas tres lineas. UNIFICAR
        """It makes a time-match between the phonemes and the envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        mne.io.array.array.RawArray
            Matrix with dimensions Phonemes_labels(in order)XSamples. The value of a given element is 1 if the phoneme is being pronounced and 0 elsewise.
        """

        # Get trial total time length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname)

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
            
            # Rename silences
            if label == 'sil' or label == 'sp':
                label = ""
            
            # Check if the phoneme is in the list
            if not(label in exp_info.ph_labels or label==""):
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
        updated_taggs = exp_info.ph_labels + [ph for ph in np.unique(labels) if ph not in exp_info.ph_labels]

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
        # Match phoneme with envelope
        for i, tagg in enumerate(phonemes_tgrid):
            phonemes[i, updated_taggs.index(tagg)] = 1

        # Creates mne raw array
        info_phonemes = mne.create_info(ch_names=updated_taggs, sfreq=self.sr, ch_types='misc')

        return mne.io.RawArray(data=phonemes.T, info=info_phonemes)
   
    def f_phonemes_onset(self, envelope:np.ndarray): #TODO: es exactamente igual a f_phonemes, salvo las últimas tres lineas. UNIFICAR
        """It makes a time-match between the phonemes and the envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        mne.io.array.array.RawArray
            Matrix with dimensions Phonemes_labels(in order)XSamples. The value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
        """

        # Get trial total time length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname)

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
            
            # Rename silences
            if label == 'sil' or label == 'sp':
                label = ""
            
            # Check if the phoneme is in the list
            if label in exp_info.ph_labels or label=="":
                labels.append(label)
                times.append((ph.xmin, ph.xmax))
                samples.append(np.round((ph.xmax - ph.xmin) * self.sr).astype("int"))
            else:
                print(f'{label} is not in not a recognized phoneme.')

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
        updated_taggs = exp_info.ph_labels + [ph for ph in np.unique(labels) if ph not in exp_info.ph_labels]

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)

        # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
        phonemes_onset = [phonemes_tgrid[0]]
        for i in range(1, len(phonemes_tgrid)):
            if phonemes_tgrid[i] == phonemes_tgrid[i-1]:
                phonemes_onset.append(0)
            else:
                phonemes_onset.append(phonemes_tgrid[i])
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
        # Match phoneme with envelope
        for i, tagg in enumerate(phonemes_onset):
            if tagg!=0:
                phonemes[i, updated_taggs.index(tagg)] = 1

        # Creates mne raw array
        info_phonemes = mne.create_info(ch_names=updated_taggs, sfreq=self.sr, ch_types='misc')

        return mne.io.RawArray(data=phonemes.T, info=info_phonemes)
        
    def f_phonemes_manual(self, envelope:np.ndarray):#TODO: es muy parecida a f_phonemes, salvo las últimas tres lineas. UNIFICAR
        """It makes a time-match between the phonemes and the envelope. But it selects manually exclutions for certain phonemes.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        mne.io.array.array.RawArray
            Matrix with dimensions Phonemes_labels(in order)XSamples. The value of a given element is 1 if the phoneme is being pronounced and 0 elsewise.
        """

        # Get trial total time length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname)

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
            if not(label in exp_info.ph_labels or label==""):
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
        updated_taggs = exp_info.ph_labels + [ph for ph in np.unique(labels) if ph not in exp_info.ph_labels]

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)
        
        # Make empty array of phonemes
        phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
        # Match phoneme with envelope
        for i, tagg in enumerate(phonemes_tgrid):
            phonemes[i, updated_taggs.index(tagg)] = 1

        # Creates mne raw array
        info_phonemes = mne.create_info(ch_names=updated_taggs, sfreq=self.sr, ch_types='misc')

        return mne.io.RawArray(data=phonemes.T, info=info_phonemes)

    def f_calculate_pitch(self): #TODO: make it usuable in any computer. Also missing folder.
        """Calculates pitch from .wav file and stores it.
        """
        # Makes directory for storing data
        if platform.system() == 'Linux':
            praatEXE = 'Praat/praat'
            output_folder = 'Datos/Pitch_threshold_{}'.format(self.SilenceThreshold)
        else:
            praatEXE = r"C:\Program Files\Praat\Praat.exe"
            output_folder = "C:/Users/joaco/Desktop/Joac/Facultad/Tesis/Código/Datos/Pitch_threshold_{}".format(
                self.SilenceThreshold)
        try:
            os.makedirs(output_folder)
        except:
            pass

        output_path = self.pitch_fname
        if self.sex == 'M':
            minPitch = 50
            maxPitch = 300
        if self.sex == 'F':
            minPitch = 75
            maxPitch = 500
        pitch_and_intensity.extractPI(inputFN=os.path.abspath(self.wav_fname), 
                                      outputFN=os.path.abspath(output_path), 
                                      praatEXE=praatEXE, 
                                      minPitch=minPitch,
                                      maxPitch=maxPitch, 
                                      sampleStep=self.sampleStep, 
                                      silenceThreshold=self.SilenceThreshold)

    def load_pitch(self, envelope:np.ndarray): #TODO: pedir a joaco los datos
        """Loads the pitch of the speaker

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        mne.io.array.array.RawArray
            Array containing pitch of the audio signal
        """
        # Reads the file and transform it in a np.ndarray
        read_file = pd.read_csv(self.pitch_fname)
        pitch = np.array(read_file['pitch'])
        # time = np.array(read_file['time'])
        # intensity = np.array(read_file['intensity'])

        # Filter undefined values
        pitch[pitch == '--undefined--'] = np.nan
        pitch = np.array(pitch, dtype=np.float32)
        if self.valores_faltantes == None:
            pitch = pitch[~np.isnan(pitch)]
        
        # Replace nans by valores_faltantes (int)
        elif np.isfinite(self.valores_faltantes):
            pitch[np.isnan(pitch)] = float(self.valores_faltantes)
        else:
            print('Invalid missing value for pitch {}'.format(self.valores_faltantes) + '\nMust be finite.')

        # Match length with envelope
        # Repeat each value of the pitch to make audio_sr*sampleStep times (in particular, a multiple of audio_sr)
        pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=np.float32)
        
        pitch = Processing.subsamplear(pitch, int(self.audio_sr/self.sr))
        pitch = pitch[:min(len(pitch), len(envelope))].reshape(-1,1)
        
        # Creates mne raw array
        info_pitch = mne.create_info(ch_names=['Pitch'], sfreq=self.sr, ch_types='misc') # TODO: check sfreq

        return mne.io.RawArray(data=pitch.T, info=info_pitch)

    def load_trial(self, stims:list): 
        """Extract EEG and calculates specified stimuli.
        Parameters
        ----------
        stims : list
            A list containing possible stimuli. Possible input values are: 
            ['Envelope', 'Pitch', 'Spectrogram', 'Phonemes', 'Phonemes-manual', 'Phonemes-discrete', 'Phonemes-onset']

        Returns
        -------
        dict
            Dictionary with EEG, info and specified stimuli as mne objects
        """
        channel = {}
        channel['EEG'] = self.f_eeg()
        channel['info'] = self.f_info()
        channel['Envelope'] = self.f_envelope()
        env = Funciones.mne_to_numpy(channel['Envelope'])

        if 'Envelope' in stims:
            return channel
        if 'Pitch' in stims:
            channel['Pitch'] = self.load_pitch(envelope=env)
        if 'Spectrogram' in stims:
            channel['Spectrogram'] = self.f_spectrogram(envelope=env)
        if 'Phonemes' in stims:
            channel['Phonemes'] = self.f_phonemes(envelope=env)
        if 'Phonemes-manual' in stims:
            channel['Phonemes-manual'] = self.f_phonemes_manual(envelope=env)
        if 'Phonemes-discrete' in stims:
            channel['Phonemes-discrete'] = self.f_phonemes_discrete(envelope=env)
        if 'Phonemes-onset' in stims:
            channel['Phonemes-onset'] = self.f_phonemes_onset(envelope=env)

        return channel

class Sesion_class: # TODO calculate pitch must be inside load pitch, only do it if pitch is a stimulus
    def __init__(self, sesion:int=21, stim:str='Envelope', Band:str='All', sr:float=128, tmin:float=-0.6, 
                 tmax:float=-0.003, valores_faltantes:int=0, Causal_filter_EEG:bool=True, Env_Filter:bool=False,
                 situacion:str='Escucha', Calculate_pitch:bool=False, SilenceThreshold:float=0.03,
                 procesed_data_path:str=f'saves/Preprocesed_Data/tmin{-0.6}_tmax{-.003}/'
                 ):
        """Construct an object for the given session containing all concerning data.

        Parameters
        ----------
        sesion : int, optional
            Session number, by default 21
        stim : str, optional
            Stimuli to use in the analysis, by default 'Envelope'. If more than one stimulus is wanted, the separator should be '_'. Allowed stimuli are:
            ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset'].
        Band : str, optional
            Neural frequency band, by default 'All'. It could be one of:
            ['Delta','Theta',Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        sr : float, optional
            Sample rate in Hz of the EEG, by default 128
        tmin : float, optional
            Minimimum window time, by default -0.6
        tmax : float, optional
            Maximum window time, by default -0.003
        valores_faltantes : int, optional
            Number to replace the missing values (nans), by default 0
        Causal_filter_EEG : bool, optional
            Whether to use or not a cusal filter, by default True
        Env_Filter : bool, optional
            Whether to use or not an envelope filter, by default False
        situacion : str, optional
            Situation considerer when performing the analysis, by default 'Escucha'. Allowed sitations are:
            ['Habla_Propia','Ambos_Habla','Escucha']
        Calculate_pitch : bool, optional
            Pitch of speaker signal, perform on envelope, by default False
        SilenceThreshold : float, optional
            Silence threshold of the dialogue, by default 0.03
        procesed_data_path : str, optional
            Path directing to procesed data, by default f'saves/Preprocesed_Data/tmin{-0.6}_tmax{-0.003}/'

        Raises
        ------
        SyntaxError
            If 'stim' is not an allowed stimulus. Allowed stimuli are:
            ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset']. 
            If more than one stimulus is wanted, the separator should be '_'.
        SyntaxError
            If 'Band' is not an allowed band frecuency. Allowed bands are:
            ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        SyntaxError
            If situacion is not an allowed situation. Allowed situations are: 
            ['Habla_Propia','Ambos_Habla','Escucha']
        """
       
        # Check if band, stim and situacion parameters where passed with the right syntax
        allowed_stims = ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset']
        allowed_band_frequencies = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        allowed_situaciones = ['Habla_Propia','Ambos_Habla','Escucha']
        for st in stim.split('_'):
            if st in allowed_stims:
                pass
            else:
                raise SyntaxError(f"{st} is not an allowed stimulus. Allowed stimuli are: {allowed_stims}. If more than one stimulus is wanted, the separator should be '_'.")
        self.stim = stim
        if Band in allowed_band_frequencies:
            self.Band = Band
        else:
            raise SyntaxError(f"{Band} is not an allowed band frecuency. Allowed bands are: {allowed_band_frequencies}")
        if situacion in allowed_situaciones:
            self.situacion = situacion
        else:
            raise SyntaxError(f"{situacion} is not an allowed situation. Allowed situations are: {allowed_situaciones}")
        
        # Define parameters
        self.sesion = sesion
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(Band)
        self.sr = sr
        self.tmin, self.tmax = tmin, tmax
        self.delays = - np.arange(np.floor(tmin * self.sr), np.ceil(tmax * self.sr), dtype=int) # TODO: preguntar por qué se pasan así
        self.valores_faltantes = valores_faltantes
        self.Causal_filter_EEG = Causal_filter_EEG
        self.Env_Filter = Env_Filter
        self.Calculate_pitch = Calculate_pitch
        self.SilenceThreshold = SilenceThreshold

        # Relevant paths
        self.procesed_data_path = procesed_data_path
        self.samples_info_path = self.procesed_data_path + f'samples_info/Sit_{self.situacion}/'
        self.phn_path = f"Datos/phonemes/S{self.sesion}/"
        self.phrases_path = f"Datos/phrases/S{self.sesion}/"

        # Define paths to export data
        self.export_paths = {}
        if self.Causal_filter_EEG:
            self.export_paths['EEG'] = self.procesed_data_path + f'EEG/Causal_Sit_{self.situacion}_Band_{self.Band}/'
        else:
            self.export_paths['EEG'] = self.procesed_data_path + f'EEG/Sit_{self.situacion}_Band_{self.Band}/'
        if self.Env_Filter:
            self.export_paths['Envelope'] = self.procesed_data_path + f'Envelope/{self.Env_Filter}_Sit_{self.situacion}/'
        else:
            self.export_paths['Envelope'] = self.procesed_data_path + f'Envelope/Sit_{self.situacion}/'
        self.export_paths['PitchMask']= self.procesed_data_path + f'Pitch_mask_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/'
        self.export_paths['Pitch'] = self.procesed_data_path + f'Pitch_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/'
        self.export_paths['Spectrogram'] = self.procesed_data_path + f'Spectrogram/Sit_{self.situacion}/'
        self.export_paths['Phonemes'] = self.procesed_data_path + f'Phonemes/Sit_{self.situacion}/'
        self.export_paths['Phonemes-manual'] = self.procesed_data_path + f'Phonemes-manual/Sit_{self.situacion}/'
        self.export_paths['Phonemes-discrete'] = self.procesed_data_path + f'Phonemes-discrete/Sit_{self.situacion}/'
        self.export_paths['Phonemes-onset'] = self.procesed_data_path + f'Phonemes-onset/Sit_{self.situacion}/'
        
    def load_from_raw(self):
        """Loads raw data, this includes EEG, info and stimuli.

        Returns
        -------
        dict
            Sessions of both subjects
        """
        
        # Subjects dictionaries, stores their data
        Sujeto_1 = {}
        Sujeto_2 = {}

        # Try to open preprocessed info of samples, if not crates raw
        try:
            samples_info = Funciones.load_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl')
            loaded_samples_info = True
        except:
            loaded_samples_info = False
            samples_info = {'keep_indexes1': [], 'keep_indexes2': [], 'trial_lengths1': [0], 'trial_lengths2': [0]}

        # Retrive number of files, i.e: trials
        trials = list(set([int(fname.split('.')[2]) for fname in os.listdir(self.phrases_path) if os.path.isfile(self.phrases_path + f'/{fname}')]))
        
        # Retrive and concatenate data of all trials
        for trial in trials:
            print(f'Trial {trial} of {len(trials)}')
            try:
                # Create trial for both channels in order to extract features and EEG signal
                channel_1 = Trial_channel(
                    s=self.sesion, 
                    trial=trial, 
                    channel=1,
                    Band=self.Band, 
                    sr=self.sr, 
                    tmin=self.tmin, 
                    tmax=self.tmax,
                    valores_faltantes=self.valores_faltantes,
                    Causal_filter_EEG=self.Causal_filter_EEG,
                    Env_Filter=self.Env_Filter,
                    SilenceThreshold=self.SilenceThreshold)
                channel_2 = Trial_channel(
                    s=self.sesion,
                    trial=trial,
                    channel=2,
                    Band=self.Band,
                    sr=self.sr,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    valores_faltantes=self.valores_faltantes,
                    Causal_filter_EEG=self.Causal_filter_EEG,
                    Env_Filter=self.Env_Filter,
                    SilenceThreshold=self.SilenceThreshold)
                if self.Calculate_pitch:# TODO calculate pitch must be inside Trial_channel, only do it if pitch is a stimulus. If that is done, then its not necessary to store channel_i, instead Trial_channel_i can be creted at first
                    channel_1.f_calculate_pitch()
                    channel_2.f_calculate_pitch()

                # Extract dictionaries with the data
                Trial_channel_1 = channel_1.load_trial(stims=self.stim.split('_'))
                Trial_channel_2 = channel_2.load_trial(stims=self.stim.split('_'))
    
                # Load data to dictionary taking stimuli and eeg from speaker. I.e: each subject predicts its own EEG
                if self.situacion == 'Habla_Propia' or self.situacion == 'Ambos_Habla':
                    Trial_sujeto_1 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys()}
                    Trial_sujeto_2 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys()}
                
                # Load data to dictionary taking stimuli from speaker and eeg from listener. I.e: predicts own EEG using stimuli from interlocutor
                else:
                    Trial_sujeto_1 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys() if key!='EEG'} 
                    Trial_sujeto_2 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys() if key!='EEG'}
                    Trial_sujeto_1['EEG'], Trial_sujeto_2['EEG'] = Trial_channel_1['EEG'], Trial_channel_2['EEG']

                # Labeling of current speaker. {3:both_speaking,2:speaks_listener,3:speaks_interlocutor,0:silence} # TODO why are channel number interchanged, shouldn't this be inside previous else?
                current_speaker_1 = self.labeling(trial=trial, channel=2)
                current_speaker_2 = self.labeling(trial=trial, channel=1)

                # Match length of speaker labels and trials with the info of its lengths
                if loaded_samples_info:
                    Trial_sujeto_1, current_speaker_1 = Sesion_class.match_lengths(dic=Trial_sujeto_1, speaker_labels=current_speaker_1, minimum_length=samples_info['trial_lengths1'][trial])
                    Trial_sujeto_2, current_speaker_2 = Sesion_class.match_lengths(dic=Trial_sujeto_2, speaker_labels=current_speaker_2, minimum_length=samples_info['trial_lengths2'][trial])

                else:
                    # If there isn't any data matches lengths of both variables comparing every key and speaker labels length (the Trial gets modify inside the function)
                    Trial_sujeto_1, current_speaker_1, minimo_largo1 = Sesion_class.match_lengths(dic=Trial_sujeto_1, speaker_labels=current_speaker_1)
                    Trial_sujeto_2, current_speaker_2, minimo_largo2 = Sesion_class.match_lengths(dic=Trial_sujeto_2, speaker_labels=current_speaker_2)
                    samples_info['trial_lengths1'].append(minimo_largo1)
                    samples_info['trial_lengths2'].append(minimo_largo2)

                    # Preprocessing: calaculates the relevant indexes for the apropiate analysis
                    keep_indexes1_trial = self.shifted_indexes_to_keep(speaker_labels=current_speaker_1)
                    keep_indexes2_trial = self.shifted_indexes_to_keep(speaker_labels=current_speaker_2)

                    # Add sum of all previous trials length. This is because at the end, all trials previous to the actual will be concatenated
                    keep_indexes1_trial += np.sum(samples_info['trial_lengths1'][:-1])
                    keep_indexes2_trial += np.sum(samples_info['trial_lengths2'][:-1])
                    
                    # Append relevant indexes to sample_info
                    samples_info['keep_indexes1'].append(keep_indexes1_trial)
                    samples_info['keep_indexes2'].append(keep_indexes2_trial)

                # Concatenates data of each subject #TODO la versión vieja excluía pitch preguntar por que
                for key in Trial_sujeto_1:
                    if key != 'info':
                        if key not in Sujeto_1:
                            Sujeto_1[key] = Trial_sujeto_1[key]
                        else:
                            if key=='EEG':
                                Sujeto_1[key] = mne.concatenate_raws(raws=[Sujeto_1[key], Trial_sujeto_1[key]])
                            else:
                                Sujeto_1[key].append([Trial_sujeto_1[key]]) 
                for key in Trial_sujeto_2:
                    if key != 'info':
                        if key not in Sujeto_2:
                            Sujeto_2[key] = Trial_sujeto_2[key]
                        else:
                            if key=='EEG':
                                Sujeto_2[key] = mne.concatenate_raws(raws=[Sujeto_2[key], Trial_sujeto_2[key]])
                            else:
                                Sujeto_2[key].append([Trial_sujeto_2[key]]) 

            # Empty trial
            except:
                print(f"Trial {trial} of session {self.sesion} couldn't be loaded.")
                samples_info['trial_lengths1'].append(0)
                samples_info['trial_lengths2'].append(0)

        # Get info of the setup that was exluded in the previous iteration
        info = Trial_channel_1['info']

        # Saves flatten relevant indexes 
        if not loaded_samples_info:
            samples_info['keep_indexes1'] = [item for sublist in samples_info['keep_indexes1'] for item in sublist]
            samples_info['keep_indexes2'] = [item for sublist in samples_info['keep_indexes2'] for item in sublist]
            os.makedirs(self.samples_info_path, exist_ok=True)
            Funciones.dump_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl', obj=samples_info, rewrite=True)

        # Construct shifted matrix 
        specific_stimuli = ['Spectrogram', 'Phonemes', 'Phonemes-manual', 'Phonemes-discrete', 'Phonemes-onset']
        for key in Sujeto_1:
            # Drops silences phoneme column
            if key.startswith('Phonemes'):
                # Remove silence row 
                silence_index_1 = Sujeto_1[key].info.get('ch_names').index('')
                data_unsilence_1 = np.delete(arr=Sujeto_1[key].get_data(), obj=silence_index_1, axis=0)

                silence_index_2 = Sujeto_2[key].info.get('ch_names').index('')
                data_unsilence_2 = np.delete(arr=Sujeto_2[key].get_data(), obj=silence_index_2, axis=0)
                
                # Recreates mne array
                ch_1 = Sujeto_1[key].ch_names
                ch_2 = Sujeto_2[key].ch_names
                ch_1.remove('')
                ch_2.remove('')
                info_1 = mne.create_info(ch_names=ch_1, sfreq=Sujeto_1[key].info.get('sfreq'), ch_types='misc')
                info_2 = mne.create_info(ch_names=ch_2, sfreq=Sujeto_2[key].info.get('sfreq'), ch_types='misc')
                Sujeto_1[key] = mne.io.RawArray(data=data_unsilence_1, info=info_1)
                Sujeto_2[key] = mne.io.RawArray(data=data_unsilence_2, info=info_2)

            # Stacked shifted matrices row by row (a given row is a feature). This is faster than hstack, also is convient to make loop separated
            if key in specific_stimuli:
                print(f'Computing shifted matrix for the {key}')
                data_1, data_2 = Sujeto_1[key].get_data(), Sujeto_2[key].get_data()
                shift_1 = np.zeros(shape=(data_1.shape[1], data_1.shape[0]*(len(self.delays))))
                shift_2 = np.zeros(shape=(data_2.shape[1], data_2.shape[0]*(len(self.delays))))
                
                for i in range(len(data_1)):
                    shift_1_i = Processing.shifted_matrix(feature=data_1[i], delays=self.delays)
                    shift_1[:,len(self.delays)*i:len(self.delays)*(i+1)] = shift_1_i
                for i in range(len(data_2)):
                    shift_2_i = Processing.shifted_matrix(feature=data_2[i], delays=self.delays)
                    shift_2[:,len(self.delays)*i:len(self.delays)*(i+1)] = shift_2_i

                # Return to mne arrays
                shift_1_ch_names, shift_2_ch_names = [], []
                for delay in self.delays:
                    shift_1_ch_names += [ch + f'_delay_{delay}' for ch in Sujeto_1[key].ch_names]
                    shift_2_ch_names += [ch + f'_delay_{delay}' for ch in Sujeto_2[key].ch_names]

                 # Keep just relevant indexes # TODO la info puede fallar x la cantidad de samples
                relevant_1 = shift_1[samples_info['keep_indexes1'], :]
                shift_1_info = mne.create_info(ch_names=shift_1_ch_names, sfreq=Sujeto_1[key].info.get('sfreq'), ch_types='misc')
                Sujeto_1[key] = mne.io.RawArray(data=relevant_1.T, info=shift_1_info)

                relevant_2 = shift_2[samples_info['keep_indexes2'], :]
                shift_2_info = mne.create_info(ch_names=shift_2_ch_names, sfreq=Sujeto_2[key].info.get('sfreq'), ch_types='misc')
                Sujeto_2[key] = mne.io.RawArray(data=relevant_2.T, info=shift_2_info)
                print(f'{key} matrix computed')

            # The rest of the stimuli is 1D, so it's not necessary to stack them). Bassically envelope.
            elif key!= 'EEG':
                print(f'Computing shifted matrix for the {key}')
                shift_1 = Processing.shifted_matrix(feature=Sujeto_1[key].get_data()[0], delays=self.delays)
                shift_2 = Processing.shifted_matrix(feature=Sujeto_2[key].get_data()[0], delays=self.delays)

                # Return to mne arrays
                shift_1_ch_names = [Sujeto_1[key].ch_names[0]+f'_delay_{delay}' for delay in self.delays]
                shift_2_ch_names = [Sujeto_2[key].ch_names[0]+f'_delay_{delay}' for delay in self.delays]
                
                # Keep just relevant indexes # TODO la info puede fallar x la cantidad de samples
                relevant_1 = shift_1[samples_info['keep_indexes1'], :]
                shift_1_info = mne.create_info(ch_names=shift_1_ch_names, sfreq=Sujeto_1[key].info.get('sfreq'), ch_types='misc')
                Sujeto_1[key] = mne.io.RawArray(data=relevant_1.T, info=shift_1_info)

                relevant_2 = shift_2[samples_info['keep_indexes2'], :]
                shift_2_info = mne.create_info(ch_names=shift_2_ch_names, sfreq=Sujeto_2[key].info.get('sfreq'), ch_types='misc')
                Sujeto_2[key] = mne.io.RawArray(data=relevant_2.T, info=shift_2_info)
                print(f'{key} matrix computed')
            
            elif key=='EEG':
                # Keep just relevant indexes # TODO la info puede fallar x la cantidad de samples
                relevant_1 = Sujeto_1[key].get_data().T[samples_info['keep_indexes1'], :]
                Sujeto_1[key] = mne.io.RawArray(data=relevant_1.T, info=Sujeto_1[key].info)

                relevant_2 = Sujeto_2[key].get_data().T[samples_info['keep_indexes2'], :]
                Sujeto_2[key] = mne.io.RawArray(data=relevant_2.T, info=Sujeto_1[key].info)

            # Save preprocesed data
            os.makedirs(self.export_paths[key], exist_ok=True)
            Funciones.dump_pickle(path=self.export_paths[key] + f'Sesion{self.sesion}.pkl', obj=[Sujeto_1[key], Sujeto_2[key]], rewrite=True)

        # Saves info of the setup                    
        Funciones.dump_pickle(path=self.procesed_data_path + 'EEG/info.pkl', obj=info, rewrite=True)

        # Redefine subjects dictionaries to return only used stimuli
        Sujeto_1_return = {key: Sujeto_1[key] for key in self.stim.split('_') + ['EEG']}
        Sujeto_2_return = {key: Sujeto_2[key] for key in self.stim.split('_') + ['EEG']}
        Sujeto_1_return['info'] = info
        Sujeto_2_return['info'] = info

        return {'Sujeto_1': Sujeto_1_return, 'Sujeto_2': Sujeto_2_return}
    
    def load_procesed(self):
        """Loads procesed data, this includes EEG, info and stimuli.

        Returns
        -------
        dict
            Sessions of both subjects.
        """
        # Define EEG path
        EEG_path = self.export_paths['EEG']+ f'Sesion{self.sesion}.pkl'

        # Load EEGs and procesed data
        eeg_sujeto_1, eeg_sujeto_2 = Funciones.load_pickle(path=EEG_path)
        info = Funciones.load_pickle(path=self.procesed_data_path + 'EEG/info.pkl')
        Sujeto_1 = {'EEG': eeg_sujeto_1, 'info': info}
        Sujeto_2 = {'EEG': eeg_sujeto_2, 'info': info}
        
        # Loads stimuli to each subject
        for stimulus in self.stim.split('_'):
            Sujeto_1[stimulus], Sujeto_2[stimulus] = Funciones.load_pickle(path=self.export_paths[stimulus]+f'Sesion{self.sesion}.pkl')
            if stimulus == 'Pitch':
                # Remove missing values
                if self.valores_faltantes == None: #TODO CREO QUE ACA VA 0 TODAVIA NO VIMOS ESTOS DATOS
                    Sujeto_1[stimulus], Sujeto_2[stimulus] = Sujeto_1[stimulus][Sujeto_1[stimulus]!=0], Sujeto_2[stimulus][Sujeto_2[stimulus]!=0]
                elif self.valores_faltantes:
                    Sujeto_1[stimulus], Sujeto_2[stimulus] = Sujeto_1[stimulus][Sujeto_1[stimulus]==0], Sujeto_2[stimulus][Sujeto_2[stimulus]==0]
        return {'Sujeto_1': Sujeto_1, 'Sujeto_2': Sujeto_2}
    
    def labeling(self, trial:int, channel:int):
        """Gives an array with speaking channel: 
            3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)

        Parameters
        ----------
        trial : int
            _description_
        channel : int
            _description_

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
        """Obtain shifted matrix indexes that match situacion

        Parameters
        ----------
        speaker_labels : np.ndarray
            Labels of type of speaking for given sample

        Returns
        -------
        np.ndarray
            Indexes to keep for the analysis
        """
        
        # Computes shifted matrix
        shifted_matrix_speaker_labels = Processing.shifted_matrix(feature=speaker_labels, delays=self.delays).astype(float)
        
        # Make the appropiate label
        if self.situacion == 'Todo':
            return

        elif self.situacion == 'Silencio':
            situacion_label = 0
        elif self.situacion == 'Escucha':
            situacion_label = 1
        elif self.situacion == 'Habla' or self.situacion == 'Habla_Propia':
            situacion_label = 2
        elif self.situacion == 'Ambos' or self.situacion == 'Ambos_Habla':
            situacion_label = 3

        # Change the value of situacion_label by 'nan
        shifted_matrix_speaker_labels[shifted_matrix_speaker_labels == situacion_label] = float("nan")
        
        # Shifted matrix index where the given situation is ocurring
        return pd.isnull(shifted_matrix_speaker_labels).all(axis=1).nonzero()[0]

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
            minimum = min([dic[key].get_data().T.shape[0] for key in dic if key!='info'] + [len(speaker_labels)])

        # Correct length and update mne array
        for key in dic:
            if key!= 'info' and key!='EEG':
                data = dic[key].get_data().T
                if data.shape[0] > minimum:
                    dic[key] = mne.io.RawArray(data=data[:minimum].T, info=dic[key].info)
            elif key=='EEG':
                data = dic[key].get_data().T
                if data.shape[0] > minimum:
                    eeg_times = dic[key].times.tolist()
                    dic[key].crop(tmin=eeg_times[0], tmax=eeg_times[minimum], verbose=True)

        if len(speaker_labels) > minimum:
            speaker_labels = speaker_labels[:minimum]
        if minimum_length:
            return dic, speaker_labels
        else:
            return dic, speaker_labels, minimum

def Load_Data(sesion:int, stim:str, Band:str, sr:float, tmin:float, tmax:float, 
              procesed_data_path:str, situacion:str='Escucha', Causal_filter_EEG:bool=True, 
              Env_Filter:bool=False, valores_faltantes:int=0, Calculate_pitch:bool=False, 
              SilenceThreshold:float=0.03):
    """Loads sessions of both subjects

    Parameters
    ----------
    sesion : int
        Session number
    stim : str
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'. Allowed stimuli are:
            ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset'].
    Band : str
        Neural frequency band. It could be one of:
            ['Delta','Theta',Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
    sr : float
        Sample rate in Hz of the EEG
    tmin : float
        Minimimum window time
    tmax : float
        Maximum window time
    procesed_data_path : str
        Path directing to procesed data
    situacion : str, optional
        Situation considerer when performing the analysis, by default 'Escucha'. Allowed sitations are:
            ['Habla_Propia','Ambos_Habla','Escucha']
    Causal_filter_EEG : bool, optional
        Whether to use or not a cusal filter, by default True
    Env_Filter : bool, optional
        Whether to use or not an envelope filter, by default False
    valores_faltantes : int, optional
        Number to replace the missing values (nans), by default 0
    Calculate_pitch : bool, optional
        Pitch of speaker signal, perform on envelope, by default False
    SilenceThreshold : float, optional
        Silence threshold of the dialogue, by default 0.03

    Returns
    -------
    tuple
        Dictionaries containing data from both subjects

    Raises
    ------
    SyntaxError
        If 'stimulus' is not an allowed stimulus. Allowed stimuli are:
            ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset']. 
        If more than one stimulus is wanted, the separator should be '_'.
    """

    # Define allowed stimuli
    allowed_stims = ['Envelope', 'Pitch', 'PitchMask', 'Spectrogram', 'Phonemes', 'Phonemes-manual', 'Phonemes-discrete', 'Phonemes-onset']

    if all(stimulus in allowed_stims for stimulus in stim.split('_')):
        sesion_obj = Sesion_class(sesion=sesion, 
                                  stim=stim, 
                                  Band=Band, 
                                  sr=sr, 
                                  tmin=tmin, 
                                  tmax=tmax,
                                  valores_faltantes=valores_faltantes, 
                                  Causal_filter_EEG=Causal_filter_EEG,
                                  Env_Filter=Env_Filter, 
                                  situacion=situacion, 
                                  Calculate_pitch=Calculate_pitch,
                                  SilenceThreshold=SilenceThreshold, 
                                  procesed_data_path=procesed_data_path)

        # Try to load procesed data, if it fails it loads raw data
        try:
            print('Loading preprocesed data\n')
            Sesion = sesion_obj.load_procesed()
            print('Data loaded succesfully\n')
        except:
            print("Couldn't load data, compute it from raw\n")
            Sesion = sesion_obj.load_from_raw()
        return Sesion['Sujeto_1'], Sesion['Sujeto_2']
    else:
        raise SyntaxError(f"{stim} is not an allowed stimulus. Allowed stimuli are: {allowed_stims}. If more than one stimulus is wanted, the separator should be '_'.")

def Estimulos(stim:str, Sujeto_1:dict, Sujeto_2:dict):
    """Extracts both subjects stimuli

    Parameters
    ----------
    stim : str
        Stimuli to use in the analysis. If more than one stimulus is wanted, the separator should be '_'. Allowed stimuli are:
            ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset']
    Sujeto_1 : dict
        _description_
    Sujeto_2 : dict
        _description_

    Returns
    -------
    tuple
        lists with asked stimuli for both subjects
    """
    # By this time, stimuli are already checked. So it just get a list with stimuli of both subjects
    dfinal_para_sujeto_1 = []
    dfinal_para_sujeto_2 = []

    for stimulus in stim.split('_'):
        dfinal_para_sujeto_1.append(Sujeto_1[stimulus])
        dfinal_para_sujeto_2.append(Sujeto_2[stimulus])

    return dfinal_para_sujeto_1, dfinal_para_sujeto_2

if __name__=='__main__':
    sesion, stim, Band, sr, tmin, tmax, situacion = 21, 'Spectrogram', 'Theta', 128, -.6,.2,'Escucha'
    procesed_data_path = f'saves/Preprocesed_Data/tmin{tmin}_tmax{tmax}/'
    Sujeto_1, Sujeto_2 = Load_Data(sesion=sesion, 
                                                stim=stim, 
                                                Band=Band, 
                                                sr=sr, 
                                                tmin=tmin, 
                                                tmax=tmax,
                                                procesed_data_path=procesed_data_path, 
                                                situacion=situacion,
                                                SilenceThreshold=0.03)
