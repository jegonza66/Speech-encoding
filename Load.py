# Standard libraries
import numpy as np, pandas as pd, os, warnings
import mne, librosa, platform, opensmile, textgrids

# Specific libraries
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
from praatio import pitch_and_intensity

# Modules
import Processing, Funciones, setup

# Review this If we want to update packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

exp_info = setup.exp_info()


class Trial_channel:

    def __init__(self, s:int=21, trial:int=1, channel:int=1, Band:str='All', sr:float=128, tmin:float=-0.6,
                 tmax:float=-0.003, valores_faltantes:int=0, Causal_filter_EEG:bool=True, 
                 Env_Filter:bool=False, SilenceThreshold:float=0.03):
        """Extract transcriptions, audio signal and EEG signal and calculates specified features.

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
        allowed_band_frecuencies = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        if Band in allowed_band_frecuencies:
            self.Band= Band
        else:
            raise SyntaxError(f"{Band} is not an allowed band frecuency. Allowed bands are: {allowed_band_frecuencies}")

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

        # Relevant paths
        self.eeg_fname = "Datos/EEG/S" + str(s) + "/s" + str(s) + "-" + str(channel) + "-Trial" + str(
            trial) + "-Deci-Filter-Trim-ICA-Pruned.set"
        self.wav_fname = "Datos/wavs/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(channel) + ".wav"
        self.pitch_fname = "Datos/Pitch_threshold_{}/S".format(SilenceThreshold) + str(s) + "/s" + str(s) + ".objects." \
                           + "{:02d}".format(trial) + ".channel" + str(channel) + ".txt"
        self.phn_fname = "Datos/phonemes/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(channel) + ".aligned_fa.TextGrid"
        self.phn_fname_manual = "Datos/phonemes/S" + str(s) + "/manual/s" + str(s) + "_objects_" + "{:02d}".format(
            trial) + "_channel" + str(channel) + "_aligned_faTAMARA.TextGrid"
        self.phrases_fname = "Datos/phrases/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(
            channel) + ".phrases"

    def f_phonemes(self, envelope:np.ndarray):
        """It makes a time-match between the phonemes and the envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        pandas.DataFrame
            Columns given by phonemes and index given by samples. The values are the amplitude of the envolpe at the given sample.
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

        # Make empty df of phonemes #TODO: change to dictionary, much faster
        phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=exp_info.ph_labels)

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)
        
        # Match phoneme with envelope
        for i, phoneme in enumerate(phonemes_tgrid):
            phonemes_df.loc[i, phoneme] = envelope[i]

        return phonemes_df
        
    def f_phonemes_discrete(self, envelope:np.ndarray): #TODO: es exactamente igual a f_phonemes, salvo las últimas tres lineas. UNIFICAR
        """It makes a time-match between the phonemes and the envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        pandas.DataFrame
            Columns given by phonemes and index given by samples. The value of a given element is 1 if the phoneme is being pronounced and 0 elsewise.
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

        # Make empty df of phonemes #TODO: change to dictionary, much faster
        phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=exp_info.ph_labels)

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)

        for i, phoneme in enumerate(phonemes_tgrid):
            phonemes_df.loc[i, phoneme] = 1

        return phonemes_df
 
    def f_phonemes_onset(self, envelope:np.ndarray): #TODO: es exactamente igual a f_phonemes, salvo las últimas tres lineas. UNIFICAR
        """It makes a time-match between the phonemes and the envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        pandas.DataFrame
            Columns given by phonemes and index given by samples. The value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
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

        # Make empty df of phonemes #TODO: change to dictionary, much faster
        phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=exp_info.ph_labels)

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)

        # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
        phonemes_onset = [phonemes_tgrid[0]]
        
        for i in range(1, len(phonemes_tgrid)):
            if phonemes_tgrid[i] == phonemes_tgrid[i-1]:
                phonemes_onset.append(0)
            else:
                phonemes_onset.append(phonemes_tgrid[i])
        
        for i, phoneme in enumerate(phonemes_onset):
            if phoneme != 0:
                phonemes_df.loc[i, phoneme] = 1
        return phonemes_df

    def phn_features(self):# TODO: no se entiende que es esto, tal vez algo que queda desarrollar
        x=1

    def f_phonemes_manual(self, envelope:np.ndarray):#TODO: es muy parecida a f_phonemes, salvo las últimas tres lineas. UNIFICAR
        """It makes a time-match between the phonemes and the envelope. But it selects manually exclutions for certain phonemes.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        pandas.DataFrame
            Columns given by phonemes and index given by samples. The values are the amplitude of the envolpe at the given sample.
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

        # Make empty df of phonemes #TODO: change to dictionary, much faster
        phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=exp_info.ph_labels)

        # Repeat each label the number of times it was sampled
        phonemes_tgrid = np.repeat(labels, samples)

        for i, phoneme in enumerate(phonemes_tgrid):
            phonemes_df.loc[i, phoneme] = 1

        return phonemes_df

    def f_eeg(self):
        """Extract eeg file to an array and downsample it to get the same rate as self.sr.

        Returns
        -------
        np.ndarray
            Downsampled EEG signal.
        """
        # Read the .set file
        eeg = mne.io.read_raw_eeglab(input_fname=self.eeg_fname, verbose=False) # TODO: warning of annotations and 'boundry' events -data discontinuities-.
        
        # Sample frequency in Hz
        eeg_freq = eeg.info.get("sfreq")
        
        eeg.load_data(verbose=False)
        # eeg = eeg.set_eeg_reference(ref_channels='average', projection=False)

        # Independent sources
        # eeg = mne.preprocessing.compute_current_source_density(eeg)

        # Apply a lowpass filter
        if self.Band:
            if self.Causal_filter_EEG:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, phase='minimum', verbose=False) # TODO: preguntar la diferencia causal_filter False, tambien quite el verbose, preguntar si esta ok
            else:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, verbose=False)

        # Convert to numpy array, excluding the time column. This gives a matrix of dimension timesxchannels
        eeg = eeg.to_data_frame()
        eeg = np.array(eeg)[:, 1:129]  

        # Downsample to get same rate as sr
        eeg = Processing.subsamplear(eeg, int(eeg_freq / self.sr))

        return np.array(eeg)

    def f_info(self):
        """Define a descriptor for the set up

        Returns
        -------
        mne.io.meas_info.Info
            Description of measurment.
        """
        # Define montage and info object
        montage = mne.channels.make_standard_montage('biosemi128')
        channel_names = montage.ch_names
        info = mne.create_info(ch_names=channel_names[:], sfreq=self.sr, ch_types='eeg').set_montage(montage)
    
        return info

    def f_envelope(self): #TODO check self.envelope assignation
        """Takes the low pass filtered -butterworth-, downsample and smoothened envelope of .wav file.# TODO: chequear

        Returns
        -------
        np.ndarray
            Envelope of wav signal.
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")

        # Calculate envelope
        envelope = np.abs(sgn.hilbert(wav))
        
        # Apply lowpass butterworth filter
        if self.Env_Filter == 'Causal':
            envelope = Processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='Causal')
        elif self.Env_Filter == 'NonCausal':
            envelope = Processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='NonCausal')
        
        # Smooth and downsample the envelope taking averages  # TODO: preguntar si esto es para suavizar la función
        window_size, stride = 125, 125
        envelope = np.array([np.mean(envelope[i:i + window_size]) for i in range(0, len(envelope), stride) if
                             i + window_size <= len(envelope)])
        
        self.envelope = envelope.ravel().flatten()
        # envelope_mat = Processing.matriz_shifteada(envelope, self.delays)  # armo la matriz shifteada
        return np.array(envelope)

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

    def load_pitch(self, envelope:np.ndarray): #TODO: f_calculate_pitch can store the final variable (a list), hay preguntas en la parte final
        """Loads the pitch of the speaker

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        np.ndarray
            Pitch of the speaker signal.
        """
        # Reads the file anf transform it in a np.ndarray
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

        # TODO: preguntar para qué esta esto
        # Repeat each value of the pitch to make audio_sr*sampleStep times
        pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=np.float32)
        
        pitch = Processing.subsamplear(pitch, 125)
        # pitch = Processing.matriz_shifteada(pitch, self.delays)
        pitch = pitch[:min(len(pitch), len(envelope))]
        # pitch_der = Processing.matriz_shifteada(pitch_der, self.delays)

        return np.array(pitch)

    def f_jitter_shimmer(self, envelope:np.ndarray):
        """Gives the jitter and shimmer matching the size of the envelope

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        tuple
            jitter and shimmer.
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
        
        # Calculate the least common multiple between envelope and jitter lengths
        mcm = Funciones.minimo_comun_multiplo(len(jitter), len(envelope))
        
        # Repeat each value the number of times it takes the length of jitter to achive the mcm. The result is that jitter length matches mcm
        jitter = np.repeat(jitter, mcm / len(jitter))

        # Subsample by the number of times it takes the length of the envelope to achive the mcm. Now it has exactly the same size as envelope
        jitter = Processing.subsamplear(jitter, mcm / len(envelope))
        # jitter = Processing.matriz_shifteada(jitter, self.delays)

        # Reassurance that the count is correct
        jitter = jitter[:min(len(jitter), len(envelope))]

        # Apply the same process with shimmer
        mcm = Funciones.minimo_comun_multiplo(len(shimmer), len(envelope))
        shimmer = np.repeat(shimmer, mcm / len(shimmer))
        shimmer = Processing.subsamplear(shimmer, mcm / len(envelope))
        # shimmer = Processing.matriz_shifteada(shimmer, self.delays)
        shimmer = shimmer[:min(len(shimmer), len(envelope))]

        return jitter, shimmer

    # def f_cssp(self):

    # snd = parselmouth.Sound(self.wav_fname)
    # data = []
    # frame_length = 0.2
    # hop_length = 1/128
    # t1s = np.arange(0, snd.duration - frame_length, hop_length)
    # times = zip(t1s, t1s + frame_length)
    #
    # for t1, t2 in times:
    #     powercepstrogram = call(snd.extract_part(t1, t2), "To PowerCepstrogram", 60, 0.0020001, 5000, 50)
    #     cpps = call(powercepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0,
    #                 "Exponential decay", "Robust")
    #     data.append(cpps)
    #
    # cssp = np.array(np.repeat(data, self.audio_sr * self.sampleStep))
    # cssp = Processing.subsamplear(cssp, 125)
    # cssp = Processing.matriz_shifteada(cssp, self.delays)

    # return cssp

    def f_spectrogram(self, envelope:np.ndarray):
        """Calculates spectrogram of .wav file between 16 Mel frecuencies.

        Parameters
        ----------
        envelope : np.ndarray
            Envelope of the audio signal using Hilbert transform.

        Returns
        -------
        np.ndarray
            _description_
        """
        # Read file
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")
        
        # Calculates the mel frecuencies spectrogram
        n_fft, hop_length, n_mels = 125, 125, 16
        S = librosa.feature.melspectrogram(y=wav,
                                           sr=self.audio_sr, 
                                           n_fft=n_fft, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels)
        
        # Transform to dB using normalization to 1
        S_DB = librosa.power_to_db(S=S, ref=np.max)
        S_DB = S_DB.transpose()

        # Match to envelope size if shorter to standarized across features
        S_DB = S_DB[:min(len(S_DB), len(envelope)), :]

        # # Shifted matrix row by row
        # spec_shift = Processing.matriz_shifteada(S_DB[0], self.delays)
        # for i in np.arange(1, len(S_DB)):
        #     spec_shift = np.hstack((spec_shift, Processing.matriz_shifteada(S_DB[i], self.delays)))
        return np.array(S_DB)

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
            Dictionary with EEG, info and specified stimuli.
        """
        channel = {}
        channel['EEG'] = self.f_eeg()
        channel['info'] = self.f_info()

        if 'Envelope' in stims:
            channel['Envelope'] = self.f_envelope()
        if 'Pitch' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Pitch'] = self.load_pitch(envelope=self.envelope)
        if 'Spectrogram' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Spectrogram'] = self.f_spectrogram(envelope=self.envelope)
        if 'Phonemes' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Phonemes'] = self.f_phonemes(envelope=self.envelope)
        if 'Phonemes-manual' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Phonemes-manual'] = self.f_phonemes_manual(envelope=self.envelope)
        if 'Phonemes-discrete' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Phonemes-discrete'] = self.f_phonemes_discrete(envelope=self.envelope)
        if 'Phonemes-onset' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Phonemes-onset'] = self.f_phonemes_onset(envelope=self.envelope)

        return channel

class Sesion_class:
    def __init__(self, sesion:int=21, stim:str='Envelope', Band:str='All', sr:float=128, tmin:float=-0.6, 
                 tmax:float=-0.003, valores_faltantes:int=0, Causal_filter_EEG:bool=True, Env_Filter:bool=False,
                 situacion:str='Escucha', Calculate_pitch:bool=False, SilenceThreshold:float=0.03,
                 procesed_data_path:str=f'saves/Preprocesed_Data/tmin{-0.6}_tmax{-.003}/'
                 ):
        """_summary_ #TODO falta llenar

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
        # Check if band, stimuli and situacion parameters where passed right
        allowed_stims = ['Envelope','Pitch','PitchMask','Spectrogram','Phonemes','Phonemes-manual','Phonemes-discrete','Phonemes-onset']
        allowed_band_frecuencies = ['Delta','Theta','Alpha','Beta_1','Beta_2','All','Delta_Theta','Delta_Theta_Alpha']
        allowed_situaciones = ['Habla_Propia','Ambos_Habla','Escucha']
        for st in stim.split('_'):
            if st in allowed_stims:
                pass
            else:
                raise SyntaxError(f"{st} is not an allowed stimulus. Allowed stimuli are: {allowed_stims}. If more than one stimulus is wanted, the separator should be '_'.")
        self.stim = stim
        if Band in allowed_band_frecuencies:
            self.Band = Band
        else:
            raise SyntaxError(f"{Band} is not an allowed band frecuency. Allowed bands are: {allowed_band_frecuencies}")
        if situacion in allowed_situaciones:
            self.situacion = situacion
        else:
            raise SyntaxError(f"{situacion} is not an allowed situation. Allowed situations are: {allowed_situaciones}")
        
        # Define the rest of the parameters
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
        self.procesed_data_path = procesed_data_path
        self.samples_info_path = self.procesed_data_path + 'samples_info/Sit_{}/'.format(self.situacion)
        self.phn_path = "Datos/phonemes/S" + str(self.sesion) + "/"
        self.phrases_path = "Datos/phrases/S" + str(self.sesion) + "/"

    def load_from_raw(self): # TODO: chequear si dataframes es lo más eficiente
        """Loads raw data, this includes EEG, info and stimuli.

        Returns
        -------
        dict
            Sessions of both subjects
        """

        # Subjects dictionaries, stores their data
        Sujeto_1 = {'EEG': pd.DataFrame()}
        Sujeto_2 = {'EEG': pd.DataFrame()}

        for stimuli in self.stim.split('_'):
            Sujeto_1[stimuli] = pd.DataFrame()
            Sujeto_2[stimuli] = pd.DataFrame()
        
        # Try to open preprocessed data, if not opens raw
        try:
            samples_info = Funciones.load_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl')
            loaded_samples_info = True
        except:
            loaded_samples_info = False
            samples_info = {'keep_indexes1': [], 'keep_indexes2': [], 'trial_lengths1': [0], 'trial_lengths2': [0]}

        # Retrive number of files, i.e: trials
        trials = list(set([int(fname.split('.')[2]) for fname in os.listdir(self.phrases_path) if os.path.isfile(self.phrases_path + f'/{fname}')]))
        
        for trial in trials:
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
                if self.Calculate_pitch:
                    channel_1.f_calculate_pitch()
                    channel_2.f_calculate_pitch()

                # Extract dictionaries with the data
                Trial_channel_1 = channel_1.load_trial(stims=self.stim.split('_'))
                Trial_channel_2 = channel_2.load_trial(stims=self.stim.split('_'))
                
                if self.situacion == 'Habla_Propia' or self.situacion == 'Ambos_Habla':
                    # Load data to dictionary taking stimuli and eeg from speaker. I.e: each subject predicts its own EEG # TODO: la definición es exactamente Trial Channel1 (chequeado)
                    Trial_sujeto_1 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys()}
                    Trial_sujeto_2 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys()}

                else:
                    # Load data to dictionary taking stimuli from speaker and eeg from listener. I.e: predicts own EEG using stimuli from interlocutor
                    Trial_sujeto_1 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys()} # TODO: la definición es exactamente Trial Channel2 (chequeado)
                    Trial_sujeto_1['EEG'] = Trial_channel_1['EEG']

                    Trial_sujeto_2 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys()}# TODO: la definición es exactamente Trial Channel1 (chequeado)
                    Trial_sujeto_2['EEG'] = Trial_channel_2['EEG']

                # Labeling of current speaker. {3:both_speaking,2:speaks_listener,3:speaks_interlocutor,0:silence}
                momentos_sujeto_1_trial = Processing.labeling(s=self.sesion, trial=trial, canal_hablante=2, sr=self.sr)
                momentos_sujeto_2_trial = Processing.labeling(s=self.sesion, trial=trial, canal_hablante=1, sr=self.sr)

                # Match length of momentos and trials with the info of its lengths
                if loaded_samples_info:
                    _ = Funciones.igualar_largos_dict_sample_data(dict=Trial_sujeto_1, momentos=momentos_sujeto_1_trial,
                                                                  minimo_largo=samples_info['trial_lengths1'][trial])
                    _ = Funciones.igualar_largos_dict_sample_data(dict=Trial_sujeto_2, momentos=momentos_sujeto_2_trial,
                                                                  minimo_largo=samples_info['trial_lengths2'][trial])

                else:
                    # If there isn't any data matches lengths of both variables comparing every key and momentos length (the Trial gets modify inside the function)
                    momentos_sujeto_1_trial, minimo_largo1 = Funciones.igualar_largos_dict(Trial_sujeto_1, momentos_sujeto_1_trial)
                    momentos_sujeto_2_trial, minimo_largo2 = Funciones.igualar_largos_dict(Trial_sujeto_2, momentos_sujeto_2_trial)

                    samples_info['trial_lengths1'].append(minimo_largo1)
                    samples_info['trial_lengths2'].append(minimo_largo2)

                    # Preprocessing: calaculates the relevant indexes for the apropiate analysis
                    keep_indexes1_trial = Processing.preproc_dict(momentos_escucha=momentos_sujeto_1_trial, delays=self.delays,
                                            situacion=self.situacion)
                    keep_indexes2_trial = Processing.preproc_dict(momentos_escucha=momentos_sujeto_2_trial, delays=self.delays,
                                            situacion=self.situacion)

                    # Add sum of all previous trials length
                    keep_indexes1_trial += np.sum(samples_info['trial_lengths1'][:-1])
                    keep_indexes2_trial += np.sum(samples_info['trial_lengths2'][:-1])

                    samples_info['keep_indexes1'].append(keep_indexes1_trial)
                    samples_info['keep_indexes2'].append(keep_indexes2_trial)

                # Turn dictionaries to DataFrame
                Funciones.make_df_dict(Trial_sujeto_1) #TODO por qué se excluye 'Phonemes'
                Funciones.make_df_dict(Trial_sujeto_2)

                # Append analysis to each subject
                if len(Trial_sujeto_1['EEG']):
                    for key in list(Sujeto_1.keys()):
                        Sujeto_1[key] = Sujeto_1[key].append(Trial_sujeto_1[key])

                if len(Trial_sujeto_2['EEG']):
                    for key in list(Sujeto_2.keys()):
                        Sujeto_2[key] = Sujeto_2[key].append(Trial_sujeto_2[key])

            except:
                # Empty trial
                samples_info['trial_lengths1'].append(0)
                samples_info['trial_lengths2'].append(0)
                # samples_info['keep_indexes1'].append([0])
                # samples_info['keep_indexes2'].append([0])

        info = Trial_channel_1['info']

        # Save instants Data
        if not loaded_samples_info:
            samples_info['keep_indexes1'] = Funciones.flatten_list(samples_info['keep_indexes1'])
            samples_info['keep_indexes2'] = Funciones.flatten_list(samples_info['keep_indexes2'])
            os.makedirs(self.samples_info_path, exist_ok=True)
            Funciones.dump_pickle(path=self.samples_info_path + f'samples_info_{self.sesion}.pkl', obj=samples_info, rewrite=True)

        # Drops silence phoneme column
        for key in Sujeto_1.keys():
            if st.startswith('Phonemes'):
                Sujeto_1[st].drop(columns="", inplace=True)
                Sujeto_2[st].drop(columns="", inplace=True)

        # Convert elements of dicts np.ndarrays
        Funciones.make_array_dict(Sujeto_1)
        Funciones.make_array_dict(Sujeto_2)

        # Construct shifted matrix
        specific_stimuli = ['Spectrogram', 'Phonemes', 'Phonemes-manual', 'Phonemes-discrete', 'Phonemes-onset']
        for st in specific_stimuli:
            if st in Sujeto_1.keys():
                Sujeto_1[st] = Sujeto_1[st].transpose()
                Sujeto_2[st] = Sujeto_2[st].transpose()
            
                # Shifted matrix row by row
                print(f'Computing shifted matrix for the {st}')
                shift_1 = Processing.matriz_shifteada(Sujeto_1[st][0], self.delays)
                shift_2 = Processing.matriz_shifteada(Sujeto_2[st][0], self.delays)
                for i in np.arange(1, len(Sujeto_1[st])):
                    shift_1 = np.hstack((shift_1, Processing.matriz_shifteada(Sujeto_1[st][i], self.delays)))
                for i in np.arange(1, len(Sujeto_2[st])):
                    shift_2 = np.hstack((shift_2, Processing.matriz_shifteada(Sujeto_2[st][i], self.delays)))
                Sujeto_1[st], Sujeto_2[st] = shift_1, shift_2

        for key in Sujeto_1.keys(): # TODO esto dice básicamente que la iteración anterior se podía hacer sobre todas las keys salvo EEG y que para algunas en particular se hago el stack
            if key not in specific_stimuli and key!= 'EEG':
                Sujeto_1[key] = Processing.matriz_shifteada(Sujeto_1[key], self.delays)
                Sujeto_2[key] = Processing.matriz_shifteada(Sujeto_2[key], self.delays)

            # Keep good time instants
            Sujeto_1[key] = Sujeto_1[key][samples_info['keep_indexes1'], :]
            Sujeto_2[key] = Sujeto_2[key][samples_info['keep_indexes2'], :]

        # Define paths to export data
        Paths = {}
        if self.Band and self.Causal_filter_EEG:
            Paths['EEG'] = self.procesed_data_path + f'EEG/Causal_Sit_{self.situacion}_Band_{self.Band}/'
        else:
            Paths['EEG'] = self.procesed_data_path + f'EEG/Sit_{self.situacion}_Band_{self.Band}/'
        Paths['Envelope'] = self.procesed_data_path + f'Envelope/Sit_{self.situacion}/'
        Paths['Pitch'] = self.procesed_data_path + f'Pitch_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/'
        Paths['Spectrogram'] = self.procesed_data_path + f'Spectrogram/Sit_{self.situacion}/'
        Paths['Phonemes'] = self.procesed_data_path + f'Phonemes/Sit_{self.situacion}/'
        Paths['Phonemes-manual'] = self.procesed_data_path + f'Phonemes-manual/Sit_{self.situacion}/'
        Paths['Phonemes-discrete'] = self.procesed_data_path + f'Phonemes-discrete/Sit_{self.situacion}/'
        Paths['Phonemes-onset'] = self.procesed_data_path + f'Phonemes-onset/Sit_{self.situacion}/'

        # Save preprocesed data
        for key in Sujeto_1.keys():
            os.makedirs(Paths[key], exist_ok=True)
            Funciones.dump_pickle(path=Paths[key] + f'Sesion{self.sesion}.pkl', obj=[Sujeto_1[key], Sujeto_2[key]], rewrite=True)
        
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
        EEG_path = self.procesed_data_path + 'EEG/'
        if self.Band and self.Causal_filter_EEG: 
            EEG_path += 'Causal_'
        EEG_path += f'Sit_{self.situacion}_Band_{self.Band}/Sesion{self.sesion}.pkl'

        # Load EEGs and procesed data
        eeg_sujeto_1, eeg_sujeto_2 = Funciones.load_pickle(path=EEG_path)
        info = Funciones.load_pickle(path=self.procesed_data_path + 'EEG/info.pkl')
        Sujeto_1 = {'EEG': eeg_sujeto_1, 'info': info}
        Sujeto_2 = {'EEG': eeg_sujeto_2, 'info': info}

        # Define stimuli paths
        stim_paths = {
            'Envelope':self.procesed_data_path + 'Envelope/'+ f'Sit_{self.situacion}/Sesion{self.sesion}.pkl',
            'Pitch':self.procesed_data_path + f'Pitch_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/Sesion{self.sesion}.pkl',
            'PitchMask':self.procesed_data_path + f'Pitch_mask_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/Sesion{self.sesion}.pkl',
            'Spectrogram':self.procesed_data_path + f'Spectrogram/Sit_{self.situacion}/Sesion{self.sesion}.pkl',
            'Phonemes':self.procesed_data_path + f'Phonemes/Sit_{self.situacion}/Sesion{self.sesion}.pkl', 
            'Phonemes-manual':self.procesed_data_path + f'Phonemes-manual/Sit_{self.situacion}/Sesion{self.sesion}.pkl',
            'Phonemes-discrete':self.procesed_data_path + f'Phonemes-discrete/Sit_{self.situacion}/Sesion{self.sesion}.pkl',
            'Phonemes-onset':self.procesed_data_path + f'Phonemes-onset/Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        }
        if self.Env_Filter:
            stim_paths['Envelope'] = self.procesed_data_path + 'Envelope/' + self.Env_Filter + '_' + f'Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        
        # Loads stimuli to each subject
        for stimuli in self.stim.split('_'):
            Sujeto_1[stimuli], Sujeto_2[stimuli] = Funciones.load_pickle(path=stim_paths[stimuli])
            if stimuli == 'Pitch':
                # Remove missing values
                if self.valores_faltantes == None: #TODO CREO QUE ACA VA 0
                    Sujeto_1[stimuli], Sujeto_2[stimuli] = Sujeto_1[stimuli][Sujeto_1[stimuli]!=0], Sujeto_2[stimuli][Sujeto_2[stimuli]!=0]
                elif self.valores_faltantes:
                    Sujeto_1[stimuli], Sujeto_2[stimuli] = Sujeto_1[stimuli][Sujeto_1[stimuli]==0], Sujeto_2[stimuli][Sujeto_2[stimuli]==0]

        # for stimuli in self.stim.split('_'): #TODO si funciona lo de arriba borrar
        #     if stimuli == 'Envelope':
        #         Envelope_path = self.procesed_data_path + 'Envelope/'
        #         if self.Env_Filter: Envelope_path += self.Env_Filter + '_'
        #         Envelope_path += 'Sit_{}/Sesion{}.pkl'.format(self.situacion, self.sesion)
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(path=Envelope_path)

        #     if stimuli == 'Pitch':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Pitch_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/Sesion{self.sesion}.pkl'
        #         )
                
        #         if self.valores_faltantes == None:
        #             stimuli_para_sujeto_1, stimuli_para_sujeto_2 = stimuli_para_sujeto_1[stimuli_para_sujeto_1 != 0], \
        #                                                            stimuli_para_sujeto_2[
        #                                                                stimuli_para_sujeto_2 != 0]  # saco 0s
        #         elif self.valores_faltantes:
        #             stimuli_para_sujeto_1[stimuli_para_sujeto_1 == 0], stimuli_para_sujeto_2[
        #                 stimuli_para_sujeto_2 == 0] = self.valores_faltantes, self.valores_faltantes  # cambio 0s

        #     if stimuli == 'PitchMask':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Pitch_mask_threshold_{self.SilenceThreshold}/Sit_{self.situacion}_Faltantes_{self.valores_faltantes}/Sesion{self.sesion}.pkl'
        #         )

        #     if stimuli == 'Spectrogram':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Spectrogram/Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        #         )

        #     if stimuli == 'Phonemes':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Phonemes/Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        #         )
                
        #     if stimuli == 'Phonemes-manual':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Phonemes-manual/Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        #         )
                
        #     if stimuli == 'Phonemes-discrete':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Phonemes-discrete/Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        #         )
                
        #     if stimuli == 'Phonemes-onset':
        #         stimuli_para_sujeto_1, stimuli_para_sujeto_2 = Funciones.load_pickle(
        #         path=self.procesed_data_path + f'Phonemes-onset/Sit_{self.situacion}/Sesion{self.sesion}.pkl'
        #         )
                
            # Sujeto_1[stimuli] = stimuli_para_sujeto_1
            # Sujeto_2[stimuli] = stimuli_para_sujeto_2
        return {'Sujeto_1': Sujeto_1, 'Sujeto_2': Sujeto_2}

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
    allowed_stims = ['Envelope', 'Pitch', 'PitchMask', 'Spectrogram', 'Phonemes', 'Phonemes-manual',
                      'Phonemes-discrete', 'Phonemes-onset']

    if all(stimulus in allowed_stims for stimulus in stim.split('_')):
        Sesion_obj = Sesion_class(sesion=sesion, 
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
            Sesion = Sesion_obj.load_procesed()
        except:
            Sesion = Sesion_obj.load_from_raw()

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

    for stimuli in stim.split('_'):
        dfinal_para_sujeto_1.append(Sujeto_1[stimuli])
        dfinal_para_sujeto_2.append(Sujeto_2[stimuli])

    return dfinal_para_sujeto_1, dfinal_para_sujeto_2