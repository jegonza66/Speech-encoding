# Standard libraries
import numpy as np, pandas as pd, os, warnings, matplotlib.pyplot as plt
import mne, librosa, librosa.display, platform, opensmile, textgrids

# Specific libraries
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
from praatio import pitch_and_intensity

# Modules
import processing, funciones, setup

# Review this If we want to update packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(verbose='CRITICAL')
exp_info = setup.exp_info()

# =====================
# PHONOlOGICAL FEATURES
phonologicalf=Phonological()
file_audio=r'Datos/wavs/S21/s21.objects.01.channel1.wav'
feats=phonologicalf.extract_features_file(file_audio, static=False, plots=True, fmt="dataframe")
print(feats)
feats.columns
feats.head()
plt.figure()
plt.plot(feats['time'], feats['labial'], label = 'labial')
plt.plot(feats['time'], feats['labial'], label = 'labial')
plt.grid(True)
plt.xlabel('Time (s)')
plt.show()

sr = 128
number_of_samples_between_times = np.diff(feats['time'].values)*sr

# #==========
# # FUNCTIONS
# def f_phonemes(envelope:np.ndarray, kind:str='Phonemes-Envelope-Manual'):
#         """It makes a time-match matrix between the phonemes and the envelope. The values and shape of given matrix depend on kind.

#         Parameters
#         ----------
#         envelope : np.ndarray
#             Envelope of the audio signal using Hilbert transform.
#         kind : str, optional
#            Kind of phoneme matrix to use, by default 'Envelope'. Available kinds are:
#             'Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete', 'Phonemes-Discrete-Manual', 'Phonemes-Onset', 'Phonemes-Onset-Manual'

#         Returns
#         -------
#         np.ndarray
#             if kind.startswith('Phonemes-Envelope'):
#                 Matrix with envelope amplitude at given sample. The matrix dimension is SamplesXPhonemes_labels(in order)
#             elif kind.startswith('Phonemes-Discrete'):
#                 Also a matrix but it has 1s and 0s instead of envelope amplitude.
#             elif kind.startswith('Phonemes-Onset'):
#                 In this case the value of a given element is 1 just if its the first time is being pronounced and 0 elsewise. It doesn't repeat till the following phoneme is pronounced.
            
#         Raises
#         ------
#         SyntaxError
#             Whether the input value of 'kind' is passed correctly. It must be a string among ['Envelope', 'Discrete', 'Onset'].
#         """
#         if kind.endswith('anual'):
#             exp_info_labels = exp_info.ph_labels_man
#         else: 
#             exp_info_labels = exp_info.ph_labels            

#         # Check if given kind is a permited input value
#         allowed_kind = ['Phonemes-Envelope', 'Phonemes-Envelope-Manual', 'Phonemes-Discrete', 'Phonemes-Discrete-Manual', 'Phonemes-Onset', 'Phonemes-Onset-Manual']
#         if kind not in allowed_kind:
#             raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phonemes are: {allowed_kind}")

#         # Get trial total time length
#         phrases = pd.read_table(r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\phrases\S21\s21.objects.01.channel1.phrases', header=None, sep="\t")
#         trial_tmax = phrases[1].iloc[-1]

#         # Load transcription
#         grid = textgrids.TextGrid(r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\phonemes\S21\s21.objects.01.channel1.aligned_fa.TextGrid')

#         # Get phonemes
#         phonemes_grid = grid['transcription : phones']

#         # Extend first silence time to trial start time
#         phonemes_grid[0].xmin = 0.

#         # Parse for labels, times and number of samples within each phoneme
#         labels = []
#         times = []
#         samples = []
        
#         for ph in phonemes_grid:
#             label = ph.text.transcode()
#             label = label.replace(' ', '')
#             label = label.replace('º', '')
#             label = label.replace('-', '')

#             # Rename silences
#             if label in ['sil','sp','sile','silsil','SP','s¡p','sils']:
#                 label = ""
            
#             # Check if the phoneme is in the list
#             if not(label in exp_info_labels or label==""):
#                 print(f'"{label}" is not in not a recognized phoneme. Will be added as silence.')
#                 label = ""
#             labels.append(label)
#             times.append((ph.xmin, ph.xmax))
#             samples.append(np.round((ph.xmax - ph.xmin) * 128).astype("int"))


#         # Extend on more phoneme of silence till end of trial 
#         labels.append("")
#         times.append((ph.xmin, trial_tmax))
#         samples.append(np.round((trial_tmax - ph.xmax) *128).astype("int"))


#         # If use envelope amplitude to make continuous stimuli: the total number of samples must match the samples use for stimuli
#         diferencia = np.sum(samples) - len(envelope)

#         if diferencia > 0:
#             # Making the way back checking when does the number of samples of the ith phoneme exceed diferencia
#             for ith_phoneme in [-i-1 for i in range(len(samples))]:
#                 if diferencia > samples[ith_phoneme]:
#                     diferencia -= samples[ith_phoneme]
#                     samples[ith_phoneme] = 0
#                 # When samples is greater than the difference, takes the remaining samples to match the envelope
#                 else:
#                     samples[ith_phoneme] -= diferencia
#                     break
#         elif diferencia < 0:
#             # In this case, the last silence is prolonged
#             samples[-1] -= diferencia
        
#         # Make a list with phoneme labels tha already are in the known set
#         updated_taggs = exp_info_labels + [ph for ph in np.unique(labels) if ph not in exp_info_labels]

#         # Repeat each label the number of times it was sampled
#         phonemes_tgrid = np.repeat(labels, samples)
        
#         # Make empty array of phonemes
#         phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
#         # Match phoneme with kind
#         if kind.startswith('Phonemes-Envelope'):
#             for i, tagg in enumerate(phonemes_tgrid):
#                 phonemes[i, updated_taggs.index(tagg)] = envelope[i]
#         elif kind.startswith('Phonemes-Discrete'):
#             for i, tagg in enumerate(phonemes_tgrid):
#                 phonemes[i, updated_taggs.index(tagg)] = 1
#         elif kind.startswith('Phonemes-Onset'):
#             # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
#             phonemes_onset = [phonemes_tgrid[0]]
#             for i in range(1, len(phonemes_tgrid)):
#                 if phonemes_tgrid[i] == phonemes_tgrid[i-1]:
#                     phonemes_onset.append(0)
#                 else:
#                     phonemes_onset.append(phonemes_tgrid[i])
#             # Match phoneme with envelope
#             for i, tagg in enumerate(phonemes_onset):
#                 if tagg!=0:
#                     phonemes[i, updated_taggs.index(tagg)] = 1
#         return phonemes

# #=========
# # ENVELOPE
# wav = wavfile.read(r'Datos/wavs/S21/s21.objects.01.channel1.wav')[1]
# wav = wav.astype("float")

# # Calculate envelope
# envelope = np.abs(sgn.hilbert(wav))

# # Resample
# window_size, stride = 125, 125
# envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)]).reshape(-1,1)

# # =====
# # PITCH

# # HIPER PARAMS
# silence_threshold = .03
# minPitch = 50
# maxPitch = 300
# sampleStep = 1/128
# praat_executable_path = 'C:/Program Files/Praat/Praat.exe'
# output_folder = f"Datos/PRUEBA_threshold_{silence_threshold}"
# os.makedirs(output_folder, exist_ok=True)


# wav_fname = r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\wavs\S21\s21.objects.01.channel1.wav'
# pitch_fname = r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\PRUEBA_threshold_0.03\S21\s21.objects.01.channel1.txt'
# pitch_and_intensity.extractPI(inputFN=os.path.abspath(wav_fname), 
#                             outputFN=os.path.abspath(pitch_fname), 
#                             praatEXE=praat_executable_path, 
#                             minPitch=minPitch,
#                             maxPitch=maxPitch, 
#                             sampleStep=sampleStep, 
#                             silenceThreshold=silence_threshold,
#                             pitchQuadInterp=False)
    
# # Loads data
# data = np.genfromtxt(os.path.abspath(pitch_fname), dtype=np.float, delimiter=',', missing_values='--undefined--', filling_values=np.inf)
# time, pitch = data[:, 0], data[:, 1]

# # Get defined indexes
# defined_indexes = np.where(pitch!=np.inf)[0]

# # Approximated window size
# window_size = 100e-3
# n_steps_in_window = np.ceil(window_size/sampleStep)
# window_size = n_steps_in_window*sampleStep


# # Interpolate relevant moments of silence
# logpitch = np.log(pitch)

# # lista_aux = []
# # for i in range(len(defined_indexes)):
# #     if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
# #         logpitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(x=time[defined_indexes[i-1]+1:defined_indexes[i]], xp=time[defined_indexes], fp=logpitch[defined_indexes])
# #         lista_aux+=list(np.arange(defined_indexes[i-1]+1,defined_indexes[i],1))

# # Load phonemes matrix, excluding '' labela
# phonemes = f_phonemes(envelope=envelope, kind='Phonemes-Discrete-Manual')
# phonemes = np.delete(arr=phonemes, obj=-1, axis=1)

# # Given that spacing between samples is 1/self.sr
# sound_indexes = np.where(phonemes.any(axis=1))[0]

# # Within window of window_size of said phoneme if there is a silence it gets interpoled

# # lista_aux = []
# # for i in range(len(defined_indexes)):
# #     if 1<(defined_indexes[i]-defined_indexes[i-1])<=n_steps_in_window:
# #         logpitch[defined_indexes[i-1]+1:defined_indexes[i]] = np.interp(x=time[defined_indexes[i-1]+1:defined_indexes[i]], xp=time[defined_indexes], fp=logpitch[defined_indexes])
# #         lista_aux+=list(np.arange(defined_indexes[i-1]+1,defined_indexes[i],1))

# # Load phonemes matrix, excluding '' labela
# phonemes = f_phonemes(envelope=envelope, kind='Phonemes-Discrete-Manual')
# phonemes = np.delete(arr=phonemes, obj=-1, axis=1)

# # Given that spacing between samples is 1/self.sr
# sound_indexes = np.where(phonemes.any(axis=1))[0]

# # Within window of window_size of said phoneme if there is a silence it gets interpoled
# lista_aux = []
# for i in range(len(sound_indexes)):
#     if 1<(sound_indexes[i]-sound_indexes[i-1])<=n_steps_in_window:
#         logpitch[sound_indexes[i-1]+1:sound_indexes[i]] = np.interp(x=time[sound_indexes[i-1]+1:sound_indexes[i]], xp=time[defined_indexes], fp=logpitch[defined_indexes])
#         lista_aux+=list(np.arange(sound_indexes[i-1]+1, sound_indexes[i], 1))

# # # Log transformation
# # logpitch = np.log(pitch).reshape(-1, 1)

# # # Set left values to zero
# # logpitch[logpitch==np.inf]=0

# plt.figure()
# aux = np.log(pitch)
# aux[~defined_indexes] = 0
# logpitch[logpitch==np.inf]=0
# # pitch[~defined_indexes] = 0
# # plt.scatter(time, pitch, s=5, label='data')
# plt.scatter(time, aux, s=5, label ='log-data')
# plt.scatter(time[lista_aux], logpitch[lista_aux], s=5, label ='interpoled')
# plt.xlabel('Time (s)')
# plt.ylabel('Pitch')
# plt.legend()
# plt.grid(visible=True)
# plt.show(block=True)

# # ===========
# # SPECTROGRAM
# # Read file
# wav = wavfile.read(r'Datos/wavs/S21/s21.objects.01.channel1.wav')[1]
# wav = wav.astype("float")

# # Calculates the mel frequencies spectrogram giving the desire sampling (match the EEG)
# audio_sr, sr = 16e3, 128
# sample_window = int(audio_sr/sr)
# S = librosa.feature.melspectrogram(y=wav,
#                                     sr=audio_sr, 
#                                     n_fft=sample_window, 
#                                     hop_length=sample_window, 
#                                     n_mels=16)

# # Transform to dB using normalization to 1
# S_DB = librosa.power_to_db(S=S, ref=np.max)

# S_DB.T

# # =====
# # MFCCS

# # load audio files with librosa
# signal, audio_sr = librosa.load(r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\wavs\S21\s21.objects.01.channel1.wav', sr=None) 
# # DIFIEREN EN UN FACTOR DE ESCALA WAV=32768*SIGNAL; siendo la normalización de punto flotante nbits=16, 2 ** (nbits - 1)=32768
# mfccs = librosa.feature.mfcc(y=wav, 
#                              n_mfcc=12, 
#                              sr=audio_sr, 
#                              n_fft=sample_window, 
#                              hop_length=sample_window,
#                              n_mels=12)



# plt.figure(figsize=(8, 4))
# librosa.display.specshow(mfccs, 
#                          x_axis="time", 
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()

# delta_mfccs = librosa.feature.delta(mfccs)
# delta2_mfccs = librosa.feature.delta(mfccs, order=2)
# mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))