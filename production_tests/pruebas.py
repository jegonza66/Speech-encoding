import numpy as np, matplotlib.pyplot as plt
from funciones import dump_pickle, load_pickle
import config
import torch
from processing import shifted_matrix_2

# mtrfs = load_pickle('C:/Users/User/Downloads/mtrfs.pkl')
mtrfs = load_pickle('C:/Users/User/Downloads/mtrfs.pkl').cpu().numpy().T
mtrfs_original = load_pickle('C:/Users/User/Downloads/mtrfs_orig.pkl')

# mtrfs = np.stack(np.split(mtrfs, 104, axis=1))
# mtrfs_original = np.stack(np.split(mtrfs_original, 104, axis=1))
# mtrfs = mtrfs.reshape(128, 104, 16)
# mtrfs_original = mtrfs_original.reshape(128, 104, 16)
plt.figure()
# plt.plot(mtrfs.mean(axis=1).mean(axis=1), label='mtrfs')
# plt.plot(mtrfs_original.mean(axis=1).mean(axis=1), label='mtrfs_original')
plt.plot(mtrfs[1,:104], label='mtrfs')
plt.plot(mtrfs_original[1,:104], label='mtrfs_original')
plt.legend()
plt.title('mtrfs')
plt.show(block=False)



X_train_a = load_pickle('C:/Users/User/Downloads/X_train_a.pkl')
y_train_a = load_pickle('C:/Users/User/Downloads/y_train_a.pkl')
X_pred_a = load_pickle('C:/Users/User/Downloads/X_pred_a.pkl')
y_test_a = load_pickle('C:/Users/User/Downloads/y_test_a.pkl')

dstims_train_val_a = load_pickle('C:/Users/User/Downloads/dstims_train_val_a.pkl')
dstims_test_a = load_pickle('C:/Users/User/Downloads/dstims_test_a.pkl')
eeg_test_a = load_pickle('C:/Users/User/Downloads/eeg_test_a.pkl')
eeg_train_val_a = load_pickle('C:/Users/User/Downloads/eeg_train_val_a.pkl')

# (np.abs(dstims_train_val_a==X_train_a.cpu().numpy())).all()
# (np.abs(dstims_test_a==X_pred_a.cpu().numpy())).all()
# (np.abs(eeg_train_val_a==y_train_a.cpu().numpy())).all()
# (np.abs(eeg_test_a==y_test_a.cpu().numpy())).all()

by_gpu = True
device = torch.device("cuda" if by_gpu and torch.cuda.is_available() else "cpu")
# ===============
# ESTANDARIZACION
# LO QUE HACE ORIGINAL
mean = np.mean(eeg_train_val_a, axis=0)
std = np.std(eeg_train_val_a, axis=0)
y_train_N = (eeg_train_val_a-mean)/std
y_test_N = (eeg_test_a-mean)/std

# LO QUE HACE EL NUEVO
y_train_a = y_train_a.to(device)
y_test_a = y_test_a.to(device)
mean_t = y_train_a.mean(dim=0)
std_t = y_train_a.std(dim=0, unbiased=False)
y_train_N2 = (y_train_a-mean_t)/(std_t + 1e-8)  
y_test_N2 = (y_test_a-mean_t)/(std_t + 1e-8)  

# =============
# NORMALIZACIÓN
# LO QUE HACE ORIGINAL
min = np.min(dstims_train_val_a, axis=0)
X_train_N = dstims_train_val_a-min
max = np.max(X_train_N, axis = 0)
X_train_N = np.divide(X_train_N, max, out=np.zeros_like(X_train_N), where=max != 0)
X_pred_N = dstims_test_a-min
X_pred_N = np.divide(X_pred_N, max, out=np.zeros_like(X_pred_N), where=max != 0)
        
# LO QUE HACE EL NUEVO
X_train_a = X_train_a.to(device)
X_pred_a = X_pred_a.to(device)
min_t = X_train_a.min(dim=0)[0]

X_train_N2 = X_train_a-min_t
max_t = X_train_N2.max(dim=0)[0]
X_train_N2 /= (max_t+1e-12)
X_pred_N2 = (X_pred_a-min_t)/(max_t+1e-12)

# mean_t_np = mean_t.cpu().numpy()
# std_t_np = std_t.cpu().numpy()
# print("mean diff:", np.max(np.abs(mean_t_np - mean)))
# print("std diff:", np.max(np.abs(std_t_np - std)))

min_t_np = min_t.cpu().numpy()
max_t_np = max_t.cpu().numpy()
print("min diff:", np.max(np.abs(min_t_np - min)))
print("max diff:", np.max(np.abs(max_t_np - max)))



# np.allclose(y_train_N2.cpu().numpy(), y_train_N, atol=1e-6)
# np.allclose(y_test_N2.cpu().numpy(), y_test_N, atol=1e-6)
# (np.abs(y_train_N2.cpu().numpy()-y_train_N)<1e6).all()
# (np.abs(y_test_N2.cpu().numpy()-y_test_N)<1e6).all()
# (np.abs(X_train_N2.cpu().numpy()-X_train_N)==0).all()
# (np.abs(dstims_test-X_pred)==0).all()
# (np.abs(y_train-eeg_train_val)==0).all()
# (np.abs(eeg_test-y_test)==0).all()

X_train = load_pickle('C:/Users/User/Downloads/X_train.pkl')
y_train = load_pickle('C:/Users/User/Downloads/y_train.pkl')
X_pred = load_pickle('C:/Users/User/Downloads/X_pred.pkl')
y_test = load_pickle('C:/Users/User/Downloads/y_test.pkl')

dstims_train_val = load_pickle('C:/Users/User/Downloads/dstims_train_val.pkl')
dstims_test = load_pickle('C:/Users/User/Downloads/dstims_test.pkl')
eeg_test = load_pickle('C:/Users/User/Downloads/eeg_test.pkl')
eeg_train_val = load_pickle('C:/Users/User/Downloads/eeg_train_val.pkl')

stim_tolerance = 1e-9
eeg_tolerance = 1e-9
(np.abs(dstims_train_val-X_train.cpu().numpy())<stim_tolerance).all()
(np.abs(dstims_test-X_pred.cpu().numpy())<stim_tolerance).all()
(np.abs(eeg_train_val-y_train.cpu().numpy())<eeg_tolerance).all()
(np.abs(eeg_test-y_test.cpu().numpy())<eeg_tolerance).all()

XTX_reg = dstims_train_val.T @ dstims_train_val + 400 *  np.eye(dstims_train_val.shape[1]) # X^T * X + alpha*I
mtrfs = np.linalg.solve(XTX_reg, dstims_train_val.T @ eeg_train_val)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=400, fit_intercept=False)
ridge.fit(dstims_train_val, eeg_train_val)
mtrfs = ridge.coef_.T


wavfile(r'Datos\wavs\S21\s21.objects.01.channel1.wav')

channel = load_pickle('C:/Users/User/Downloads/channel.pkl')

X_train_2 = np.array(load_pickle('C:/Users/User/Downloads/X_train2.pkl')[:len(channel)])

relevant_indexes_1 = len(load_pickle('C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/Internal_BS/tmin-0.2_tmax0.6/samples_info/samples_info_21.pkl')['keep_indexes1'])
relevant_indexes_1 = np.cumsum(load_pickle('C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/Internal_BS/tmin-0.2_tmax0.6/samples_info/samples_info_21.pkl')['trial_lengths1'])

X_train_2[relevant_indexes_1][:, 0]
X_train[:, :104][:, 26]

for i in [21,22,23,24,25,26,27,29,30]:
    # len(load_pickle(f'C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/External_BS/tmin-0.2_tmax0.6/samples_info/samples_info_{i}.pkl')['keep_indexes1'])/128#/60
    len(load_pickle(f'C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/Internal_BS/tmin-0.2_tmax0.6/samples_info/samples_info_{i}.pkl')['keep_indexes1'])/128#/60
    len(load_pickle(f'C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/External/tmin-0.2_tmax0.6/samples_info/samples_info_{i}.pkl')['keep_indexes1'])/128/.8
    # len(load_pickle(f'C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/Internal/tmin-0.2_tmax0.6/samples_info/samples_info_{i}.pkl')['keep_indexes2'])/128/.8
    
    # len(load_pickle(f'C:/Users/User/repos/Speech-encoding/saves/preprocessed_data/Internal/tmin-0.2_tmax0.6/samples_info/samples_info_{i}.pkl')['keep_indexes1'])/128/60

# Read file
import librosa
from scipy.io import wavfile
wav_fname = r'Datos\wavs\S21\s21.objects.01.channel1.wav'
audio_sr = 16e3
wav = wavfile.read(wav_fname)[1]
wav = wav.astype("float")
len
# Calculates the mel frequencies spectrogram giving the desire sampling (match the EEG)
sample_window = int(audio_sr/config.sr)
S = librosa.feature.melspectrogram(
    y=wav,
    sr=audio_sr, 
    n_fft=sample_window, 
    hop_length=sample_window, 
    n_mels=16
    )
# Transform to dB using normalization to 1
S_DB = librosa.power_to_db(S=S, ref=np.max)

S_DB = S_DB.T


wavo = wavfile.read(wav_fname)[1]
wavo = wavo.astype("float")

n_fft = 125
hop_length = 125
n_mels = 16

S_2 = librosa.feature.melspectrogram(wavo, sr=audio_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB_2 = librosa.power_to_db(S_2, ref=np.max)
S_DB_2 = S_DB_2.transpose()

# Match to Envelope size if shorter to standarized across features
S_DB_2 = S_DB_2[:min(len(S_DB_2), 9168), :]
S_DB_2 =np.array(S_DB_2)


# ESTAA ACA!!!!JAJAJA
sX_train_2 = shifted_matrix_3(features=X_train_2, delays=config.delays, use_gpu=True,indices_to_keep=relevant_indexes_1)
sX_train_2b = shifted_matrix_3(features=X_train_2, delays=config.delays, use_gpu=True,indices_to_keep=relevant_indexes_1)

(sX_train_2-sX_train_2b).max()

X_train.astype(np.float32).mean()
X_train.astype(np.float32).std()
X_train_2.astype(np.float32).mean()
X_train_2.astype(np.float32).std()

'C:/Users/User/Downloads/eeg_train_index_new.pkl'

# # Standard libraries
# import numpy as np, pandas as pd, os, warnings, matplotlib.pyplot as plt
# import mne, librosa, librosa.display, platform, opensmile, textgrids

# # Specific libraries
# import scipy.io.wavfile as wavfile
# from scipy import signal as sgn
# from praatio import pitch_and_intensity
# from disvoice.phonological.phonological import Phonological

# # Modules
# import processing, funciones, config

# # Review this If we want to update packages
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# mne.set_log_level(verbose='CRITICAL')
# exp_info = config.Exp_info()

# # Modules
# from funciones  import load_pickle, dump_pickle
# bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']
# # bands = ['Theta']
# sesiones = [21,22,23,24,25,26,27,29,30]
# root = r'saves\preprocessed_data\External\tmin-0.2_tmax0.6'

# for sesion in sesiones:
#     # sesion=24
#     stimuli = []
#     for folder in os.listdir(root):
#         if folder == 'EEG':
#             eeg1 = {}
#             eeg2 = {}
#             for band in bands:
#                 eeg1[band], eeg2[band] = load_pickle(os.path.join(root, folder, band,  'Causal', f'Sesion{sesion}.pkl'))
#         elif folder == 'samples_info':
#             samples_inf = load_pickle(os.path.join(root, folder, f'samples_info_{sesion}.pkl'))
#             index_1, index_2 = samples_inf['keep_indexes1'], samples_inf['keep_indexes2']
#         else:
#             stimulus_1, stimulus_2 = load_pickle(os.path.join(root, folder, f'Sesion{sesion}.pkl'))
#             stimuli.append((folder, stimulus_1, stimulus_2))
        
#     # Hacemos chequeo de shape
#     for band in bands:
#         # band='Theta'
#         for (folder, stimulus_1, stimulus_2) in stimuli:
#             # if folder=='Phonemes-Discrete-Phonet':
#             if (stimulus_1[:].shape[0]==eeg1[band][:].shape[0], stimulus_2[:].shape[0]==eeg2[band][:].shape[0]) != (True, True):
#             # if (stimulus_1[index_1].shape[0]==eeg1[band][index_1].shape[0], stimulus_2[index_2].shape[0]==eeg2[band][index_2].shape[0]) != (True, True):
#                 print('\n\nFALLA', band, sesion, folder)
#                 stimulus_1[:].shape[0], eeg1[band][:].shape[0]


# ind = load_pickle(r'C:\Users\User\repos\Speech-encoding\saves\preprocessed_data\External\tmin-0.2_tmax0.6\samples_info\samples_info_21.pkl')['keep_indexes2']
# atr = load_pickle(r'C:\Users\User\repos\Speech-encoding\saves\preprocessed_data\External\tmin-0.2_tmax0.6\Phonemes-Discrete-Phonet\Sesion21.pkl')[0]
# eeg = load_pickle(r'C:\Users\User\repos\Speech-encoding\saves\preprocessed_data\External\tmin-0.2_tmax0.6\EEG\Theta\Causal\Sesion21.pkl')[1]
# atr[ind].shape
# eeg[ind].shape

# #=========
# # ENVELOPE
# wav = wavfile.read(r'Datos/wavs/S21/s21.objects.01.channel1.wav')[1]
# wav = wav.astype("float")

# # Calculate envelope
# envelope = np.abs(sgn.hilbert(wav))

# # Resample
# window_size, stride = 125, 125
# envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)]).reshape(-1,1)

# # =====================
# # PHONOlOGICAL FEATURES
# phonologicalf = Phonological()
# file_audio = r'Datos/wavs/S21/s21.objects.01.channel1.wav'
# feats=phonologicalf.extract_features_file(file_audio, static=False, plots=False, fmt="dataframe")
# print(feats)
# phon_feats = [feat for feat in feats.columns if feat!='time']
# sr = 128
# desire_time = np.linspace(0, envelope.shape[0]/sr + 1/sr , envelope.shape[0])
# phonological_features = []
# for phon_feat in phon_feats:
#     phonological_features.append(np.interp(desire_time, feats['time'].values, feats[f'{phon_feat}'].values))
# phonological_features = np.stack(phonological_features, axis=0).T

# plt.figure()
# plt.plot(feats['time'], feats['vocalic'], label = 'labial')
# plt.plot(desire_time, phonological_features[0], label = 'labial')
# plt.grid(True)
# plt.xlabel('Time (s)')
# plt.show()


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

# =====
# PITCH

# HIPER PARAMS
silence_threshold = .03
minPitch = 50
maxPitch = 300
sampleStep = 1/128
praat_executable_path = 'C:/Program Files/Praat/Praat.exe'
output_folder = f"Datos/PRUEBA_threshold_{silence_threshold}"
os.makedirs(output_folder, exist_ok=True)


wav_fname = r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\wavs\S21\s21.objects.01.channel1.wav'
pitch_fname = r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\PRUEBA_threshold_0.03\S21\s21.objects.01.channel1.txt'


wav_fname='Datos/wavs/S21/s21.objects.04.channel1.wav'
pitch_fname='Datos/pitch_threshold_0.03/S21/s21.objects.04.channel1.txt'
praat_executable_path = 'C:\\Program Files\\Praat\\Praat.exe'
silence_threshold = .03
minPitch = 50
maxPitch = 300
sampleStep = 1/128
pitch_and_intensity.extractPI(inputFN=os.path.abspath(wav_fname), 
                            outputFN=os.path.abspath(pitch_fname), 
                            praatEXE=praat_executable_path, 
                            minPitch=minPitch,
                            maxPitch=maxPitch, 
                            sampleStep=sampleStep, 
                            silenceThreshold=silence_threshold,
                            pitchQuadInterp=False)
    
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