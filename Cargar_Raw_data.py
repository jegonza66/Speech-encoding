import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import mne
from scipy import signal as sgn
import Processing
from numpy.fft import fft, fftfreq
import parselmouth
import opensmile
from parselmouth.praat import call
import textgrids
import os
import Funciones
import seaborn as sn

## PARAMETROS
s = 21
trial = 1
channel = 1

valores_faltantes_pitch = np.nan
audio_sr = 16000
sampleStep = 0.01

sr = 128
audio_sr = 16000
tmin, tmax = -0.6, -0.003
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)


## PHN features



sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]

all_labels = []

for sesion in sesiones:

    phn_path = "Datos/phonemes/S" + str(sesion) + "/"
    trials = list(set([int(fname.split('.')[2]) for fname in os.listdir(phn_path) if
                       os.path.isfile(phn_path + f'/{fname}')]))
    for trial in trials:
        for channel in [1, 2]:
            try:
                phn_fname = "Datos/phonemes/S" + str(sesion) + "/s" + str(sesion) + ".objects." + "{:02d}".format(trial) + ".channel" + \
                            str(channel) + ".aligned_fa.TextGrid"
                # phn_fname = "Datos/phonemes/S" + str(sesion) + "/manual/s" + str(sesion) + "_objects_" + "{:02d}".format(
                #     trial) + "_channel" + str(channel) + "_aligned_faTAMARA.TextGrid"

                phrases_fname = "Datos/phrases/S" + str(sesion) + "/s" + str(sesion) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
                        channel) + ".phrases"

                # Get trial total length
                phrases = pd.read_table(phrases_fname, header=None, sep="\t")
                trial_tmax = phrases[1].iloc[-1]

                # Phonemes
                grid = textgrids.TextGrid(phn_fname)

                phonemes = grid['transcription : phones']
                phonemes[0].xmin = 0.

                # Parse
                labels = []
                times = []
                samples = []

                for ph in phonemes:
                    label = ph.text.transcode()
                    label = label.replace(' ', '')
                    label = label.replace('º', '')
                    label = label.replace('-', '')
                    # Rename silences
                    if label == 'sil' or label == 'sp' or label == 'sile' or label == 'silsil'\
                            or label == 'SP' or label == 's¡p' or label == 'sils':
                        label = ""
                    labels.append(label)
                    times.append((ph.xmin, ph.xmax))
                    samples.append(np.round((ph.xmax - ph.xmin) * sr).astype("int"))

                labels.append("")
                times.append((ph.xmin, trial_tmax))
                samples.append(np.round((trial_tmax - ph.xmax) * sr).astype("int"))

                # Get unique phonemes
                labels_set = set(labels)
                unique_labels = (list(labels_set))

                all_labels.append(unique_labels)
            except:
                print(f'Could not upload Sesion: {sesion} - Trial {trial} - Channel {channel}')



dfvowels = pd.read_csv('vowels.csv', header=None, index_col=0)
dfconsonants = pd.read_csv('consonants.csv', header=None, index_col=0)
dfarpabet = pd.concat([pd.get_dummies(dfconsonants), pd.get_dummies(dfvowels)]).fillna(0)
dfarpabet.index.name = 'arpabet'
dfarpabet = dfarpabet.sort_values('arpabet')
plt.figure(figsize=(12, 12))
sn.heatmap(dfarpabet)


## Phonemes

sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]

all_labels = []

for sesion in sesiones:

    phn_path = "Datos/phonemes/S" + str(sesion) + "/"
    trials = list(set([int(fname.split('.')[2]) for fname in os.listdir(phn_path) if
                       os.path.isfile(phn_path + f'/{fname}')]))
    for trial in trials:
        for channel in [1, 2]:
            try:
                phn_fname = "Datos/phonemes/S" + str(sesion) + "/s" + str(sesion) + ".objects." + "{:02d}".format(trial) + ".channel" + \
                            str(channel) + ".aligned_fa.TextGrid"
                # phn_fname = "Datos/phonemes/S" + str(sesion) + "/manual/s" + str(sesion) + "_objects_" + "{:02d}".format(
                #     trial) + "_channel" + str(channel) + "_aligned_faTAMARA.TextGrid"

                phrases_fname = "Datos/phrases/S" + str(sesion) + "/s" + str(sesion) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
                        channel) + ".phrases"

                # Get trial total length
                phrases = pd.read_table(phrases_fname, header=None, sep="\t")
                trial_tmax = phrases[1].iloc[-1]

                # Phonemes
                grid = textgrids.TextGrid(phn_fname)

                phonemes = grid['transcription : phones']
                phonemes[0].xmin = 0.

                # Parse
                labels = []
                times = []
                samples = []

                for ph in phonemes:
                    label = ph.text.transcode()
                    label = label.replace(' ', '')
                    label = label.replace('º', '')
                    label = label.replace('-', '')
                    # Rename silences
                    if label == 'sil' or label == 'sp' or label == 'sile' or label == 'silsil'\
                            or label == 'SP' or label == 's¡p' or label == 'sils':
                        label = ""
                    labels.append(label)
                    times.append((ph.xmin, ph.xmax))
                    samples.append(np.round((ph.xmax - ph.xmin) * sr).astype("int"))

                labels.append("")
                times.append((ph.xmin, trial_tmax))
                samples.append(np.round((trial_tmax - ph.xmax) * sr).astype("int"))

                # Get unique phonemes
                labels_set = set(labels)
                unique_labels = (list(labels_set))

                all_labels.append(unique_labels)
            except:
                print(f'Could not upload Sesion: {sesion} - Trial {trial} - Channel {channel}')

all_labels = Funciones.flatten_list(all_labels)

labels_set = set(all_labels)
unique_labels = (list(labels_set))


unique_labels.sort()

ph_labels = ['', 'CH', 'NY', 'R', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'x', 'y']
ph_labels_man = ['', '(d)o', 'CH', 'NY', 'R', 'a', 'ap', 'b', 'chas', 'd', 'e', 'es', 'f', 'g', 'i', 'k', 'l', 'lg',
                 'm', 'n', 'ns', 'o', 'p', 'r', 's', 't', 'u', 'x', 'y']

ph_labels_man = ['', '(d)o', 'A', 'AH', 'CH', 'F', 'NY', 'R', 'SP', 'Y', 'a', 'ap', 'b', 'br', 'c', 'chas', 'd', 'de',
                 'e', 'es', 'f', 'g', 'h', 'i', 'k', 'l', 'l-', 'lg', 'm', 'n', 'ns', 'o', 'p', 'r', 's', 'si', 'sils',
                 's¡p', 'sº', 't', 'u', 'v', 'x', 'y']


# WAV
wav_fname = "Datos/wavs/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
    channel) + ".wav"
wav = wavfile.read(wav_fname)[1]
wav = wav.astype("float")

# ENVELOPE
envelope = np.abs(sgn.hilbert(wav))
window_size = 125
stride = 125
envelope = np.array(
    [np.mean(envelope[i:i + window_size]) for i in range(0, len(envelope), stride) if i + window_size <= len(envelope)])
envelope = envelope.ravel().flatten()


diferencia = np.sum(samples) - len(envelope)
if diferencia > 0:
    samples[-1] -= diferencia
elif diferencia < 0:
    samples[-1] += diferencia

# Make empty df of phonemes
df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=unique_labels)

#
phonemes_tgrid = np.repeat(labels, samples)

for i, phoneme in enumerate(phonemes_tgrid):
    df.loc[i, phoneme] = envelope[i]
    # df.iloc[i][phoneme] = 1





## EEG
Band = None
l_freq_eeg, h_freq_eeg = Processing.band_freq(Band)

eeg_fname = "Datos/EEG/S" + str(s) + "/s" + str(s) + "-" + str(channel) + "-Trial" + str(trial) + "-Deci-Filter-Trim-ICA-Pruned.set"
eeg = mne.io.read_raw_eeglab(eeg_fname)
eeg_freq = eeg.info.get("sfreq")
info = eeg.info
eeg.load_data()
if Band: eeg = eeg.filter(l_freq=l_freq_eeg, h_freq=h_freq_eeg, phase='minimum')
# eeg.plot()

## Source estimation
from mne.datasets import fetch_fsaverage
import os

montage = mne.channels.make_standard_montage('biosemi128')
eeg.set_montage(montage)
eeg.set_eeg_reference(projection=True)  # needed for inverse modeling

# Download fsaverage files
fs_dir = r'C:\Users\joaco\OneDrive - The University of Nottingham\MEGEYEHS\DATA\MRI_DATA\FreeSurfer_out\fsaverage'
subjects_dir = os.path.dirname(fs_dir)
os.environ["SUBJECTS_DIR"] = subjects_dir


# The files live in:
subject = 'fsaverage'
trans = 'trans.fif'  # MNE has a built-in fsaverage transformation
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
# Check that the locations of EEG electrodes is correct with respect to MRI

eeg_info_path = 'eeg_info.fif'
eeg.save(eeg_info_path)


mne.gui.coregistration(subject='fsaverage', subjects_dir=subjects_dir, inst=eeg_info_path)

mne.viz.plot_alignment(eeg.info, src=src, eeg=['original', 'projected'], trans=trans,
                       show_axes=False, mri_fiducials=False)

# eeg = eeg.to_data_frame()
# eeg = np.array(eeg)[:, 1:129]  # paso a array y tomo tiro la primer columna de tiempos

# eeg = np.array(eeg._data * 1e6).transpose()  # 1e6 es por el factor de scaling del eeg.to_data_frame()
#
# # Downsample
# eeg = Processing.subsamplear(eeg, int(eeg_freq / sr))


# plt.savefig('gráficos/Raw_EEG.png')
# plt.savefig('{}Theta.png'.format(s))

## PSD
psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(eeg._data, sfreq=eeg_freq, fmin=1, fmax=60)

fig, ax = plt.subplots()
evoked = mne.EvokedArray(psds_welch_mean, info)
evoked.times = freqs_mean
evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s',
            show=False, spatial_colors=True, unit=False, units='w', axes=ax)
ax.set_xlabel('Frequency [Hz]')
ax.grid()
# plt.savefig('NO_CAUSAL.png')

## WAV
wav_fname = "Datos/wavs/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
    channel) + ".wav"
wav = wavfile.read(wav_fname)[1]
wav = wav.astype("float")

## ENVELOPE

envelope = np.abs(sgn.hilbert(wav))

window_size = 125
stride = 125
envelope = np.array(
    [np.mean(envelope[i:i + window_size]) for i in range(0, len(envelope), stride) if i + window_size <= len(envelope)])
envelope = envelope.ravel().flatten()

envelope_x = np.linspace(0, len(wav)/audio_sr, len(envelope))
wav_x = np.linspace(0, len(wav)/audio_sr, len(wav))
fig, ax = plt.subplots()
plt.plot(wav_x, wav, label='Audio signal')
plt.plot(envelope_x, envelope, label='Envelope')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
plt.grid()
plt.legend()



## PLOT spectre
sp = fft(envelope)
freq = fftfreq(envelope.shape[-1], d=1 / 128)
plt.figure()
plt.plot(freq, np.abs(sp))
plt.xlim([0, 30])
plt.ylim([0, 3e4])
plt.grid()
plt.title('Hilbert + Butter + Prom')
plt.xlabel('Frecuency [Hz]')
plt.ylabel('Amplitud')

## PITCH Pablo
path = r"C:\Users\joaco\Desktop\Joac\Facultad\Tesis\Código\Datos\Pitch\Pablo\tracks_generales"
subject_letter = 'A' if channel else 'B'
pitch_name = path + '\{}_{}.csv'.format(s,subject_letter)

df = pd.read_csv(pitch_name)
trial_changes = [i for i, value in enumerate(np.diff(df['time'].values)<0) if value]
pitch = df['pitch'].values

pitch_trial = pitch[:trial_changes[0]]
plt.plot(pitch_trial, label='Pablo')

## PITCH
pitch_fname = "Datos/Pitch_threshold_0.03/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
    channel) + ".txt"

read_file = pd.read_csv(pitch_fname)

time = np.array(read_file['time'])
pitch = np.array(read_file['pitch'])
intensity = np.array(read_file['intensity'])

pitch[pitch == '--undefined--'] = np.nan
pitch = np.array(pitch, dtype=float)

pitch_der = Funciones.sliding_window(df=pd.DataFrame(pitch))

if not valores_faltantes_pitch:
    pitch[np.isnan(pitch)] = valores_faltantes_pitch
    pitch_der[np.isnan(pitch_der)] = valores_faltantes_pitch
else:
    pitch[np.isnan(pitch)] = float(valores_faltantes_pitch)
    pitch_der[np.isnan(pitch_der)] = float(valores_faltantes_pitch)
    pitch[np.isnan(pitch)] = float(valores_faltantes_pitch)
    pitch_der[np.isnan(pitch_der)] = float(valores_faltantes_pitch)
# else:
#     print('Invalid missing value for pitch {}'.format(valores_faltantes_pitch) + '\nMust be finite.')

pitch = np.array(np.repeat(pitch, audio_sr * sampleStep), dtype=float)
pitch = Processing.subsamplear(pitch, 125)

pitch_der = np.array(np.repeat(pitch_der, audio_sr * sampleStep), dtype=float)
pitch_der = Processing.subsamplear(pitch_der, 125)

fig, ax = plt.subplots()
plt.plot(np.linspace(0, len(wav)/audio_sr, len(wav)), wav, label='Audio signal [Amplitude]')
plt.plot(np.linspace(0,len(wav)/audio_sr, len(pitch)), pitch, label='Pitch [Hz]')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Magnitude')
plt.grid()
plt.legend()

## Normalice features Leave nans for plot

no_nan_pitch = pitch[~np.isnan(pitch)]
no_nan_pitch_der = pitch_der[~np.isnan(pitch_der)]

norm = Processing.normalizar()
norm.normalize_01(envelope)
norm.normalize_01(no_nan_pitch)
norm.normalize_11(no_nan_pitch_der)
no_nan_pitch_der -= no_nan_pitch_der.mean()

pitch[~np.isnan(pitch)] = no_nan_pitch
pitch_der[~np.isnan(pitch_der)] = no_nan_pitch_der

pitch = np.array(np.repeat(pitch, audio_sr * sampleStep), dtype=float)
pitch = Processing.subsamplear(pitch, 125)

pitch_der = np.array(np.repeat(pitch_der, audio_sr * sampleStep), dtype=float)
pitch_der = Processing.subsamplear(pitch_der, 125)

norm.normalize_11(wav)
wav -= wav.mean()

## Spectrogram
import librosa
import librosa.display

n_fft = 125
hop_length = 125
n_mels = 16

S = librosa.feature.melspectrogram(wav, sr=audio_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)

plt.figure()
librosa.display.specshow(S_DB, sr=audio_sr, hop_length=hop_length, x_axis='time', y_axis='mel', )
plt.colorbar(format='%+2.0f dB')

# Shifted matrix row by row
spec_shift = Processing.matriz_shifteada(S_DB[0], delays)
for i in np.arange(1, len(S_DB)):
    spec_shift = np.hstack((spec_shift, Processing.matriz_shifteada(S_DB[i], delays)))

fmin = n_fft/2
mel_f = librosa.mel_frequencies(n_mels+2, fmin=fmin, fmax=8000)
low = mel_f[:-2]
center = mel_f[1:-1]
high = mel_f[2:]

## PLOT
time = np.arange(len(wav)) / 16000

# plt.ion()
# plt.figure()
# plt.plot(time, wav, label='Audio signal')
plt.plot(np.linspace(0, time[-1], len(envelope)), envelope, label='Envelope')
plt.plot(np.linspace(0, time[-1], len(S_DB[-1])), S_DB[-1], label='Spectrogram')
# plt.plot(np.linspace(0, time[-1], len(pitch)), pitch, label='Pitch [Hz]')
# plt.plot(np.linspace(0, time[-1], len(pitch_der)), pitch_der, label='Pitch derivate [Hz/s]')
plt.ylabel('Magnitude')
plt.xlabel('Time [s]')
# plt.xlim([41,43])
plt.title('Scaled audio features')
plt.legend()

## SHIMMER
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    sampling_rate=16000)

y = smile.process_file(wav_fname)
y.index = y.index.droplevel(0)
y.index = y.index.map(lambda x: x[0].total_seconds())

shimmer = y['shimmerLocaldB_sma3nz']

mcm = Funciones.minimo_comun_multiplo(len(shimmer), len(envelope))
shimmer = np.repeat(shimmer, mcm/len(shimmer))
shimmer = Processing.subsamplear(shimmer, mcm/len(envelope))

norm.normalize_11(wav)
wav -= wav.mean()

fig, ax = plt.subplots()
plt.plot(np.linspace(0, len(wav)/audio_sr, len(wav)), wav*5, label='Audio signal')
plt.plot(np.linspace(0,len(wav)/audio_sr, len(shimmer)), shimmer, label='Shimmer')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
plt.grid()
plt.legend()



## JITTER

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

y = smile.process_file(wav_fname)
y.index = y.index.droplevel(0)
y.index = y.index.map(lambda x: x[0].total_seconds())
jitter = y['jitterLocal_sma3nz']

mcm = Funciones.minimo_comun_multiplo(len(jitter), len(envelope))
jitter = np.repeat(jitter, mcm/len(jitter))
jitter = Processing.subsamplear(jitter, mcm/len(envelope))

plt.figure(figsize=(16, 4))
plt.plot(jitter)
plt.title('Jitter')

## CSSP

snd = parselmouth.Sound(wav_fname)

data = []
frame_length = 0.2
hop_length = 1 / 128
t1s = np.arange(0, snd.duration - frame_length, hop_length)
times = zip(t1s, t1s + frame_length)

for t1, t2 in times:
    powercepstrogram = call(snd.extract_part(t1, t2), "To PowerCepstrogram", 60, 0.0020001, 5000, 50)
    cpps = call(powercepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0,
                "Exponential decay", "Robust")
    data.append(cpps)

data = np.array(np.repeat(data, audio_sr * sampleStep))
data = Processing.subsamplear(data, 125)
data = Processing.matriz_shifteada(data, delays)

cssp = np.array(data)
print(cssp.shape)

plt.figure(figsize=(16, 4))
plt.plot(cssp[:,0])
plt.title('CPPS')
