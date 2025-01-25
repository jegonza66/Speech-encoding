import mne, matplotlib.pyplot as plt, numpy as np, os, pandas as pd
from scipy.io import wavfile
from scipy import signal as sgn
import librosa
from funciones import load_pickle
import config

params = {
        'legend.fontsize': 16,
        'legend.title_fontsize': 16,
        'figure.figsize': (10, 5),
        'figure.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize':16,
        'ytick.labelsize':16
        }
import matplotlib.pylab as pylab
pylab.rcParams.update(params)
plt.style.use([plt.style.available[23]])

# =============
# Espectrograma
SpectrogramPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Spectrogram/Sesion21.pkl"
NumberOfTicks = 16

spectrogram = load_pickle(path=SpectrogramPath)[0][:9168]
WindowLeft, WindowRight = 0, len(spectrogram)/config.sr

time_spectrogram = np.arange(0, len(spectrogram)/config.sr, 1/config.sr)
window_spectrogram = (WindowLeft <= time_spectrogram) & (time_spectrogram <= WindowRight)

bands_center = librosa.mel_frequencies(
    n_mels=NumberOfTicks+2, 
    fmin=62, 
    fmax=8000
    )[1:-1]

tags = [int(bands_center[i]) for i in np.arange(1, len(bands_center)+1, 2)]
ticks = np.arange(0, NumberOfTicks, 2)

fig = plt.figure(
    tight_layout=True,
    figsize=(6, 5)
    )
im = plt.imshow(
    spectrogram.T,
    aspect='auto',  # Ajusta el aspecto
    extent=[WindowLeft, WindowRight, 0, 16],  # Ajusta los límites de los ejes
    origin='lower',  # Ajusta el origen
    cmap='RdBu'  # Ajusta el mapa de colores
    )
plt.colorbar(
    im,
    label='Amplitud (dB)'
    )

plt.yticks(
    ticks=ticks, 
    labels=tags
    )
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')  
fig.savefig(
    f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_espectrograma.svg',
    transparent=True
    )
# fig.show()

# # ========================
# # Tono de voz del hablante
# WavPath = 'Datos/wavs/S21/s21.objects.01.channel1.wav'
# PitchPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Pitch-Log-Raw_threshold_0.03/Sesion21.pkl"
# WindowLeft, WindowRight = 32, 34

# sr, audio = wavfile.read(WavPath)
# pitch = load_pickle(path=PitchPath)[0][:9168].reshape(-1)

# time_audio = np.arange(0, len(audio)/sr, 1/sr)
# time_pitch = np.arange(0, len(pitch)/config.sr, 1/config.sr)

# window_audio = (WindowLeft <= time_audio) & (time_audio <= WindowRight)
# window_pitch = (WindowLeft <= time_pitch) & (time_pitch <= WindowRight)

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )
# plt.plot(
#     time_pitch[window_pitch], 
#     1e2*pitch[window_pitch], 
#     label=r'$\propto$ Log (Tono de voz)',
#     color='C4',
#     linewidth=2
#     )
# plt.plot(
#     time_audio[window_audio], 
#     audio[window_audio], 
#     label='Audio',
#     color='C0',
#     linewidth=1,
#     zorder=-1
#     )
# plt.legend(loc=(.15,.81))
# plt.yticks([])
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud (U.A)')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_tono.svg',
#     transparent=True
#     )
# # fig.show()

# # ===============================
# # Envlovente de la señal de audio
# WavPath = 'Datos/wavs/S21/s21.objects.01.channel1.wav'
# WindowLeft, WindowRight, EegSr = 32, 34, 128

# sr, audio = wavfile.read(WavPath)
# window_size, stride = int(sr/EegSr), int(sr/EegSr)
# envelope = np.abs(sgn.hilbert(audio))
# envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
# # audio = np.array([np.mean(audio[i:i+WindowSize]) for i in range(0, len(audio), Stride) if i+window_size<=len(audio)])
# # sr = EegSr

# time_audio = np.arange(0, len(audio)/sr, 1/sr)
# time_envelope = np.arange(0, len(envelope)/EegSr, 1/EegSr)

# window_audio = (WindowLeft <= time_audio) & (time_audio <= WindowRight)
# window_envelope = (WindowLeft <= time_envelope) & (time_envelope <= WindowRight)

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )
# plt.plot(
#     time_audio[window_audio], 
#     audio[window_audio], 
#     label='Audio',
#     color='C0',
#     linewidth=1
#     )
# plt.plot(
#     time_envelope[window_envelope], 
#     envelope[window_envelope], 
#     label='Envolvente',
#     color='C4',
#     linewidth=2
#     )
# plt.legend(loc=(.16,.8))
# plt.yticks([])
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud (U.A)')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_envolvente.svg',
#     transparent=True
#     )
# # fig.show()

# # =======================================================
# # Ejemplo EEG y PSD (power spectral density) de un sujeto
# sesion, sujeto = 21, 2
# RawEegPath = f'Datos/EEG/S{sesion}/s{sesion}-{sujeto}-Trial1-Deci-Filter-Trim-ICA-Pruned.set'
# # EegPath = 'saves/preprocessed_data/External/tmin-0.2_tmax0.6/EEG/All/Causal/Sesion21.pkl'

# raw = mne.io.read_raw_eeglab(
#         RawEegPath, 
#         preload=True,
#         verbose='CRITICAL'
#         )
# raw = raw.filter(l_freq=.1, h_freq=40)
# raw.plot(
#     scalings=dict(eeg=2e-5)
# )

# psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(raw._data, sfreq=128, fmin=0.1, fmax=90)

# fig, ax = plt.subplots()
# for i, psd in enumerate(psds_welch_mean):
#     ax.plot(freqs_mean, psd, alpha=.5) 

# ax.set_xlabel('Frequency [Hz]')
# ax.set_xlim(0,40)
# # ax.set_ylim(0,1)
# # ax.set_xticks(freqs_mean[::5])
# # ax.set_xticklabels(freqs_mean[::5])
# ax.grid()
# fig.show()


# # eeg = load_pickle(path=EegPath)[0]
# # raw = mne.io.RawArray(data=eeg.T*1e-6, info=raw_raw.info)
# spectrum = raw.compute_psd(
#     method='welch',
#     fmin=.1, 
#     fmax=40, 
#     )

# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=2,
#     figsize=(10, 4),
#     # tight_layout=True
#     )

# spectrum.plot(
#     dB=False,
#     spatial_colors=True,
#     sphere=.14,
#     axes=axes[1]
#     )
# axes[1].set(
#     title='PSD (Power Spectral Density)',
#     ylabel='U.A',
#     xlabel='Frecuencia (Hz)',
#     ylim=(-1,20),
#     xlim=(0,40)
# )
# fig.show()

# =================================
# Tarea comportamental: EEG y audio
# for sujeto in [1,2]:
#     EegPath = f'Datos/EEG/S21/s21-{sujeto}-Trial13-Deci-Filter-Trim-ICA-Pruned.set'
#     WavPath = f'Datos/wavs/S21/s21.objects.01.channel{sujeto}.wav'
#     raw = mne.io.read_raw_eeglab(
#         EegPath, 
#         preload=True,
#         verbose='CRITICAL'
#         )
#     sr, audio = wavfile.read(WavPath)
#     fig, axes = plt.subplots(
#         nrows=1,
#         ncols=2,
#         figsize=(10, 5),
#         tight_layout=True
#         )
#     texto = 'Hiper-registro del primer sujeto' if sujeto == 1 else 'Hiper-registro del segundo sujeto'
#     fig.suptitle(texto, fontsize=20)

#     for k, ch in enumerate(raw.get_data()[::25,:]):
#         axes[0].plot(
#             ch[6000:16000]*1e1 + k*2.5e-3, 
#             color='C0'
#             )
        
#     axes[0].set(
#         title='EEG',
#         # ylabel='Canales de EEG',
#         # yticks=[k*2.5e-3 for k in range(6)],
#         yticks=[],
#         # yticklabels=[f'C{k}' for k in range(0, 129, 25)],
#         xticks=[],
#         )

#     axes[1].plot(
#         audio[100:], 
#         color='C4'
#         )   
#     axes[1].set(
#         title='Audio',
#         yticks=[],
#         xticks=[],
#         )
#     fig.savefig(
#         f'G:/My Drive/tesis_licenciatura/figuras/UBA_GAMES_s{sujeto}.png', 
#         transparent=True, 
#         dpi=600
#         )
    # # fig.show()