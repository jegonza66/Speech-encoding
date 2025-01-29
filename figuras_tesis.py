import mne, matplotlib.pyplot as plt, numpy as np, os, pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib.pylab as pylab

from matplotlib import rc
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

pylab.rcParams.update(params)

rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{subscript}')
import scienceplots
import matplotlib.pyplot as plt 
plt.style.use(['science'])

# =============
# EJEMPLOS TFCE
SpectrogramTfcePath = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram_4096.pkl'
PhonemesTfcePath = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Discrete-Phonet_4096.pkl'
_, sp_pvalue_tfce = load_pickle(path=SpectrogramTfcePath)
_, ph_pvalue_tfce = load_pickle(path=PhonemesTfcePath)

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 6),
    tight_layout=True
    )

# Spectrogram
NumberOfFeats = 16

# Mask and transformation
sp_pvalue_tfce[sp_pvalue_tfce>config.significance] = 1
sp_pvalue_tfce = -np.log10(sp_pvalue_tfce)

im_sp = axes[0].pcolormesh(
    config.times*1e3, # x
    np.arange(NumberOfFeats), # y
    sp_pvalue_tfce.T, # z
    shading='auto',
    cmap='inferno'
    )

bands_center = librosa.mel_frequencies(n_mels=NumberOfFeats+2, fmin=62, fmax=8000)[1:-1]
# tags = [int(bands_center[i]) for i in np.arange(1, len(bands_center)+1, 1)]
# tags = [int(bands_center[i]) for i in np.arange(1, len(bands_center)+1, 2)]
tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center))]
ticks = np.arange(0, NumberOfFeats, 1)

axes[0].set(
    xlabel='Tiempo (ms)', 
    ylabel='Frecuencia (Hz)', 
    yticks=ticks, 
    yticklabels=tags
    )
fig.colorbar( 
    orientation='vertical', 
    label=r"$-log_{10}(p_{values})$",
    aspect=15, 
    shrink=1, 
    mappable=im_sp,
    ax=axes[0]
    )

# Phonemes
NumberOfFeats = 21
significant_channels = np.zeros(shape=(NumberOfFeats, len(config.times)))

# Iteate over columns to get number of channels per feature that passes the threshold
for feature in range(NumberOfFeats):
    for delay in range(len(config.times)):
        # Count how many channels pass the threshold for a given feature and delay
        ppval = ph_pvalue_tfce[feature][delay]
        significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
number_of_ticks = significant_channels.shape[0]
y, z = np.arange(number_of_ticks), significant_channels

imph = axes[1].pcolormesh(
    config.times*1e3, # x
    y, # y
    z, # z
    shading='auto',
    cmap='inferno'
    )
tags = ['/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/', '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/']
ticks = np.arange(0, NumberOfTicks, 1)#+.5
axes[1].set(
    xlabel='Tiempo (ms)', 
    # ylabel='Fonemas', 
    yticks=np.arange(0, NumberOfFeats, 1),
    yticklabels=tags
    )

fig.colorbar( 
    orientation='vertical', 
    label="Número de canales significativos",
    aspect=15, 
    shrink=1, 
    mappable=imph,
    ax=axes[1]
    )
fig.savefig(
    f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/ejemplo_TFCE.svg',
    )
# fig.show()
# # =============================
# # DIAGRAMA  DE MATRIZ DE DISEÑO
# channel = 0
# WindowLeft, WindowRight = 6.20, 8.20 # seconds
# envelope = load_pickle('saves/preprocessed_data/External/tmin-0.2_tmax0.6/Envelope/Sesion21.pkl')[1][:9168]
# raw = load_pickle('saves/preprocessed_data/External/tmin-0.2_tmax0.6/EEG/Theta/Causal/Sesion21.pkl')[1][:9168]
# mTRF = load_pickle('saves/mtrf/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Envelope/total_weights_per_subject.pkl')['average_weights_subjects'].mean(axis=0).mean(axis=1)[channel]

# times = np.arange(0, len(envelope)/(config.sr), 1/config.sr)
# mask = (WindowLeft<=times)&(times<=WindowRight)
# times, envelope, raw = times[mask], envelope[mask].reshape(-1), raw[mask][:, channel]

# # Normalize stimuli
# envelope /= envelope.max()
# raw /= raw.max()

# # Make time simmetric
# times -= (times[0] + times[-1])/2
# HalfSpan = (times[-1]-times[0])/2

# span_window = times[(-610e-3<=times) & (times<=200e-3)]

# ### COMPLEto
# fig, axes = plt.subplots(
#     nrows=3,
#     ncols=1,
#     figsize=(5, 6),
#     tight_layout=True    
#     )

# # Plot the data
# axes[0].plot(
#     times, 
#     envelope, 
#     color ='orange', 
#     linewidth=1.5
#     )

# # axes[0].scatter(
# #     times[(-HalfSpan/2<=times)&(times<=HalfSpan/5)],#[::2], 
# #     envelope[(-HalfSpan/2<=times)&(times<=HalfSpan/5)],#[::2], 
# #     color ='black', 
# #     s=2, 
# #     zorder=10
# #     )

# axes[0].axvspan(
#     span_window[0], 
#     span_window[-1], 
#     -0.15, 
#     1.2, 
#     color='grey', 
#     alpha=0.3
#     )

# # Hide the default x-axis
# axes[0].spines['bottom'].set_position('zero')  # Move x-axis to the center
# axes[0].spines['left'].set_position('zero')   # Move y-axis to the center
# axes[0].spines['bottom'].set_visible(False)
# axes[0].spines['top'].set_visible(False)
# axes[0].spines['right'].set_visible(False)

# # Add an arrow for the x-axis
# axes[0].annotate(
#     '', 
#     xy=(HalfSpan, 0), 
#     xytext=(-HalfSpan, 0),
#     arrowprops=dict(
#         arrowstyle='->', 
#         color='black'
#         )
#     )

# # Add the x-axis label at the end of the arrow
# axes[0].text(
#     HalfSpan, 
#     0, 
#     't', 
#     fontsize=16, 
#     verticalalignment='center', 
#     horizontalalignment='left'
#     )

# # Customize the y-axis (optional)
# axes[0].annotate(
#     '', 
#     xy=(0, 1.5), 
#     xytext=(0, -1),
#     arrowprops=dict(
#         arrowstyle='->', 
#         color='black'
#         )
#     )
# axes[0].text(
#     .2, 
#     1.5, 
#     r'$S(t)$', 
#     fontsize=16, 
#     verticalalignment='bottom', 
#     horizontalalignment='center'
#     )

# # Set plot limits and display
# axes[0].tick_params(axis='x', which='minor', length=0)

# axes[0].set_xlim(-HalfSpan, HalfSpan)
# # axes[0].set_Ylim(0, envelope.max())

# axes[0].set_xticks([span_window[0], span_window[-1]])
# axes[0].set_xticklabels(['-600 ms', '200 ms'])
# axes[0].set_yticks([])
# axes[0].set_ylim(-1, 1.5)

# # Plot the mTRF
# axes[1].plot(
#     span_window, 
#     mTRF[::-1], 
#     color ='gray', 
#     linewidth=1.5
#     )
# axes[1].plot(
#     times, 
#     np.zeros(times.shape), 
#     color ='gray', 
#     linewidth=.025
#     )

# # Find max
# t_min = span_window[mTRF.tolist().index(mTRF.min())]
# # axes[1].vlines(x=t_min, ymin=mTRF.min(), ymax=0, color='black')
# # axes[1].scatter(
# #     t_min,
# #     mTRF.min(),
# #     s=15,
# #     color='black'
# #     )

# # Hide the default x-axis
# axes[1].spines['bottom'].set_position('zero')  # Move x-axis to the center
# axes[1].spines['left'].set_position('zero')   # Move y-axis to the center
# axes[1].spines['bottom'].set_visible(False)
# axes[1].spines['top'].set_visible(False)
# axes[1].spines['right'].set_visible(False)

# # Add an arrow for the x-axis
# axes[1].annotate(
#     '', 
#     xy=(span_window[-1]+.25, 0), 
#     xytext=(span_window[0]-.25, 0),
#     arrowprops=dict(
#             arrowstyle='->', 
#             color='black'
#             )
#         )

# # Add the x-axis label at the end of the arrow
# axes[1].text(
#     span_window[-1]+.25, 
#     0, 
#     r'$\tau$', 
#     fontsize=16, 
#     verticalalignment='center', 
#     horizontalalignment='left'
#     )

# # Customize the y-axis (optional)
# axes[1].annotate(
#     '', 
#     xy=(0, 0.15), 
#     xytext=(0, -.25),
#     arrowprops=dict(
#         arrowstyle='->', 
#         color='black'
#         )
#     )
# axes[1].text(
#     .25,# 1, 
#     0.15,#.5, 
#     r'$W(\tau)$', 
#     fontsize=16, 
#     verticalalignment='bottom', 
#     horizontalalignment='center'
#     )

# # Set plot limits and display
# # axes[1].set_xlim(-HalfSpan, HalfSpan)
# axes[1].tick_params(axis='x', which='minor', length=0)

# axes[1].set_yticks([])
# # axes[1].set_xticks([span_window[0], span_window[-1]])
# # axes[1].set_xticklabels(['', ''])
# axes[1].set_xticks([])
# axes[1].set_xticklabels([])
# axes[1].set_yticks([])

# # Plot the data
# axes[2].plot(
#     times, 
#     raw, 
#     color='black',
#     linewidth=1.5
#     )
# axes[2].scatter(
#     0, 
#     raw[(-.1<=times)&(times<=.1)].max(), 
#     color='black', 
#     s=15
#     )
# # axes[2].annotate('', xy=(-HalfSpan/4, 1.75), xytext=(0, eeg.max()),
# #             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),zorder=1)
# # axes[2].annotate('', xy=(HalfSpan/5, 1.75), xytext=(0, eeg.max()),
# #             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),zorder=1)

# # Hide the default x-axis
# axes[2].spines['bottom'].set_position('zero')  # Move x-axis to the center
# axes[2].spines['left'].set_position('zero')   # Move y-axis to the center
# axes[2].spines['bottom'].set_visible(False)
# axes[2].spines['top'].set_visible(False)
# axes[2].spines['right'].set_visible(False)

# # Add an arrow for the x-axis
# axes[2].annotate(
#     '', 
#     xy=(HalfSpan, 0), 
#     xytext=(-HalfSpan, 0),
#     arrowprops=dict(
#         arrowstyle='->', 
#         color='black'
#         )
#     )

# # Add the x-axis label at the end of the arrow
# axes[2].text(
#     HalfSpan, 
#     0, 
#     't', 
#     fontsize=16, 
#     verticalalignment='center', 
#     horizontalalignment='left'
#     )

# # Customize the y-axis (optional)
# axes[2].annotate(
#     '', 
#     xy=(0, 1.5), 
#     xytext=(0, -1),
#     arrowprops=dict(
#         arrowstyle='->', 
#         color='black'
#         )
#     )
# axes[2].text(.25, 1.5, r'$\hat{EEG}(t)$', fontsize=16, verticalalignment='bottom', horizontalalignment='center')

# axes[2].text(
#     -.0025, 
#     -1.3, 
#     '$t_0$', 
#     fontsize=16, 
#     verticalalignment='center', 
#     horizontalalignment='left'
#     )
# # Set plot limits and display
# axes[2].set_xlim(-HalfSpan, HalfSpan)

# axes[2].set_xticks([])
# axes[2].set_yticks([])

# axes[2].set_ylim(-1, 1.5)
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/diagrama_matriz_diseño.png',
#     transparent=True,
#     dpi=600
#     )
# # fig.show()

# #### PESOS
# fig,ax=plt.subplots(
#     nrows=1,
#     ncols=1,
#     figsize=(5,3)
#     )

# # Plot the mTRF
# ax.plot(
#     -span_window, 
#     mTRF[::-1], 
#     color ='C0', 
#     linewidth=1.5
#     )
# ax.plot(
#     times, 
#     np.zeros(times.shape), 
#     color ='gray', 
#     linewidth=.025
#     )

# # Hide the default x-axis
# ax.spines['bottom'].set_position('zero')  # Move x-axis to the center
# ax.spines['left'].set_position('zero')   # Move y-axis to the center
# ax.spines['bottom'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Add an arrow for the x-axis
# ax.annotate(
#     '', 
#     xy=(-span_window[0]+.25, 0), 
#     xytext=(-span_window[-1]-.25, 0),
#     arrowprops=dict(
#             arrowstyle='->', 
#             color='black'
#             )
#         )

# # Add the x-axis label at the end of the arrow
# ax.text(
#     -span_window[0]+.25, 
#     0, 
#     r'$\tau$', 
#     fontsize=16, 
#     verticalalignment='center', 
#     horizontalalignment='left'
#     )

# # Customize the y-axis (optional)
# ax.annotate(
#     '', 
#     xy=(0, 0.15), 
#     xytext=(0, -.25),
#     arrowprops=dict(
#         arrowstyle='->', 
#         color='black'
#         )
#     )
# ax.text(
#     .25,# 1, 
#     0.15,#.5, 
#     r'$TRF(\tau)$', 
#     fontsize=16, 
#     verticalalignment='bottom', 
#     horizontalalignment='center'
#     )

# # Set plot limits and display
# # ax.set_xlim(-HalfSpan, HalfSpan)
# ax.tick_params(axis='x', which='minor', length=0)

# ax.set_yticks([])
# # ax.set_xticks([span_window[0], span_window[-1]])
# # ax.set_xticklabels(['', ''])
# ax.set_xticks([])
# ax.set_xticklabels([])
# ax.set_yticks([])
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/diagrama_matriz_diseño2.png',
#     transparent=True,
#     dpi=600
#     )

# # =======================================
# # ENVOLVENETE + AUDIO + EEG PARA DIAGRAMA DE PIPELINE
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
#     color='C0',
#     linewidth=1
#     )

# plt.axis('off')
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_envolvente_diagrama1.png',
#     transparent=True
#     )
# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )
# plt.plot(
#     time_envelope[window_envelope], 
#     envelope[window_envelope], 
#     color='C4',
#     linewidth=2
#     )
# plt.axis('off')
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_envolvente_diagrama2.png',
#     transparent=True
#     )
# fig.show()

# EegPath = f'Datos/EEG/S21/s21-{1}-Trial13-Deci-Filter-Trim-ICA-Pruned.set'
# WavPath = f'Datos/wavs/S21/s21.objects.01.channel{1}.wav'
# raw = mne.io.read_raw_eeglab(
#     EegPath, 
#     preload=True,
#     verbose='CRITICAL'
#     )
# sr, audio = wavfile.read(WavPath)
# fig = plt.figure(
#     figsize=(6, 5),
#     tight_layout=True
#     )
# for k, ch in enumerate(raw.get_data()[::25,:]):
#     plt.plot(
#         ch[6000:16000]*1e1 + k*2.5e-3, 
#         color='gray'
#         )
# plt.axis('off')
# # fig.show()
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_envolvente_diagrama3.png',
#     transparent=True
#     )

# # Get average weights across subjects
# situation='External'
# mtrf_path = os.path.normpath(f'saves/{config.model}/{situation}/weights//stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
# weights = load_pickle(path=os.path.join(mtrf_path, 'Theta', 'Envelope', 'total_weights_per_subject.pkl'))['average_weights_subjects'].mean(axis=0).mean(axis=1)

# # Create figure and title
# fig, axes = plt.subplots(
#                         nrows=1, 
#                         ncols=1, 
#                         layout="constrained",
#                         figsize=(6, 5),
#                         )
# evoked = mne.EvokedArray(data=weights, info=config.info_mne)
# evoked.shift_time(config.times[0], relative=True)
# ev = evoked.plot(
#             scalings={'eeg':1}, 
#             zorder='std', 
#             # gfp=True,
#             time_unit='ms',
#             show=False, 
#             spatial_colors=True, 
#             # unit=False, 
#             titles=None,
#             window_title=None,
#             units='mTRF (a.u.)',
#             axes=axes,
#             # legend=False
#             )
# ev.delaxes(ev.axes[-1])  # Eliminarlo
        
# axes.plot(
#         config.times*1000, #ms
#         evoked._data.mean(0), 
#         'k-', 
#         zorder=130, 
#         linewidth=2
#         )

# # frame = axes.legend(loc="upper right", edgecolor="#d9ead3ff")
# for text in axes.texts:
#     text.set_visible(False)
# # Graph properties
# axes.set_title("")
# axes.axis('off')
# # fig.show()
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_envolvente_diagrama4.png',
#     transparent=True
#     )
# # =====
# # Phones
# PhonesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phones-Discrete-Phonet/Sesion21.pkl"
# NumberOfTicks = 34

# phones = load_pickle(path=PhonesPath)[0][:9168]
# WindowLeft, WindowRight = 0, len(phones)/config.sr

# time_phones = np.arange(0, len(phones)/config.sr, 1/config.sr)
# window_phones = (WindowLeft <= time_phones) & (time_phones <= WindowRight)

# ph_labels_phonet = ['B', 'D', 'F', 'G', 'N', 'T', 'a', 'b', 'd', 'e', \
#                     'f', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', \
#                     'r', 'rr', 's', 't', 'tS', 'u', 'w', 'x', 'z', 'Z', \
#                     'g', 'S', 'J', 'L', 'sil', '<p:>']
# ph_labels_phonet.remove('sil')
# ph_labels_phonet.remove('<p:>')

# tags = [r'b\textsubscript{2}', r'd\textsubscript{2}', r'f\textsubscript{2}', r'g\textsubscript{2}', r'n\textsubscript{2}', r'tS\textsubscript{3}', r'a', r'b\textsubscript{1}', r'd\textsubscript{1}', r'e', \
#         r'f\textsubscript{1}', r'i\textsubscript{1}', r'i\textsubscript{2}', r'x\textsubscript{2}', r'k', r'l', r'm', r'n\textsubscript{1}', r'o', r'p', \
#         r'R', r'r', r's\textsubscript{1}', r't', r'tS\textsubscript{1}', r'u\textsubscript{1}', r'u\textsubscript{2}', r'x\textsubscript{1}', r's\textsubscript{3}', r's\textsubscript{4}', \
#         r'g\textsubscript{1}', r'tS\textsubscript{2}', r'x\textsubscript{3}', r'L']

# ticks = np.arange(0, NumberOfTicks, 1)+.5

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(8, 8)
#     )
# im = plt.imshow(
#     phones[window_phones].T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu'  # Ajusta el mapa de colores
#     )
# plt.colorbar(
#     im,
#     label='Amplitud'
#     )

# plt.yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Fonos')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_phones.svg',
#     transparent=True,
#     )
# # fig.show()

# =====
# Phonemes
PhonemesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonemes-Discrete-Phonet/Sesion21.pkl"
NumberOfTicks = 21

phonemes = load_pickle(path=PhonemesPath)[0][:9168]
WindowLeft, WindowRight = 0, len(phonemes)/config.sr

time_phonemes = np.arange(0, len(phonemes)/config.sr, 1/config.sr)
window_phonemes = (WindowLeft <= time_phonemes) & (time_phonemes <= WindowRight)

# tags = config.Exp_info().phonemes_phonet.copy()
# tags.remove('/sil/')
tags = ['/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/', '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/']
ticks = np.arange(0, NumberOfTicks, 1)+.5

fig = plt.figure(
    tight_layout=True,
    figsize=(8, 6)
    )
im = plt.imshow(
    phonemes[window_phonemes].T,
    aspect='auto',  # Ajusta el aspecto
    extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
    origin='lower',  # Ajusta el origen
    cmap='RdBu'  # Ajusta el mapa de colores
    )
plt.colorbar(
    im,
    label='Amplitud'
    )

plt.yticks(
    ticks=ticks, 
    labels=tags
    )
plt.xlabel('Tiempo (s)')
plt.ylabel('Fonemas')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_phonemes.svg',
#     transparent=True,
#     )
fig.show()

# # =====
# # Phonological
# PhonologicalPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonological/Sesion21.pkl"
# NumberOfTicks = 18

# phonological = load_pickle(path=PhonologicalPath)[0][:9168]
# WindowLeft, WindowRight = 0, len(phonological)/config.sr

# time_phonological = np.arange(0, len(phonological)/config.sr, 1/config.sr)
# window_phonological = (WindowLeft <= time_phonological) & (time_phonological <= WindowRight)

# tags = list(config.Exp_info().phonological_labels.keys())
# ticks = np.arange(0, NumberOfTicks, 1)+.5

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(8, 6)
#     )
# im = plt.imshow(
#     phonological.T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu'  # Ajusta el mapa de colores
#     )
# plt.colorbar(
#     im,
#     label='Amplitud (U.A)'
#     )

# plt.yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Características fonológicas')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_phonological.svg',
#     transparent=True
#     )
# # fig.show()

# # =====
# # Mfccs
# MfccsPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Mfccs/Sesion21.pkl"
# NumberOfTicks = 16

# mfccs = load_pickle(path=MfccsPath)[0][:9168]
# WindowLeft, WindowRight = 0, len(mfccs)/config.sr

# time_mfccs = np.arange(0, len(mfccs)/config.sr, 1/config.sr)
# window_mfccs = (WindowLeft <= time_mfccs) & (time_mfccs <= WindowRight)

# tags = [f'M{i}' for i in np.arange(1, NumberOfTicks, 2)]
# ticks = np.arange(0, NumberOfTicks, 2)+.5

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )
# im = plt.imshow(
#     mfccs.T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu'  # Ajusta el mapa de colores
#     )
# plt.colorbar(
#     im,
#     label='Amplitud (U.A)'
#     )

# plt.yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Coeficientes Mel')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_mfccs.svg',
#     transparent=True
#     )
# # fig.show()

# # =============
# # Espectrograma
# SpectrogramPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Spectrogram/Sesion21.pkl"
# NumberOfTicks = 16

# spectrogram = load_pickle(path=SpectrogramPath)[0][:9168]
# WindowLeft, WindowRight = 0, len(spectrogram)/config.sr

# time_spectrogram = np.arange(0, len(spectrogram)/config.sr, 1/config.sr)
# window_spectrogram = (WindowLeft <= time_spectrogram) & (time_spectrogram <= WindowRight)

# bands_center = librosa.mel_frequencies(
#     n_mels=NumberOfTicks+2, 
#     fmin=62, 
#     fmax=8000
#     )[1:-1]

# # tags = [int(bands_center[i]) for i in np.arange(1, len(bands_center)+1, 2)]
# # ticks = np.arange(0, NumberOfTicks, 2)+.5
# tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center))]
# ticks = np.arange(0, NumberOfTicks)+.5

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )
# im = plt.imshow(
#     spectrogram.T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, 16],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu'  # Ajusta el mapa de colores
#     )
# plt.colorbar(
#     im,
#     label='Amplitud (dB)'
#     )

# plt.yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Frecuencia (Hz)')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_espectrograma.svg',
#     transparent=True
#     )
# # fig.show()

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

# ===============================
# Envlovente de la señal de audio
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
# # fig.savefig(
# #     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_envolvente.svg',
# #     transparent=True
# #     )
# fig.show()

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