import numpy as np, os, pandas as pd

from scipy.io import wavfile
from scipy import signal
import librosa
import mne

from matplotlib.collections import PathCollection
from matplotlib.ticker import ScalarFormatter
from matplotlib_venn import venn3, venn2
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt 
from matplotlib import rc
import seaborn as sns
import scienceplots

from processing import clustering_by_correlation
from funciones import load_pickle
from plot import define_ticks
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
plt.style.use(['science'])

# # ===============================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: SPECTROGRAM
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Spectrogram.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram/total_weights_per_subject.pkl'

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, singificant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# # average_correlation_subjects = np.where((singificant_channels_subjects==1), average_correlation_subjects, np.nan)
# # np.nanmean(average_correlation_subjects, axis=0).mean()
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] # (18, 128, 1, 104)

# # Crear una figura
# fig = plt.figure(
#     figsize=(12, 9),
#     tight_layout=True
#     )

# # Definir la cuadrícula usando GridSpec
# # 2 filas y 2 columnas, con la segunda columna dividida en dos partes en la primera fila
# gs = gridspec.GridSpec(
#     nrows=2, 
#     ncols=2, 
#     width_ratios=[1, 1.2], 
#     height_ratios=[1, 2]
#     )

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# ax1 = plt.subplot(gs[0, 0])
# weights = average_weights_subjects.mean(axis=0).mean(axis=1)
# evoked = mne.EvokedArray(data=weights, info=config.info_mne)
# evoked.shift_time(config.times[0], relative=True)
# evoked_plot = evoked.plot(
#     scalings={'eeg':1}, 
#     zorder='std', 
#     time_unit='ms',
#     show=False, 
#     spatial_colors=True, 
#     # unit=False, 
#     units='mTRFs (U.A)',
#     axes=ax1,
#     gfp=False
#     )
# # Eliminar la etiqueta "Nave"
# for text in evoked_plot.axes[0].texts:
#     if "ave" in text.get_text():
#         text.set_visible(False)  # Ocultar el texto
# ax1.plot(
#     config.times*1e3, #ms
#     evoked._data.mean(axis=0), 
#     'black', 
#     label='Valor medio', 
#     zorder=130, 
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in ax1.get_lines()[:len(evoked.ch_names)]]

# # Eliminar el esquema de la cabeza original
# for ax in fig.axes:
#     # Verificar si el eje contiene un objeto de tipo "PathCollection" (los puntos de los canales)
#     for artist in ax.get_children():
#         if isinstance(artist, PathCollection):
#             ax.remove()  # Eliminar el eje que contiene el esquema de la cabeza original
#             break

# # Obtener las posiciones de los sensores en 2D
# montage = evoked.info.get_montage()
# pos = montage.get_positions()['ch_pos']  # Diccionario con las posiciones de los canales

# # Crear un eje adicional para la cabecita sin sensores
# ax_head_outline = fig.add_axes([.3, 0.831, 0.12, 0.12])  # [x, y, width, height]

# # Graficar solo el contorno de la cabeza (sin sensores)
# mne.viz.plot_topomap(
#     np.zeros(len(evoked.ch_names)),  # Datos ficticios (todos ceros)
#     evoked.info,
#     axes=ax_head_outline,
#     show=False,
#     sensors=False,  # No graficar los sensores
#     outlines='head'  # Graficar solo el contorno de la cabeza
# )
# ax_head_outline.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_outline.axis('off')  # Ocultar los ejes

# # Crear un eje adicional para graficar los sensores
# ax_head = fig.add_axes([.31, 0.833, 0.1, 0.1])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.55,.17))

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)
# feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# im = ax2.pcolormesh(
#     config.times * 1e3, 
#     np.arange(feat_weights.shape[0]), 
#     feat_weights, 
#     cmap='RdBu_r', 
#     shading='auto',
#     vmin=feat_weights.min(),
#     vmax=np.abs(feat_weights).max()
#     )

# # Set figure configuration
# bands_center = librosa.mel_frequencies(n_mels=feat_weights.shape[0]+2, fmin=0, fmax=16000/2)[1:-1]
# # tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)]
# tags = [int(bands_center[i]) for i in np.arange(len(bands_center))]
# ticks = np.arange(feat_weights.shape[0])
# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600], 
#     ylabel='Frecuencia (Hz)', 
#     yticks=ticks, 
#     yticklabels=tags
#     )

# # Configure colorbar
# fig.colorbar(
#     im, 
#     ax=ax2, 
#     orientation='horizontal', 
#     shrink=1, 
#     label='Amplitud (U.A)', 
#     fraction=.05,
#     aspect=50
#     )

# # Dividir la primera fila de la segunda columna en dos partes HORIZONTALES
# # Usar GridSpecFromSubplotSpec para dividir la celda (0, 1) en 2 columnas
# gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.4)

# # Tercer gráfico en la primera subcolumna de la segunda columna (primera fila)
# ax3 = plt.subplot(gs_sub[0])
# mean_average_correlation = average_correlation_subjects.mean(axis=0)
# im = mne.viz.plot_topomap(
#         data=mean_average_correlation, 
#         pos=config.info_mne, 
#         cmap='Reds',
#         vlim=(mean_average_correlation.min(), mean_average_correlation.max()),
#         show=False, 
#         sphere=0.07, 
#         axes=ax3
#         )
# fig.colorbar(
#         im[0],
#         ax=ax3, 
#         shrink=0.85,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min().round(decimals=3), mean_average_correlation.max().round(decimals=3), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2)
#         )
# ax3.axis('off')  # Desactivar ejes
# ax3.set(title=r'Correlación: $('+ f'{mean_average_correlation.mean():.2f}\pm{mean_average_correlation.std():.2f}'+r')$')

# # Cuarto gráfico en la segunda subcolumna de la segunda columna (primera fila)
# ax4 = plt.subplot(gs_sub[1])

# n_subjects, n_chan, _, n_delays = average_weights_subjects.shape
# average_weights = average_weights_subjects.mean(axis=2)# across delays
# correlation_matrices = np.zeros(shape=(n_chan, n_subjects, n_subjects))

# # Calculate correlation betweem subjects
# for channel in range(n_chan):
#     matrix = average_weights[:,channel,:] 
#     correlation_matrices[channel] = np.corrcoef(matrix)

# # Correlacion por canal
# absolute_correlation_per_channel = np.zeros(n_chan)
# for channel in range(n_chan):
#     channel_corr_values = correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)]
#     absolute_correlation_per_channel[channel] = np.mean(np.abs(channel_corr_values))

# im = mne.viz.plot_topomap(
#     data=absolute_correlation_per_channel, 
#     pos=config.info_mne, 
#     axes=ax4, 
#     show=False, 
#     sphere=0.07,
#     cmap='Greens', 
#     vlim=(absolute_correlation_per_channel.min(),absolute_correlation_per_channel.max())    
#     )
        
# # Make colorbar
# fig.colorbar(
#     im[0], 
#     ax=ax4, 
#     shrink=0.85, 
#     orientation='horizontal', 
#     boundaries=np.linspace(absolute_correlation_per_channel.min().round(decimals=3), absolute_correlation_per_channel.max().round(decimals=3), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2)
#     )
#     # boundaries=np.linspace(mean_average_correlation.min().round(decimals=3), mean_average_correlation.max().round(decimals=3), 100),
#     # ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2)
#     # )
# ax4.set(title=r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.2f}\pm{absolute_correlation_per_channel.std():.2f}'+r')$')
# ax4.axis('off')  # Desactivar ejes

# # Quinto gráfico en la segunda columna (segunda fila)
# ax5 = plt.subplot(gs[1, 1])

# weights_across_features = average_weights_subjects.mean(axis=2) # nsubjects, nchans, ndelays
# mean_across_subjects = weights_across_features.mean(axis=0) # nchans, ndelays

# # To store correlation matrix of each channel, it has an extra dimension to correlate against whole average
# correlation_matrices_of_each_channel = np.zeros(
#     shape=(n_chan, n_subjects+1, n_subjects+1)
#     ) 

# # Add mean of all subjects to the weights
# weights_across_features_plus_mean = np.concatenate(
#     (weights_across_features, 
#         mean_across_subjects.reshape(1, n_chan, n_delays) # to match weight_across_features shape
#         ),
#     axis=0
#     )

# # For each channel, correlation across time delays is computed to get a matrix of n_subjects+1 x n_subjects+1
# for channel in range(n_chan):
#     matrix = weights_across_features_plus_mean[:,channel,:] # nsubjects+1, ndelays
#     correlation_matrices_of_each_channel[channel] = np.corrcoef(matrix)

# # Take average across all channels and exclude whole average
# correlation_matrix = correlation_matrices_of_each_channel.mean(axis=0)[:-1, :-1]

# # Now get the vector of correlations of channels vs whole average, excluding whole vs whole
# correlation_of_channel_vs_average = correlation_matrices_of_each_channel.mean(axis=0)[-1][:-1]

# # Change diagonal for values with correlations of channels vs whole average. By doing so, it's very unlinkely to find a 1 in the diagonal.
# for i in range(n_subjects):
#     correlation_matrix[i, i] = correlation_of_channel_vs_average[i]

# # Get a list of subjects
# subject_names = np.arange(1, n_subjects+1).tolist()

# # Make mask for lower triangle of correlation matrix (this is a symmetric matrix)
# mask = np.ones_like(correlation_matrix)
# mask[np.tril_indices_from(mask)] = False

# # Take average
# correlation_mean, correlation_std = np.mean(np.abs(correlation_of_channel_vs_average)), np.std(np.abs(correlation_of_channel_vs_average))

# sns.heatmap(
#     correlation_matrix, 
#     mask=mask, 
#     cmap="RdBu_r", 
#     fmt='.1f', 
#     ax=ax5,
#     annot=True, 
#     center=0, 
#     xticklabels=True, 
#     annot_kws={"size": 10},
#     cbar=False
#     )
# ax5.set_yticklabels(['Media'] + subject_names[1:], rotation=35)
# ax5.set_xticklabels(subject_names[:-1] + ['Media'], rotation=35)
# cbar = fig.colorbar(
#     ax5.collections[0],  # Usamos la primera colección mappable del heatmap
#     orientation="horizontal",
#     ax=ax5,
#     fraction=.05,
#     aspect=50,
#     label='Correlación'
#     )
# fig.savefig(
#     'C:/Users/jocta/Documents/tesis_escrita/imagenes/resultados/espectrograma_completo.svg',
#     transparent=True
#     )
# # fig.show()

# # ===========================
# # EJEMPLO DE DIAGRAMA DE VENN
# fig = plt.figure(
#     figsize=(4,4),
#     layout='tight'
#     )

# # Make plot
# venn2(
#     subsets=(.3,.1,.1), # left area diagran, right area diagram, shared area <--> (10, 01, 11)
#     set_labels=('Atributo 1', 'Atributo 2'),
#     set_colors=('C0', 'C1'), 
#     alpha=0.45
#     )
# # Save figure
# fig.savefig(
#     'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/ejemplo_venn2.svg',
#     transparent=True
#     )
# # fig.show()
# fig = plt.figure(
#     figsize=(4,4),
#     layout='tight'
#     )

# # Make plot
# venn3(
#     subsets=(.3,
#              .1,
#              .3,
#              .12,
#              .1,
#              .1,
#              .13), # the order should be(100, 010, 110, 001, 101, 011, 111)
#     set_labels=('Atributo 1', 'Atributo 2', 'Atributo 3'),
#     set_colors=('C0', 'C1', 'C2'), 
#     alpha=0.45
#     )
# # Save figure
# fig.savefig(
#     'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/ejemplo_venn3.svg',
#     transparent=True
#     )
# # fig.show()

# # ===============================
# # EJEMPLO PRUEBA DE PERMUTACIONES
# from sklearn.model_selection import KFold
# from funciones import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
# from model_implementations import fold_model
# from processing import tfce 
# from load import load_data
# import config, plot

# situation, band, stim, sesion, sujeto = 'External', 'Theta', 'Pitch-Log-Raw', 24, 1
# preprocessed_data_path = f'saves/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/'
# path_null = f'saves/{config.model}/{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
# path_validation = f'saves/{config.model}/{situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
# alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')
# average_weights_subjects = []
# average_correlation_subjects = []
# average_rmse_subjects = []
# pvalues_corr_subjects = []
# pvalues_rmse_subjects = []
# repeated_good_correlation_channels_subjects = []
# repeated_good_rmse_channels_subjects = []
# print(f'\n------->\tStart of session {sesion}\n')

# # Load data by subject, EEG and info
# sujeto_1, sujeto_2, samples_info = load_data(
#                                 sesion=sesion,
#                                 stim=stim,
#                                 band=band,
#                                 sr=config.sr,
#                                 delays=config.delays,
#                                 preprocessed_data_path=preprocessed_data_path,
#                                 praat_executable_path=config.praat_executable_path,
#                                 situation=situation
#                                 )
# eeg, info = sujeto_1['EEG'], sujeto_1['info']

# # Load stimuli by subject (i.e: concatenated stimuli features)
# stims = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')])
# n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
# delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]
# relevant_indexes = samples_info['keep_indexes1'].copy()
# weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
# correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
# rmse_per_channel = np.zeros((config.n_folds, info['nchan']))
# topo_pvalues_corr_per_fold = np.zeros((config.n_folds, info['nchan']))
# topo_pvalues_rmse_per_fold = np.zeros((config.n_folds, info['nchan']))
# proba_correlation_per_channel = np.ones((config.n_folds, info['nchan']))
# proba_rmse_per_channel = np.ones((config.n_folds, info['nchan']))
# print(f'\n\t······  Running model for Subject {sujeto}\n')
# if config.set_alpha is None:
#     try:
#         alphas = load_pickle(path=alphas_path)
#         alpha = alphas[sesion][sujeto]
#     except:
#         alpha = config.default_alpha
# else:
#     alpha = config.set_alpha
# kf_test = KFold(config.n_folds, shuffle=False)
# relevant_eeg = eeg[relevant_indexes]
# k_models_output = []
# for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
#     print(f'\n\t······  [{fold+1}/{config.n_folds}]')
#     k_models_output.append(
#                     fold_model(
#                         fold=fold,
#                         alpha=np.float32(alpha),#TODO adapt inside
#                         stims=stims,
#                         eeg=eeg,
#                         relevant_indexes=relevant_indexes,
#                         train_indexes=train_indexes,
#                         test_indexes=test_indexes,
#                         validation=False,
#                         statistical_test=config.statistical_test,
#                         path_null=path_null,
#                         session=sesion,
#                         subject=sujeto,                              
#                         )
#                     )
# for output_k in k_models_output:
#     fold, weights, correlation_matrix, root_mean_square_error = output_k[:4]
#     weights_per_fold[fold] = weights
#     correlation_per_channel[fold] = correlation_matrix
#     rmse_per_channel[fold] = root_mean_square_error 
#     p_corr, p_rmse, null_correlation_per_channel = output_k[4:]
#     proba_correlation_per_channel[fold][p_corr < config.significance_threshold] = p_corr[p_corr < config.significance_threshold]
#     proba_rmse_per_channel[fold][p_rmse < config.significance_threshold] = p_rmse[p_rmse < config.significance_threshold]
#     topo_pvalues_corr_per_fold[fold] = p_corr
#     topo_pvalues_rmse_per_fold[fold] = p_rmse
# print(f'\n\t······  Run model\n')
# for k, weight in enumerate(weights_per_fold):
#     if (weight==0).all():
#         weights_per_fold[k] = np.full(shape=weight.shape, fill_value=np.nan)
#         print(
#             f'\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>\n'
#             f'\t\tFold {k+1}/{config.n_folds} weights are empty\n'
#             f'\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>'
#             )
# average_weights = np.nanmean(weights_per_fold, axis=0) # info['nchan'], np.sum(n_feats), len(delays)
# average_weights = np.nan_to_num(average_weights)
# average_correlation = np.nanmean(correlation_per_channel, axis=0)
# average_correlation = np.nan_to_num(average_correlation)
# average_rmse = rmse_per_channel.mean(axis=0)
# corr_good_channel_indexes = []
# rmse_good_channel_indexes = []
# repeated_good_correlation_channels = np.zeros(info['nchan'])
# repeated_good_rmse_channels = np.zeros(info['nchan'])
# # Find good indexes by checking where all folds (at the same time) are significant
# try:
#     corr_good_channel_indexes, = np.where(
#                                 np.all((proba_correlation_per_channel < 1), axis=0)
#                                 )
#     rmse_good_channel_indexes, = np.where(
#                                 np.all((proba_rmse_per_channel < 1), axis=0)
#                                 )
# except:
#     corr_good_channel_indexes = []
#     rmse_good_channel_indexes = []
#     print('No significant channels found')   

# # Saves passing channels by subject
# repeated_good_correlation_channels[corr_good_channel_indexes] += 1 # binary array with ones where significant
# repeated_good_rmse_channels[rmse_good_channel_indexes] += 1
# average_correlation = correlation_per_channel.mean(axis=0)
# channels = np.arange(len(average_correlation))
# null_correlation_per_channel_min = null_correlation_per_channel.min(axis=1).min(axis=0)
# null_correlation_per_channel_max = null_correlation_per_channel.max(axis=1).max(axis=0) 

# # Create figure and title
# fig, ax = plt.subplots(
#     nrows=1, 
#     ncols=1, 
#     figsize=(6, 4), 
#     layout='tight'
#     )
# ax.scatter(
#     corr_good_channel_indexes, 
#     .28*np.ones(shape=len(corr_good_channel_indexes)), 
#     marker='*', 
#     s=15,
#     color='black', 
#     label="Valores significativos"
#     )

# ax.fill_between(
#     x=channels, 
#     y1=null_correlation_per_channel_min,
#     y2=null_correlation_per_channel_max, 
#     alpha=.5,
#     label='Distribución nula',
#     color='orange'
#     )

# ax.fill_between(
#     x=channels, 
#     y1=np.percentile(null_correlation_per_channel.min(axis=0), 50-25, axis=0),
#     y2=np.percentile(null_correlation_per_channel.min(axis=0), 50+25, axis=0), 
#     alpha=.8,
#     # label=r'50 \% de la distribución nula',
#     color='orange'
#     )
# # Add shadow between min and max
# ax.fill_between(
#     x=channels, 
#     y1=correlation_per_channel.min(axis=0), # min across all folds
#     y2=correlation_per_channel.max(axis=0), 
#     alpha=.5,
#     label='Distribucion de la correlación',
#     color='C0'
#     )
# ax.scatter(
#     channels,
#     average_correlation, 
#     s=5,
#     color='C0', 
#     alpha=1,
#     label="Correlación media entre particiones"
#     )

# # Graph properties
# ax.grid(visible=True)
# ax.set(
#     xlim=[-1, 129],
#     xlabel='Canales de EEG',
#     ylabel='Correlación'
#     )
# ax.legend(loc=(.2,.2))
# fig.savefig(
#     'C:/Users/User/Documents/tesis_escrita/imagenes/metodos/prueba_permutaciones.svg', # 'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/prueba_permutaciones.svg',
#     transparent=True
#     )
# # fig.show()

# # ===================
# # EJEMPLOS VALIDACIÓN
# from load import load_data
# from tqdm import tqdm 
# from model_implementations import fold_model
# from sklearn.model_selection import KFold

# situation, band, sesion = 'External', 'Theta', 21
# preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/')

# correlations_T, correlations_std_T, alpha_subject_T = [],[],[]
# for stim in ['Envelope', 'Spectrogram']:
#     sujeto_1, sujeto_2, samples_info = load_data(
#                                                 sesion=sesion,
#                                                 stim=stim,
#                                                 band=band,
#                                                 sr=config.sr,
#                                                 delays=config.delays,
#                                                 preprocessed_data_path=preprocessed_data_path,
#                                                 praat_executable_path=config.praat_executable_path,
#                                                 situation=situation
#                                                 )
#     eeg_sujeto_1, eeg_sujeto_2, info = sujeto_1['EEG'], sujeto_2['EEG'], sujeto_1['info']
#     stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')]) 
#     stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])
#     n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
#     delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]
#     relevant_indexes_1 = samples_info['keep_indexes1'].copy()
#     relevant_indexes_2 = samples_info['keep_indexes2'].copy()
#     subject, eeg, stims, relevant_indexes = 1, eeg_sujeto_1, stims_sujeto_1, relevant_indexes_1
#     print(f'\n\n\t······  Running model for Subject {subject}\n')
#     correlations = np.zeros(len(config.alphas_swept))
#     correlations_std = np.zeros(len(config.alphas_swept))

#     # Make sweep
#     for i_alpha, alpha in tqdm(enumerate(config.alphas_swept), total=len(config.alphas_swept), desc='Sweeping progress'):
#         weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
#         correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
#         kf_test = KFold(config.n_folds, shuffle=False)
#         relevant_eeg = eeg[relevant_indexes]
#         k_models_output = []
#         for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
#             k_models_output.append(
#                             fold_model(
#                             fold=fold,
#                             alpha=alpha,
#                             stims=stims,
#                             eeg=eeg,
#                             relevant_indexes=relevant_indexes,
#                             train_indexes=train_indexes,
#                             test_indexes=test_indexes,                              
#                             ) 
#                             )     
#         for fold, weights, correlation_matrix, root_mean_square_error in k_models_output:
#             weights_per_fold[fold] = weights
#             correlation_per_channel[fold] = correlation_matrix
#         correlations[i_alpha] = np.nan_to_num(np.nanmean(correlation_per_channel))
#         correlations_std[i_alpha] = np.nan_to_num(np.nanstd(correlation_per_channel))

#     # Find all indexes where the relative difference between the correlation and its maximum is within corr_limit_percent
#     relative_difference = abs((correlations.max() - correlations)/correlations.max())
#     good_indexes_range = np.where(relative_difference < config.val_correlation_limit_percentage)[0]

#     # Get the very last one, because the greater the alpha, the smoothest the signal gets
#     alpha_subject = config.alphas_swept[int(good_indexes_range[-1])]
#     correlations_T.append(correlations)
#     correlations_std_T.append(correlations_std)
#     alpha_subject_T.append(alpha_subject)

# # Create figure and plot
# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=2,
#     figsize=(12,4), 
#     tight_layout=True,
#     sharey=True
#     )

# # Plot alphas vs correlations as dots with errorbars
# axes[0].plot(
#     config.alphas_swept, 
#     correlations_T[1], 
#     'o--'
#     )
# axes[0].errorbar(
#     config.alphas_swept, 
#     correlations_T[1], 
#     yerr=correlations_std_T[1]/np.sqrt(5), 
#     fmt='none', 
#     ecolor='black',
#     elinewidth=0.5, 
#     capsize=0.5
#     )

# # Make vlines for maximum correlation and selected alpha
# axes[0].vlines(
#     config.alphas_swept[correlations_T[1].argmax()], 
#     # axes[0].get_ylim()[0], 
#     # axes[0].get_ylim()[1],
#     0,
#     1, 
#     linestyle='dashed',
#     color='black', 
#     linewidth=1.5, 
#     label='Máxima correlación'
#     )

# # Find relevant range within correlation_limit_percentage
# relative_difference = abs((correlations_T[1].max() - correlations_T[1])/correlations_T[1].max())
# good_indexes_range = np.where(relative_difference < config.val_correlation_limit_percentage)[0]    

# # Make green box of range within config.val_correlation_limit_percentage
# if good_indexes_range.size > 1:
#     axes[0].axvspan(
#         config.alphas_swept[good_indexes_range[0]], config.alphas_swept[good_indexes_range[-1]], 
#         alpha=0.2, 
#         color='orange',
#         label=f'{100-int(config.val_correlation_limit_percentage*100)}'+r'\% de la máxima'
#         )
# axes[0].vlines(
#     alpha_subject_T[1], 
#     # axes[0].get_ylim()[0], 
#     # axes[0].get_ylim()[1],
#     0,
#     1, 
#     color='red', 
#     alpha=.8,
#     linewidth=1.5, 
#     label=f'Valor seleccionado'
#     )
# # Axes parameters
# axes[0].set(
#     title='Espectrograma', 
#     xlabel=r'Parámetro de regularización $\alpha$', 
#     ylabel='Correlación promedio', 
#     xscale='log', 
#     xlim=([config.alphas_swept[0], config.alphas_swept[-1]]),
#     ylim=(0.2,0.6),
#     yticks=np.arange(.25,.6,.05)
#     )
# axes[0].grid(visible=True)
# axes[0].legend(loc='best', fontsize=14)

# # Plot alphas vs correlations as dots with errorbars
# axes[1].plot(
#     config.alphas_swept, 
#     correlations_T[0], 
#     'o--'
#     )
# axes[1].errorbar(
#     config.alphas_swept, 
#     correlations_T[0], 
#     yerr=correlations_std_T[0]/np.sqrt(5), 
#     fmt='none', 
#     ecolor='black',
#     elinewidth=0.5, 
#     capsize=0.5
#     )

# # Make vlines for maximum correlation and selected alpha
# axes[1].vlines(
#     config.alphas_swept[correlations_T[0].argmax()], 
#     # axes[1].get_ylim()[0], 
#     # axes[1].get_ylim()[1], 
#     0,
#     1,
#     linestyle='dashed',
#     color='black', 
#     linewidth=1.5, 
#     label='Máxima correlación'
#     )

# # Find relevant range within correlation_limit_percentage
# relative_difference = abs((correlations_T[0].max() - correlations_T[0])/correlations_T[0].max())
# good_indexes_range = np.where(relative_difference < config.val_correlation_limit_percentage)[0]    

# # Make green box of range within config.val_correlation_limit_percentage
# if good_indexes_range.size > 1:
#     axes[1].axvspan(
#         config.alphas_swept[good_indexes_range[0]], config.alphas_swept[good_indexes_range[-1]], 
#         alpha=0.2, 
#         color='orange',
#         label=f'{100-int(config.val_correlation_limit_percentage*100)}'+r'\% de la máxima'
#         )
# axes[1].vlines(
#     alpha_subject_T[0], 
#     # axes[1].get_ylim()[0], 
#     # axes[1].get_ylim()[1], 
#     0,
#     1,
#     color='red', 
#     alpha=.8,
#     linewidth=1.5, 
#     label=f'Valor seleccionado'
#     )

# # Axes parameters
# axes[1].set(
#     title='Envolvente', 
#     xlabel=r'Parámetro de regularización $\alpha$', 
#     # ylabel='Correlación promedio', 
#     xscale='log', 
#     xlim=([config.alphas_swept[0], config.alphas_swept[-1]]),
#     ylim=(0.2, 0.6)
#     )
# axes[1].grid(visible=True)
# axes[1].legend(loc='upper right', fontsize=14)
# fig.savefig(
#     'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/validacion.svg',
#     transparent=True
#     )
# # fig.show()
            
# # =============
# # EJEMPLOS TFCE
# SpectrogramTfcePath = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram_4096.pkl'
# PhonemesTfcePath = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Discrete-Phonet_4096.pkl'
# _, sp_pvalue_tfce = load_pickle(path=SpectrogramTfcePath)
# _, ph_pvalue_tfce = load_pickle(path=PhonemesTfcePath)

# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=2,
#     figsize=(14, 6)
#     )

# # Spectrogram
# NumberOfFeats = 16

# # Mask and transformation
# pvals_for_graph = sp_pvalue_tfce.copy()
# pvals_for_graph[sp_pvalue_tfce>config.significance] = 1
# pvals_for_graph = -np.log10(pvals_for_graph)

# im_sp = axes[0].pcolormesh(
#     config.times*1e3, # x
#     np.arange(pvals_for_graph.shape[1]), # y
#     pvals_for_graph.T, # z
#     shading='auto',
#     cmap='inferno'
#     )

# bands_center = librosa.mel_frequencies(n_mels=NumberOfFeats+2, fmin=0, fmax=8000)[1:-1]
# tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center))]
# ticks = np.arange(0, NumberOfFeats)

# axes[0].set(
#     xlabel='Tiempo (ms)', 
#     ylabel='Frecuencia (Hz)', 
#     yticks=ticks, 
#     yticklabels=tags,
#     # xticklabels=[-200,0,200,400,600]
#     )
# fig.colorbar( 
#     orientation='vertical', 
#     label=r"$-log_{10}(p_{values})$",
#     aspect=15, 
#     shrink=1, 
#     mappable=im_sp,
#     ax=axes[0]
#     )

# # Phonemes
# NumberOfFeats = 21
# significant_channels = np.zeros(shape=(NumberOfFeats, len(config.times)))

# # Iteate over columns to get number of channels per feature that passes the threshold
# for feature in range(NumberOfFeats):
#     for delay in range(len(config.times)):
#         # Count how many channels pass the threshold for a given feature and delay
#         ppval = ph_pvalue_tfce[feature][delay]
#         significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# # Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
# number_of_ticks = significant_channels.shape[0]
# y, z = np.arange(number_of_ticks), significant_channels

# imph = axes[1].pcolormesh(
#     config.times*1e3, # x
#     y, # y
#     z, # z
#     shading='auto',
#     cmap='inferno'
#     )
# tags = config.Exp_info().phonemes_phonet
# tags.remove('/sil/')
# axes[1].set(
#     xlabel='Tiempo (ms)', 
#     ylabel='Fonemas', 
#     yticks=np.arange(0, NumberOfFeats, 1),
#     yticklabels=tags
#     )

# fig.colorbar( 
#     orientation='vertical', 
#     label="Número de canales significativos",
#     aspect=15, 
#     shrink=1, 
#     mappable=imph,
#     ax=axes[1]
#     )
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/ejemplo_TFCE.svg',
#     transparent=True
#     )
# # fig.show()
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
# envelope = np.abs(signal.hilbert(audio))
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
#     cmap='RdBu_r'  # Ajusta el mapa de colores
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
# plt.ylabel('Fonos')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_phones.svg',
#     transparent=True,
#     )
# # fig.show()

# # ========
# # Phonemes
# PhonemesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonemes-Discrete-Phonet/Sesion21.pkl"
# NumberOfTicks = 21

# phonemes = load_pickle(path=PhonemesPath)[0][:9168]
# WindowLeft, WindowRight = 0, len(phonemes)/config.sr

# time_phonemes = np.arange(0, len(phonemes)/config.sr, 1/config.sr)
# window_phonemes = (WindowLeft <= time_phonemes) & (time_phonemes <= WindowRight)

# # tags = config.Exp_info().phonemes_phonet.copy()
# # tags.remove('/sil/')
# tags = ['/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/', '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/']
# ticks = np.arange(0, NumberOfTicks, 1)+.5

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(8, 6)
#     )
# im = plt.imshow(
#     phonemes[window_phonemes].T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu_r'  # Ajusta el mapa de colores
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
# plt.ylabel('Fonemas')  
# fig.savefig(
#     f'C:/Users/jocta/Documents/tesis_escrita/imagenes/metodos/sample_phonemes.svg',
#     transparent=True,
#     )
# # fig.show()

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
#     cmap='RdBu_r'  # Ajusta el mapa de colores
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
#     cmap='RdBu_r'  # Ajusta el mapa de colores
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
#     fmin=0, 
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
#     cmap='RdBu_r'  # Ajusta el mapa de colores
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
# envelope = np.abs(signal.hilbert(audio))
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

# =======================================================
# Ejemplo EEG y PSD (power spectral density) de un sujeto
sesion, sujeto = 21, 2
RawEegPath = f'Datos/EEG/S{sesion}/s{sesion}-{sujeto}-Trial1-Deci-Filter-Trim-ICA-Pruned.set'
# EegPath = 'saves/preprocessed_data/External/tmin-0.2_tmax0.6/EEG/All/Causal/Sesion21.pkl'

raw = mne.io.read_raw_eeglab(
        RawEegPath, 
        preload=True,
        verbose='CRITICAL',
        )
# raw = raw.filter(l_freq=.1, h_freq=40)
# raw.resample(sfreq=128)
raw.plot(
    scalings=dict(eeg=2e-5)
)

from processing import subsample


fmin, fmax = 0, 40

fig, ax = plt.subplots()
eeg = raw.get_data().T*1e6  # paso a array y tiro la primer columna de tiempo
eeg = subsample(x=eeg, step=int(raw.info.get("sfreq")/ 128))
psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(
        eeg.T, 
        128, 
        fmin, 
        fmax
        )
evoked = mne.EvokedArray(psds_welch_mean, config.info_mne)
evoked.times = freqs_mean
evoked.plot(
        scalings=dict(eeg=1, grad=1, mag=1), 
        zorder='std', 
        time_unit='s',
        show=False, 
        spatial_colors=True, 
        unit=False, 
        units='w', 
        axes=ax
        )
ax.set_xlabel('Frequency [Hz]')
ax.grid()
fig.show()













# eeg = load_pickle(path=EegPath)[0]
# raw = mne.io.RawArray(data=eeg.T*1e-6, info=raw_raw.info)
spectrum = raw.compute_psd(
    method='welch',
    fmin=.1, 
    fmax=40, 
    
    # n_fft=1096,       # Aumenta el tamaño de la FFT para mayor resolución
    # n_overlap=125 # Mayor solapamiento para suavizar el espectro
    )

fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(10, 4),
    # tight_layout=True
    )

spectrum.plot(
    dB=False,
    spatial_colors=True,
    sphere=.14,
    axes=axes
    )
axes.set(
    title='PSD (Power Spectral Density)',
    ylabel='U.A',
    xlabel='Frecuencia (Hz)',
    ylim=(-1,20),
#     xlim=(0,40)
)
fig.show()

# # =================================
# # Tarea comportamental: EEG y audio
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
#     # fig.savefig(
#     #     f'G:/My Drive/tesis_licenciatura/figuras/UBA_GAMES_s{sujeto}.png', 
#     #     transparent=True, 
#     #     dpi=600
#     #     )
#     fig.show()