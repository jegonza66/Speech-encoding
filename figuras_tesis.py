import numpy as np, os, pandas as pd

from scipy.stats import mannwhitneyu, wilcoxon
from statannot import add_stat_annotation
from scipy.optimize import curve_fit
from scipy.io import wavfile
from scipy import signal
import librosa
import mne

from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.collections import PathCollection
from matplotlib.ticker import ScalarFormatter
from matplotlib_venn import venn3, venn2
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull
import matplotlib.pylab as pylab
from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import rc, cm

import seaborn as sns
import scienceplots

from funciones import load_pickle, all_possible_combinations, get_maximum_correlation_channels
from processing import clustering_by_correlation
from plot import define_ticks
import config

pylab.rcParams.update({
        'legend.fontsize': 16,
        'legend.title_fontsize': 16,
        'figure.figsize': (10, 5),
        'figure.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize':16,
        'ytick.labelsize':16
        }
        )
rc('text', usetex=True)
plt.style.use(['science'])

tesis_path = os.path.normpath(os.path.join('C:\\Users', 'jocta', 'Documents', 'tesis_escrita', 'imagenes'))
# tesis_path = os.path.normpath(os.path.join('C:\\Users', 'User', 'Documents', 'tesis_escrita', 'imagenes'))
tesis_path = os.path.normpath(os.path.join('figures','figuras_tesis' ))
figformat, dpi = 'png', 350

# # ==================
# # CONVEX HULL # TODO PENDIENTE 'Phonemes-Phonet_Phonological_Spectrogram'
# situation='External'
# correlations_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/')
# renombre = {'Spectrogram':'Espectrograma', 'Phonological':'Características fonológicas', 'Mfccs':'Coeficientes Mel',
#             'Pitch-Log-Raw':'Tono de voz','Envelope':'Envolvente', 'Phonemes-Phonet':'Fonemas'}
# colores_st = {'Spectrogram':'C0', 'Phonological':'C1', 'Mfccs':'C2', 'Pitch-Log-Raw':'C3','Envelope':'C4', 'Phonemes-Phonet':'C5', 'Phones-Phonet':'C6'}

# bands = ['Theta']
# # stims = 'Spectrogram_Phonological'
# # stims = 'Spectrogram_Mfccs'
# # stims = 'Spectrogram_Envelope_Pitch-Log-Raw'
# stims = 'Phonemes-Phonet_Phonological_Spectrogram'

# stims = '_'.join(sorted(stims.split('_')))
# substims = stims.split('_')
# avg_corr, good_ch = {}, {}

# # Iterate over bands
# for band in bands:
#     # Fill dictionaries with data
#     for stim in substims + [stims]:
#         data = load_pickle(path=os.path.join(correlations_path, band, stim +'.pkl'))
#         if config.relevant_channels:
#             filter_relevant_channels_filter = get_maximum_correlation_channels(data['average_correlation_subjects'].mean(axis=0), number_of_lat_channels=config.relevant_channels)
#             good_ch[stim] = data['repeated_good_correlation_channels_subjects'][:,filter_relevant_channels_filter].ravel()
#             avg_corr[stim] = data['average_correlation_subjects'][:,filter_relevant_channels_filter].ravel()
#         else:
#             good_ch[stim] = data['repeated_good_correlation_channels_subjects'].ravel()
#             avg_corr[stim] = data['average_correlation_subjects'].ravel()

#     # Make plot
#     plt.figure(
#         figsize=(8,6),
#         layout='tight'
#         )
#     for i, stim in enumerate(substims):

#         # Identify Hull (la cáscara de los datos, i.e, el borde)
#         stim_points = np.array([avg_corr[stim], avg_corr[stims]]).transpose()
#         stim_hull = ConvexHull(stim_points)

#         # Plot bad channels for each stim
#         plt.plot(
#             avg_corr[stim][good_ch[stim] == 0],
#             avg_corr[stims][good_ch[stim] == 0],
#             '.',
#             color='grey',
#             alpha=0.5,
#             label='Prueba de permutaciones fallidas',
#             markersize=10
#             )

#         # Plot good channels for each stim
#         plt.plot(
#             avg_corr[stim][good_ch[stim] != 0],
#             avg_corr[stims][good_ch[stim] != 0],
#             '.',
#             color=colores_st[stim],
#             label=renombre[stim],
#             ms=10
#             )
#         plt.fill(
#             stim_points[stim_hull.vertices, 0],
#             stim_points[stim_hull.vertices, 1],
#             color=colores_st[stim],
#             alpha=0.3,
#             linewidth=0
#             )

#     # Get limits
#     xlimit, ylimit = plt.xlim(), plt.ylim()
#     # plt.plot([xlimit[0], ylimit[1]], [xlimit[0], ylimit[1]], 'k--', zorder=0)
#     plt.plot([xlimit[0], xlimit[1]], [ylimit[0], ylimit[1]], 'k--', zorder=0)


#     # plt.hlines(0, xlimit[0], xlimit[1], color='grey', linestyle='dashed')
#     # plt.vlines(0, ylimit[0], ylimit[1], color='grey', linestyle='dashed')

#     plt.xlabel(r'Correlación del modelo individual')
#     plt.ylabel(r'Correlación del modelo conjunto')

#     # Legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), markerscale=2)

#     plt.grid(visible=True)
#     plt.savefig(
#     os.path.join(tesis_path,'resultados', f'convex_{band}_{stims}.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
#     plt.show()


# # =================
# # DIAGRAMAS DE VENN #TODO REHACER CON PHONEMES-PHONET y PHONOLOGICAL NUEVO
# situation='External'
# correlations_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/')
# # 'Envelope_Pitch-Log-Raw',
# # 'Envelope_Spectrogram',
# # 'Pitch-Log-Raw_Spectrogram',
# # 'Envelope_Pitch-Log-Raw_Spectrogram',
# # 'Phonemes-Discrete-Phonet_Spectrogram',
# # 'Phonological_Spectrogram',
# # 'Phonological_Phonemes-Discrete-Phonet',
# # 'Phonological_Phonemes-Discrete-Phonet_Spectrogram',
# bands = ['Delta','Theta','Alpha','Beta1','Beta2','All']
# stimuli_list = [
#     # ['Spectrogram', 'Mfccs'],
#     # ['Envelope', 'Pitch-Log-Raw'],
#     # ['Phones-Phonet', 'Phonemes-Phonet'],
#     ['Spectrogram', 'Envelope', 'Pitch-Log-Raw'],
#     ['Phonological', 'Phonemes-Phonet', 'Spectrogram']
# ]
# areas_dic = {band:{} for band in bands}
# mean_correlations = {band:{} for band in bands}
# for stimuli in stimuli_list:
#     for band in bands:
#         stimuli = sorted(stimuli)
#         double_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==2]
#         triple_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==3]
#         all_stimuli = stimuli + double_combinations + triple_combinations
#         for stim in all_stimuli:
#             # Get average correlation of each stimulus
#             data = load_pickle(path=os.path.join(correlations_path, band, stim +'.pkl'))['average_correlation_subjects']
#             mean_correlations[band][stim] = data.mean()
#             # if config.relevant_channels:
#             #     filter_relevant_channels_filter = get_maximum_correlation_channels(data.mean(axis=0), number_of_lat_channels=config.relevant_channels)
#             #     mean_correlations[stim] = data[:,filter_relevant_channels_filter].mean()
#             # else:
#             #     mean_correlations[stim] = data.mean()
#         for stim12 in double_combinations:
#             stim1, stim2 = stim12.split('_')

#             # Get squared correlation of stimuli
#             variance_1 = mean_correlations[band][stim1] ** 2
#             variance_2 = mean_correlations[band][stim2] ** 2
#             variance_12 = mean_correlations[band][stim12] ** 2 # this represent the union of the two

#             # This represent the shared variances explained by the intersections of sets 1 and 2 (intersection between 1 and 2)
#             variance_intersection_12 = variance_1 + variance_2 - variance_12 #11

#             # This is the realtive complemente of 1 and 2: portion of the variance solely explained by 1 and 2, respectively
#             variance_explained_by_1 = variance_12 - variance_2 #10
#             variance_explained_by_2 = variance_12 - variance_1 #01

#             # Get list with areas
#             areas = [ #(10, 01, 11)
#                 variance_explained_by_1,
#                 variance_explained_by_2,
#                 variance_intersection_12
#                 ] # note that the sum gives shared model
#             areas = [0 if area<0 else area.round(3) for area in areas]
#             areas_dic[band][stim12]=areas
#         if triple_combinations:
#             # Get squared correlation of stimuli
#             variance_1 = mean_correlations[band][all_stimuli[0]]**2
#             variance_2 = mean_correlations[band][all_stimuli[1]]**2
#             variance_3 = mean_correlations[band][all_stimuli[2]]**2
#             variance_12 = mean_correlations[band][all_stimuli[3]]**2
#             variance_13 = mean_correlations[band][all_stimuli[4]]**2
#             variance_23 = mean_correlations[band][all_stimuli[5]]**2
#             variance_123 = mean_correlations[band][all_stimuli[6]]**2

#             # Shared without each stimulus
#             variance_shared_with_1 = variance_123 - variance_23 #100
#             variance_shared_with_2 = variance_123 - variance_13 #010
#             variance_shared_with_3 = variance_123 - variance_12 #001

#             # Explained by subshared, but not by all shared model
#             variance_shared_with_12 = variance_13 + variance_23 - variance_3 - variance_123 #110
#             variance_shared_with_13 = variance_12 + variance_23 - variance_2 - variance_123 #101
#             variance_shared_with_23 = variance_12 + variance_13 - variance_1 - variance_123 #011

#             # Explained by one, two, three and full shared model but not by subshared models
#             variance_int_complement_submodels = variance_123 + variance_1 + variance_2 + variance_3 - variance_12 - variance_13 - variance_23 #111

#             areas = [ # the order should be(100, 010, 110, 001, 101, 011, 111)
#                 variance_shared_with_1,
#                 variance_shared_with_2,
#                 variance_shared_with_12,
#                 variance_shared_with_3,
#                 variance_shared_with_13,
#                 variance_shared_with_23,
#                 variance_int_complement_submodels
#                 ]
#             areas = [0 if area<0 else area.round(3) for area in areas] # note that the sum gives shared model variance_123
#             areas_dic[band][triple_combinations[0]]=areas
        
# normalizer = {band:{} for band in bands}
# for band in bands:
#     max_shared_area = max([sum(areas_dic[band][stim]) for stim in areas_dic[band]])
#     for k, stim in enumerate(areas_dic[band]):
#         if sum(areas_dic[band][stim])==max_shared_area:
#             normalizer[band][stim] = 1
#         else:
#             normalizer[band][stim] = (sum(areas_dic[band][stim])/max_shared_area)
# #================
# # LAS REDUNDANTES
# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=2,
#     figsize=(10, 4),
#     constrained_layout=True
# )

# fig.text(.1, .95, 'a)', fontsize=18, va='top', ha='right')
# venn = venn2(
#     subsets=areas_dic['Theta']['Mfccs_Spectrogram'], # left area diagran, right area diagram, shared area <--> (10, 01, 11)
#     set_labels=('C. Mel', 'Espectrograma'), # stim1, stim2
#     set_colors=('C0', 'C1'),
#     alpha=0.45,
#     normalize_to=normalizer['Theta']['Mfccs_Spectrogram'],
#     ax=axes[0]
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(15)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(18)

# label_conjunto1 = venn.get_label_by_id('A')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] - 0.5, label_conjunto1.get_position()[1] + 0.4))
# label_conjunto1 = venn.get_label_by_id('B')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] + 0.45, label_conjunto1.get_position()[1] + 0.9))

# fig.text(.6, .95, 'b)', fontsize=18, va='top', ha='right')
# venn = venn2(
#     subsets=areas_dic['Theta']['Phonemes-Phonet_Phones-Phonet'], # left area diagran, right area diagram, shared area <--> (10, 01, 11) 'Phonemes-Phonet_Phones-Phonet'
#     set_labels=('Fonemas', 'Fonos'), # stim1, stim2
#     set_colors=('C2', 'C3'),
#     alpha=0.45,
#     ax=axes[1],
#     normalize_to=normalizer['Theta']['Phonemes-Phonet_Phones-Phonet']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(15)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(18)
# label_conjunto1 = venn.get_label_by_id('A')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] - 0.5, label_conjunto1.get_position()[1] + 0.4))
# label_conjunto1 = venn.get_label_by_id('B')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] + 0.49, label_conjunto1.get_position()[1] + 0.95))

# # # Save figure
# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'modelos_conjuntos_externa_redundante.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # =====
# # 'Envolvente', 'Tono de voz', 'Espectrograma'
# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=4,
#     figsize=(14, 10),
#     layout='constrained'
# )

# # Delta
# venn = venn3(
#     subsets=areas_dic['Delta']['Envelope_Pitch-Log-Raw_Spectrogram'],
#     set_labels=('Envolvente', 'Tono de voz', 'Espectrograma'),
#     set_colors=('C4', 'C5', 'C1'),
#     alpha=0.45,
#     ax=axes[0],
#     normalize_to=normalizer['Delta']['Envelope_Pitch-Log-Raw_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(16)
# # label_conjunto1 = venn.get_label_by_id('A')
# # label_conjunto1.set_position((label_conjunto1.get_position()[0] + .2, label_conjunto1.get_position()[1] + 0.05))

# # Theta
# venn = venn3(
#     subsets=areas_dic['Theta']['Envelope_Pitch-Log-Raw_Spectrogram'],
#     set_labels=('Envolvente', 'Tono de voz', 'Espectrograma'),
#     set_colors=('C4', 'C5', 'C1'),
#     alpha=0.45,
#     ax=axes[1],
#     normalize_to=normalizer['Theta']['Envelope_Pitch-Log-Raw_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(16)

# # Alpha
# venn = venn3(
#     subsets=areas_dic['Alpha']['Envelope_Pitch-Log-Raw_Spectrogram'],
#     set_labels=('Envolvente', 'Tono de voz', 'Espectrograma'),
#     set_colors=('C4', 'C5', 'C1'),
#     alpha=0.45,
#     ax=axes[2],
#     normalize_to=normalizer['Alpha']['Envelope_Pitch-Log-Raw_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
#         if label.get_text() =='0.01':
#             pos_actual = label.get_position()
#             # Ajusta los valores según lo que necesites; por ejemplo, sumar 0.1 a x y restar 0.05 a y
#             nueva_pos = (pos_actual[0] + 0.3, pos_actual[1] + 0.05)
#             label.set_position(nueva_pos)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(16)
# # label_conjunto1 = venn.get_label_by_id('A')
# # label_conjunto1.set_position((label_conjunto1.get_position()[0] + .2, label_conjunto1.get_position()[1] + 0.05))

# # Ancha
# venn = venn3(
#     subsets=areas_dic['All']['Envelope_Pitch-Log-Raw_Spectrogram'],
#     set_labels=('Envolvente', 'Tono de voz', 'Espectrograma'),
#     set_colors=('C4', 'C5', 'C1'),
#     alpha=0.45,
#     ax=axes[3],
#     normalize_to=normalizer['All']['Envelope_Pitch-Log-Raw_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(16)
# # label_conjunto1 = venn.get_label_by_id('A')
# # label_conjunto1.set_position((label_conjunto1.get_position()[0] + .2, label_conjunto1.get_position()[1] + 0.05))

# for idx, banda in enumerate(['Delta', 'Theta', 'Alpha', 'Ancha']):
#     axes[idx].set_title(r'\textbf{'+banda+r'}', fontsize=18)
    
# # fig.set_constrained_layout_pads(hspace=0.01)#, wspace=0.05, h_pad=0.1, w_pad=0.1)
# # fig.subplots_adjust(hspace=0.0003, wspace=0.001)
# # fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, hspace=0.00001, wspace=0.00001)


# # left: Posición del borde izquierdo de los subplots (como fracción del ancho total de la figura).
# # right: Posición del borde derecho (fracción del ancho total).
# # bottom: Posición del borde inferior (fracción de la altura total).
# # top: Posición del borde superior (fracción de la altura total).
# # wspace: Espacio (ancho) entre columnas de subplots, expresado como fracción del ancho medio de los subplots.
# # hspace: Espacio (alto) entre filas de subplots, expresado como fracción de la altura media de los subplots.

# # Save figure
# # fig.text(.1, .95, 'a)', fontsize=18, va='top', ha='right')
# fig.savefig(
#     os.path.join(tesis_path,'resultados', f'modelos_conjuntos_externa_bandas1.{figformat}'),
#     bbox_inches='tight',
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()

# # =====
# # 'Envolvente', 'Tono de voz', 'Espectrograma'
# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=4,
#     figsize=(15, 10),
#     layout='constrained'
# )

# # Delta
# venn = venn3(
#     subsets=areas_dic['Delta']['Phonemes-Phonet_Phonological_Spectrogram'], # the order should be(100, 010, 110, 001, 101, 011, 111)
#     set_labels=('Fonemas', 'C. Fonológicas', 'Espectrograma'),
#     set_colors=('C2', 'C6', 'C1'),
#     alpha=0.45,
#     ax=axes[0],
#     normalize_to=normalizer['Delta']['Phonemes-Phonet_Phonological_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(15)
# label_conjunto1 = venn.get_label_by_id('A')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] - .2, label_conjunto1.get_position()[1] - 0.2))

# # Theta
# venn = venn3(
#     subsets=areas_dic['Theta']['Phonemes-Phonet_Phonological_Spectrogram'], # the order should be(100, 010, 110, 001, 101, 011, 111)
#     set_labels=('Fonemas', 'C. Fonológicas', 'Espectrograma'),
#     set_colors=('C2', 'C6', 'C1'), 
#     alpha=0.45,
#     ax=axes[1],
#     normalize_to=normalizer['Theta']['Phonemes-Phonet_Phonological_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(15)
# label_conjunto1 = venn.get_label_by_id('A')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] - .2, label_conjunto1.get_position()[1] - 0.2))

# # Alpha
# venn = venn3(
#     subsets=areas_dic['Alpha']['Phonemes-Phonet_Phonological_Spectrogram'], # the order should be(100, 010, 110, 001, 101, 011, 111)
#     set_labels=('Fonemas', 'C. Fonológicas', 'Espectrograma'),
#     set_colors=('C2', 'C6', 'C1'),
#     alpha=0.45,
#     ax=axes[2],
#     normalize_to=normalizer['Alpha']['Phonemes-Phonet_Phonological_Spectrogram']
#     )
# for label in venn.subset_labels:
#     label
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
#         # if label =='0.001':
#         #     pos_actual = label.get_position()
#         #     # Ajusta los valores según lo que necesites; por ejemplo, sumar 0.1 a x y restar 0.05 a y
#         #     nueva_pos = (pos_actual[0] + 0.5, pos_actual[1] + 0.05)
#         #     label.set_position(nueva_pos)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(15)
# label_conjunto1 = venn.get_label_by_id('A')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] - .2, label_conjunto1.get_position()[1] - 0.2))

# # Ancha
# venn = venn3(
#     subsets=areas_dic['All']['Phonemes-Phonet_Phonological_Spectrogram'], # the order should be(100, 010, 110, 001, 101, 011, 111)
#     set_labels=('Fonemas', 'C. Fonológicas', 'Espectrograma'),
#     set_colors=('C2', 'C6', 'C1'),
#     alpha=0.45,
#     ax=axes[3],
#     normalize_to=normalizer['All']['Phonemes-Phonet_Phonological_Spectrogram']
#     )
# for label in venn.subset_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(13)
# for label in venn.set_labels:
#     if label:  # Verificar que la etiqueta no sea None
#         label.set_fontsize(15)
# label_conjunto1 = venn.get_label_by_id('A')
# label_conjunto1.set_position((label_conjunto1.get_position()[0] - .2, label_conjunto1.get_position()[1] - 0.2))

# for idx, banda in enumerate(['Delta', 'Theta', 'Alpha', 'Ancha']):
#     axes[idx].set_title(r'\textbf{'+banda+r'}', fontsize=18)
    
# # fig.set_constrained_layout_pads(hspace=0.01)#, wspace=0.05, h_pad=0.1, w_pad=0.1)
# # fig.subplots_adjust(hspace=0.0003, wspace=0.001)
# # fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, hspace=0.00001, wspace=0.00001)


# # left: Posición del borde izquierdo de los subplots (como fracción del ancho total de la figura).
# # right: Posición del borde derecho (fracción del ancho total).
# # bottom: Posición del borde inferior (fracción de la altura total).
# # top: Posición del borde superior (fracción de la altura total).
# # wspace: Espacio (ancho) entre columnas de subplots, expresado como fracción del ancho medio de los subplots.
# # hspace: Espacio (alto) entre filas de subplots, expresado como fracción de la altura media de los subplots.

# # Save figure
# # fig.text(.1, .95, 'a)', fontsize=18, va='top', ha='right')
# fig.savefig(
#     os.path.join(tesis_path,'resultados', f'modelos_conjuntos_externa_bandas2.{figformat}'),
#     bbox_inches='tight',
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()

# # ==================================================================
# # Violin plots para todos los atributos las distintas bandas de frecs
# situation='External'
# correlations_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/')

# # Relevant parameters
# bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']
# stimuli = ['Pitch-Log-Raw', 'Envelope', 'Spectrogram', 'Mfccs', 'Phones-Phonet', 'Phonemes-Phonet', 'Phonological'] #'Phonemes-Discrete-Phonet', 
# stimulus_tostr = {'Pitch-Log-Raw':'tono', 'Envelope':'envolvente', 'Spectrogram':'espectrograma', 'Mfccs':'mfccs', 'Phones-Phonet':'fonos', 'Phonemes-Phonet':'fonemas', 'Phonological':'fonologicas'}
# avg_corr_0 = {stimulus:{} for stimulus in stimuli}
# avg_corr_1 = {stimulus:{} for stimulus in stimuli}
# p_vals = {stimulus:{} for stimulus in stimuli}
# for stimulus in stimuli:
#     for i, band in enumerate(bands):
#         data = load_pickle(path=os.path.join(correlations_path, band, stimulus +'.pkl'))
#         avg_corr_0[stimulus][band] = data['average_correlation_subjects'].mean(axis=0)
#         avg_corr_1[stimulus][band] = data['average_correlation_subjects'].mean(axis=1)
#         stat, p_val = wilcoxon(avg_corr_1[stimulus][band], zero_method='wilcox')
#         p_vals[stimulus][band] = p_val

# for stimulus in stimuli:
#     # Crear figura y GridSpec
#     fig = plt.figure(
#         figsize=(10, 6),
#         tight_layout=True
#         )
#     gs = gridspec.GridSpec(2, len(bands), height_ratios=[1, 1])  # Proporción entre violin y topomaps

#     # Crear eje para el violin plot que ocupa toda la primera fila
#     ax_violin = fig.add_subplot(gs[0, :])  # Ocupar todas las columnas de la primera fila
#     ax_violin.grid(visible=True)
#     violin = sns.violinplot(
#         data=pd.DataFrame(avg_corr_1[stimulus]),
#         palette={band: f'C1' for h, band in enumerate(bands)},
#         ax=ax_violin
#     )
#     sns.stripplot(
#         data=pd.DataFrame(avg_corr_1[stimulus]),
#         jitter=.1,  # Sin dispersión horizontal,
#         size=3,
#         color='black',
#         ax=ax_violin
#         )
#     for collection in violin.collections:
#         collection.set_alpha(0.5)
#     ax_violin.set_xticklabels(['Delta', 'Theta', 'Alpha', r'Beta$_1$', r'Beta$_2$', 'Ancha'])
    
#     if stimulus.startswith('Phon'):
#         ax_violin.set_yticks([-0.25, 0, 0.3, 0.7,  0.9 ])
#     elif stimulus in ['Envelope', 'Pitch-Log-Raw']:
#         ax_violin.set_yticks([-0.2, 0, 0.3, 0.6])
#     else:
#         ax_violin.set_yticks([-0.25, 0, 0.3, 0.7,  0.9])


#     ax_violin.set_xticklabels(['Delta', 'Theta', 'Alpha', r'Beta$_1$', r'Beta$_2$', 'Ancha'])
#     ax_violin.set_ylabel("Correlación sujetos")
#     ax_violin.set_axisbelow(True)

#     # Crear ejes para los topomaps en la segunda fila
#     axs = [fig.add_subplot(gs[1, i]) for i in range(len(bands))]

#     for i, band in enumerate(bands):

#         # Dibujar topomap
#         im = mne.viz.plot_topomap(avg_corr_0[stimulus][band].ravel(),
#                                 config.info_mne,
#                                 axes=axs[i],
#                                 show=False,
#                                 sphere=0.07,
#                                 cmap='Reds',
#                                 vlim=(avg_corr_0[stimulus][band].min(), avg_corr_0[stimulus][band].max())
#         )

#         # Agregar colorbar debajo de cada topomap
#         cbar = plt.colorbar(im[0], ax=axs[i], orientation='horizontal', shrink=0.7)
#         min_a = avg_corr_0[stimulus][band].min()
#         max_a = avg_corr_0[stimulus][band].max()
#         mid = (max_a+min_a)/2
#         cbar.set_ticks([min_a, mid, max_a])
#         cbar.set_ticklabels([f'{min_a:.2f}', f'{mid:.2f}', f'{max_a:.2f}'])
#         # if i==0:
#         #     axs[i].set_ylabel("Correlación canales")
#     fig.text(0.05, 1, 'a)', fontsize=18, va='top', ha='right')
#     fig.text(0.05, .47, 'b)', fontsize=18, va='top', ha='right')
#     fig.text(.025, .25, 'Correlación canales', fontsize=18, rotation=90, va='center', ha='center')
#     fig.savefig(
#     os.path.join(tesis_path,'resultados', f'violin_correlacion_{stimulus_tostr[stimulus]}.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
#     fig.show()

# # Caso especial con fonemas
# stimulus1, stimulus2 = 'Phonemes-Discrete-Phonet', 'Phonemes-Phonet'
# p_vals = {band: wilcoxon(avg_corr_1[stimulus1][band], avg_corr_1[stimulus2][band])[1] 
#           for band in bands}

# # Reestructurar los datos para cada estímulo (asumiendo que avg_corr_1[stimulusX] es un diccionario con arrays o listas por banda)
# df1 = pd.DataFrame(avg_corr_1[stimulus1])
# df1 = df1.melt(var_name='Band', value_name='Correlación')
# df1['Representación'] = 'Ocurrencias'

# df2 = pd.DataFrame(avg_corr_1[stimulus2])
# df2 = df2.melt(var_name='Band', value_name='Correlación')
# df2['Representación'] = 'PLLR'

# df = pd.concat([df1, df2], ignore_index=True)

# # # Ordenar las bandas si es necesario:
# # # (Asegúrate de que los nombres de las columnas en el DataFrame coincidan con estos, o ajústalos según corresponda.)

# # Crear la figura y el GridSpec
# fig = plt.figure(figsize=(10, 6), tight_layout=True)
# gs = plt.GridSpec(2, len(bands), height_ratios=[1, 1])  # Violin arriba, topomaps abajo

# # Eje para el violin plot que ocupará toda la primera fila
# ax_violin = fig.add_subplot(gs[0, :])
# ax_violin.grid(True)

# # Crear el violin plot comparativo con split (requiere dos niveles en 'Representación')
# violin = sns.violinplot(
#     data=df,
#     x='Band',
#     y='Correlación',
#     hue='Representación',
#     split=True,
#     inner='quartile',
#     palette={'Ocurrencias': 'C2', 'PLLR': 'C1'},
#     alpha=.25,
#     order=bands,
#     ax=ax_violin
# )

# # Ajustar manualmente la transparencia de los violines (por si la paleta no funciona correctamente)
# for collection in violin.collections:
#     collection.set_alpha(0.5)

# # Agregar también los puntos de cada observación (stripplot)
# sns.stripplot(
#     data=df,
#     x='Band',
#     y='Correlación',
#     hue='Representación',
#     dodge=True,
#     color='black',
#     order=bands,
#     ax=ax_violin,
#     jitter=0.1,
#     size=3
# )

# # Es importante quitar la leyenda duplicada (ya que tanto violinplot como stripplot la generan)
# handles, labels = ax_violin.get_legend_handles_labels()
# for handle in handles:
#     handle.set_alpha(0.5) 
# if len(handles) > 2:
#     ax_violin.legend(handles[0:2], labels[0:2], title='Representación', loc=(.6, .45))

# ax_violin.set_xticklabels(['Delta', 'Theta', 'Alpha', r'Beta$_1$', r'Beta$_2$', 'Ancha'])
# ax_violin.set_ylabel("Correlación sujetos")
# ax_violin.set_xlabel("")
# ax_violin.set_axisbelow(True)

# # Configurar los ticks en y según el estímulo (como en tu código original)
# if stimulus1.startswith('Phon'):
#     ax_violin.set_yticks([-0.25, 0, 0.3, 0.7,  0.9])
# elif stimulus1 in ['Envelope', 'Pitch-Log-Raw']:
#     ax_violin.set_yticks([-0.2, 0, 0.3, 0.6])
# else:
#     ax_violin.set_yticks([-0.25, 0, 0.3, 0.7,  0.9])

# for i, band in enumerate(bands):
#     p = p_vals.get(band, 1)
#     if p < 0.005:
#         sig = '***'
#     elif p < 0.01:
#         sig = '**'
#     elif p < 0.05:
#         sig = '*'
#     else:
#         sig = ''
#     if sig:
#         # Filtrar los datos para la banda actual y determinar el máximo
#         max_val = df[df['Band'] == band]['Correlación'].max()
#         ypos = ax_violin.get_ylim()[1]-0.08
#         ax_violin.text(i, ypos, sig, ha='center', va='bottom', color='black', fontsize=14)

# # Crear ejes para los topomaps en la segunda fila
# axs = [fig.add_subplot(gs[1, i]) for i in range(len(bands))]

# for i, band in enumerate(bands):
#     im = mne.viz.plot_topomap(
#         avg_corr_0[stimulus2][band].ravel(),  # o ajusta según el estímulo que quieras mostrar
#         config.info_mne,
#         axes=axs[i],
#         show=False,
#         sphere=0.07,
#         cmap='Reds',
#         vlim=(avg_corr_0[stimulus2][band].min(), avg_corr_0[stimulus2][band].max())
#     )
#     cbar = plt.colorbar(im[0], ax=axs[i], orientation='horizontal', shrink=0.7)
#     min_a = avg_corr_0[stimulus2][band].min()
#     max_a = avg_corr_0[stimulus2][band].max()
#     mid = (max_a + min_a) / 2
#     cbar.set_ticks([min_a, mid, max_a])
#     cbar.set_ticklabels([f'{min_a:.2f}', f'{mid:.2f}', f'{max_a:.2f}'])

# fig.text(0.05, 1, 'a)', fontsize=18, va='top', ha='right')
# fig.text(0.05, 0.47, 'b)', fontsize=18, va='top', ha='right')
# fig.text(0.025, 0.25, 'Correlación canales', fontsize=18, rotation=90, va='center', ha='center')
# fig.savefig(
# os.path.join(tesis_path,'resultados', f'violin_correlacion_especial_fonemas.{figformat}'),
# transparent=False,
# dpi=dpi
# )
# # ===================================================================================================================
# # TOPOGRAPHIC DISTRIBUTION HEATMAPS: make heatmaps with topographic information across features, situations and bands # TODO SUMAR CANALES SIGNIFICATIVOS PARA EL ESTADISTICO ENTRE SUJETOS EN VEZ DE CANALES
# # ===================================================================================================================
# situation = 'External'
# correlations_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/')

# # Relevant parameters
# bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']
# stimuli = ['Envelope', 'Pitch-Log-Raw', 'Spectrogram', 'Phonemes-Phonet', 'Phonological'] # 'Envelope_Phonemes-Discrete-Manual'
# n_stims, n_bands = len(stimuli), len(bands)

# # Get mean correlations across subjects and total max and min
# correlations = {(stim,band):load_pickle(path=os.path.join(correlations_path, band, stim +'.pkl'))['average_correlation_subjects'] for stim in stimuli for band in bands}
# good_chs = {(stim,band):load_pickle(path=os.path.join(correlations_path, band, stim +'.pkl'))['repeated_good_correlation_channels_subjects'] for stim in stimuli for band in bands}

# # Get groups to make average correlations
# n_groups_x, n_groups_y = 12, 12
# posicion_x = np.array([config.montage._get_ch_pos()[ch][0] for ch in config.montage._get_ch_pos()])
# posicion_y = np.array([config.montage._get_ch_pos()[ch][1] for ch in config.montage._get_ch_pos()])
# bins_x = np.linspace(posicion_x.min(), posicion_x.max(), n_groups_x)
# bins_y = np.linspace(posicion_y.min(), posicion_y.max(), n_groups_y)
# groups_x, groups_y = [], []
# for i in range(1, n_groups_y):
#     group_y = []
#     for ch in config.montage._get_ch_pos():
#         y_value = config.montage._get_ch_pos()[ch][1]
#         if bins_y[i-1]<=y_value<=bins_y[i]:
#             group_y.append(config.info_mne.ch_names.index(ch))
#     groups_y.append(group_y)

# for i in range(1, n_groups_x):
#     group_x = []
#     for ch in config.montage._get_ch_pos():
#         x_value = config.montage._get_ch_pos()[ch][0]
#         y_value = config.montage._get_ch_pos()[ch][1]
#         if (bins_x[i-1]<=x_value<=bins_x[i]) and (y_value>=0) :
#             group_x.append(config.info_mne.ch_names.index(ch))
#     groups_x.append(group_x)

# # # Para ver si está balanceado
# # plt.figure()
# # plt.title('bins_x')
# # plt.hist(posicion_x, bins_x)
# # plt.grid(visible=True)
# # plt.yticks(ticks=np.arange(0,20))
# # plt.show(block=False)
# # plt.figure()
# # plt.title('bins_y')
# # plt.hist(posicion_y, bins_y)
# # plt.grid(visible=True)
# # plt.yticks(ticks=np.arange(0,20))
# # plt.show(block=False)

# # =======================
# # HEATMAP LATERALIZATION #TODO CHARLAR CON JUAN:  COMO NO USO CANALES SIGNIFICATIVOS, INTRA SUJETOS NO DA
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 4), constrained_layout='True')
# upper_channels = np.concatenate(groups_y[4:]).tolist()

# # left_channels = np.concatenate(groups_x[:4]).tolist()
# # right_channels = np.concatenate(groups_x[7:]).tolist()

# # left_channels = ['C32', 'C31', 'C30', 'C29', 'C28', 'C27', 'C26', 'C25', 'C24', 'D7', 'D6',\
# #     'D5', 'D4', 'D3', 'D2', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13']
# # right_channels = ['C10', 'C9', 'C8', 'C16', 'C15', 'C14', 'C13', 'C12', 'C11', 'C7', 'C6',\
# #     'C5', 'C4', 'C3', 'C2', 'B27', 'B28', 'B29', 'B30', 'B31','B32']
# left_channels = ['C32', 'C31', 'C30', 'D7', 'D6',\
#     'D5', 'D4', 'D3', 'D8', 'D9', 'D10', 'D11']
# right_channels = ['C10', 'C9', 'C8', 'C7', 'C6',\
#     'C5', 'C4', 'C3', 'B27', 'B28', 'B29', 'B30']
# # left_channels = ['C32', 'C31', 'C30', 'D7', 'D6',\
# #     'D5', 'D8', 'D9', 'D10', 'D11']
# # right_channels = ['C10', 'C9', 'C8', 'C7', 'C6',\
# #     'C5', 'B27', 'B28', 'B29', 'B30']

# left_channels = [config.info_mne['ch_names'].index(ch) for ch in left_channels]
# right_channels = [config.info_mne['ch_names'].index(ch) for ch in right_channels]
# left_channels = [ch for ch in left_channels if ch in upper_channels]
# right_channels = [ch for ch in right_channels if ch in upper_channels]

# # number_of_channels = 21
# # left_corrs = sorted(average_corr[left_channels])[-number_of_channels:]
# # right_corrs = sorted(average_corr[right_channels])[-number_of_channels:]
# # left_chs = [average_corr.tolist().index(corr) for corr in left_corrs]
# # right_chs = [average_corr.tolist().index(corr) for corr in right_corrs]

# cmap = colormaps['magma']
# colors = cmap(np.linspace(0, 1, len(groups_x)))
# custom_cmap = ListedColormap([colors[2], colors[-2]])

# axes[0].set_title('Selección de grupos')
# mne.viz.plot_sensors(
#         info=config.info_mne,
#         show_names=False,
#         block=False,
#         pointsize=35,
#         cmap=custom_cmap,
#         axes=axes[0],
#         ch_groups=[left_channels, right_channels],
#         linewidth=0,
#         show=False
#         )
# # mne.viz.plot_sensors(
# #         info=config.info_mne,
# #         show_names=True,
# #         block=False,
# #         pointsize=35,
# #         cmap=custom_cmap,
# #         # axes=axes[0],
# #         # ch_groups=[left_channels, right_channels],
# #         linewidth=0,
# #         show=True
# #         )

# axes[0].text(
#             -.076, -.02, 'Izquierda', color=colors[2],
#             verticalalignment="center", horizontalalignment="left", fontsize=13
#         )
# axes[0].text(
#             .026, -.02, 'Derecha', color=colors[-2],
#             verticalalignment="center", horizontalalignment="left", fontsize=13
#         )

# axes[1].set_title('C. Fonológicas - Theta',
#                    y=1.05,
#                    verticalalignment="top")
# axes[1].grid(visible=True)
# axes[1].set_axisbelow(True)

# data_plot = {'Izquierda':correlations[('Phonological','Theta')][:, left_channels].mean(axis=1),
#              'Derecha':correlations[('Phonological','Theta')][:, right_channels].mean(axis=1)}
# data_plot = pd.DataFrame(data=data_plot)

# # Ahora usamos Seaborn para crear el violin plot
# sns.boxplot(
#     data=data_plot,
#     # x='Grupos de electrodos',
#     # y='Correlación',
#     palette={'Izquierda':colors[2], 'Derecha':colors[-2]},
#     ax=axes[1],
# )
# add_stat_annotation(
#                     ax=axes[1],
#                     data=data_plot,
#                     box_pairs=[('Izquierda', 'Derecha')],
#                     test='Wilcoxon',
#                     text_format='star',
#                     loc='inside',
#                     fontsize='xx-large',
#                     verbose=0
#                     )

# for patch in axes[1].artists:
#     r, g, b, alpha = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, .8))
# sns.swarmplot(data=data_plot, color=".25", ax=axes[1])

# # # Move the legend
# axes[1].grid(visible=True)
# axes[1].set_xticks([0, 1])
# axes[1].set_xticklabels(['Izquierda', 'Derecha'])
# # axes[1].set_ylim([0.42, 0.455])
# # axes[1].set_yticks(np.linspace(correlations[('Phonological', 'Theta')].min(), correlations[('Phonological', 'Theta')].max(), 5))

# axes[1].grid(False)

# # axes[1].set_xlabel('Grupos de electrodos', fontsize=15)
# axes[1].set_ylabel('Correlación promedio', fontsize=15)
# axes[1].tick_params(which='minor', bottom=False, left=True, right=True, top=False)

# z = np.zeros(shape=(len(bands),len(stimuli)))
# z_pv = np.zeros(shape=(len(bands),len(stimuli)))
# for i, band in enumerate(bands):
#     for j, stim in enumerate(stimuli):
#         average_correlation = correlations[(stim,band)]
#         # left_corrs = sorted(average_correlation.mean(axis=0)[left_channels])[-number_of_channels:]
#         # right_corrs = sorted(average_correlation.mean(axis=0)[right_channels])[-number_of_channels:]
#         # left_chs = [average_correlation.mean(axis=0).tolist().index(corr) for corr in left_corrs]
#         # right_chs = [average_correlation.mean(axis=0).tolist().index(corr) for corr in right_corrs]

#         z[i,j] = average_correlation[:, right_channels].mean()-average_correlation[:, left_channels].mean()
#         stat, p_val = wilcoxon(average_correlation[:, right_channels].mean(axis=1), average_correlation[:, left_channels].mean(axis=1), alternative='two-sided')
#         z_pv[i,j] = p_val

# # axes[2].set_title('Lateralization: right(G[0-3])-left(G[7-10])')
# axes[2].set_title('Lateralización: diferencia de correlación', x=.65)
# im = axes[2].imshow(
#             z,
#             vmin=z.min(),
#             vmax=z.max(),
#             cmap='magma'
#             )
# for i in range(z_pv.shape[0]):
#     for j in range(z_pv.shape[1]):
#         p_val = z_pv[i, j]
#         if p_val < 0.005:
#             annot = '***'
#         elif p_val < 0.01:
#             annot = '**'
#         elif p_val < 0.05:
#             annot = '*'
#         else:
#             annot = ''
#         if annot:
#             if z[i,j]<-.003:
#                 axes[2].text(j, i, annot, ha='center', va='center', color='white', fontsize=20, fontweight='bold')
#             else:
#                 axes[2].text(j, i, annot, ha='center', va='center', color='black', fontsize=20, fontweight='bold')

# axes[2].set_xticks(np.arange(len(stimuli)))
# axes[2].set_xticklabels(['Envolvente', 'Tono de voz', 'Espectrograma', 'Fonemas', 'C. Fonológicas'], minor=False, fontsize=12, rotation=35)
# axes[2].set_yticks(np.arange(len(bands)))
# axes[2].set_yticklabels(['Delta', 'Theta', 'Alpha', r'Beta$_1$', r'Beta$_2$', 'Ancha'], minor=False, fontsize=12)
# axes[2].minorticks_off()

# fig.text(.025, 1, 'a)', fontsize=18, va='top', ha='right')
# fig.text(.33, 1, 'b)', fontsize=18, va='top', ha='right')
# fig.text(.62, 1, 'c)', fontsize=18, va='top', ha='right')
# cbar = fig.colorbar(im, ax=axes[2])
# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'distribucion_lateralizacion.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ======================
# # HEATMAP CENTRALIZATION #TODO CHARLAR CON JUAN:  COMO NO USO CANALES SIGNIFICATIVOS, INTRA SUJETOS NO DA
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11,4), constrained_layout='True')
# cmap = colormaps['turbo']
# colors = cmap(np.linspace(0, 1, len(groups_x)))
# cmap_gr = colormaps['magma']
# colors_gr = cmap_gr(np.linspace(0, 1, len(groups_x)))

# axes[0].set_title('Selección de grupos')
# mne.viz.plot_sensors(
#         info=config.info_mne,
#         show_names=False,
#         block=False,
#         pointsize=35,
#         cmap='turbo',
#         axes=axes[0],
#         ch_groups=groups_x,
#         linewidth=0,
#         show=False
#         )
# # Obtener coordenadas de los sensores en el eje Y
# x_start = -0.086
# x_end = 0.086

# # Definir coordenadas de la flecha
# ylim = axes[0].get_ylim()
# y_arrow = ylim[0] + 0 * (ylim[1] - ylim[0])  # Un poco a la derecha del borde

# # Dibujar la flecha
# axes[0].annotate(
#     "",  # Sin texto
#     xy=(x_end, y_arrow), xytext=(x_start, y_arrow),
#     arrowprops=dict(arrowstyle="->", linewidth=2, color="black"),
#     annotation_clip=False  # Para que se vea si está fuera de los límites
# )

# # Definir etiquetas en los valores exactos de y_start y y_end
# x_positions = np.linspace(x_start-.02, x_end, 11)  # 11 valores, incluyendo los extremos

# for i, x in enumerate(x_positions):
#     if i in [4,5,6]:
#         axes[0].text(
#             x, y_arrow-0.02, f"G{i}", color=colors[i],
#             verticalalignment="center", horizontalalignment="left", fontsize=13
#         )
#     else:
#         axes[0].text(
#             x, y_arrow-0.02, f"G{i}", color=colors[i],
#             verticalalignment="center", horizontalalignment="left", fontsize=13
#         )
# # axes[0].text(
# #             np.mean(x_positions)-.01, y_arrow-0.04, f"Centro", color=colors_gr[0],
# #             verticalalignment="center", horizontalalignment="left", fontsize=13
# #         )
# # axes[0].text(
# #             x_positions[-1]-.045, y_arrow-0.04, f"Costados", color=colors_gr[-2],
# #             verticalalignment="center", horizontalalignment="left", fontsize=13
# #         )
# # axes[0].text(
# #             x_positions[0]+.01, y_arrow-0.04, f"Costados", color=colors_gr[-2],
# #             verticalalignment="center", horizontalalignment="left", fontsize=13
# #         )

# axes[1].set_title('Fonemas - Theta',
#                    y=1.05,
#                    verticalalignment="top")
# axes[1].grid(visible=True)
# axes[1].set_axisbelow(True)

# data_plot = {'Grupos de electrodos':[], 'Correlación':[]}
# for i, group in enumerate(groups_x):
#     corr_group = correlations[('Phonemes-Phonet','Theta')][:, group].mean(axis=1)
#     for corr in corr_group:
#         data_plot['Grupos de electrodos'].append(f'G{i}')
#         data_plot['Correlación'].append(corr)

# # Ahora usamos Seaborn para crear el violin plot
# sns.boxplot(
#     data=data_plot,
#     x='Grupos de electrodos',
#     y='Correlación',
#     palette={f'G{k}': colors[k] for k in range(len(groups_x))},
#     ax=axes[1],
# )
# # Move the legend
# axes[1].grid(visible=True)
# axes[1].set_xticks(np.arange(n_groups_x-1))
# axes[1].set_xticklabels([f'{i}'for i in np.arange(n_groups_x-1)])
# xticks = axes[1].get_xticklabels()
# for l, label in enumerate(xticks):
#     if l in [4,5,6]:
#         label.set_color(colors_gr[0])
#     else:
#         label.set_color(colors_gr[-2])
# axes[1].set_yticks(np.linspace(correlations[('Phonemes-Phonet','Theta')].min(), correlations[('Phonemes-Phonet','Theta')].max(), 5).round(3))
# axes[1].grid(False)

# axes[1].set_xlabel('Grupos de electrodos', fontsize=15)
# axes[1].set_ylabel('Correlación promedio', fontsize=15)
# axes[1].tick_params(which='minor', bottom=False, left=True, right=True, top=False)

# z = np.zeros(shape=(len(bands),len(stimuli)))
# z_pv = np.zeros(shape=(len(bands),len(stimuli)))
# for i, band in enumerate(bands):
#     for j, stim in enumerate(stimuli):
#         average_correlation = correlations[(stim,band)]
#         sides_group, center_group = [], []
#         for l in range(n_groups_x-1):
#             if 4<=l<=6:
#                 for ch in groups_x[l]:
#                     center_group.append(ch)
#             else:
#                 for ch in groups_x[l]:
#                     sides_group.append(ch)
#         z[i,j] = average_correlation[:, sides_group].mean()-average_correlation[:, center_group].mean()
#         stat, p_val = wilcoxon(average_correlation[:, sides_group].mean(axis=1), average_correlation[:, center_group].mean(axis=1), alternative='two-sided')
#         z_pv[i,j] = p_val

# # axes[2].set_title('Lateralization: right(G[0-3])-left(G[7-10])')
# axes[2].set_title('Centralización: diferencia de correlación', x=.65)
# im = axes[2].imshow(
#             z,
#             vmin=z.min(),
#             vmax=z.max(),
#             cmap='magma'
#             )
# for i in range(z_pv.shape[0]):
#     for j in range(z_pv.shape[1]):
#         p_val = z_pv[i, j]
#         if p_val < 0.005:
#             annot = '***'
#         elif p_val < 0.01:
#             annot = '**'
#         elif p_val < 0.05:
#             annot = '*'
#         else:
#             annot = ''
#         if annot:
#             if z[i,j]<-.002:
#                 axes[2].text(j, i, annot, ha='center', va='center', color='white', fontsize=20, fontweight='bold')
#             else:
#                 axes[2].text(j, i, annot, ha='center', va='center', color='black', fontsize=20, fontweight='bold')

# axes[2].set_xticks(np.arange(len(stimuli)))
# axes[2].set_xticklabels(['Tono de voz', 'Envolvente', 'Espectrograma', 'Fonemas', 'C. Fonológicas'], minor=False, fontsize=12, rotation=35)
# axes[2].set_yticks(np.arange(len(bands)))
# axes[2].set_yticklabels(['Delta', 'Theta', 'Alpha', r'Beta$_1$', r'Beta$_2$', 'Ancha'], minor=False, fontsize=12)
# axes[2].minorticks_off()

# # axes[2].xaxis.tick_top()
# cbar = fig.colorbar(im,
#                     ax=axes[2])
# cbar.ax.set(yticks=[-.006,  0.   ,  0.006,  0.009], yticklabels=[-.006,  0.   ,  0.006,  0.009])
# # cbar.ax.tick_params(labelsize=15)
# fig.text(.025, 1, 'a)', fontsize=18, va='top', ha='right')
# fig.text(.33, 1, 'b)', fontsize=18, va='top', ha='right')
# fig.text(.62, 1, 'c)', fontsize=18, va='top', ha='right')
# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'distribucion_centralizacion.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ==========================
# # HEATMAP ANTERIOR-POSTERIOR
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11,4), constrained_layout='True')
# axes[0].set_title('Selección de grupos')
# im = mne.viz.plot_sensors(
#         info=config.info_mne,
#         show_names=False,
#         block=False,
#         pointsize=35,
#         cmap='cividis',
#         axes=axes[0],
#         ch_groups=groups_y,
#         linewidth=0,
#         show=False
#         )
# cmap = colormaps['cividis']
# colors = cmap(np.linspace(0, 1, len(groups_y)))

# # Obtener coordenadas de los sensores en el eje Y
# y_start = -0.076
# y_end = 0.076

# # Definir coordenadas de la flecha
# xlim = axes[0].get_xlim()
# x_arrow = xlim[0] + 0 * (xlim[1] - xlim[0])  # Un poco a la derecha del borde

# # Dibujar la flecha
# axes[0].annotate(
#     "",  # Sin texto
#     xy=(x_arrow, y_end), xytext=(x_arrow, y_start),
#     arrowprops=dict(arrowstyle="->", linewidth=2, color="black"),
#     annotation_clip=False  # Para que se vea si está fuera de los límites
# )

# # Definir etiquetas en los valores exactos de y_start y y_end
# y_positions = np.linspace(y_start, y_end, 11)  # 11 valores, incluyendo los extremos

# for i, y in enumerate(y_positions):
#     axes[0].text(
#         x_arrow - 0.025, y, f"G{i}", color=colors[i],
#         verticalalignment="center", horizontalalignment="left", fontsize=14
#     )
# axes[1].set_title('C. Fonológicas - Alpha',
#                    y=1.05,
#                    verticalalignment="top")
# axes[1].grid(visible=True)
# axes[1].set_axisbelow(True)

# for i, (group, color) in enumerate(zip(groups_y, colors)):
#     corr_group = correlations[('Phonological','Alpha')][:, group].mean()
#     axes[1].scatter(i, corr_group, color=color, s=35)

# # # Move the legend
# axes[1].grid(visible=True)
# axes[1].set_xticks(np.arange(n_groups_y-1))
# axes[1].set_xticklabels([f'{i}'for i in np.arange(n_groups_y-1)])
# # axes[1].set_yticks(np.linspace(correlations[('Phonological', 'Alpha')].min(), correlations[('Phonological', 'Alpha')].max(), 5))
# axes[1].grid(visible=True)

# axes[1].set_xlabel('Grupos de electrodos', fontsize=15)
# axes[1].set_ylabel('Correlación promedio', fontsize=15)
# axes[1].tick_params(which='minor', bottom=False, left=True, right=True, top=False)

# z = np.zeros(shape=(len(bands),len(stimuli)))
# for i, band in enumerate(bands):
#     for j, stim in enumerate(stimuli):
#         average_correlation = correlations[(stim,band)]
#         z[i,j] = np.corrcoef([average_correlation[:, group].mean() for group in groups_y], np.arange(n_groups_y-1))[1,0]

# # axes[2].set_title('Lateralization: right(G[0-3])-left(G[7-10])')
# axes[2].set_title('Correlación de Pearson')
# im = axes[2].imshow(
#             z,
#             vmin=z.min(),
#             vmax=z.max(),
#             cmap='magma'
#             )
# axes[2].set_xticks(np.arange(len(stimuli)))
# axes[2].set_xticklabels(['Envolvente', 'Tono de voz', 'Espectrograma', 'Fonemas', 'C. Fonológicas'], minor=False, fontsize=12, rotation=35)
# axes[2].set_yticks(np.arange(len(bands)))
# axes[2].set_yticklabels(['Delta', 'Theta', 'Alpha', r'Beta$_1$', r'Beta$_2$', 'Ancha'], minor=False, fontsize=12)
# axes[2].minorticks_off()

# cbar = fig.colorbar(im, ax=axes[2])
# # cbar.ax.tick_params(labelsize=15)
# fig.text(.05, 1, 'a)', fontsize=18, va='top', ha='right')
# fig.text(.35, 1, 'b)', fontsize=18, va='top', ha='right')
# fig.text(.68, 1, 'c)', fontsize=18, va='top', ha='right')
# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'distribucion_anterior_posterior.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ====================================
# # PERFIL ESPECTRAL DE GRUPOS FONEMICOS # TODO REHACER CON PHONEMES-PHONET y PHONOLOGICAL NUEVO
# phonemes_path = 'saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonemes-Discrete-Phonet'
# spectrogram_path = 'saves/preprocessed_data/External/tmin-0.2_tmax0.6/Spectrogram'

# phonemes = config.Exp_info().phonemes_phonet.copy()
# phonemes.remove('/sil/')

# group1 = ['/a/', '/e/', '/i/', '/o/', '/u/', '/l/', '/m/', '/b/', '/R/']
# group2 = ['/k/', '/f/', '/t/', '/s/', '/x/', '/tS/']

# group1_index = [phonemes.index(ph) for ph in group1]
# group2_index = [phonemes.index(ph) for ph in group2]

# sp_group1, sp_group2 = [], []
# for sesion in config.sesiones:
#     ph_fname = os.path.join(phonemes_path, f'Sesion{sesion}.pkl')
#     sp_fname = os.path.join(spectrogram_path, f'Sesion{sesion}.pkl')
#     ph_1, ph_2 = load_pickle(path=ph_fname)
#     sp_1, sp_2 = load_pickle(path=sp_fname)

#     sp_group1_ses = []
#     sp_group2_ses = []
#     for ph, sp, in zip([ph_1, ph_2], [sp_1, sp_2]):
#         spectrogram_group1 = []
#         for col in ph[:, group1_index].T:
#             spectrogram_group1.append(sp[(col==1)])

#         max_len_group1 = max([len(sp) for sp in spectrogram_group1])

#         spectrogram_group1_padded = []
#         for spg in spectrogram_group1:
#             if len(spg)!=max_len_group1:
#                 new_sp = np.full(shape=(max_len_group1, 16), fill_value=spg.min())
#                 pad = (max_len_group1 - spg.shape[0]) // 2
#                 new_sp[pad:pad + spg.shape[0], :] = spg
#                 spectrogram_group1_padded.append(new_sp)
#             else:
#                 spectrogram_group1_padded.append(spg)

#         spectrogram_group2 = []
#         for col in ph[:, group2_index].T:
#             spectrogram_group2.append(sp[(col==1)])
#         max_len_group2 = max([len(sp) for sp in spectrogram_group2])

#         spectrogram_group2_padded = []
#         for spg in spectrogram_group2:
#             if len(spg)!=max_len_group2:
#                 new_sp = np.full(shape=(max_len_group2, 16), fill_value=spg.min())
#                 pad = (max_len_group2 - spg.shape[0]) // 2
#                 new_sp[pad:pad + spg.shape[0], :] = spg
#                 spectrogram_group2_padded.append(new_sp)
#             else:
#                 spectrogram_group2_padded.append(spg)

#         sp_group1_ses.append(np.stack(spectrogram_group1_padded).mean(axis=0))
#         sp_group2_ses.append(np.stack(spectrogram_group2_padded).mean(axis=0))
#     sp_group1.append(np.stack(spectrogram_group1_padded))
#     sp_group2.append(np.stack(spectrogram_group2_padded))

# sp_group1 = [sp_group1[i].mean(axis=0) for i in range(len(config.sesiones))]
# sp_group2 = [sp_group2[i].mean(axis=0) for i in range(len(config.sesiones))]

# max_len_group1 = max([len(sp) for sp in sp_group1])
# max_len_group2 = max([len(sp) for sp in sp_group2])

# sp_group1_padded = []
# for spg in sp_group1:
#     if len(spg)!=max_len_group1:
#         new_sp = np.full(shape=(max_len_group1, 16), fill_value=spg.min())
#         pad = (max_len_group1 - spg.shape[0]) // 2
#         new_sp[pad:pad + spg.shape[0], :] = spg
#         sp_group1_padded.append(new_sp)
#     else:
#         sp_group1_padded.append(spg)

# sp_group2_padded = []
# for spg in sp_group2:
#     if len(spg)!=max_len_group2:
#         new_sp = np.full(shape=(max_len_group2, 16), fill_value=spg.min())
#         pad = (max_len_group2 - spg.shape[0]) // 2
#         new_sp[pad:pad + spg.shape[0], :] = spg
#         sp_group2_padded.append(new_sp)
#     else:
#         sp_group2_padded.append(spg)

# sp_group1, sp_group2 = np.stack(sp_group1_padded).mean(axis=0), np.stack(sp_group2_padded).mean(axis=0)

# ###########
# phonological_path = 'saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonological'
# spectrogram_path = 'saves/preprocessed_data/External/tmin-0.2_tmax0.6/Spectrogram'

# phonological = list(config.Exp_info().phonological_labels).copy()
# phonological.remove('trill')
# phonological.remove('pause')

# group1 = ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal']
# group2 = ['dental', 'consonantal', 'velar', 'flap', 'close', 'strident', 'continuant']


# group1_index = [phonological.index(ph) for ph in group1]
# group2_index = [phonological.index(ph) for ph in group2]

# fsp_group1, fsp_group2 = [], []
# for sesion in config.sesiones:
#     ph_fname = os.path.join(phonological_path, f'Sesion{sesion}.pkl')
#     sp_fname = os.path.join(spectrogram_path, f'Sesion{sesion}.pkl')
#     ph_1, ph_2 = load_pickle(path=ph_fname)
#     sp_1, sp_2 = load_pickle(path=sp_fname)

#     fsp_group1_ses = []
#     fsp_group2_ses = []
#     for ph, sp, in zip([ph_1, ph_2], [sp_1, sp_2]):
#         spectrogram_group1 = []
#         ph_nuevo = np.zeros(ph.shape)
#         for i, col in enumerate(np.argmax(ph, axis=1)):
#             ph_nuevo[i, col]=1
#         ph = ph_nuevo

#         for col in ph[:, group1_index].T:
#             spectrogram_group1.append(sp[(col==1)])

#         max_len_group1 = max([len(sp) for sp in spectrogram_group1])

#         spectrogram_group1_padded = []
#         for spg in spectrogram_group1:
#             if len(spg)!=max_len_group1:
#                 new_sp = np.full(shape=(max_len_group1, 16), fill_value=spg.min())
#                 pad = (max_len_group1 - spg.shape[0]) // 2
#                 new_sp[pad:pad + spg.shape[0], :] = spg
#                 spectrogram_group1_padded.append(new_sp)
#             else:
#                 spectrogram_group1_padded.append(spg)

#         spectrogram_group2 = []
#         for col in ph[:, group2_index].T:
#             spectrogram_group2.append(sp[(col==1)])
#         max_len_group2 = max([len(sp) for sp in spectrogram_group2])

#         spectrogram_group2_padded = []
#         for spg in spectrogram_group2:
#             if len(spg)!=max_len_group2:
#                 new_sp = np.full(shape=(max_len_group2, 16), fill_value=spg.min())
#                 pad = (max_len_group2 - spg.shape[0]) // 2
#                 new_sp[pad:pad + spg.shape[0], :] = spg
#                 spectrogram_group2_padded.append(new_sp)
#             else:
#                 spectrogram_group2_padded.append(spg)

#         fsp_group1_ses.append(np.stack(spectrogram_group1_padded).mean(axis=0))
#         fsp_group2_ses.append(np.stack(spectrogram_group2_padded).mean(axis=0))
#     fsp_group1.append(np.stack(spectrogram_group1_padded))
#     fsp_group2.append(np.stack(spectrogram_group2_padded))

# fsp_group1 = [fsp_group1[i].mean(axis=0) for i in range(len(config.sesiones))]
# fsp_group2 = [fsp_group2[i].mean(axis=0) for i in range(len(config.sesiones))]

# max_len_group1 = max([len(sp) for sp in fsp_group1])
# max_len_group2 = max([len(sp) for sp in fsp_group2])

# fsp_group1_padded = []
# for spg in fsp_group1:
#     if len(spg)!=max_len_group1:
#         new_sp = np.full(shape=(max_len_group1, 16), fill_value=spg.min())
#         pad = (max_len_group1 - spg.shape[0]) // 2
#         new_sp[pad:pad + spg.shape[0], :] = spg
#         fsp_group1_padded.append(new_sp)
#     else:
#         fsp_group1_padded.append(spg)

# fsp_group2_padded = []
# for spg in fsp_group2:
#     if len(spg)!=max_len_group2:
#         new_sp = np.full(shape=(max_len_group2, 16), fill_value=spg.min())
#         pad = (max_len_group2 - spg.shape[0]) // 2
#         new_sp[pad:pad + spg.shape[0], :] = spg
#         fsp_group2_padded.append(new_sp)
#     else:
#         fsp_group2_padded.append(spg)

# fsp_group1, fsp_group2 = np.stack(fsp_group1_padded).mean(axis=0), np.stack(fsp_group2_padded).mean(axis=0)


# bands_center = librosa.mel_frequencies(
#     n_mels=16+2,
#     fmin=0,
#     fmax=8000
#     )[1:-1]
# # tags = [int(bands_center[i]) for i in np.arange(1, len(bands_center)+1, 2)]
# # ticks = np.arange(0, NumberOfTicks, 2)+.5
# tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center))]
# ticks = np.arange(0, 16)

# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=2,
#     figsize=(10, 8),
#     sharey='row',
#     tight_layout=True
#     )
# im = axes[0, 0].pcolormesh(
#     np.arange(sp_group1.shape[0]),
#     np.arange(16),
#     sp_group1.T,
#     cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     vmin=sp_group1.min(),
#     vmax=sp_group1.max()
#     )
# # im = axes[0, 0].imshow(
# #     sp_group1.T,
# #     aspect='auto',  # Ajusta el aspecto
# #     # extent=[WindowLeft, WindowRight, 0, 16],  # Ajusta los límites de los ejes
# #     origin='lower',  # Ajusta el origen
# #     cmap='Greens',  # Ajusta el mapa de colores
# #     vmin=sp_group1.min(),
# #     vmax=sp_group1.max()
# #     )

# fig.colorbar(
#     im,
#     label='Amplitud (dB)'
#     )
# axes[0, 0].set_xticks(
#     ticks=[sp_group1.shape[0]//2-2000, sp_group1.shape[0]//2, sp_group1.shape[0]//2+2000],
#     labels=['', '', '']
#     )
# axes[0, 0].set_xlim(sp_group1.shape[0]//2-2000, sp_group1.shape[0]//2+2000)
# axes[0, 0].set_yticks(
#     ticks=ticks,
#     labels=tags
#     )
# axes[0, 0].set_ylabel('Frecuencia (Hz)')
# axes[0, 0].set_title('Fonemas (1)')
# axes[0, 0].text(-.1, 1.1, 'a)', transform=axes[0, 0].transAxes, fontsize=18, va='top', ha='right')

# im = axes[1, 0].pcolormesh(
#     np.arange(sp_group2.shape[0]),
#     np.arange(16),
#     sp_group2.T,
#     cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     vmin=sp_group2.min(),
#     vmax=sp_group2.max()
#     )

# fig.colorbar(
#     im,
#     label='Amplitud (dB)'
#     )

# axes[1, 0].set_xticks(
#     ticks=[
#         sp_group2.shape[0]//2-2000, sp_group2.shape[0]//2-1500, sp_group2.shape[0]//2-1000, sp_group2.shape[0]//2-500,\
#         sp_group2.shape[0]//2,\
#         sp_group2.shape[0]//2+500, sp_group2.shape[0]//2+1000, sp_group2.shape[0]//2+1500, sp_group2.shape[0]//2+2000
#         ],
#     labels=[f'-{2000/config.sr:.0f}', f'-{1500/config.sr:.0f}', f'-{1000/config.sr:.0f}', f'-{500/config.sr:.0f}', '0',\
#         f'{500/config.sr:.0f}', f'{1000/config.sr:.0f}', f'{1500/config.sr:.0f}', f'{2000/config.sr:.0f}']
#     )

# axes[1, 0].set_xlim(sp_group2.shape[0]//2-2000, sp_group2.shape[0]//2+2000)
# axes[1, 0].set_yticks(
#     ticks=ticks,
#     labels=tags
#     )
# axes[1, 0].set_xlabel('Tiempo (s)')
# axes[1, 0].set_ylabel('Frecuencia (Hz)')
# axes[1, 0].set_title('Fonemas (2)')
# axes[1, 0].text(-.1, 1.1, 'b)', transform=axes[1, 0].transAxes, fontsize=18, va='top', ha='right')

# im = axes[0, 1].pcolormesh(
#     np.arange(fsp_group1.shape[0]),
#     np.arange(16),
#     fsp_group1.T,
#     cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     vmin=fsp_group1.min(),
#     vmax=fsp_group1.max()
#     )

# fig.colorbar(
#     im,
#     label='Amplitud (dB)'
#     )

# axes[0, 1].set_xticks(
#     ticks=[fsp_group1.shape[0]//2-2000, fsp_group1.shape[0]//2, fsp_group1.shape[0]//2+2000],
#     labels=['', '', '']
#     )
# axes[0, 1].set_xlim(fsp_group1.shape[0]//2-2000, fsp_group1.shape[0]//2+2000)
# axes[0, 1].set_yticks(
#     ticks=ticks,
#     labels=tags
#     )
# # axes[0, 1].set_ylabel('Frecuencia (Hz)')
# axes[0, 1].set_title('C. Fonológicas (1)')
# axes[0, 1].text(-.1, 1.1, 'c)', transform=axes[0, 1].transAxes, fontsize=18, va='top', ha='right')
# im = axes[1, 1].pcolormesh(
#     np.arange(fsp_group2.shape[0]),
#     np.arange(16),
#     fsp_group2.T,
#     cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     vmin=fsp_group2.min(),
#     vmax=fsp_group2.max()
#     )
# fig.colorbar(
#     im,
#     label='Amplitud (dB)'
#     )
# axes[1, 1].set_xticks(
#     ticks=[
#         fsp_group2.shape[0]//2-2000, fsp_group2.shape[0]//2-1500, fsp_group2.shape[0]//2-1000, fsp_group2.shape[0]//2-500,\
#         fsp_group2.shape[0]//2,\
#         fsp_group2.shape[0]//2+500, fsp_group2.shape[0]//2+1000, fsp_group2.shape[0]//2+1500, fsp_group2.shape[0]//2+2000
#         ],
#     labels=[f'-{2000/config.sr:.0f}', f'-{1500/config.sr:.0f}', f'-{1000/config.sr:.0f}', f'-{500/config.sr:.0f}', '0',\
#         f'{500/config.sr:.0f}', f'{1000/config.sr:.0f}', f'{1500/config.sr:.0f}', f'{2000/config.sr:.0f}']
#     )

# axes[1, 1].set_xlim(fsp_group2.shape[0]//2-2000, fsp_group2.shape[0]//2+2000)
# # axes[1, 1].set_yticks(
# #     ticks=ticks,
# #     labels=tags
# #     )
# axes[1, 1].set_xlabel('Tiempo (s)')
# # axes[1, 1].set_ylabel('Frecuencia (Hz)')
# axes[1, 1].set_title('C. Fonológicas (2)')
# axes[1, 1].text(-.1, 1.1, 'd)', transform=axes[1, 1].transAxes, fontsize=18, va='top', ha='right')

# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'perfiles_grupos.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ==========================================
# # MATRIZ CORRELACIONES Y SIMILARIDAD CABEZAS 
# situation = 'External'
# correlations_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/')
# mtrf_path = os.path.normpath(f'saves/{config.model}/{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
# bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']
# stimuli = ['Pitch-Log-Raw', 'Envelope', 'Mfccs', 'Spectrogram', 'Phonemes-Phonet', 'Phonological']

# # Cálculo de correlaciones (como en tu código)
# correlations = {
#     (stim, band): load_pickle(path=os.path.join(correlations_path, band, stim + '.pkl'))['average_correlation_subjects'].mean(axis=0)
#     for stim in stimuli for band in bands
# }
# minimum_cor = min([corr.min() for corr in correlations.values()])
# maximum_cor = max([corr.max() for corr in correlations.values()])
# normalizer_c = Normalize(vmin=np.round(minimum_cor, 2), vmax=np.round(maximum_cor, 2))
# im_c = cm.ScalarMappable(norm=normalizer_c, cmap='Reds')

# n_stims, n_bands = len(stimuli), len(bands)

# # Get mean correlations across subjects and total max and min
# correlations = {(stim,band):load_pickle(path=os.path.join(correlations_path, band, stim +'.pkl'))['average_correlation_subjects'].mean(axis=0) for stim in stimuli for band in bands}
# minimum_cor, maximum_cor = min([correlation.min() for correlation in correlations.values()]), max([correlation.max() for correlation in correlations.values()])

# # Create figure and title
# fig, axes = plt.subplots(
#         figsize=(8,8),
#         nrows=n_bands,
#         ncols=n_stims,
#         layout="constrained"
#         )

# # Configure axis
# for ax, col in zip(axes[:,0], stimuli):
#     if col=='Phonemes-Phonet':
#         col = 'Fonemas'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Pitch-Log-Raw':
#         col = 'Tono de voz'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Envelope':
#         col = 'Envolvente'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Phonological':
#         col = 'C. Fonológicas'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Spectrogram':
#         col = 'Espectrograma'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Mfccs':
#         col = 'C. Mel'
#         ax.set_ylabel(col, rotation=90)
#     else:
#         ax.set_ylabel(col, rotation=90)
# for ax, band in zip(axes[0], bands):
#     if band=='Beta1':
#         band=r'Beta$_1$'

#     if band=='Beta2':
#         band=r'Beta$_2$'

#     if band=='All':
#         band='Ancha'
#     ax.set_title(band)

# # Build scale
# normalizer = Normalize(vmin=np.round(minimum_cor,2), vmax=np.round(maximum_cor,2))
# im = cm.ScalarMappable(norm=normalizer, cmap='Reds')

# # Iterate over bands
# for j, band in enumerate(bands):
#     for i, stim in enumerate(stimuli):
#         # Get average correlation of each stimulus across subjects
#         average_correlation = correlations[(stim,band)]

#         # Plot topomap
#         mne.viz.plot_topomap(
#                 data=average_correlation,
#                 pos=config.info_mne,
#                 axes=axes[i, j],
#                 show=False,
#                 sphere=0.07,
#                 cmap='Reds',
#                 # vlim=(minimum_cor, maximum_cor),
#                 cnorm=normalizer
#                 )

# # Make colorbar
# cbar = fig.colorbar(im, ax=axes.ravel().tolist())
# cbar.ax.tick_params(labelsize=15)
# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'matriz_corr_externa.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # =========================
# # PESOS FONOLOG POR GRUPOS # TODO REHACER CON PHONEMES-PHONET y PHONOLOGICAL NUEVO
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonological/total_weights_per_subject.pkl'
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)
# order_old, null_indexes = clustering_by_correlation(weights=average_weights_subjects.mean(axis=0).mean(axis=0))

# # Classify labels for categorization
# tags = list(config.Exp_info().phonological_labels).copy()
# tags_1 = ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal']
# tags_2 = ['dental', 'consonantal', 'pause', 'velar', 'flap', 'close', 'strident', 'continuant']
# group1 = [tags.index(ph) for ph in  tags_1]
# group2 = [tags.index(ph) for ph in tags_2]

# average_weights_subjects_1 = average_weights_subjects[:,:, group1, :]
# average_weights_subjects_2 = average_weights_subjects[:,:, group2, :]

# fig = plt.figure(
#     figsize=(12, 7),
#     tight_layout=True
#     )

# # Definir la cuadrícula usando GridSpec
# # 2 filas y 2 columnas, con la segunda columna dividida en dos partes en la primera fila
# gs = gridspec.GridSpec(
#     nrows=2,
#     ncols=2,
#     width_ratios=[1, 1],
#     height_ratios=[1, 2]
#     )

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# ax1 = plt.subplot(gs[0, 0])
# weights_1 = average_weights_subjects_1.mean(axis=0).mean(axis=1)
# evoked = mne.EvokedArray(data=weights_1, info=config.info_mne)
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

# # # Eliminar el esquema de la cabeza original
# # for ax in fig.axes:
# #     # Verificar si el eje contiene un objeto de tipo "PathCollection" (los puntos de los canales)
# #     for artist in ax.get_children():
# #         if isinstance(artist, PathCollection):
# #             ax.remove()  # Eliminar el eje que contiene el esquema de la cabeza original
# #             break


# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.1))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)

# feat_weights_1 = average_weights_subjects_1.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights_1)
# feat_weights_1 = feat_weights_1[order]

# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights_1.shape[0]),
#     feat_weights_1,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights_1).max(),
#     vmax=np.abs(feat_weights_1).max()
#     )

# # Set figure configuration
# tags_1 = [tags_1[i] for i in order]

# ticks = np.arange(feat_weights_1.shape[0])
# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel='Grupo de c. fonológicas 1',
#     yticks=ticks,
#     yticklabels=tags_1
#     )

# # Configure colorbar
# fig.colorbar(
#     im,
#     ax=ax2,
#     orientation='horizontal',
#     shrink=1,
#     label='Amplitud (U.A)',
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

# for ax in fig.axes:
#     if ax.get_subplotspec() == gs[0, 1]:
#         ax.remove()

# ax3 = plt.subplot(gs[0, 1], sharey=ax1)
# weights_2 = average_weights_subjects_2.mean(axis=0).mean(axis=1)
# evoked_2 = mne.EvokedArray(data=weights_2, info=config.info_mne)
# evoked_2.shift_time(config.times[0], relative=True)
# evoked_plot_2 = evoked_2.plot(
#     scalings={'eeg':1},
#     zorder='std',
#     time_unit='ms',
#     show=False,
#     spatial_colors=True,
#     # unit=False,
#     units='mTRFs (U.A)',
#     axes=ax3,
#     gfp=False
#     )


# ax3.plot(
#     config.times*1e3, #ms
#     evoked_2._data.mean(axis=0),
#     'black',
#     label='Valor medio',
#     zorder=130,
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in ax3.get_lines()[:len(evoked_2.ch_names)]]

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
# ax_head_outline = fig.add_axes([.31, 0.82, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.32, 0.822, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# # Obtener las posiciones de los sensores en 2D
# montage = evoked_2.info.get_montage()
# pos = montage.get_positions()['ch_pos']  # Diccionario con las posiciones de los canales

# # Crear un eje adicional para la cabecita sin sensores
# ax_head_outline_2 = fig.add_axes([.75, 0.82, 0.11, 0.11])  # [x, y, width, height]

# # Graficar solo el contorno de la cabeza (sin sensores)
# mne.viz.plot_topomap(
#     np.zeros(len(evoked_2.ch_names)),  # Datos ficticios (todos ceros)
#     evoked_2.info,
#     axes=ax_head_outline_2,
#     show=False,
#     sensors=False,  # No graficar los sensores
#     outlines='head'  # Graficar solo el contorno de la cabeza
# )
# ax_head_outline_2.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_outline_2.axis('off')  # Ocultar los ejes

# # Crear un eje adicional para graficar los sensores
# ax_head_2 = fig.add_axes([.76, 0.822, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked_2.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head_2.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head_2.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_2.axis('off')  # Ocultar los ejes

# ax3.grid(visible=True)
# ax3.set(xlabel='', xticklabels=[], ylabel='', yticklabels=['','','','',''], title='EEG (128 canales)')
# ax1.set(ylabel='mTRFs', yticks=[-0.02, -0.01,  0.  ,  0.01,  0.02], yticklabels=[-0.02, -0.01,  0.  ,  0.01,  0.02])
# ax3.tick_params(axis='x', which='both', labelbottom=False)
# ax3.tick_params(axis='y', labelleft=False)
# ax3.legend(loc=(.57,.1))
# ax3.text(-.1, 1.1, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')


# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax4 = plt.subplot(gs[1, 1], sharex=ax3)

# feat_weights_2 = average_weights_subjects_2.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights_2)
# feat_weights_2 = feat_weights_2[order]

# im = ax4.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights_2.shape[0]),
#     feat_weights_2,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights_2).max(),
#     vmax=np.abs(feat_weights_2).max()
#     )

# # Set figure configuration
# ticks = np.arange(feat_weights_2.shape[0])

# # Set figure configuration
# tags_2 = [tags_2[i] for i in order]

# ax4.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel='Grupo de c. fonológicas 2',
#     yticks=ticks,
#     yticklabels=tags_2
#     )

# # Configure colorbar
# fig.colorbar(
#     im,
#     ax=ax4,
#     orientation='horizontal',
#     shrink=1,
#     label='Amplitud (U.A)',
#     fraction=.075,
#     aspect=20
#     )
# ax4.text(-.1, 1.1, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Eliminar la etiqueta "Nave"
# for txt in fig.findobj(mtext.Text):
#     if "ave" in txt.get_text():
#          txt.remove()
# fig.savefig(
#     os.path.join(tesis_path,'resultados', f'fonologicas_grupos.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()

# # =========================
# # PESOS FONEMAS POR GRUPOS # TODO REHACER CON PHONEMES-PHONET y PHONOLOGICAL NUEVO
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Discrete-Phonet/total_weights_per_subject.pkl'
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)
# order_old, null_indexes = clustering_by_correlation(weights=average_weights_subjects.mean(axis=0).mean(axis=0))

# # Classify labels for categorization
# tags = config.Exp_info().phonemes_phonet.copy()
# tags.remove('/sil/')
# tags_1 = ['/a/', '/e/', '/i/', '/o/', '/u/', '/l/', '/m/', '/b/', '/R/']
# tags_2 = ['/k/', '/f/', '/t/', '/s/', '/x/', '/tS/']
# group1 = [tags.index(ph) for ph in tags_1]
# group2 = [tags.index(ph) for ph in tags_2]

# average_weights_subjects_1 = average_weights_subjects[:,:, group1, :]
# average_weights_subjects_2 = average_weights_subjects[:,:, group2, :]

# fig = plt.figure(
#     figsize=(12, 7),
#     tight_layout=True
#     )

# # Definir la cuadrícula usando GridSpec
# # 2 filas y 2 columnas, con la segunda columna dividida en dos partes en la primera fila
# gs = gridspec.GridSpec(
#     nrows=2,
#     ncols=2,
#     width_ratios=[1, 1],
#     height_ratios=[1, 2]
#     )

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# ax1 = plt.subplot(gs[0, 0])
# weights_1 = average_weights_subjects_1.mean(axis=0).mean(axis=1)
# evoked = mne.EvokedArray(data=weights_1, info=config.info_mne)
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

# # # Eliminar el esquema de la cabeza original
# # for ax in fig.axes:
# #     # Verificar si el eje contiene un objeto de tipo "PathCollection" (los puntos de los canales)
# #     for artist in ax.get_children():
# #         if isinstance(artist, PathCollection):
# #             ax.remove()  # Eliminar el eje que contiene el esquema de la cabeza original
# #             break


# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.1))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)

# feat_weights_1 = average_weights_subjects_1.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights_1)
# feat_weights_1 = feat_weights_1[order]

# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights_1.shape[0]),
#     feat_weights_1,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights_1).max(),
#     vmax=np.abs(feat_weights_1).max()
#     )

# # Set figure configuration
# tags_1 = [tags_1[i] for i in order]

# ticks = np.arange(feat_weights_1.shape[0])
# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel='Grupo de fonemas 1',
#     yticks=ticks,
#     yticklabels=tags_1
#     )

# # Configure colorbar
# fig.colorbar(
#     im,
#     ax=ax2,
#     orientation='horizontal',
#     shrink=1,
#     label='Amplitud (U.A)',
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

# for ax in fig.axes:
#     if ax.get_subplotspec() == gs[0, 1]:
#         ax.remove()

# ax3 = plt.subplot(gs[0, 1], sharey=ax1)
# weights_2 = average_weights_subjects_2.mean(axis=0).mean(axis=1)
# evoked_2 = mne.EvokedArray(data=weights_2, info=config.info_mne)
# evoked_2.shift_time(config.times[0], relative=True)
# evoked_plot_2 = evoked_2.plot(
#     scalings={'eeg':1},
#     zorder='std',
#     time_unit='ms',
#     show=False,
#     spatial_colors=True,
#     # unit=False,
#     units='mTRFs (U.A)',
#     axes=ax3,
#     gfp=False
#     )


# ax3.plot(
#     config.times*1e3, #ms
#     evoked_2._data.mean(axis=0),
#     'black',
#     label='Valor medio',
#     zorder=130,
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in ax3.get_lines()[:len(evoked_2.ch_names)]]

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
# ax_head_outline = fig.add_axes([.31, 0.82, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.32, 0.822, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# # Obtener las posiciones de los sensores en 2D
# montage = evoked_2.info.get_montage()
# pos = montage.get_positions()['ch_pos']  # Diccionario con las posiciones de los canales

# # Crear un eje adicional para la cabecita sin sensores
# ax_head_outline_2 = fig.add_axes([.75, 0.82, 0.11, 0.11])  # [x, y, width, height]

# # Graficar solo el contorno de la cabeza (sin sensores)
# mne.viz.plot_topomap(
#     np.zeros(len(evoked_2.ch_names)),  # Datos ficticios (todos ceros)
#     evoked_2.info,
#     axes=ax_head_outline_2,
#     show=False,
#     sensors=False,  # No graficar los sensores
#     outlines='head'  # Graficar solo el contorno de la cabeza
# )
# ax_head_outline_2.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_outline_2.axis('off')  # Ocultar los ejes

# # Crear un eje adicional para graficar los sensores
# ax_head_2 = fig.add_axes([.76, 0.822, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked_2.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head_2.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head_2.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_2.axis('off')  # Ocultar los ejes

# ax3.grid(visible=True)
# ax3.set(xlabel='', xticklabels=[], ylabel='', yticklabels=['','','','',''], title='EEG (128 canales)')
# ax1.set(ylabel='mTRFs', yticks=[-0.02, -0.01,  0.  ,  0.01,  0.02], yticklabels=[-0.02, -0.01,  0.  ,  0.01,  0.02])
# ax3.tick_params(axis='x', which='both', labelbottom=False)
# ax3.tick_params(axis='y', labelleft=False)
# ax3.legend(loc=(.57,.1))
# ax3.text(-.1, 1.1, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')


# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax4 = plt.subplot(gs[1, 1], sharex=ax3)

# feat_weights_2 = average_weights_subjects_2.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights_2)
# feat_weights_2 = feat_weights_2[order]

# im = ax4.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights_2.shape[0]),
#     feat_weights_2,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights_2).max(),
#     vmax=np.abs(feat_weights_2).max()
#     )

# # Set figure configuration
# ticks = np.arange(feat_weights_2.shape[0])

# # Set figure configuration
# tags_2 = [tags_2[i] for i in order]

# ax4.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel='Grupo de fonemas 2',
#     yticks=ticks,
#     yticklabels=tags_2
#     )

# # Configure colorbar
# fig.colorbar(
#     im,
#     ax=ax4,
#     orientation='horizontal',
#     shrink=1,
#     label='Amplitud (U.A)',
#     fraction=.075,
#     aspect=20
#     )
# ax4.text(-.1, 1.1, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Eliminar la etiqueta "Nave"
# for txt in fig.findobj(mtext.Text):
#     if "ave" in txt.get_text():
#          txt.remove()
# fig.savefig(
#     os.path.join(tesis_path,'resultados', f'fonemas_grupos.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()

# # ===============================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: Pitch-Log-Raw
# s_channels=False
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Pitch-Log-Raw.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Pitch-Log-Raw/total_weights_per_subject.pkl'
# # if s_channels:
#     # path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Pitch-Log-Raw_4096_s.pkl'
# # else:
#     # path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Pitch-Log-Raw_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# # average_correlation_subjects = np.where((significant_channels_subjects==1), average_correlation_subjects, np.nan)
# # np.nanmean(average_correlation_subjects, axis=0).mean()
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

# # Crear una figura
# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=2,
#     figsize=(10, 7),
#     constrained_layout=True,
#     )

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)
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
#     axes=axes[0,0],
#     gfp=False
#     )
# # Eliminar la etiqueta "Nave"
# for text in evoked_plot.axes[0].texts:
#     if "ave" in text.get_text():
#         text.set_visible(False)  # Ocultar el texto
# axes[0,0].plot(
#     config.times*1e3, #ms
#     evoked._data.mean(axis=0),
#     'black',
#     label='Valor medio',
#     zorder=130,
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in axes[0,0].get_lines()[:len(evoked.ch_names)]]

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
# ax_head_outline = fig.add_axes([.35, 0.83, 0.12, 0.12])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.36, 0.832, 0.1, 0.1])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# axes[0,0].grid(visible=True)
# axes[0,0].set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# axes[0,0].tick_params(axis='x', which='both', labelbottom=False)
# axes[0,0].legend(loc=(.5,.19))
# axes[0,0].text(-.1, 1.1, 'a)', transform=axes[0,0].transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# significant_channels = np.zeros(shape=(average_weights_subjects.shape[2], len(config.times)))

# # Iteate over columns to get number of channels per feature that passes the threshold
# for feature in range(average_weights_subjects.shape[2]):
#     for delay in range(len(config.times)):
#         # Count how many channels pass the threshold for a given feature and delay
#         ppval = pvalue_tfce[feature][delay]
#         significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# number_of_ticks = significant_channels.shape[0]
# y, z = (np.arange(2), np.concatenate((significant_channels,significant_channels)))
# im2 = axes[1,0].pcolormesh(
#     config.times*1e3, # x
#     y, # y
#     z, #z
#     shading='auto',
#     cmap='inferno'
#     )
# axes[1,0].set(
#     xlabel='Tiempo (ms)',
#     yticklabels=['' for i in range(axes[1,0].get_yticks().shape[0])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600]
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=im2,
#     ax=axes[1,0],
#     orientation='horizontal',
#     shrink=1,
#     label='Número de canales significativos',
#     fraction=.075,
#     aspect=20
#     )
# # pos = axes[1,0].get_position()

# # # Reducir el ancho al 80% de su tamaño original:
# # pos = axes[1,0].get_position()
# # new_y0 = pos.y0 + (pos.height - pos.height * 0.5) / 2
# # axes[1,0].set_position([pos.x0, new_y0, pos.width, pos.height * 0.5])
# # plt.draw()

# axes[1,0].text(-.1, 1.1, 'b)', transform=axes[1,0].transAxes, fontsize=18, va='top', ha='right')

# # Tercer gráfico en la primera subcolumna de la segunda columna (primera fila)
# mean_average_correlation = average_correlation_subjects.mean(axis=0) #---> # n_subj, n_chans # TODO TOPOMAPS
# im = mne.viz.plot_topomap(
#         data=mean_average_correlation,
#         pos=config.info_mne,
#         cmap='Reds',
#         vlim=(mean_average_correlation.min(), mean_average_correlation.max()),
#         show=False,
#         sphere=0.07,
#         axes=axes[0,1]
#         )
# cbar = fig.colorbar(
#         im[0],
#         ax=axes[0,1],
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# axes[0,1].axis('off')  # Desactivar ejes
# axes[0,1].set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std()/np.sqrt(config.sr):.3f}'+r')$', fontsize=15)
# axes[0,1].text(-.35, 1.1, 'c)', transform=axes[0,1].transAxes, fontsize=18, va='top', ha='right')

# # Cuarto gráfico en la segunda subcolumna de la segunda columna (primera fila)
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
#     axes=axes[1,1],
#     show=False,
#     sphere=0.07,
#     cmap='Greens',
#     vlim=(absolute_correlation_per_channel.min(),absolute_correlation_per_channel.max())
#     )

# # Make colorbar
# cbar = fig.colorbar(
#     im[0],
#     ax=axes[1,1],
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal',
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# axes[1,1].set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std()/np.sqrt(correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)].shape[0]):.3f}'+r')$', fontsize=15)

# axes[1,1].axis('off')  # Desactivar ejes
# axes[1,1].text(-.35, 1.1, 'd)', transform=axes[1,1].transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'tonodevoz_completo_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'tonodevoz_completo.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # ============================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: Envelope # TODO FALTA TFCE SIGNIFICATIVO
# s_channels=False
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Envelope.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Envelope/total_weights_per_subject.pkl'
# # if s_channels:
#     # path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Envelope_4096_s.pkl'
# # else:
#     # path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Envelope_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

# # Crear una figura
# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=2,
#     figsize=(10, 7),
#     constrained_layout=True,
#     )

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)

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
#     axes=axes[0,0],
#     gfp=False
#     )
# # Eliminar la etiqueta "Nave"
# for text in evoked_plot.axes[0].texts:
#     if "ave" in text.get_text():
#         text.set_visible(False)  # Ocultar el texto
# axes[0,0].plot(
#     config.times*1e3, #ms
#     evoked._data.mean(axis=0),
#     'black',
#     label='Valor medio',
#     zorder=130,
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in axes[0,0].get_lines()[:len(evoked.ch_names)]]

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
# ax_head_outline = fig.add_axes([.35, 0.83, 0.12, 0.12])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.36, 0.832, 0.1, 0.1])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# axes[0,0].grid(visible=True)
# axes[0,0].set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# axes[0,0].tick_params(axis='x', which='both', labelbottom=False)
# axes[0,0].legend(loc=(.5,.19))
# axes[0,0].text(-.1, 1.1, 'a)', transform=axes[0,0].transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# significant_channels = np.zeros(shape=(average_weights_subjects.shape[2], len(config.times)))

# # Iteate over columns to get number of channels per feature that passes the threshold
# for feature in range(average_weights_subjects.shape[2]):
#     for delay in range(len(config.times)):
#         # Count how many channels pass the threshold for a given feature and delay
#         ppval = pvalue_tfce[feature][delay]
#         significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# number_of_ticks = significant_channels.shape[0]
# y, z = (np.arange(2), np.concatenate((significant_channels,significant_channels)))
# im2 = axes[1,0].pcolormesh(
#     config.times*1e3, # x
#     y, # y
#     z, #z
#     shading='auto',
#     cmap='inferno'
#     )
# axes[1,0].set(
#     xlabel='Tiempo (ms)',
#     yticklabels=['' for i in range(axes[1,0].get_yticks().shape[0])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600]
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=im2,
#     ax=axes[1,0],
#     orientation='horizontal',
#     shrink=1,
#     label='Número de canales significativos',
#     fraction=.075,
#     aspect=20
#     )

# axes[1,0].text(-.1, 1.1, 'b)', transform=axes[1,0].transAxes, fontsize=18, va='top', ha='right')

# # Tercer gráfico en la primera subcolumna de la segunda columna (primera fila)
# mean_average_correlation = average_correlation_subjects.mean(axis=0)
# im = mne.viz.plot_topomap(
#         data=mean_average_correlation,
#         pos=config.info_mne,
#         cmap='Reds',
#         vlim=(mean_average_correlation.min(), mean_average_correlation.max()),
#         show=False,
#         sphere=0.07,
#         axes=axes[0,1]
#         )
# cbar = fig.colorbar(
#         im[0],
#         ax=axes[0,1],
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# axes[0,1].axis('off')  # Desactivar ejes
# axes[0,1].set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std()/np.sqrt(config.sr):.3f}'+r')$', fontsize=15)
# axes[0,1].text(-.35, 1.1, 'c)', transform=axes[0,1].transAxes, fontsize=18, va='top', ha='right')

# # Cuarto gráfico en la segunda subcolumna de la segunda columna (primera fila)
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
#     axes=axes[1,1],
#     show=False,
#     sphere=0.07,
#     cmap='Greens',
#     vlim=(absolute_correlation_per_channel.min(),absolute_correlation_per_channel.max())
#     )

# # Make colorbar
# cbar = fig.colorbar(
#     im[0],
#     ax=axes[1,1],
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal',
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# axes[1,1].set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std()/np.sqrt(correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)].shape[0]):.3f}'+r')$', fontsize=15)

# axes[1,1].axis('off')  # Desactivar ejes
# axes[1,1].text(-.35, 1.1, 'd)', transform=axes[1,1].transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'envolvente_completo_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'envolvente_completo.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # ================================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: PHONOLOGICAL # TODO FALTA TFCE SIGNIFICATIVO
# s_channels=True
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Phonological.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonological/total_weights_per_subject.pkl'
# # if s_channels:
# #     # path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonological_4096_s.pkl'
# # # else:
# #     # path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonological_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)
# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

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
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)
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
# ax_head_outline = fig.add_axes([.33, 0.84, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.34, 0.842, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.1))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)
# if s_channels:
#     feat_weights = average_weights_subjects[significant_channels_subjects.astype(bool)].mean(axis=0)
# else:
#     feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights)
# feat_weights = feat_weights[order]

# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights.shape[0]),
#     feat_weights,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
#     vmax=np.abs(feat_weights).max()
#     )

# # Set figure configuration
# tags = list(config.Exp_info().phonological_labels)
# tags.remove('trill')
# tags.remove('pause')

# ticks = np.arange(feat_weights.shape[0])
# tags = tags if order is None else [tags[i] for i in order]

# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel='Características fonológicas',
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
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#         im[0],
#         ax=ax3,
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# ax3.axis('off')  # Desactivar ejes
# ax3.set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std()/np.sqrt(config.sr):.3f}'+r')$', fontsize=15)
# ax3.text(-.2, 1.15, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#     im[0],
#     ax=ax4,
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal',
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# ax4.set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std()/np.sqrt(correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)].shape[0]):.3f}'+r')$', fontsize=15)

# ax4.axis('off')  # Desactivar ejes
# ax4.text(-.2, 1.15, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Quinto gráfico en la segunda columna (segunda fila)
# ax5 = plt.subplot(gs[1, 1])

# significant_channels = np.zeros(shape=(average_weights_subjects.shape[2], len(config.times)))

# # Iteate over columns to get number of channels per feature that passes the threshold
# for feature in range(average_weights_subjects.shape[2]):
#     for delay in range(len(config.times)):
#         # Count how many channels pass the threshold for a given feature and delay
#         ppval = pvalue_tfce[feature][delay]
#         significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# significant_channels = significant_channels[order]

# # Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
# number_of_ticks = significant_channels.shape[0]
# y, z = np.arange(number_of_ticks), significant_channels

# imph = ax5.pcolormesh(
#     config.times*1e3, # x
#     y, # y
#     z, # z
#     shading='auto',
#     cmap='inferno'
#     )
# ax5.set(
#     xlabel='Tiempo (ms)',
#     # yticks=np.arange(0, average_weights_subjects.shape[2], 1),
#     yticklabels=['' for i in range(average_weights_subjects.shape[2])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600]
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=imph,
#     ax=ax5,
#     orientation='horizontal',
#     shrink=1,
#     label='Número de canales significativos',
#     fraction=.075,
#     aspect=20
#     )

# ax5.text(-.06, 1.1, 'e)', transform=ax5.transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'fonologicas_completo_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'fonologicas_completo.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # ===============================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: PHONEMES-DISCRETE # TODO REHACER CON PHONEMES-PHONET y PHONOLOGICAL NUEVO
# s_channels=True
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Phonemes-Discrete-Phonet.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Discrete-Phonet/total_weights_per_subject.pkl'
# path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Discrete-Phonet_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)


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
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)
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
# ax_head_outline = fig.add_axes([.33, 0.84, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.34, 0.842, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.1))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)

# if s_channels:
#     feat_weights = average_weights_subjects[significant_channels_subjects.astype(bool)].mean(axis=0)
# else:
#     feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights)
# feat_weights = feat_weights[order]

# im = ax2.pcolormesh(
#     config.times * 1e3, 
#     np.arange(feat_weights.shape[0]), 
#     feat_weights, 
#     cmap='RdBu_r', 
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
#     vmax=np.abs(feat_weights).max()
#     )

# # Set figure configuration
# tags = config.Exp_info().phonemes_phonet
# tags.remove('/sil/')
# ticks = np.arange(feat_weights.shape[0])
# tags = tags if order is None else [tags[i] for i in order]

# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600], 
#     ylabel='Fonemas', 
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
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#         im[0],
#         ax=ax3, 
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# ax3.axis('off')  # Desactivar ejes
# ax3.set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std():.3f}'+r')$', fontsize=15)
# ax3.text(-.2, 1.15, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#     im[0], 
#     ax=ax4, 
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal', 
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# ax4.set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std():.3f}'+r')$', fontsize=15)

# ax4.axis('off')  # Desactivar ejes
# ax4.text(-.2, 1.15, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Quinto gráfico en la segunda columna (segunda fila)
# ax5 = plt.subplot(gs[1, 1])

# significant_channels = np.zeros(shape=(average_weights_subjects.shape[2], len(config.times)))

# # Iteate over columns to get number of channels per feature that passes the threshold
# for feature in range(average_weights_subjects.shape[2]):
#     for delay in range(len(config.times)):
#         # Count how many channels pass the threshold for a given feature and delay
#         ppval = pvalue_tfce[feature][delay]
#         significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# significant_channels = significant_channels[order]

# # Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
# number_of_ticks = significant_channels.shape[0]
# y, z = np.arange(number_of_ticks), significant_channels

# imph = ax5.pcolormesh(
#     config.times*1e3, # x
#     y, # y
#     z, # z
#     shading='auto',
#     cmap='inferno'
#     )
# ax5.set(
#     xlabel='Tiempo (ms)', 
#     # yticks=np.arange(0, average_weights_subjects.shape[2], 1),
#     yticklabels=['' for i in range(average_weights_subjects.shape[2])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600] 
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=imph, 
#     ax=ax5, 
#     orientation='horizontal', 
#     shrink=1, 
#     label='Número de canales significativos', 
#     fraction=.075,
#     aspect=20
#     )

# ax5.text(-.06, 1.1, 'e)', transform=ax5.transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'fonemas_completo_discreto_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'fonemas_completo_discreto.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # ===================================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: PHONEMES-PHONET # TODO FATLA ESTADISTICA Y TFCE SIGNIFICATIVO
# s_channels=False
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Phonemes-Phonet.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Phonet/total_weights_per_subject.pkl'
# # if s_channels:
# #     path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Phonet_4096_s.pkl'
# # else:
# #     path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Phonet_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

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
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)
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
# ax_head_outline = fig.add_axes([.33, 0.84, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.34, 0.842, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.1))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)
# if s_channels:
#     feat_weights = average_weights_subjects[significant_channels_subjects.astype(bool)].mean(axis=0)
# else:
#     feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights)
# feat_weights = feat_weights[order]

# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights.shape[0]),
#     feat_weights,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
#     vmax=np.abs(feat_weights).max()
#     )

# # Set figure configuration
# tags = config.Exp_info().phonemes_phonet
# tags.remove('/sil/')
# ticks = np.arange(feat_weights.shape[0])
# tags = tags if order is None else [tags[i] for i in order]

# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel='Fonemas',
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
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#         im[0],
#         ax=ax3,
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# ax3.axis('off')  # Desactivar ejes
# ax3.set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std()/np.sqrt(config.sr):.3f}'+r')$', fontsize=15)
# ax3.text(-.2, 1.15, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#     im[0],
#     ax=ax4,
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal',
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# ax4.set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std()/np.sqrt(correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)].shape[0]):.3f}'+r')$', fontsize=15)

# ax4.axis('off')  # Desactivar ejes
# ax4.text(-.2, 1.15, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Quinto gráfico en la segunda columna (segunda fila)
# ax5 = plt.subplot(gs[1, 1])

# significant_channels = np.zeros(shape=(average_weights_subjects.shape[2], len(config.times)))

# # Iteate over columns to get number of channels per feature that passes the threshold
# for feature in range(average_weights_subjects.shape[2]):
#     for delay in range(len(config.times)):
#         # Count how many channels pass the threshold for a given feature and delay
#         ppval = pvalue_tfce[feature][delay]
#         significant_channels[feature, delay] = len(ppval[ppval<config.significance])

# significant_channels = significant_channels[order]

# # Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
# number_of_ticks = significant_channels.shape[0]
# y, z = np.arange(number_of_ticks), significant_channels

# imph = ax5.pcolormesh(
#     config.times*1e3, # x
#     y, # y
#     z, # z
#     shading='auto',
#     cmap='inferno'
#     )
# ax5.set(
#     xlabel='Tiempo (ms)',
#     # yticks=np.arange(0, average_weights_subjects.shape[2], 1),
#     yticklabels=['' for i in range(average_weights_subjects.shape[2])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600]
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=imph,
#     ax=ax5,
#     orientation='horizontal',
#     shrink=1,
#     label='Número de canales significativos',
#     fraction=.075,
#     aspect=20
#     )

# ax5.text(-.06, 1.1, 'e)', transform=ax5.transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'fonemas_completo_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'fonemas_completo.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # =========================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: MFCCS
# s_channels = False
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Mfccs.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Mfccs/total_weights_per_subject.pkl'
# if s_channels:
#     path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Mfccs_4096_s.pkl'
# else:
#     path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Mfccs_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

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
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)
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
# ax_head_outline = fig.add_axes([.3, 0.84, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.31, 0.842, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.19))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)
# if s_channels:
#     feat_weights = average_weights_subjects[significant_channels_subjects.astype(bool)].mean(axis=0)
# else:
#     feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights.shape[0]),
#     feat_weights,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
#     vmax=np.abs(feat_weights).max()
#     )

# # Set figure configuration
# tags = [r'$M_{{{}}}$'.format(int(i)) for i in np.arange(1, feat_weights.shape[0]+1)]
# ticks = np.arange(feat_weights.shape[0])
# ax2.set(
#     xlabel='Tiempo (ms)',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     ylabel="Coeficientes Mel",
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
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#         im[0],
#         ax=ax3,
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# ax3.axis('off')  # Desactivar ejes
# ax3.set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std()/np.sqrt(config.sr):.3f}'+r')$', fontsize=15)
# ax3.text(-.23, 1.15, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#     im[0],
#     ax=ax4,
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal',
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# ax4.set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std()/np.sqrt(correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)].shape[0]):.3f}'+r')$', fontsize=15)

# ax4.axis('off')  # Desactivar ejes
# ax4.text(-.23, 1.15, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Quinto gráfico en la segunda columna (segunda fila)
# ax5 = plt.subplot(gs[1, 1])

# # Mask and transformation
# pvals_for_graph = pvalue_tfce.copy()
# pvals_for_graph[pvalue_tfce>config.significance] = 1
# pvals_for_graph = -np.log10(pvals_for_graph)

# imph = ax5.pcolormesh(
#     config.times*1e3, # x
#     np.arange(pvals_for_graph.shape[1]), # y
#     pvals_for_graph.T, # z
#     shading='auto',
#     cmap='inferno'
#     )

# ax5.set(
#     xlabel='Tiempo (ms)',
#     yticklabels=['' for i in range(average_weights_subjects.shape[2])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600]
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=imph,
#     ax=ax5,
#     orientation='horizontal',
#     shrink=1,
#     label=r"$-log_{10}(p_{valores})$",
#     fraction=.075,
#     aspect=20
#     )
# ax5.text(-.06, 1.1, 'e)', transform=ax5.transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'mfccs_completo_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'mfccs_completo.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # ===============================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: SPECTROGRAM
# s_channels = False
# path_correlations = 'saves/mtrf_ridge_torch/External/correlations/tmin-0.2_tmax0.6/Theta/Spectrogram.pkl'
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram/total_weights_per_subject.pkl'
# if s_channels:
#     path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram_4096_s.pkl'
# else:
#     path_tfce = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram_4096.pkl'
# _, pvalue_tfce = load_pickle(path=path_tfce)

# correlations = load_pickle(path=path_correlations)
# average_correlation_subjects, significant_channels_subjects = correlations['average_correlation_subjects'], correlations['repeated_good_correlation_channels_subjects']
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

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
# if s_channels:
#     weights = np.empty(average_weights_subjects.shape[1:])
#     filter_ch = significant_channels_subjects.astype(bool)
#     for j in range(128):
#         subset = average_weights_subjects[filter_ch[:, j], j, :, :]  # Esto tiene forma (n_true, n_feats, n_delays)

#         # Si no hay ningún True, podemos decidir asignar NaN o algún valor por defecto
#         if subset.size == 0:
#             weights[j] = np.nan
#         else:
#             # Promediamos sobre el eje 0 (las filas filtradas)
#             weights[j] = subset.mean(axis=0)
#     weights = weights.mean(axis=1)
# else:
#     weights = average_weights_subjects.mean(axis=0).mean(axis=1)
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
# ax_head_outline = fig.add_axes([.3, 0.84, 0.11, 0.11])  # [x, y, width, height]

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
# ax_head = fig.add_axes([.31, 0.842, 0.09, 0.09])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.19))
# ax1.text(-.1, 1.1, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)
# if s_channels:
#     feat_weights = average_weights_subjects[significant_channels_subjects.astype(bool)].mean(axis=0)
# else:
#     feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights.shape[0]),
#     feat_weights,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
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
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'b)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#         im[0],
#         ax=ax3,
#         fraction=.075,
#         aspect=20,
#         # label='Correlación',
#         orientation='horizontal',
#         boundaries=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 100),
#         ticks=np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3)
#         )
# cbar.set_ticklabels(np.linspace(mean_average_correlation.min(), mean_average_correlation.max(), 3).round(decimals=2))

# ax3.axis('off')  # Desactivar ejes
# ax3.set_title(r'Correlación: $('+ f'{mean_average_correlation.mean():.3f}\pm{mean_average_correlation.std()/np.sqrt(config.sr):.3f}'+r')$', fontsize=15)
# ax3.text(-.23, 1.15, 'c)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')

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
# cbar = fig.colorbar(
#     im[0],
#     ax=ax4,
#     fraction=.075,
#     aspect=20,
#     orientation='horizontal',
#     boundaries=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 100),
#     ticks=np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3)
#     )
# cbar.set_ticklabels(np.linspace(absolute_correlation_per_channel.min(), absolute_correlation_per_channel.max(), 3).round(decimals=2))

# ax4.set_title(r'Similaridad: $('+ f'{absolute_correlation_per_channel.mean():.3f}\pm{absolute_correlation_per_channel.std()/np.sqrt(correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)].shape[0]):.3f}'+r')$', fontsize=15)

# ax4.axis('off')  # Desactivar ejes
# ax4.text(-.23, 1.15, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Quinto gráfico en la segunda columna (segunda fila)
# ax5 = plt.subplot(gs[1, 1])

# # Mask and transformation
# pvals_for_graph = pvalue_tfce.copy()
# pvals_for_graph[pvalue_tfce>config.significance] = 1
# pvals_for_graph = -np.log10(pvals_for_graph)

# imph = ax5.pcolormesh(
#     config.times*1e3, # x
#     np.arange(pvals_for_graph.shape[1]), # y
#     pvals_for_graph.T, # z
#     shading='auto',
#     cmap='inferno'
#     )

# ax5.set(
#     xlabel='Tiempo (ms)',
#     yticklabels=['' for i in range(average_weights_subjects.shape[2])],
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600]
#     )

# # Make colorbar
# fig.colorbar(
#     mappable=imph,
#     ax=ax5,
#     orientation='horizontal',
#     shrink=1,
#     label=r"$-log_{10}(p_{valores})$",
#     fraction=.075,
#     aspect=20
#     )

# ax5.text(-.06, 1.1, 'e)', transform=ax5.transAxes, fontsize=18, va='top', ha='right')
# # if s_channels:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'espectrograma_completo_s.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# # else:
# #     fig.savefig(
# #         os.path.join(tesis_path,'resultados', f'espectrograma_completo.{figformat}'),
# #         transparent=False,
# #         dpi=dpi
# #         )
# fig.show()

# # ===============================================================
# # PESOS + TOPOMAPS CORR + SIMILARITY + MATRIZ: THETA: PHONES # TODO TAMBIEN FALTA TFCE EVITANDO LA DIMENSION PROBLEMATICA
# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phones-Discrete-Phonet/total_weights_per_subject.pkl'
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)
# # rightindices = [7, 8, 10, 30, 27, 33, 17, 22, 24, 5, 6, 0, 1, 9, 2, 3, 11, 12, 13, 14, 15, 16, 4, 18, 19, 20, 21, 28, 23, 31, 25, 26, 32, 29]
# # average_weights_subjects = average_weights_subjects[:, :, rightindices,:]

# # Crear una figura
# fig = plt.figure(
#     figsize=(13, 10),
#     tight_layout=True
#     )

# # Definir la cuadrícula usando GridSpec
# # 2 filas y 2 columnas, con la segunda columna dividida en dos partes en la primera fila
# gs = gridspec.GridSpec(
#     nrows=2,
#     ncols=2,
#     width_ratios=[1, 1],
#     height_ratios=[1, 3.5]
#     )

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# ax1 = plt.subplot(gs[0, 0])
# weights = average_weights_subjects.mean(axis=0).mean(axis=1)
# evoked_1 = mne.EvokedArray(data=weights, info=config.info_mne)
# evoked_1.shift_time(config.times[0], relative=True)
# evoked_plot_1 = evoked_1.plot(
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
# for text in evoked_plot_1.axes[0].texts:
#     if "ave" in text.get_text():
#         text.set_visible(False)  # Ocultar el texto
# ax1.plot(
#     config.times*1e3, #ms
#     evoked_1._data.mean(axis=0),
#     'black',
#     label='Valor medio',
#     zorder=130,
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in ax1.get_lines()[:len(evoked_1.ch_names)]]
# ax1.grid(visible=True)
# ax1.set(xlabel='', xticklabels=[], title='EEG (128 canales)')
# ax1.tick_params(axis='x', which='both', labelbottom=False)
# ax1.legend(loc=(.5,.02))
# ax1.text(-.1, 1.2, 'a)', transform=ax1.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax2 = plt.subplot(gs[1, 0], sharex=ax1)

# feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights)
# feat_weights = feat_weights[order]

# im = ax2.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights.shape[0]),
#     feat_weights,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
#     vmax=np.abs(feat_weights).max()
#     )

# # # Set figure configuration
# # ph_labels_phonet =  config.Exp_info().ph_labels_phonet.copy()
# # ph_labels_phonet.remove('sil')
# # ph_labels_phonet.remove('<p:>')
# # tags = []
# # for phone in ph_labels_phonet:
# #     tags.append(config.Exp_info().phones_to_phonemes[phone].replace('/', ''))
# # order_tag, tags_counts = np.unique(tags, return_counts=True)
# # tuples = order_tag[tags_counts==2].tolist()
# # triples = order_tag[tags_counts==3].tolist()
# # quadruples = order_tag[tags_counts==4].tolist()
# # final_tags = []
# # auxiliary_tags = []
# # for l, tag in enumerate(tags):
# #     auxiliary_tags.append(tag)
# #     if tag in tuples+triples+quadruples:
# #         repetitions = np.sum(np.array(auxiliary_tags)==tag)
# #         final_tags.append(tag+r'\textsubscript' + r'{' + f'{repetitions}'+r'}')
# #     else:
# #         final_tags.append(tag)
# # tags = final_tags       
# tags = [r'b\textsubscript{2}', r'd\textsubscript{2}', r'f\textsubscript{2}', r'g\textsubscript{2}', r'n\textsubscript{2}', r's\textsubscript{2}', r'a', r'b\textsubscript{1}', r'd\textsubscript{1}', r'e', \
#         r'f\textsubscript{1}', r'i\textsubscript{1}', r'i\textsubscript{2}', r'x\textsubscript{2}', r'k', r'l', r'm', r'n\textsubscript{1}', r'o', r'p', \
#         r'R', r'r', r's\textsubscript{1}', r't', r'tS\textsubscript{1}', r'u\textsubscript{1}', r'u\textsubscript{2}', r'x\textsubscript{1}', r's\textsubscript{3}', r's\textsubscript{4}', \
#         r'g\textsubscript{1}', r'tS\textsubscript{2}', r'x\textsubscript{3}', r'L']     

# ticks = np.arange(feat_weights.shape[0])
# tags = tags if order is None else [tags[i] for i in order]

# ax2.set(
#     xlabel='Tiempo (ms)',
#     ylabel='Fonos',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     yticks=ticks,
#     # yticklabels=tags,
#     )
# ax2.set_yticklabels(tags, fontsize=14)

# # Configure colorbar
# fig.colorbar(
#     im,
#     ax=ax2,
#     orientation='horizontal',
#     shrink=1,
#     label='Amplitud (U.A)',
#     fraction=.075,
#     aspect=20
#     )
# ax2.text(-.1, 1.1, 'c)', transform=ax2.transAxes, fontsize=18, va='top', ha='right')


# path_mtrfs = 'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phones-Phonet/total_weights_per_subject.pkl'
# average_weights_subjects = load_pickle(path=path_mtrfs)['average_weights_subjects'][:, :, :, :] #(n_sub, n_chans, n_feats, n_delays)

# # Primer gráfico en la primera columna (comparte el eje x con el segundo gráfico)
# ax3 = plt.subplot(gs[0, 1])
# weights = average_weights_subjects.mean(axis=0).mean(axis=1)
# evoked_2 = mne.EvokedArray(data=weights, info=config.info_mne)
# evoked_2.shift_time(config.times[0], relative=True)
# evoked_plot_2 = evoked_2.plot(
#     scalings={'eeg':1},
#     zorder='std',
#     time_unit='ms',
#     show=False,
#     spatial_colors=True,
#     # unit=False,
#     units='mTRFs (U.A)',
#     axes=ax3,
#     gfp=False
#     )

# ax3.plot(
#     config.times*1e3, #ms
#     evoked_2._data.mean(axis=0),
#     'black',
#     label='Valor medio',
#     zorder=130,
#     linewidth=2
#     )

# # Extraer los colores de los canales
# colors = [line.get_color() for line in ax3.get_lines()[:len(evoked_2.ch_names)]]

# # Eliminar el esquema de la cabeza original
# for ax in fig.axes:
#     # Verificar si el eje contiene un objeto de tipo "PathCollection" (los puntos de los canales)
#     for artist in ax.get_children():
#         if isinstance(artist, PathCollection):
#             ax.remove()  # Eliminar el eje que contiene el esquema de la cabeza original
#             break

# # Obtener las posiciones de los sensores en 2D
# montage = evoked_1.info.get_montage()
# pos = montage.get_positions()['ch_pos']  # Diccionario con las posiciones de los canales

# # Crear un eje adicional para la cabecita sin sensores
# ax_head_outline = fig.add_axes([.32, 0.85, 0.09, 0.09])  # [x, y, width, height]

# # Graficar solo el contorno de la cabeza (sin sensores)
# mne.viz.plot_topomap(
#     np.zeros(len(evoked_1.ch_names)),  # Datos ficticios (todos ceros)
#     evoked_1.info,
#     axes=ax_head_outline,
#     show=False,
#     sensors=False,  # No graficar los sensores
#     outlines='head'  # Graficar solo el contorno de la cabeza
# )
# ax_head_outline.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_outline.axis('off')  # Ocultar los ejes

# # Crear un eje adicional para graficar los sensores
# ax_head = fig.add_axes([.327, 0.853, 0.075, 0.075])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked_1.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head.axis('off')  # Ocultar los ejes

# # Obtener las posiciones de los sensores en 2D
# montage = evoked_2.info.get_montage()
# pos = montage.get_positions()['ch_pos']  # Diccionario con las posiciones de los canales

# # Crear un eje adicional para la cabecita sin sensores
# ax_head_outline_2 = fig.add_axes([.86, 0.866, 0.08, 0.08])  # [x, y, width, height]

# # Graficar solo el contorno de la cabeza (sin sensores)
# mne.viz.plot_topomap(
#     np.zeros(len(evoked_2.ch_names)),  # Datos ficticios (todos ceros)
#     evoked_2.info,
#     axes=ax_head_outline_2,
#     show=False,
#     sensors=False,  # No graficar los sensores
#     outlines='head'  # Graficar solo el contorno de la cabeza
# )
# ax_head_outline_2.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_outline_2.axis('off')  # Ocultar los ejes

# # Crear un eje adicional para graficar los sensores
# ax_head_2 = fig.add_axes([.866, 0.868, 0.068, 0.068])  # [x, y, width, height]

# # Convertir las posiciones a un array 2D (x, y)
# pos_2d = np.array([pos[ch][:2] for ch in evoked_2.ch_names])  # Solo tomamos las coordenadas x e y
# ax_head_2.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
# ax_head_2.set_aspect('equal')  # Mantener la proporción de aspecto
# ax_head_2.axis('off')  # Ocultar los ejes

# ax3.grid(visible=True)
# ax3.set(xlabel='', xticklabels=[], ylabel='', title='EEG (128 canales)')
# # ax1.get_yticks()
# ax1.set(ylabel='', yticks=[-0.0005,  0.    ,  0.0005], yticklabels=[-0.0005,  0.    ,  0.0005])
# ax3.set(ylabel='', yticks=[-0.0005,  0.    ,  0.0005])
# ax3.tick_params(axis='x', which='both', labelbottom=False)
# ax3.tick_params(axis='y', labelleft=False)
# ax3.legend(loc=(.57,.1))
# ax3.text(-.1, 1.2, 'b)', transform=ax3.transAxes, fontsize=18, va='top', ha='right')

# # Segundo gráfico en la primera columna (comparte el eje x con el primer gráfico)
# ax4 = plt.subplot(gs[1, 1], sharex=ax3)

# feat_weights = average_weights_subjects.mean(axis=0).mean(axis=0)
# order, null_indexes = clustering_by_correlation(weights=feat_weights)
# feat_weights = feat_weights[order]

# im = ax4.pcolormesh(
#     config.times * 1e3,
#     np.arange(feat_weights.shape[0]),
#     feat_weights,
#     cmap='RdBu_r',
#     shading='auto',
#     vmin=-np.abs(feat_weights).max(),
#     vmax=np.abs(feat_weights).max()
#     )

# # Set figure configuration
# ph_labels_phonet =  config.Exp_info().ph_labels_phonet.copy()
# ph_labels_phonet.remove('sil')
# ph_labels_phonet.remove('<p:>')
# tags = []
# for phone in ph_labels_phonet:
#     tags.append(config.Exp_info().phones_to_phonemes[phone].replace('/', ''))
# order_tag, tags_counts = np.unique(tags, return_counts=True)
# tuples = order_tag[tags_counts==2].tolist()
# triples = order_tag[tags_counts==3].tolist()
# quadruples = order_tag[tags_counts==4].tolist()
# final_tags = []
# auxiliary_tags = []
# for l, tag in enumerate(tags):
#     auxiliary_tags.append(tag)
#     if tag in tuples+triples+quadruples:
#         repetitions = np.sum(np.array(auxiliary_tags)==tag)
#         final_tags.append(tag+r'\textsubscript' + r'{' + f'{repetitions}'+r'}')
#     else:
#         final_tags.append(tag)
# tags = final_tags            

# ticks = np.arange(feat_weights.shape[0])
# tags = tags if order is None else [tags[i] for i in order]

# ax4.set(
#     xlabel='Tiempo (ms)',
#     ylabel='Fonos',
#     xticks=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     xticklabels=[-200, -100, 0, 100, 200, 300, 400, 500, 600],
#     yticks=ticks,
#     # yticklabels=tags,
#     )
# ax4.set_yticklabels(tags, fontsize=14)

# # Configure colorbar
# fig.colorbar(
#     im,
#     ax=ax4,
#     orientation='horizontal',
#     shrink=1,
#     label='Amplitud (U.A)',
#     fraction=.075,
#     aspect=20
#     )
# ax4.text(-.1, 1.1, 'd)', transform=ax4.transAxes, fontsize=18, va='top', ha='right')

# # Eliminar la etiqueta "Nave"
# for txt in fig.findobj(mtext.Text):
#     if "ave" in txt.get_text():
#          txt.remove()
# # fig.savefig(
# #     os.path.join(tesis_path,'resultados', f'fonos_completo.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

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
#     os.path.join(tesis_path,'metodos', f'ejemplo_venn2.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()
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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'ejemplo_venn3.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

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
#                         statistical_test=True,
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
#     figsize=(7, 4),
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
#     os.path.join(tesis_path,'metodos', f'prueba_permutaciones.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()

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
#     figsize=(13,4),
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
#     os.path.join(tesis_path,'metodos', f'validacion.{figformat}'),
#     transparent=False,
#     dpi=dpi
#     )
# fig.show()

# # =============
# # EJEMPLOS TFCE 
# SpectrogramTfcePath = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Spectrogram_4096.pkl'
# PhonemesTfcePath = 'saves/mtrf_ridge_torch/External/TFCE/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/Theta/Phonemes-Phonet_4096.pkl'
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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'ejemplo_TFCE.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'diagrama_matriz_diseño.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'diagrama_matriz_diseño2.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_envolvente_diagrama1.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_envolvente_diagrama2.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_envolvente_diagrama3.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_envolvente_diagrama4.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ======
# # Phones 
# # ======
# fig, axes = plt.subplots(
#     nrows=1, 
#     ncols=2,
#     tight_layout=True,
#     figsize=(14, 8),
#     # sharey='row'
#     )

# # PLLR
# PhonesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phones-Phonet/Sesion21.pkl"
# NumberOfTicks = 34

# phones = load_pickle(path=PhonesPath)[0][:9168]
# WindowLeft, WindowRight = 30,40 #0, len(phones)/config.sr

# time_phones = np.arange(0, len(phones)/config.sr, 1/config.sr)
# window_phones = (WindowLeft <= time_phones) & (time_phones <= WindowRight)

# ph_labels_phonet =  config.Exp_info().ph_labels_phonet.copy()
# ph_labels_phonet.remove('sil')
# ph_labels_phonet.remove('<p:>')

# tags = []
# for phone in ph_labels_phonet:
#     tags.append(config.Exp_info().phones_to_phonemes[phone].replace('/', ''))

# order_tag, tags_counts = np.unique(tags, return_counts=True)
# tuples = order_tag[tags_counts==2].tolist()
# triples = order_tag[tags_counts==3].tolist()
# quadruples = order_tag[tags_counts==4].tolist()

# final_tags = []
# auxiliary_tags = []
# for l, tag in enumerate(tags):
#     auxiliary_tags.append(tag)
#     if tag in tuples+triples+quadruples:
#         repetitions = np.sum(np.array(auxiliary_tags)==tag)
#         final_tags.append(tag+r'\textsubscript' + r'{' + f'{repetitions}'+r'}')
#     else:
#         final_tags.append(tag)
# tags=final_tags            
# ticks = np.arange(0, NumberOfTicks, 1)+.5

# norm = TwoSlopeNorm(vmin=phones.min(), vcenter=0, vmax=phones.max())
# im = axes[1].imshow(
#     phones.T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu_r',  # Ajusta el mapa de colores
#     norm=norm
#     )
# cbar = fig.colorbar(
#     im,
#     label='Probability Loglikelihood Ratio',
#     ax=axes[1]
#     )

# axes[1].set_yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# axes[1].set_xlabel('Tiempo (s)')

# # ===============
# # Phones-Discrete
# PhonesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phones-Discrete-Phonet/Sesion21.pkl"
# NumberOfTicks = 34

# phones = load_pickle(path=PhonesPath)[0][:9168]
# WindowLeft, WindowRight = 30,40 #0, len(phones)/config.sr

# time_phones = np.arange(0, len(phones)/config.sr, 1/config.sr)
# window_phones = (WindowLeft <= time_phones) & (time_phones <= WindowRight)

# rightindices = [7, 8, 10, 30, 27, 33, 17, 22, 24, 5, 6, 0, 1, 9, 2, 3, 11, 12, 13, 14, 15, 16, 4, 18, 19, 20, 21, 28, 23, 31, 25, 26, 32, 29]
# phones = phones[:,rightindices]
# # tags_discr = [r'b\textsubscript{2}', r'd\textsubscript{2}', r'f\textsubscript{2}', r'g\textsubscript{2}', r'n\textsubscript{2}', r's\textsubscript{2}', r'a', r'b\textsubscript{1}', r'd\textsubscript{1}', r'e', \
# #         r'f\textsubscript{1}', r'i\textsubscript{1}', r'i\textsubscript{2}', r'x\textsubscript{2}', r'k', r'l', r'm', r'n\textsubscript{1}', r'o', r'p', \
# #         r'R', r'r', r's\textsubscript{1}', r't', r'tS\textsubscript{1}', r'u\textsubscript{1}', r'u\textsubscript{2}', r'x\textsubscript{1}', r's\textsubscript{3}', r's\textsubscript{4}', \
# #         r'g\textsubscript{1}', r'tS\textsubscript{2}', r'x\textsubscript{3}', r'L']
# # rightindices=[]
# # for el in tags:
# #     rightindices.append(tags_discr.index(el))

# ticks = np.arange(0, NumberOfTicks, 1)+.5

# norm = TwoSlopeNorm(vmin=phones.min(), vcenter=0.5, vmax=phones.max())
# im = axes[0].imshow(
#     phones[window_phones].T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap=ListedColormap(["white", "gray"]),  # Ajusta el mapa de colores
#     vmin=phones.min(),
#     vmax=phones.max()
#     )
# cbar = fig.colorbar(
#     im,
#     label='Ocurrencias',
#     ax=axes[0]
#     )
# ticks_b = [0, 1]#phonological.min(), phonological.max()
# cbar.set_ticks(ticks_b)

# axes[0].set_yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# axes[0].set_xlabel('Tiempo (s)')
# axes[0].set_ylabel('Fonos')
# fig.text(0.02, 1, 'a)', fontsize=18, va='top', ha='right')
# fig.text(0.5, 1, 'b)', fontsize=18, va='top', ha='right')
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_phones.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ========
# # Phonemes
# # ========
# fig, axes = plt.subplots(
#     nrows=1, 
#     ncols=2,
#     tight_layout=True,
#     figsize=(14, 6),
#     # sharey='row'
#     )
# # =====
# # PLLRS
# PhonemesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonemes-Phonet/Sesion21.pkl"
# NumberOfTicks = 21
# phonemes = load_pickle(path=PhonemesPath)[0][:9168]
# WindowLeft, WindowRight = 30, 40 #0, len(phonemes)/config.sr

# time_phonemes = np.arange(0, len(phonemes)/config.sr, 1/config.sr)
# window_phonemes = (WindowLeft <= time_phonemes) & (time_phonemes <= WindowRight)

# tags = config.Exp_info().phonemes_phonet.copy()
# tags.remove('/sil/')

# ticks = np.arange(0, NumberOfTicks, 1)+.5
# norm = TwoSlopeNorm(vmin=phonemes.min(), vcenter=0, vmax=phonemes.max())
# im = axes[1].imshow(
#     phonemes.T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap='RdBu_r',  # Ajusta el mapa de colores
#     norm=norm
#     )
# cbar = fig.colorbar(
#     im,
#     label='Probability Loglikelihood Ratio',
#     ax=axes[1]
#     )

# axes[1].set_yticks(
#     ticks=ticks, 
#     labels=tags
#     )
# axes[1].set_xlabel('Tiempo (s)')
# axes[1].set_xlim(WindowLeft, WindowRight)

# # ========
# # DISCRETE
# PhonemesPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonemes-Discrete-Phonet/Sesion21.pkl"
# NumberOfTicks = 21

# phonemes = load_pickle(path=PhonemesPath)[0][:9168]
# WindowLeft, WindowRight = 30, 40 #3, len(phonemes)/config.sr

# time_phonemes = np.arange(0, len(phonemes)/config.sr, 1/config.sr)
# window_phonemes = (WindowLeft <= time_phonemes) & (time_phonemes <= WindowRight)

# ticks = np.arange(0, NumberOfTicks, 1)+.5

# im = axes[0].imshow(
#     phonemes[window_phonemes].T,
#     aspect='auto',  # Ajusta el aspecto
#     extent=[WindowLeft, WindowRight, 0, NumberOfTicks],  # Ajusta los límites de los ejes
#     origin='lower',  # Ajusta el origen
#     cmap=ListedColormap(["white", "gray"]),  # Ajusta el mapa de colores
#     vmin=phonemes.min(),
#     vmax=phonemes.max()
#     )
# cbar = fig.colorbar(
#     im,
#     label='Ocurrencias',
#     ax=axes[0]
#     )
# ticks_b = [0, 1]#phonological.min(), phonological.max()
# cbar.set_ticks(ticks_b)
# axes[0].set_yticks(
#     ticks=ticks, 
#     labels=tags
#     )

# axes[0].set_xlabel('Tiempo (s)')
# axes[0].set_ylabel('Fonemas')  
# fig.text(0.02, 1, 'a)', fontsize=18, va='top', ha='right')
# fig.text(0.5, 1, 'b)', fontsize=18, va='top', ha='right')
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_phonemes.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ============
# # Phonological
# PhonologicalPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Phonological/Sesion21.pkl"
# NumberOfTicks = 16

# phonological = load_pickle(path=PhonologicalPath)[0][:9168]
# WindowLeft, WindowRight = 30, 40 #0, len(phonological)/config.sr

# time_phonological = np.arange(0, len(phonological)/config.sr, 1/config.sr)
# window_phonological = (WindowLeft <= time_phonological) & (time_phonological <= WindowRight)

# tags = list(config.Exp_info().phonological_labels.keys())
# tags.remove('trill')
# tags.remove('pause')
# ticks = np.arange(0, NumberOfTicks, 1)

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(8, 6)
#     )
# # phonological.T[17] = np.zeros(9168)
# # phonological.T[11] = np.zeros(9168)
# # phonological = phonological[:, [i for i in np.arange(18) if i not in [11, 17]]]

# norm = TwoSlopeNorm(vmin=phonological.min(), vcenter=0, vmax=phonological.max())
# im = plt.pcolormesh(
#     time_phonological,
#     np.arange(16),
#     phonological.T,
#     cmap='RdBu_r',#LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     norm=norm
#     )

# cbar = plt.colorbar(
#     im,
#     label='Probability Loglikelihood Ratio'
#     )
# # ticks_b = [-8, -4, -2, 0, 2, 4, 6, 8]#phonological.min(), phonological.max()
# # cbar.set_ticks(ticks_b)
# plt.yticks(
#     ticks=ticks,
#     labels=tags
#     )
# plt.xlim(WindowLeft, WindowRight)
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Características fonológicas')
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_phonological.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # =====
# # Mfccs
# MfccsPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Mfccs/Sesion21.pkl"
# NumberOfTicks = 16

# mfccs = load_pickle(path=MfccsPath)[0][:9168]
# WindowLeft, WindowRight = 30, 40#0, len(mfccs)/config.sr

# time_mfccs = np.arange(0, len(mfccs)/config.sr, 1/config.sr)
# window_mfccs = (WindowLeft <= time_mfccs) & (time_mfccs <= WindowRight)

# tags = [f'M{i}' for i in np.arange(1, NumberOfTicks, 2)]
# ticks = np.arange(0, NumberOfTicks, 2)

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )
# norm = TwoSlopeNorm(vmin=mfccs.min(), vcenter=0, vmax=mfccs.max())
# im = plt.pcolormesh(
#     time_mfccs,
#     np.arange(16),
#     mfccs.T,
#     cmap='RdBu_r',#LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     norm=norm
#     )

# cbar = plt.colorbar(
#     im,
#     label='Amplitud (U.A)'
#     )
# ticks_b = [-40, -20, 0, 40, 80, 120, 180]#mfccs.min(), mfccs.max()
# cbar.set_ticks(ticks_b)

# plt.yticks(
#     ticks=ticks,
#     labels=tags
#     )
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Coeficientes Mel')
# plt.xlim(WindowLeft, WindowRight)
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_mfccs.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # =============
# # Espectrograma
# SpectrogramPath = "saves/preprocessed_data/External/tmin-0.2_tmax0.6/Spectrogram/Sesion21.pkl"
# NumberOfTicks = 16

# spectrogram = load_pickle(path=SpectrogramPath)[0][:9168]
# WindowLeft, WindowRight = 30, 40 #0, len(spectrogram)/config.sr

# time_spectrogram = np.arange(0, len(spectrogram)/config.sr, 1/config.sr)
# window_spectrogram = (WindowLeft <= time_spectrogram) & (time_spectrogram <= WindowRight)

# bands_center = librosa.mel_frequencies(n_mels=16+2, fmin=0, fmax=8000)[1:-1]

# # tags = [int(bands_center[i]) for i in np.arange(1, len(bands_center)+1, 2)]
# # ticks = np.arange(0, NumberOfTicks, 2)+.5
# tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center))]
# ticks = np.arange(0, NumberOfTicks)

# fig = plt.figure(
#     tight_layout=True,
#     figsize=(6, 5)
#     )

# im = plt.pcolormesh(
#     time_spectrogram,
#     np.arange(16),
#     spectrogram.T,
#     cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", "gray"]),  # Ajusta el mapa de colores
#     shading='auto',
#     vmin=spectrogram.min(),
#     vmax=spectrogram.max()
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
# plt.xlim(WindowLeft, WindowRight)

# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_espectrograma.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
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
# # fig.savefig(
# #     os.path.join(tesis_path,'metodos', f'sample_tono.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# # ===============================
# # Envlovente de la señal de audio
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
# #     os.path.join(tesis_path,'metodos', f'sample_envolvente.{figformat}'),
# #     transparent=False,
# #     dpi=dpi
# #     )
# fig.show()

# =======================================================
# Ejemplo EEG y PSD (power spectral density) de un sujeto# TODO SIGUE SIN DAR CHARLAR CON JOACO
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
# raw.plot(
#     scalings=dict(eeg=2e-5)
# )

from processing import subsample



psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(
    raw._data,
    sfreq=1024,#raw.info.get("sfreq"),
    fmin=1,
    fmax=60,
    n_fft=2048,
    n_per_seg=2048*16,
    n_ovelap=32
    )

fig, ax = plt.subplots(figsize=(10,5))
evoked = mne.EvokedArray(psds_welch_mean, config.info_mne)
# evoked.times = freqs_mean

evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s',
            show=False, spatial_colors=True, unit=False, units='w', axes=ax)
ax.set_xlabel('Frequency [Hz]')
ax.grid()
fig.show()



raw = mne.io.read_raw_eeglab(
        RawEegPath,
        preload=True,
        verbose='CRITICAL',
        )

fmin, fmax = 1, 15

montage = mne.channels.make_standard_montage('biosemi128')
info = mne.create_info(ch_names=montage.ch_names[:], sfreq=1024, ch_types='eeg').set_montage(montage)
raw = mne.io.RawArray(raw._data, info)

spectrum = raw.compute_psd(
    method='welch',
    fmin=fmin,
    fmax=fmax,
    # n_fft=2048,#048,
    # n_per_seg=2048*16,
    # n_ovelap=32
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



# fig, ax = plt.subplots()
# eeg = raw.get_data().T*1e6  # paso a array y tiro la primer columna de tiempo
# # eeg = subsample(x=eeg, step=int(raw.info.get("sfreq")/ 128))
# psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(
#         eeg.T,
#         raw.info.get("sfreq"),
#         fmin,
#         fmax
#         )



# evoked = mne.EvokedArray(psds_welch_mean, config.info_mne)
# evoked.times = freqs_mean
# evoked.plot(
#         scalings=dict(eeg=1, grad=1, mag=1),
#         zorder='std',
#         time_unit='s',
#         show=False,
#         spatial_colors=True,
#         unit=False,
#         units='w',
#         axes=ax
#         )
# ax.set_xlabel('Frequency [Hz]')
# ax.grid()
# fig.show()

# eeg = load_pickle(path=EegPath)[0]
# raw = mne.io.RawArray(data=eeg.T*1e-6, info=raw_raw.info)
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
#     #     transparent=False,
#     #     dpi=dpi
#     #     )
#     fig.show()


# # ==============
# # STACK DE PESOS
# stimuli = [
#     'Envelope',
#     'Pitch-Log-Raw',
#     'Spectrogram',
#     'Mfccs',
#     'Phonemes-Discrete-Phonet',
#     'Phonological'
#     ]
# bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']

# fig, axes = plt.subplots(
#     nrows=len(stimuli),
#     ncols=len(bands),
#     figsize=(len(bands)*1.5, len(stimuli)*1.5),
#     tight_layout=True,
#     sharex=True,
#     sharey='row'
#     )

# for j, band in enumerate(bands):
#     for i, stimulus in enumerate(stimuli):
#         path_mtrfs = f'saves/mtrf_ridge_torch/External/weights/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/{band}/{stimulus}/total_weights_per_subject.pkl'
#         weights = load_pickle(
#             path=path_mtrfs
#             )['average_weights_subjects'].mean(axis=0).mean(axis=1)
#         evoked = mne.EvokedArray(data=weights, info=config.info_mne)
#         evoked.shift_time(config.times[0], relative=True)
#         axes[i, j].plot(
#             config.times*1e3,
#             evoked._data.mean(axis=0),
#             'black',
#             label='Valor medio',
#             zorder=130,
#             linewidth=2
#             )
#         axes[i, j].grid(visible=True)
#         if band=='Theta':
#             axes[i, j].set_ylim(evoked._data.mean(axis=0).min(), evoked._data.mean(axis=0).max())

# for ax, col in zip(axes[:,0], stimuli):
#     if col=='Phonemes-Discrete-Phonet':
#         col = 'Fonemas'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Pitch-Log-Raw':
#         col = 'Tono de voz'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Envelope':
#         col = 'Envolvente'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Phonological':
#         col = 'C. Fonológicas'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Spectrogram':
#         col = 'Espectrograma'
#         ax.set_ylabel(col, rotation=90)
#     elif col=='Mfccs':
#         col = 'Coef. Mel'
#         ax.set_ylabel(col, rotation=90)

# for ax, band in zip(axes[0], bands):
#     ax.set_title(band)

# for ax in axes[-1]:
#     ax.set_xlabel('Tiempo (ms)')

# fig.show()
