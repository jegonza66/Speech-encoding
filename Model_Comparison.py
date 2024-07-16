# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn, copy

# Specific libraries
# from statsmodels.stats.multitest import fdrcorrection
from matplotlib_venn import venn3, venn2  # venn3_circles
from scipy.spatial import ConvexHull
from scipy.stats import wilcoxon

# Modules
from funciones import load_pickle, cohen_d, all_possible_combinations

# Default size is 10 pts, the scalings (10pts*scale) are:
#'xx-small':0.579,'x-small':0.694,'small':0.833,'medium':1.0,'large':1.200,'x-large':1.440,'xx-large':1.728,None:1.0}
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'legend.title_fontsize': 'x-large',
          'figure.figsize': (8, 6),
          'figure.titlesize': 'xx-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Model parametrs
model ='mtrf'
situation = 'Escucha'
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'
tmin, tmax = -.2, .6

# Relevant paths
path_figures = os.path.normpath(f'figures/{model}/model_comparison/{situation}/{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')
final_corr_path = os.path.normpath(f'saves/{model}/{situation}/correlations/tmin{tmin}_tmax{tmax}/')
preprocesed_data_path = os.path.normpath(f'saves/preprocessed_data/tmin-0.2_tmax0.6/') 

# Code parameters
save_figures = True
# ================================================================================
# CONVEX HULL:  it contrasts the correlation of each part against the joint model.
# ================================================================================
path_convex_hull = os.path.join(path_figures,'convex_hull')

# Relevant parameters
bands = ['Theta']
stims = 'Phonological_Deltas_Spectrogram'
stims = '_'.join(sorted(stims.split('_'))) 
substims = stims.split('_')
avg_corr, good_ch = {}, {}

# Iterate over bands
for band in bands:
    # Fill dictionaries with data
    for stim in substims + [stims]:
        data = load_pickle(path=os.path.join(final_corr_path, f'{stim}_EEG_{band}.pkl'))
        good_ch[stim] = data['repeated_good_correlation_channels_subjects'].ravel()
        avg_corr[stim] = data['average_correlation_subjects'].ravel()
        
    # Make plot
    plt.ioff()
    plt.figure(layout='tight')
    plt.title(band)

    for i, stim in enumerate(substims):
        # Identify Hull (la cáscara de los datos, i.e, el borde)
        stim_points = np.array([avg_corr[stims], avg_corr[stim]]).transpose()
        stim_hull = ConvexHull(stim_points)

        # Plot good channels for each stim
        plt.plot(avg_corr[stims][good_ch[stim] != 0], 
                 avg_corr[stim][good_ch[stim] != 0], 
                 '.', 
                 color=f'C{i}',
                 label=stim, 
                 ms=2.5)

        # Plot bad channels for each stim
        plt.plot(avg_corr[stims][good_ch[stim] == 0], 
                 avg_corr[stim][good_ch[stim] == 0], 
                 '.', 
                 color='grey',
                alpha=0.5, 
                label='Failed permutation test', 
                markersize=2)
        plt.fill(stim_points[stim_hull.vertices, 0], 
                 stim_points[stim_hull.vertices, 1], 
                 color=f'C{i}',
                 alpha=0.3, 
                 linewidth=0)

    # Get limits
    xlimit, ylimit = plt.xlim(), plt.ylim()
    plt.plot([xlimit[0], 0.7], [xlimit[0], 0.7], 'k--', zorder=0)
    plt.hlines(0, xlimit[0], xlimit[1], color='grey', linestyle='dashed')
    plt.vlines(0, ylimit[0], ylimit[1], color='grey', linestyle='dashed')
    
    plt.ylabel('Individual model (r)')
    plt.xlabel('Full model (r)')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), markerscale=3)

    # Save
    if save_figures:
        temp_path = os.path.join(path_convex_hull, f'{band}')
        os.makedirs(temp_path, exist_ok=True)
        plt.savefig(os.path.join(temp_path, f'{stims}.png'))
        plt.savefig(os.path.join(temp_path, f'{stims}.svg'))
    plt.close()

# =========================================================================================================
# VENN DIAGRAMS: it makes diagrams explaining correlation of each part of a shared model (upto 3 features).
# =========================================================================================================
path_venn_diagrams = os.path.join(path_figures,'venn_diagrams')

# Relevant parameters
bands = ['Theta']
stimuli = ['Spectrogram', 'Phonological']

# Arrange shared stimuli
stimuli = sorted(stimuli)
double_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==2]
triple_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==3]
all_stimuli = stimuli + double_combinations + triple_combinations
mean_correlations = {}

# Iterate over bands
for band in bands:
    for stim in all_stimuli:
        # Get average correlation of each stimulus
        mean_correlations[stim] = load_pickle(path=os.path.join(final_corr_path, f'{stim}_EEG_{band}.pkl'))['average_correlation_subjects'].mean()
        # HACER SOBRE LOS CANALES BUENOS y/O LOS MEJORES PUTNTUADOS r2 VER COMO DEFINIR 12 MEJORES CANALEs (EN EL CODIGO ESTA POR ALGUN LADO)

    for stim12 in double_combinations:
        stim1, stim2 = stim12.split('_')

        # Get squared correlation of stimuli
        rsquared_1 = mean_correlations[stim1] ** 2
        rsquared_2 = mean_correlations[stim2] ** 2
        rsquared_12 = mean_correlations[stim12] ** 2

        # Explained by one and two, but not by shared model
        rsquared_complement_12 = rsquared_1 + rsquared_2 - rsquared_12 #11

        # Shared without each stimulus
        rsquared_shared_with_1 = rsquared_12 - rsquared_2 #10
        rsquared_shared_with_2 = rsquared_12 - rsquared_1 #01
        
        # Get list with areas
        areas = [rsquared_shared_with_1, rsquared_shared_with_2, rsquared_complement_12] # note that the sum gives shared model
        areas = [0 if area<0 else area.round(3) for area in areas]
        
        # Create figure and title
        title = stim12.replace('_', ' and ')
        plt.ioff()
        plt.figure(layout='tight')
        plt.title(f'Diagram {band} band - {title}')

        # Make plot
        venn2(subsets=areas, # left area diagran, right area diagram, shared area <--> (10, 01, 11)
              set_labels=(stim1, stim2),
              set_colors=('C0', 'C1'), 
              alpha=0.45)

        # Save figure
        if save_figures:
            temp_path = os.path.join(path_venn_diagrams, f'{band}')
            os.makedirs(temp_path, exist_ok=True)
            plt.savefig(os.path.join(temp_path, f'{stim12}.png'))
            plt.savefig(os.path.join(temp_path, f'{stim12}.svg'))
        plt.close()
        
        # Make a print with information
        stim1_percent = (rsquared_shared_with_1 * 100 /np.sum(areas)).round(2)
        stim2_percent = (rsquared_shared_with_2 * 100 /np.sum(areas)).round(2)
        shared_percent = (rsquared_complement_12 * 100 /np.sum(areas)).round(2)
        print(f'\nPercentage explained by {stim1} is {stim1_percent} %',
              f'\nPercentage explained by {stim2} is {stim2_percent} %',
              f'\nShared percentage explained is {shared_percent} %')
        #TODO Y la variancia no explicada?

    if triple_combinations:
        # Get squared correlation of stimuli
        rsquared_1 = mean_correlations[all_stimuli[0]]**2
        rsquared_2 = mean_correlations[all_stimuli[1]]**2
        rsquared_3 = mean_correlations[all_stimuli[2]]**2
        rsquared_12 = mean_correlations[all_stimuli[3]]**2
        rsquared_13 = mean_correlations[all_stimuli[4]]**2
        rsquared_23 = mean_correlations[all_stimuli[5]]**2
        rsquared_123 = mean_correlations[all_stimuli[6]]**2

        # Shared without each stimulus
        rsquared_shared_with_1 = rsquared_123 - rsquared_23 #100
        rsquared_shared_with_2 = rsquared_123 - rsquared_13 #010
        rsquared_shared_with_3 = rsquared_123 - rsquared_12 #001
        
        # Explained by subshared, but not by all shared model
        rsquared_shared_with_12 = rsquared_13 + rsquared_23 - rsquared_3 - rsquared_123 #110
        rsquared_shared_with_13 = rsquared_12 + rsquared_23 - rsquared_2 - rsquared_123 #101
        rsquared_shared_with_23 = rsquared_12 + rsquared_13 - rsquared_1 - rsquared_123 #011

        # Explained by one, two, three and full shared model but not by subshared models
        rsquared_int_complement_submodels = rsquared_123 + rsquared_1 + rsquared_2 + rsquared_3 - rsquared_12 - rsquared_13 - rsquared_23 #111

        areas = [rsquared_shared_with_1, rsquared_shared_with_2, rsquared_shared_with_3, \
                 rsquared_shared_with_12, rsquared_shared_with_13, rsquared_shared_with_23,\
                 rsquared_int_complement_submodels] 
        areas = [0 if area<0 else area.round(3) for area in areas] # note that the sum gives shared model rsquared_123

        # Create figure and title
        title = all_stimuli[-1].replace('_', ', ')
        plt.ioff()
        plt.figure(layout='tight')
        plt.title(f'Diagram {band} band - {title}')

        # Make plot
        venn3(subsets=areas, # left area diagran, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
              set_labels=(all_stimuli[0], all_stimuli[1], all_stimuli[2]), 
              set_colors=('C0', 'C1', 'purple'), 
              alpha=0.45)

        if save_figures:
            temp_path = os.path.join(path_venn_diagrams, f'{band}')
            os.makedirs(temp_path, exist_ok=True)
            plt.savefig(os.path.join(temp_path, f'{all_stimuli[-1]}.png'))
            plt.savefig(os.path.join(temp_path, f'{all_stimuli[-1]}.svg'))
        plt.close()

        # # Make a print with information
        # stim1_percent = (rsquared_shared_with_1 * 100 /np.sum(areas)).round(2)
        # stim2_percent = (rsquared_shared_with_2 * 100 /np.sum(areas)).round(2)
        # stim3_percent = (rsquared_shared_with_2 * 100 /np.sum(areas)).round(2)
        # shared_percent = (rsquared_complement_12 * 100 /np.sum(areas)).round(2)
        # print(f'\nExclusive Percentage explained by {stim1} is {stim1_percent} %',
        #       f'\nExclusive Percentage explained by {stim2} is {stim2_percent} %',
        #       f'\nshared percentage explained is {shared_percent} %')


# # ===================
# # VIOLIN/TOPO (BANDS)
# path_violin = os.path.join(path_figures,'violin_topo')

# # Relevant parameters
# colors_violin = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta1': 'darkgrey'}
# info = load_pickle(path=os.path.join(preprocesed_data_path,'/EEG/info.pkl'))
# bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']
# stim = 'Spectrogram'
# avg_corr = {}


# # Make figure
# fig, axs = plt.subplots(figsize=(14, 5), ncols=5, nrows=2, layout='tight')

# # Iterate over bands
# for i, band in enumerate(bands):
#     data = load_pickle(path=os.path.join(final_corr_path, f'{stim}_EEG_{band}.pkl'))
#     avg_corr[band] = data['average_correlation_subjects'].mean(axis=0) # TODO CHECK mean
    
#     # Make topomap
#     im = mne.viz.plot_topomap(avg_corr[band].ravel(), 
#                               info, 
#                               axes=axs[0, i], 
#                               show=False, 
#                               sphere=0.07, 
#                               cmap='Reds',
#                               vmin=avg_corr[band].min(), 
#                               vmax=avg_corr[band].max())
#     # And colorbar
#     cbar = plt.colorbar(im[0], ax=axs[0, i], orientation='vertical', shrink=0.5)

# for ax_row in axs[1:]:
#     for ax in ax_row:
#         ax.remove()


# plt.suptitle(situation)#TODO PONER CON FIG ARRIBA CREO
# ax = fig.add_subplot(2, 1, (2, 3))
# sn.violinplot(data=pd.DataFrame(avg_corr), palette=colors_violin, ax=ax)

# ax.set(ylabel='Correlation', 
#        xticklabels=['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'],
#        ylim=[0, 0.5])
# ax.grid(visible=True)

# if save_figures:
#     os.makedirs(path_violin, exist_ok=True)
#     plt.savefig(os.path.join(path_violin + f'{stim}.png'))
#     plt.savefig(os.path.join(path_violin + f'{stim}.svg'))

# # ===================
# # VIOLIN/MTRF (BANDS)

# info_path = 'saves/preprocessed_data/tmin-0.6_tmax-0.003/EEG/info.pkl'
# f = open(info_path, 'rb')
# info = pickle.load(f)
# f.close()

# tmin_corr, tmax_corr = -0.6, -0.003
# tmin_w, tmax_w = -0.6, 0.2
# delays = - np.arange(np.floor(tmin_w * info['sfreq']), np.ceil(tmax_w * info['sfreq']), dtype=int)
# times = np.linspace(delays[0] * np.sign(tmin_w) * 1 / info['sfreq'], np.abs(delays[-1]) * np.sign(tmax_w) * 1 / info['sfreq'], len(delays))
# times = np.flip(-times)

# model = 'Ridge'
# situacion = 'Ambos_Habla'

# Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin mTRF/'.format(model, situacion, tmin_corr, tmax_corr)
# Save_fig = True
# Correlaciones = {}
# mTRFs = {}

# stim = 'Spectrogram'
# Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']

# for Band in Bands:
#     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin_corr, tmax_corr, stim, Band), 'rb')
#     Corr, Pass = pickle.load(f)
#     f.close()
#     Correlaciones[Band] = Corr.mean(0)

# f = open('saves/{}/{}/Original/Stims_Normalize_EEG_Standarize/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.format(model, situacion, tmin_w, tmax_w, stim, 'Theta', stim, 'Theta'), 'rb')
# mTRFs_Theta = pickle.load(f)
# f.close()

# my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta1': 'darkgrey'}

# fontsize = 17
# plt.rcParams.update({'font.size': fontsize})
# fig, axs = plt.subplots(figsize=(8, 5), nrows=2, gridspec_kw={'height_ratios': [1, 1]})

# plt.suptitle(situacion)
# sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal, ax=axs[0])
# axs[0].set_ylabel('Correlation')
# axs[0].set_ylim([-0.1, 0.5])
# axs[0].grid()
# axs[0].set_xticklabels(['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'])


# spectrogram_weights_chanels = mTRFs_Theta.reshape(info['nchan'], 16, len(times)).mean(1)

# # Adapt for ERP
# spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

# evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
# evoked.times = times
# evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
#             show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[1])
# # axs[1].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
# axs[1].set_ylim([-0.016, 0.013])
# if times[0] < 0:
#     # ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')
#     axs[1].axvline(x=0, ymin=0, ymax=1, color='grey')

# # axs[1].grid()

# fig.tight_layout()
# plt.show()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + '{}.png'.format(stim))
#     plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

# # =======================
# # WILCOXON TEST TOPOPLOTS
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import os
# from scipy.stats import wilcoxon
# import mne
# # from statsmodels.stats.multitest import fdrcorrection

# model = 'Ridge'
# Band = 'Theta'
# stim = 'Spectrogram'
# situaciones = ['Escucha', 'Habla_Propia', 'Ambos', 'Ambos_Habla', 'Silencio']
# tmin, tmax = -0.6, -0.003
# stat_test = 'log'  # fdr/log/cohen
# mask = False
# Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/{}'.format(Band, stim, tmin, tmax, stat_test)
# if mask and stat_test != 'cohen':
#     Run_graficos_path += '_mask/'
# else:
#     Run_graficos_path += '/'

# Save_fig = True
# Display_fig = False
# if Display_fig:
#     plt.ion()
# else:
#     plt.ioff()

# montage = mne.channels.make_standard_montage('biosemi128')
# channel_names = montage.ch_names
# info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# Correlaciones = {}

# for situacion in situaciones:
#     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
#     Corr, Pass = pickle.load(f)
#     f.close()

#     Correlaciones[situacion] = Corr.transpose()

# # Calculate wilcoxon test over rows of the dictionary
# stats = {}
# pvals = {}

# for sit1, sit2 in zip(('Escucha', 'Escucha', 'Escucha', 'Escucha', 'Habla_Propia', 'Habla_Propia', 'Ambos', 'Ambos', 'Ambos', 'Ambos_Habla'),
#                       ('Habla_Propia', 'Ambos', 'Silencio', 'Ambos_Habla', 'Silencio', 'Ambos_Habla', 'Ambos_Habla', 'Silencio', 'Habla_Propia', 'Silencio')):
#     print(sit1, sit2)

#     dist1 = Correlaciones[sit1]
#     dist2 = Correlaciones[sit2]

#     stat = []
#     pval = []
#     for i in range(len(dist1)):
#         res = wilcoxon(dist1[i], dist2[i])
#         stat.append(res[0])
#         pval.append(res[1])

#     stats[f'{sit1}-{sit2}'] = stat
#     pvals[f'{sit1}-{sit2}'] = pval

#     if stat_test == 'fdr':
#         # passed, corrected_pval = fdrcorrection(pvals=pval, alpha=0.05, method='p')

#         if mask:
#             corrected_pval[~passed] = 1

#         # Plot pvalue
#         fig, ax = plt.subplots()
#         fig.suptitle(f'FDR p-value: {sit1}-{sit2}\n'
#                      f'mean: {round(np.mean(corrected_pval), 6)} - '
#                      f'min: {round(np.min(corrected_pval), 6)} - '
#                      f'max: {round(np.max(corrected_pval), 6)}\n'
#                      f'passed: {sum(passed)}', fontsize=17)
#         im = mne.viz.plot_topomap(corrected_pval, vmin=0, vmax=1, pos=info, axes=ax, show=Display_fig, sphere=0.07,
#                                   cmap='Reds_r')
#         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
#         cbar.ax.yaxis.set_tick_params(labelsize=17)
#         cbar.ax.set_ylabel(ylabel='FDR corrected p-value', fontsize=17)

#         fig.tight_layout()

#         if Save_fig:
#             os.makedirs(Run_graficos_path, exist_ok=True)
#             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
#             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

#     elif stat_test == 'log':
#         log_pval = np.log10(pval)
#         if mask:
#             log_pval[np.array(pval) > 0.05/128] = 0

#         # Plot pvalue
#         fig, ax = plt.subplots()
#         fig.suptitle(f'p-value: {sit1}-{sit2}\n'
#                      f'mean: {round(np.mean(pvals[f"{sit1}-{sit2}"]), 6)} - '
#                      f'min: {round(np.min(pvals[f"{sit1}-{sit2}"]), 6)} - '
#                      f'max: {round(np.max(pvals[f"{sit1}-{sit2}"]), 6)}\n'
#                      f'passed: {sum(np.array(pval)< 0.05/128)}', fontsize=17)
#         im = mne.viz.plot_topomap(log_pval, vmin=-6, vmax=-2, pos=info, axes=ax, show=Display_fig, sphere=0.07, cmap='Reds_r')
#         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85, ticks=[-6, -5, -4, -3, -2])
#         cbar.ax.yaxis.set_tick_params(labelsize=17)
#         cbar.ax.set_yticklabels(['<10-6', '10-5', '10-4', '10-3', '10-2'])
#         cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)


#         fig.tight_layout()

#         if Save_fig:
#             os.makedirs(Run_graficos_path, exist_ok=True)
#             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
#             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

#         # Plot statistic
#         fig, ax = plt.subplots()
#         fig.suptitle(f'stat: {sit1}-{sit2}\n'
#                      f'Mean: {round(np.mean(stats[f"{sit1}-{sit2}"]), 3)}', fontsize=19)
#         im = mne.viz.plot_topomap(stats[f'{sit1}-{sit2}'], info, axes=ax, show=Display_fig, sphere=0.07, cmap='Reds')
#         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
#         cbar.ax.yaxis.set_tick_params(labelsize=17)
#         cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)
#         fig.tight_layout()

#         if Save_fig:
#             os.makedirs(Run_graficos_path, exist_ok=True)
#             plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.png'.format(Band))
#             plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.svg'.format(Band))

#     elif stat_test == 'cohen':
#         cohen_ds = []
#         for i in range(len(dist1)):
#             cohen_ds.append(cohen_d(dist1[i], dist2[i]))

#         # Plot pvalue
#         fig, ax = plt.subplots()
#         fig.suptitle(f'Cohen d: {sit1}-{sit2}\n'
#                      f'mean: {round(np.mean(cohen_ds), 2)} +- {round(np.std(cohen_ds), 2)} -\n'
#                      f'min: {round(np.min(cohen_ds), 2)} - '
#                      f'max: {round(np.max(cohen_ds), 2)}', fontsize=17)
#         im = mne.viz.plot_topomap(cohen_ds, vmin=0, vmax=4, pos=info, axes=ax, show=Display_fig, sphere=0.07,
#                                   cmap='Reds')
#         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
#         cbar.ax.yaxis.set_tick_params(labelsize=17)
#         cbar.ax.set_ylabel(ylabel='Cohens d', fontsize=17)

#         fig.tight_layout()

#         if Save_fig:
#             os.makedirs(Run_graficos_path, exist_ok=True)
#             plt.savefig(Run_graficos_path + f'cohen_{sit1}-{sit2}.png'.format(Band))
#             plt.savefig(Run_graficos_path + f'cohen_{sit1}-{sit2}.svg'.format(Band))


# # ===================
# ## Violin Plot Stims

# Save_fig = True
# model = 'Ridge'
# situacion = 'Escucha'
# tmin, tmax = -0.6, 0
# Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Plots/'.format(model, situacion, tmin, tmax)

# Band = 'Theta'
# Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

# Correlaciones = {}
# for stim in Stims:
#     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
#     Corr, Pass = pickle.load(f)
#     f.close()

#     Correlaciones[stim] = Corr.mean(0)

# # Violin plot
# plt.ion()
# plt.figure(figsize=(19, 5))
# sn.violinplot(data=pd.DataFrame(Correlaciones))
# plt.ylabel('Correlation', fontsize=24)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=24)
# plt.grid()
# plt.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + '{}.png'.format(Band))
#     plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

# # Box plot
# # my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta1': 'C0'}

# fig, ax = plt.subplots()
# ax = sn.boxplot(data=pd.DataFrame(Correlaciones), width=0.35)
# ax.set_ylabel('Correlation')
# # ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
# for patch in ax.artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, .4))

# # Create an array with the colors you want to use
# # colors = ["C0", "grey"]
# # # Set your custom color palette
# # palette = sn.color_palette(colors)
# sn.swarmplot(data=pd.DataFrame(Correlaciones), size=2, alpha=0.4)
# plt.tick_params(labelsize=13)
# ax.xaxis.label.set_size(15)
# ax.yaxis.label.set_size(15)
# plt.grid()
# fig.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + '{}.png'.format(Band))
#     plt.savefig(Run_graficos_path + '{}.svg'.format(Band))
# # ==========================
# ## Violin Plot Situation


# model = 'Ridge'
# Band = 'Theta'
# stim = 'Spectrogram'
# situaciones = ['Escucha', 'Habla_Propia', 'Ambos', 'Ambos_Habla', 'Silencio']
# tmin, tmax = -0.6, -0.003
# fdr = True
# if fdr:
#     Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/fdr/'.format(Band, stim, tmin, tmax)
# else:
#     Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/log/'.format(Band, stim, tmin, tmax)
# Save_fig = True
# Display_fig = False
# if Display_fig:
#     plt.ion()
# else:
#     plt.ioff()

# montage = mne.channels.make_standard_montage('biosemi128')
# channel_names = montage.ch_names
# info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# Correlaciones = {}

# for situacion in situaciones:
#     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
#     Corr, Pass = pickle.load(f)
#     f.close()

#     Correlaciones[situacion] = Corr.transpose().ravel()

# my_pal = {'Escucha': 'darkgrey', 'Habla_Propia': 'darkgrey', 'Ambos': 'darkgrey', 'Ambos_Habla': 'darkgrey', 'Silencio': 'darkgrey'}

# plt.figure(figsize=(19, 5))
# sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal)
# plt.ylabel('Correlation', fontsize=24)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=24)
# plt.grid()
# ax = plt.gca()
# ax.set_xticklabels(['Listening', 'Speech\nproduction', 'Listening \n (speaking)', 'Speaking \n (listening)',
#                     'Silence'], fontsize=24)
# plt.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + 'Violin_plot.png'.format(Band))
#     plt.savefig(Run_graficos_path + 'Violin_plot.svg'.format(Band))

# ## TRF amplitude

# model = 'Ridge'
# Band = 'Theta'
# stim = 'Spectrogram'
# sr = 128
# tmin, tmax = -0.6, -0.003
# delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
# times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
# times = np.flip(-times)
# Stims_preprocess = 'Normalize'
# EEG_preprocess = 'Standarize'
# Save_fig = True

# Listening_25_folds_path = 'saves/25_folds/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
#     format(model, 'Escucha', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
# Listening_path = 'saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
#     format(model, 'Escucha', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
# Ambos_path = 'saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
#     format(model, 'Ambos', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)

# Run_graficos_path = 'gráficos/Amplitude_Comparison/{}/{}/tmin{}_tmax{}/'.format(Band, stim, tmin, tmax)

# # Load TRFs
# f = open(Listening_25_folds_path, 'rb')
# TRF_25 = pickle.load(f)
# f.close()
# TRF_25 = np.flip(TRF_25.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

# f = open(Listening_path, 'rb')
# TRF_escucha = pickle.load(f)
# f.close()
# TRF_escucha = np.flip(TRF_escucha.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

# f = open(Ambos_path, 'rb')
# TRF_ambos = pickle.load(f)
# f.close()
# TRF_ambos = np.flip(TRF_ambos.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

# montage = mne.channels.make_standard_montage('biosemi128')
# channel_names = montage.ch_names
# info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# mean_25 = TRF_25.mean(0)
# mean_escucha = TRF_escucha.mean(0)
# mean_ambos = TRF_ambos.mean(0)

# plt.ion()

# plt.figure(figsize=(15, 5))
# plt.plot(times*1000, mean_escucha, label='L|O')
# plt.plot(times*1000, mean_25, label='L|O downsampled')
# plt.plot(times*1000, mean_ambos, label='L|B')
# plt.xlim([(times*1000).min(), (times*1000).max()])
# plt.xlabel('Time (ms)')
# plt.ylabel('TRF (a.u.)')
# plt.grid()
# plt.legend()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + 'Plot.png'.format(Band))
#     plt.savefig(Run_graficos_path + 'Plot.svg'.format(Band))

# fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)
# for i, plot_data, title in zip(range(3), [TRF_escucha, TRF_25, TRF_ambos], ['L|O', 'L|O downsampled', 'L | B']):
#     evoked = mne.EvokedArray(plot_data, info)
#     evoked.times = times
#     evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
#                 show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[i])
#     axs[i].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
#     axs[i].xaxis.label.set_size(14)
#     axs[i].yaxis.label.set_size(14)
#     axs[i].set_ylim([-0.016, 0.015])
#     axs[i].tick_params(axis='both', labelsize=14)
#     axs[i].grid()
#     axs[i].legend(fontsize=12)
#     axs[i].set_title(f'{title}', fontsize=15)
#     if i != 2:
#         axs[i].set_xlabel('', fontsize=14)
#     else:
#         axs[i].set_xlabel('Time (ms)', fontsize=14)

# fig.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + 'Subplots.png'.format(Band))
#     plt.savefig(Run_graficos_path + 'Subplots.svg'.format(Band))



# ## Box Plot Bandas Decoding

# tmin, tmax = -0.4, 0.2
# model = 'Decoding'
# situacion = 'Escucha'

# Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Box Plots/'.format(model, situacion, tmin, tmax)
# Save_fig = False

# stim = 'Envelope'
# Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']

# Correlaciones = pd.DataFrame(columns=['Corr', 'Sig'])

# for Band in Bands:
#     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
#     Corr_Pass = pickle.load(f)
#     f.close()
#     Corr = Corr_Pass[0]
#     Pass = Corr_Pass[1]
#     temp_df = pd.DataFrame({'Corr': Corr.ravel(), 'Sig': Pass.ravel(), 'Band': [Band]*len(Corr.ravel())})
#     Correlaciones = pd.concat((Correlaciones, temp_df))

# Correlaciones['Permutations test'] = np.where(Correlaciones['Sig'] == 1, 'NonSignificant', 'Significant')

# my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta1': 'C0'}

# fig, ax = plt.subplots()
# ax = sn.boxplot(x='Band', y='Corr', data=Correlaciones, width=0.35, palette=my_pal)
# # ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
# for patch in ax.artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, .4))

# # Create an array with the colors you want to use
# colors = ["C0", "grey"]
# # Set your custom color palette
# palette = sn.color_palette(colors)
# sn.swarmplot(x='Band', y='Corr', data=Correlaciones, hue='Permutations test', size=3, palette=palette)
# plt.tick_params(labelsize=13)
# ax.xaxis.label.set_size(15)
# ax.yaxis.label.set_size(15)
# fig.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + '{}.png'.format(stim))
#     plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

# for Band in Bands:
#     print(f'\n{Band}')
#     Passed_folds = (Correlaciones.loc[Correlaciones['Band'] == Band]['Permutations test'] == 'Significant').sum()
#     Passed_subj = 0
#     for subj in range(len(Pass)):
#         Passed_subj += all(Pass[subj] < 1)
#     print(f'Passed folds: {Passed_folds}/{len(Correlaciones.loc[Correlaciones["Band"] == Band])}')
#     print(f'Passed subjects: {Passed_subj}/{len(Pass)}')

# ## Heatmaps
# model = 'Ridge'
# situacion = 'Escucha'
# tmin, tmax = -0.6, -0.003
# Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Heatmaps/'.format(model, situacion, tmin, tmax)
# Save_fig = True

# Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']
# Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

# Corrs_map = np.zeros((len(Stims), len(Bands)))
# Sig_map = np.zeros((len(Stims), len(Bands)))

# for i, stim in enumerate(Stims):
#     Corr_stim = []
#     Sig_stim = []
#     for Band in Bands:
#         f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
#         Corr_Band, Sig_Band = pickle.load(f)
#         f.close()
#         Corr_stim.append(Corr_Band.mean())
#         Sig_stim.append(Sig_Band.mean(1).sum(0))
#     Corrs_map[i] = Corr_stim
#     Sig_map[i] = Sig_stim

# fig, ax = plt.subplots()
# plt.imshow(Corrs_map)
# plt.title('Correlation', fontsize=15)
# ax.set_yticks(np.arange(len(Stims)))
# ax.set_yticklabels(Stims, fontsize=13)
# ax.set_xticks(np.arange(len(Bands)))
# ax.set_xticklabels(Bands, fontsize=13)
# # ax.xaxis.tick_top()
# cbar = plt.colorbar(shrink=0.7, aspect=15)
# cbar.ax.tick_params(labelsize=13)
# fig.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + 'Corr.png'.format(Band))
#     plt.savefig(Run_graficos_path + 'Corr.svg'.format(Band))

# fig, ax = plt.subplots()
# plt.imshow(Sig_map)
# plt.title('Significance', fontsize=15)
# ax.set_yticks(np.arange(len(Stims)))
# ax.set_yticklabels(Stims, fontsize=13)
# ax.set_xticks(np.arange(len(Bands)))
# ax.set_xticklabels(Bands, fontsize=13)
# # ax.xaxis.tick_top()
# cbar = plt.colorbar(shrink=0.7, aspect=15)
# cbar.ax.tick_params(labelsize=13)
# fig.tight_layout()

# if Save_fig:
#     os.makedirs(Run_graficos_path, exist_ok=True)
#     plt.savefig(Run_graficos_path + 'Stat.png'.format(Band))
#     plt.savefig(Run_graficos_path + 'Stat.svg'.format(Band))


# ## Plot por subjects
# import mne

# montage = mne.channels.make_standard_montage('biosemi128')
# channel_names = montage.ch_names
# info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# model = 'Ridge'
# situacion = 'Escucha'
# alpha = 100
# tmin, tmax = -0.6, -0.003

# Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/'.format(model, situacion, tmin, tmax)
# Save_fig = False

# Bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']
# Bands = ['Theta']
# for Band in Bands:

#     f = open('saves/Ridge/correlations/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
#     Corr_Envelope, Pass_Envelope = pickle.load(f)
#     f.close()

#     f = open('saves/Ridge/correlations/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
#     Corr_Pitch, Pass_Pitch = pickle.load(f)
#     f.close()

#     f = open(
#         'saves/Ridge/correlations/tmin{}_tmax{}/Envelope_Pitch_Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band),
#         'rb')
#     Corr_Envelope_Pitch_Spectrogram, Pass_Envelope_Pitch_Spectrogram = pickle.load(f)
#     f.close()

#     f = open(
#         'saves/Ridge/correlations/tmin{}_tmax{}/Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
#     Corr_Spectrogram, Pass_Spectrogram = pickle.load(f)
#     f.close()

#     plt.ion()
#     plt.figure()
#     plt.title(Band)
#     plt.plot([plt.xlim()[0], 0.55], [plt.xlim()[0], 0.55], 'k--')
#     for i in range(len(Corr_Envelope)):
#         plt.plot(Corr_Envelope[i], Corr_Pitch[i], '.', label='Subject {}'.format(i + 1))
#     plt.legend()
#     plt.grid()
