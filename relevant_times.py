# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn, copy
from matplotlib.colors import Normalize
from datetime import datetime
import matplotlib.cm as cm
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

# Specific libraries

# Modules
from funciones import load_pickle, dump_pickle
from processing import tfce
from setup import exp_info
phonological_labels = list(exp_info().phonological_labels)

# ===========
# PARAMETERS
tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

montage = mne.channels.make_standard_montage('biosemi128')
info_mne = mne.create_info(ch_names=montage.ch_names[:], sfreq=sr, ch_types='eeg').set_montage(montage)
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'
model = 'mtrf'

# TFCE parameters
pval_tresh, n_permutations = .05, 2500

# Code parameters
save_figures = True
#==================
# FEATURE SELECTION
stimuli = ['Envelope', 'Spectrogram', 'Deltas', 'Phonological', 'Mfccs', 'Mfccs-Deltas', 'Phonological_Spectrogram','Phonological_Deltas']
stimuli+= ['Phonological_Deltas_Spectrogram','Pitch-Log-Raw','Phonemes-Discrete', 'Phonemes-Onset']
stimuli+= ['Phonemes-Discrete_Pitch-Log-Raw_Envelope', 'Phonemes-Discrete_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw']
stimuli+= ['Envelope_Phonemes-Onset', 'Envelope_Phonemes-Discrete']
stimuli = ['Envelope', 'Spectrogram', 'Deltas', 'Phonological', 'Mfccs', 'Mfccs-Deltas', 'Pitch-Log-Raw', 'Phonemes-Discrete']

bands = ['Delta','Theta', 'Alpha', 'Beta1', 'Beta2']
# bands = ['Theta']

situation = 'External'

# Relevant paths
path_figures = os.path.normpath(f'figures/{model}/relevant_times/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')
final_corr_path = os.path.normpath(f'saves/{model}/{situation}/correlations/tmin{tmin}_tmax{tmax}/')
weights_path = os.path.normpath(f'saves/{model}/{situation}/weights//stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')
path_TFCE = f'saves/{model}/{situation}/TFCE/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/'

# =========
# RUN TFCE
significant_n_channels = {band:{stimulus:None for stimulus in stimuli} for band in bands}
pvalues = {band:{stimulus:None for stimulus in stimuli} for band in bands}

for band in bands:
    for stimulus in stimuli:

        # Sort stimuli and bands
        ordered_stims, ordered_band = sorted(stimulus.split('_')), sorted(band.split('_'))
        stimulus, band = '_'.join(ordered_stims), '_'.join(ordered_band)

        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stimulus+'\n','Status: ' + situation+'\n','\n===========================\n')

        # Loads TFCE
        try:
            print("\nLoading data")
            tvalue_tfce, pvalue_tfce = load_pickle(path=os.path.join(path_TFCE, band, stimulus + f'_{n_permutations}.pkl'))
            pvalues[band][stimulus] = pvalue_tfce
            print('\n===========================\n','\nBand: ' + str(band)+'\n','Stimulus: ' + stimulus+'\n','Status: ' + situation+'\n','\n\tLoad succesful\n','\n===========================\n')
        except:
            print("\nLoad fail", "\nComputing TFCE")

            # Load weights (n_subjects, n_chan, n_feats, n_delays) 
            average_weights_subjects = load_pickle(path=os.path.join(weights_path, band, stimulus, 'total_weights_per_subject.pkl'))['average_weights_subjects']

            # Compute TFCE to get p-value
            tvalue_tfce, pvalue_tfce = tfce(
                                            average_weights_subjects=average_weights_subjects,
                                            stimulus=stimulus, 
                                            n_jobs=-1, 
                                            n_permutations=n_permutations
                                            )
            
            # Save TFCE
            os.makedirs(os.path.join(path_TFCE, band), exist_ok=True)
            dump_pickle(path=os.path.join(path_TFCE, band, stimulus + f'_{n_permutations}.pkl'), obj=(tvalue_tfce, pvalue_tfce), rewrite=True)
            
            # Fill dictionary
            pvalues[band][stimulus] = pvalue_tfce

        if pvalues[band][stimulus].ndim==3:
            n_features, n_delays, n_channels = pvalues[band][stimulus].shape
            significant_delays = np.zeros(shape=(n_features, n_delays))

            # Iteate over columns to get number of channels per feature that passes the threshold
            for feature in range(n_features):
                for delay in range(n_delays):
                    # Count how many channels pass the threshold for a given feature and delay
                    ppval = pvalues[band][stimulus][feature][delay]
                    significant_delays[feature, delay] = len(ppval[ppval<pval_tresh])
            significant_n_channels[band][stimulus] = significant_delays

            plt.figure()
            plt.title(stimulus+' '+ band)
            if n_features==1:
                img = plt.pcolormesh(
                                    # times*1e3, # x
                                    # np.arange(n_features), # y
                                    significant_n_channels[band][stimulus], # z
                                    shading='auto'
                                    )
                plt.yticks([])
                plt.xticks(ticks=np.arange(n_delays)[::25], labels=[int(t) for t in (times[::25]*1e3).round(-1)])
            else:
                img = plt.pcolormesh(
                                    times*1e3, # x
                                    np.arange(n_features), # y
                                    significant_n_channels[band][stimulus], # z
                                    shading='auto'
                                    )
                plt.ylabel('Features')
                
            cbar = plt.colorbar( 
                                orientation="vertical", 
                                label=r"#of significant channels",
                                fraction=0.05, 
                                pad=0.025, 
                                mappable=img
                                )
            if stimulus.startswith('Phonolog'):
                plt.yticks(ticks=np.arange(significant_n_channels[band][stimulus].shape[0]).tolist(),labels=phonological_labels)
            # plt.xticks(ticks=[0,26,103], labels=[-200,0,600])
            # plt.yticks(ticks=[])
            plt.xlabel('Time (ms)')
            plt.show(block=False)

        else: #TODO ojo sobre éstos tipos de pvalues no se puede calcular el número de canales significantes ¿no habría que unificar el criterio si pretendemos comparar la latencia?
            n_delays, n_features = pvalues[band][stimulus].shape
            significant_delays = np.zeros(shape=(n_features, n_delays))


        
        # pvalue_tfce[0][pvalue_tfce[0]>0.05]=1
        # plt.figure()
        # plt.pcolormesh(-np.log10(pvalue_tfce[0]).T)
        # plt.show(block=False)

        # Make pvalue figures


for stimulus in stimuli:
    data = significant_n_channels['Theta'][stimulus]
    

# # ============================================
# # Order stimuli according latence in each band






# pval_tresh=0.01 # TODO: discutir

# # Make figure
# weights = average_weights_subjects[:,:, 0,:].mean(axis=0)
# evoked = mne.EvokedArray(data=weights, info=info_mne)
# evoked.shift_time(times[0], relative=True)


# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# fig.suptitle('Envelope')

# # Plot
# evoked.plot(
#     scalings={'eeg':1}, 
#     zorder='std', 
#     time_unit='ms',
#     show=False, 
#     spatial_colors=True, 
#     # unit=False, 
#     units='mTRF (a.u.)',
#     axes=ax[1],
#     gfp=False)
# ax[1].grid(visible=True)

# # Add mean of all channels
# ax[1].plot(
#         times * 1000, #ms
#         evoked._data.mean(0), 
#         'k--', 
#         label='Mean', 
#         zorder=130, 
#         linewidth=2)
# ax[1].legend()
# for i, pval_tresh in enumerate([.00025, .0005,.005,.01,.05]):

    

#     ax[0].scatter(
#                 times*1000, 
#                 significant_delays.reshape(-1), 
#                 color=f'C{i}', 
#                 s=2, 
#                 label=f'thresh: {pval_tresh}'
#                 )

#     ax[0].plot(
#                 times*1000, 
#                 significant_delays.reshape(-1), 
#                 color=f'C{i}'
#                 )
# ax[0].grid(visible=True)
# ax[0].set_ylabel('# of significant channels')
# ax[0].legend(fontsize=8, loc='lower center')
# fig.show()

# # plt.close(fig)

# tvalue_tfce[0].mean(1)


# # # # Filter threshold
# # pvalue_tfce[pvalue_tfce < pval_tresh] = 1

# # Plot t and p values
# plt.figure()
# for i in range(pvalue_tfce.shape[0]):
#     pvalue_feat_i = pvalue_tfce[i]
#     # pvalue_feat_i[pvalue_feat_i<pval_tresh].shape
#     # logp = -np.log10(np.maximum(pvalue_tfce[i], pval_tresh))
#     # plt.plot(times, logp)
#     plt.plot(times[pvalue_feat_i<pval_tresh], pvalue_feat_i[pvalue_feat_i<pval_tresh])
# plt.show()
# plt.close()
# import plot

# plot.plot_pvalue_tfce(average_weights_subjects=average_weights_subjects, pvalue=pvalue_tfce, times=times, info=info,
#                       trf_subjects_shape=trf_subjects_shape, n_feats=n_feats, band=band, stim=stim, pval_tresh=.05, 
#                       save_path=path_figures, display_interactive_mode=display_interactive_mode, save=save_figures, 
#                       no_figures=no_figures)
        
# # Save results
# if save_results and total_number_of_subjects==18:
#     os.makedirs(save_results_path, exist_ok=True)
#     os.makedirs(path_weights, exist_ok=True)
#     dump_pickle(path=save_results_path+f'{stim}.pkl', 
#                 obj={'average_correlation_subjects':average_correlation_subjects,
#                     'repeated_good_correlation_channels_subjects':repeated_good_correlation_channels_subjects},
#                 rewrite=True)
#     dump_pickle(path=path_weights+'total_weights_per_subject.pkl', 
#                 obj={'average_weights_subjects':average_weights_subjects},
#                 rewrite=True)