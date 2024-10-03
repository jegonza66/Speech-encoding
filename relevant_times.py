# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn, copy
from matplotlib.colors import Normalize
from datetime import datetime
import matplotlib.cm as cm

# Specific libraries

# Modules
from funciones import load_pickle, get_maximum_correlation_channels
from processing import tfce

# Default size is 10 pts, the scalings (10pts*scale) are:
#'xx-small':0.579,'x-small':0.694,'s
# mall':0.833,'medium':1.0,'large':1.200,'x-large':1.440,'xx-large':1.728,None:1.0}
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
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'

tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

montage = mne.channels.make_standard_montage('biosemi128')
info_mne = mne.create_info(ch_names=montage.ch_names[:], sfreq=sr, ch_types='eeg').set_montage(montage)

#==================
# FEATURE SELECTION
# Whether to use or not just relevant channels
pval_tresh, n_permutations = .05, 100
situation = 'External'
bands = ['Theta']
stimuli = ['Envelope', 'Spectrogram', 'Deltas', 'Phonological', 'Mfccs', 'Mfccs-Deltas', 'Phonological_Spectrogram','Phonological_Deltas']
stimuli+= ['Phonological_Deltas_Spectrogram','Pitch-Log-Raw','Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual']
stimuli+= ['Phonemes-Discrete-Manual_Pitch-Log-Raw_Envelope', 'Phonemes-Discrete-Manual_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw']
stimuli+= ['Envelope_Phonemes-Onset-Manual', 'Envelope_Phonemes-Discrete-Manual']

# Relevant paths
path_figures = os.path.normpath(f'figures/{model}/relevant_times/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')
final_corr_path = os.path.normpath(f'saves/{model}/{situation}/correlations/tmin{tmin}_tmax{tmax}/')
weights_path = os.path.normpath(f'saves/{model}/{situation}/weights//stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')

# Code parameters
save_figures = True

# ============
# RUN ANALYSIS
# ============
stimuli =['Envelope']
pvalues = {band:{stimulus:None for stimulus in stimuli} for band in bands}
for band in bands:
    for stim in stimuli:
        # Sort stimuli and bands
        ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
        stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)

        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + situation+'\n','\n===========================\n')
        
        # Load weights (n_subjects, n_chan, n_feats, n_delays) 
        average_weights_subjects = load_pickle(path=os.path.join(weights_path, band, stim, 'total_weights_per_subject.pkl'))['average_weights_subjects']

        # Compute TFCE to get p-value
        tvalue_tfce, pvalue_tfce = tfce(
                                        average_weights_subjects=average_weights_subjects,
                                        stimulus=stim, 
                                        n_jobs=-1, 
                                        n_permutations=n_permutations
                                        )
        pvalues[band][stim] = pvalue_tfce
        
        pvalue_tfce[0][pvalue_tfce[0]>0.05]=1
        plt.figure()
        plt.pcolormesh(-np.log10(pvalue_tfce[0]).T)
        plt.show(block=False)

        # Make pvalue figures

# ============================================
# Order stimuli according latence in each band
pval_tresh=0.01 # TODO: discutir

# Make figure
weights = average_weights_subjects[:,:, 0,:].mean(axis=0)
evoked = mne.EvokedArray(data=weights, info=info_mne)
evoked.shift_time(times[0], relative=True)


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.suptitle('Envelope')

# Plot
evoked.plot(
    scalings={'eeg':1}, 
    zorder='std', 
    time_unit='ms',
    show=False, 
    spatial_colors=True, 
    # unit=False, 
    units='mTRF (a.u.)',
    axes=ax[1],
    gfp=False)
ax[1].grid(visible=True)

# Add mean of all channels
ax[1].plot(
        times * 1000, #ms
        evoked._data.mean(0), 
        'k--', 
        label='Mean', 
        zorder=130, 
        linewidth=2)
ax[1].legend()
for i, pval_tresh in enumerate([.00025, .0005,.005,.01,.05]):
    significant_n_channels = {band:{stimulus:None for stimulus in stimuli} for band in bands}

    # Iteate over columns to get number of channels per feature that passes the threshold
    for band in bands:
        for stimulus in stimuli:
            n_features, n_delays, n_channels = pvalues[band][stimulus].shape
            significant_delays = np.zeros(shape=(n_features, n_delays))
            for feature in range(n_features):
                for delay in range(n_delays):
                    # Count how many channels pass the threshold for a given feature and delay
                    ppval=pvalues[band][stimulus][feature][delay]
                    significant_delays[feature, delay] = len(ppval[ppval<pval_tresh])
            significant_n_channels[band][stimulus] = significant_delays

    ax[0].scatter(
                times*1000, 
                significant_delays.reshape(-1), 
                color=f'C{i}', 
                s=2, 
                label=f'thresh: {pval_tresh}'
                )

    ax[0].plot(
                times*1000, 
                significant_delays.reshape(-1), 
                color=f'C{i}'
                )
ax[0].grid(visible=True)
ax[0].set_ylabel('# of significant channels')
ax[0].legend(fontsize=8, loc='lower center')
fig.show()

# plt.close(fig)

tvalue_tfce[0].mean(1)


# # # Filter threshold
# pvalue_tfce[pvalue_tfce < pval_tresh] = 1

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