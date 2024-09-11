# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn, copy
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Specific libraries
# from statsmodels.stats.multitest import fdrcorrection
from matplotlib_venn import venn3, venn2  # venn3_circles
from scipy.spatial import ConvexHull
from scipy.stats import wilcoxon

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
pval_tresh, n_permutations = .05, 1536
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
stimuli = ['Phonological']
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
        tvalue_tfce, pvalue_tfce, trf_subjects_shape = tfce(average_weights_subjects=average_weights_subjects, n_jobs=-1, n_permutations=n_permutations)
        
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