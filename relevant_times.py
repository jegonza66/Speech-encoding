# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn, copy
from matplotlib.colors import Normalize
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.ticker as pticker

params = {
        'legend.fontsize': 'x-large',
        'legend.title_fontsize': 'x-large',
        'figure.figsize': (8, 6),
        'figure.titlesize': 'xx-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large'
        }
pylab.rcParams.update(params)

# Modules
from utils.funciones import load_pickle, dump_pickle
from utils.processing import tfce
from config import Exp_info
phonological_labels = list(Exp_info().phonological_labels)

# ============
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
        print('\n\t===========================\n','\t\tPARAMETERS\n\n','\t\tModel: ' + model+'\n','\t\tBand: ' + str(band)+'\n','\t\tStimulus: ' + stimulus+'\n','\t\tCondition: ' + situation+'\n','\n\t===========================\n')

        # Loads TFCE
        try:
            print("\nLoading data")
            tvalue_tfce, pvalue_tfce = load_pickle(path=os.path.join(path_TFCE, band, stimulus + f'_{n_permutations}.pkl'))
            pvalues[band][stimulus] = pvalue_tfce
            print('\n===========================\n','\n\tBand: ' + str(band)+'\n','\tStimulus: ' + stimulus+'\n','\tCondition: ' + situation+'\n','\n\tLoad succesful\n','\n===========================\n')
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