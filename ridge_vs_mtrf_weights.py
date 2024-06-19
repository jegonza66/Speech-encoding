import numpy as np, os, mne, matplotlib.pyplot as plt
from funciones import load_pickle

# ==================================================================
# Read and extract weights to compute porcentual relative difference
path_mtrf_weights = 'saves/mtrf/Escucha/Original/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/stim_Envelope_EEG_band_Theta/total_weights_per_subject_Envelope_Theta.pkl'
path_ridge_weights = 'saves/Ridge/Escucha/Original/Stims_Normalize_EEG_Standarize/tmin-0.6_tmax0.2/Stim_Envelope_EEG_Band_Theta/Pesos_Totales_Envelope_Theta.pkl'

# Compute porcentual relative difference
w_mtrf = load_pickle(path=path_mtrf_weights)['average_weights_subjects'].mean(axis=0)[:,0,:]
w_ridge = np.flip(load_pickle(path=path_ridge_weights), axis=1)
w_rel_diff = 100*(w_ridge-w_mtrf)/(np.abs(w_mtrf+w_ridge)/2)

# TODO: there are values that goes to inf and -inf. Son 5
number_of_infs = np.unique(w_rel_diff==-np.inf, return_counts=True)[1][1] + np.unique(w_rel_diff==np.inf, return_counts=True)[1][1]
w_rel_diff[w_rel_diff==np.inf]=0
w_rel_diff[w_rel_diff==-np.inf]=0
 
# w_rel_diff = np.abs(w_ridge+w_mtrf)/2
# w_difference = w_ridge-w_mtrf

# ========================================
# Define hyperparameters used in the model

# Times and delays
sr, tmin, tmax = 128, -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# Topographic information
montage = mne.channels.make_standard_montage('biosemi128')
info = mne.create_info(ch_names=montage.ch_names, sfreq=sr, ch_types='eeg').set_montage(montage)

# ========================================
# Make plot of weights difference

fig, ax = plt.subplots(nrows=3, ncols=1, layout='tight', figsize=(6,8), sharex=True)

# Ridge
evoked_ridge = mne.EvokedArray(data=w_ridge, info=info)

# Relabel time 0
evoked_ridge.shift_time(times[0], relative=True)

# Plot
evoked_ridge.plot(
    scalings={'eeg':1}, 
    zorder='std', 
    time_unit='ms',
    show=False, 
    spatial_colors=True, 
    # unit=False, 
    units='mTRF (a.u.)',
    axes=ax[0],
    gfp=False)

# Add mean of all channels
ax[0].plot(
    times * 1000, #ms
    evoked_ridge._data.mean(0), 
    'k--', 
    label='Mean', 
    zorder=130, 
    linewidth=2)

# Graph properties
ax[0].set(xlabel='')
ax[0].legend()
ax[0].grid(visible=True)
ax[0].set_title('RIDGE')

# Mne mtrf
evoked_mtrf = mne.EvokedArray(data=w_mtrf, info=info)

# Relabel time 0
evoked_mtrf.shift_time(times[0], relative=True)

# Plot
evoked_mtrf.plot(
    scalings={'eeg':1}, 
    zorder='std', 
    time_unit='ms',
    show=False, 
    spatial_colors=True, 
    # unit=False, 
    units='mTRF (a.u.)',
    axes=ax[1],
    gfp=False)

# Add mean of all channels
ax[1].plot(
    times * 1000, #ms
    evoked_mtrf._data.mean(0), 
    'k--', 
    label='Mean', 
    zorder=130, 
    linewidth=2)

# Graph properties
ax[1].set(xlabel='')
ax[1].legend()
# ax[1].set_ylabel('')
ax[1].grid(visible=True)
ax[1].set_title('MTRF')

# Porcentual relative difference
evoked_diff = mne.EvokedArray(data=w_rel_diff, info=info)

# Relabel time 0
evoked_diff.shift_time(times[0], relative=True)

# Plot
evoked_diff.plot(
    scalings={'eeg':1}, 
    exclude='bads',
    zorder='std', 
    time_unit='ms',
    show=False, 
    spatial_colors=True, 
    # unit=False, 
    units='mTRF (a.u.)',
    axes=ax[2],
    gfp=False)

# Add mean of all channels
ax[2].plot(
    times * 1000, #ms
    evoked_diff._data.mean(0), 
    'k--', 
    label='Mean', 
    zorder=130, 
    linewidth=2)

# Graph properties
ax[2].legend()
ax[2].set_ylabel(r'$\epsilon_{\%}$')
ax[2].set(xlabel='Time(ms)')
ax[2].grid(visible=True)
ax[2].set_title('Relative Porcentual difference')
ymin = evoked_diff._data.mean(axis=0).min()
ymax = evoked_diff._data.mean(axis=0).max()
ax[2].set_ylim((ymin, ymax))

plt.show()