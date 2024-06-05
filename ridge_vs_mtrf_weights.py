import numpy as np, os, mne, matplotlib.pyplot as plt
from Funciones import load_pickle

path_mtrf_weights = 'saves/mtrf/Escucha/Original/stims_Normalize_EEG_Standarize/tmin-0.2_tmax0.6/stim_Envelope_EEG_band_Theta/total_weights_per_subject_Envelope_Theta.pkl'
path_ridge_weights = 'saves/Ridge/Escucha/Original/Stims_Normalize_EEG_Standarize/tmin-0.6_tmax0.2/Stim_Envelope_EEG_Band_Theta/Pesos_Totales_Envelope_Theta.pkl'

w_mtrf = load_pickle(path=path_mtrf_weights)['average_weights_subjects'].mean(axis=0)[:,0,:]
w_ridge = np.flip(load_pickle(path=path_ridge_weights), axis=1)
w_resta = w_ridge-w_mtrf

sr=128
tmin, tmax = -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

montage = mne.channels.make_standard_montage('biosemi128')
info = mne.create_info(ch_names=montage.ch_names, sfreq=sr, ch_types='eeg').set_montage(montage)

# Make plot of weights difference
fig, ax = plt.subplots(nrows=1, ncols=3, layout='tight', figsize=(12,4), sharey=True)

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
ax[1].legend()
ax[1].set_ylabel('')
ax[1].grid(visible=True)
ax[1].set_title('MTRF')

# Difference
evoked_resta = mne.EvokedArray(data=w_resta, info=info)

# Relabel time 0
evoked_resta.shift_time(times[0], relative=True)

# Plot
evoked_resta.plot(
    scalings={'eeg':1}, 
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
    evoked_resta._data.mean(0), 
    'k--', 
    label='Mean', 
    zorder=130, 
    linewidth=2)

# Graph properties
ax[2].legend()
ax[2].set_ylabel('')
ax[2].grid(visible=True)
ax[2].set_title('RESTA')

plt.show()