import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.text as mtext

import numpy as np
np.random.seed(42)
import mne
import os

# from ..utils.general_functions import load_pickle
from utils.general_functions import load_pickle, dump_pickle
import config

# # Make plot to identify selection of electrodes
# plt.figure()
# montage = config.info_mne.get_montage()
# montage.plot(kind='topomap', show_names=True, show=False)
# plt.show()
selection = [
    "D10", "D11", "D12","D20", "D21", "D22", # left
    "B29", "B30", "B31","B22", "B23", "B24" # right
    ]
selection = [
    config.info_mne['ch_names'].index(ch) 
    for ch in selection
]
hypothetical_kernel = load_pickle(
    path=rf'output\DiLib\weights\All\Envelope\total_weights_per_subject.pkl'
)['average_weights_subjects'].mean(axis=0).mean(axis=1)[selection, :].mean(axis=0)

# Make it smoother
# hypothetical_kernel = np.convolve(hypothetical_kernel, np.ones(10)/10, mode='same')

# Plot adjusted trfs
trfs = load_pickle(
    path=rf'output_simulated\mtrf_ridge_torch\All\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\All\Envelope\total_weights_per_subject.pkl'
)['average_weights_subjects'].mean(axis=0).mean(axis=1)

fig, ax = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(10, 5), 
    # sharex=True,
    tight_layout=True
)

# Plot the hypothetical kernel
ax[0].plot(
    config.times*1e3, 
    hypothetical_kernel, 
    label='Hypothetical Kernel', 
    color='black', 
    linewidth=1
)
ax[0].legend()
  
evoked = mne.EvokedArray(
    data=trfs, 
    info=config.info_mne
)     
evoked.shift_time(
    config.times[0], 
    relative=True
)
evoked_plot = evoked.plot(
    scalings={'eeg':1},
    zorder='std',
    time_unit='ms',
    show=False,
    spatial_colors=True,
    # unit=False,
    gfp=True,
    units='mTRFs (U.A)',
    axes=ax[1],
)
ax[1].plot(
    config.times*1e3,
    evoked._data.mean(axis=0), 
    label='mean TRF', 
    color='black', 
    linewidth=1.2
)
ax[1].set_title('')
ax[1].legend()

# Eliminar la etiqueta "Nave"
for txt in fig.findobj(mtext.Text):
    if "ave" in txt.get_text():
            txt.remove()

fig.show()

from scipy.signal import correlate

# Get the mean TRF across channels
mean_trf = trfs.mean(axis=0)

# Compute cross-correlation
corr = correlate(mean_trf, hypothetical_kernel, mode='full')
lags = np.arange(-len(mean_trf) + 1, len(hypothetical_kernel))

# Find the lag with the maximum correlation
max_corr_idx = np.argmax(corr)
delay_samples = lags[max_corr_idx]

# Convert delay from samples to milliseconds
sample_interval_ms = (config.times[1] - config.times[0]) * 1e3
delay_ms = delay_samples * sample_interval_ms

print(f"Estimated delay: {delay_samples} samples ({delay_ms:.2f} ms)")

# Plot the cross-correlation
fig_cc, ax_cc = plt.subplots(
    figsize=(8, 4),
    tight_layout=True
)
ax_cc.plot(lags * sample_interval_ms, corr)
ax_cc.axvline(delay_ms, color='red', linestyle='--', label=f'Max Corr: {delay_ms:.2f} ms')
ax_cc.set_xlabel('Lag (ms)')
ax_cc.set_ylabel('Cross-correlation')
ax_cc.set_title('Cross-correlation between mean TRF and hypothetical kernel')
ax_cc.legend()
fig_cc.show()