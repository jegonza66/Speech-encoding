"""
This code analyzes the external condition of the envelope stimulus using a moving filter window approach. 
It computes the transfer functions (TRFs) and correlations for custom frequency bands ranging from 1 to 15 Hz, with a specified window size. 
The results are then visualized by plotting the average Pearson correlation for each band.
I used inside config.py:
center_low, center_high = 2, 15
window_size = 2 # Hz # -> 1
bands = [
    f'Custom-{center-window_size/2}#{center+window_size/2}' # Custom band from 1 to 15 Hz
    for center in range(center_low, center_high + 1)
]
window_size = 4 # Hz # -> 2
center_low, center_high = 2, 15
bands += [
    f'Custom-{center-window_size/2}#{center+window_size/2}' # Custom band from 1 to 15 Hz
    for center in range(center_low, center_high + 1)
]
window_size = 5 # Hz  # -> 2.5
center_low, center_high = 3, 15
bands += [
    f'Custom-{center-window_size/2}#{center+window_size/2}' # Custom band from 1 to 15 Hz
    for center in range(center_low, center_high + 1)
]
window_size = 6 # Hz  # -> 3
center_low, center_high = 3, 15
bands += [
    f'Custom-{center-window_size/2}#{center+window_size/2}' # Custom band from 1 to 15 Hz
    for center in range(center_low, center_high + 1)
]

"""

# Standard libraries
import os, numpy as np

from utils.general_functions import load_pickle
from scipy.signal.windows import gaussian
import mne

import config
# ============
# RUN ANALYSIS
# ============
situation = 'External'
stim = 'Envelope'

# center_low, center_high = 2, 15
# window_size = 2 # Hz # -> 1
# config.bands = [
#     f'Custom-{center-window_size/2}#{center+window_size/2}' # Custom band from 1 to 15 Hz
#     for center in range(center_low, center_high + 1)
# ]

correlations = {band:None for band in config.bands}
trfs = {band:None for band in config.bands}

for band in config.bands:
    preprocessed_data_path = os.path.normpath(f'{config.saves_dir}/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/')
    path_weights = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
    save_results_path = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/correlations/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/'
    
    trfs[band] = load_pickle(
        path=path_weights+'total_weights_per_subject.pkl'
    )
    
    correlations[band] = load_pickle(
        path=save_results_path+'Envelope.pkl'
    )

# Compute average correlation for each band and plot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# =========================================
# Analyze average correlations across bands
avg_corrs = {}

for band in config.bands:
    # Extract width and band center
    l_fr, h_fr = [float(fr) for fr in band.split('-')[1].split('#')]
    band_width = h_fr - l_fr
    avg_corrs[band_width] = {}
    
for band in config.bands:
    # Extract width and band center
    l_fr, h_fr = [float(fr) for fr in band.split('-')[1].split('#')]
    band_width = h_fr - l_fr
    center = l_fr + band_width / 2
    
    avg_corrs[band_width][center] = np.mean(
        correlations[band]['average_correlation_subjects']
    )

plt.figure(
    figsize=(8, 4),
    tight_layout=True
)
for width in avg_corrs:
    plt.plot(
        list(avg_corrs[width].keys()), 
        list(avg_corrs[width].values()), 
        marker='o', 
        label=f'Width {width} Hz'
    )
plt.title('Average Pears. correlation per Band', fontsize=10)
plt.ylabel('Average Pears. correlation')
plt.xlabel('Band center (Hz)')
plt.grid(True)
plt.legend()
plt.show(block=False)

# =================================================================
# Analyze average TRFs across bands in contrast with filter kernels
center_freq = 6.0 # Hz
avg_trfs = {}

for band in config.bands:
    # Extract width and band center
    l_fr, h_fr = [float(fr) for fr in band.split('-')[1].split('#')]
    band_width = h_fr - l_fr
    center = l_fr + band_width / 2
    if center==center_freq:
        avg_trfs[band_width] = trfs[band]['average_weights_subjects'].mean(
            axis=0
            ).mean(
                axis=0
                ).mean(
                    axis=0
                    )
    
fig, axes = plt.subplots(
    nrows=2, 
    ncols=1, 
    figsize=(8, 6),
    sharex=True,
    tight_layout=True
)

# First axis: TRFs for Different Bands
colors = plt.cm.berlin(np.linspace(0, 1, len(avg_trfs)))
trf_lines = []
trf_labels = []

for i, band_range in enumerate(avg_trfs):
    trf = avg_trfs[band_range]
    line, = axes[0].plot(
        config.times*1e3, 
        trf/trf.max(), 
        color=colors[i], 
        label=f'Width {band_range} Hz'
    )
    trf_lines.append(line)
    trf_labels.append(f'Width {band_range} Hz')
    
axes[0].set_ylabel('TRF Amplitude')
axes[0].set_title(f'TRFs filtered: center {center_freq} Hz')
axes[0].legend(trf_lines, trf_labels, ncol=2, loc='upper right', fontsize='small')
axes[0].grid(True)

# Create impulse signal
impulse = np.zeros(config.delays.shape)
impulse[config.delays==0] = 1.0  # Impulse at the center

# Create a smoother impulse (e.g., a short Gaussian pulse instead of a single-sample spike)
width_samples = 5
center_idx = np.where(config.delays == 0)[0][0]
impulse = np.zeros(config.delays.shape)
gauss = gaussian(M=width_samples, std=width_samples/3)
start = max(center_idx - width_samples // 2, 0)
end = start + width_samples
impulse[start:end] = gauss[:impulse[start:end].shape[0]]
impulse /= impulse.sum()  # Normalize area to 1

# Create MNE RawArray for impulse
info_impulse = mne.create_info(ch_names=['impulse'], sfreq=128, ch_types=['eeg'])
impulse_raw = mne.io.RawArray(impulse.reshape(1, -1), info_impulse)

band_ranges = []
for band in config.bands:
    # Extract width and band center
    l_fr, h_fr = [float(fr) for fr in band.split('-')[1].split('#')]
    band_width = h_fr - l_fr
    center = l_fr + band_width / 2
    if center==center_freq:
        band_ranges.append((l_fr, h_fr))

kernel_lines = []
kernel_labels = []
for i, band_range in enumerate(band_ranges):
    l_freq, r_freq = band_range
    filtered_impulse = impulse_raw.copy().filter(
            l_freq=l_freq,
            h_freq=r_freq,
            **{'method': 'iir', 'iir_params': {
                    "ftype": "cheby2",
                    "order": 4,
                    "rs": 20
                }}
        )
    kernel = filtered_impulse.get_data().flatten()
    line, = axes[1].plot(config.times*1e3, kernel/kernel.max(), color=colors[i], label=f'Band {band_range}')
    kernel_lines.append(line)
    kernel_labels.append(f'Width {r_freq-l_freq} Hz')

axes[1].set_ylabel('Filter Kernel')
axes[1].set_title('Filter Kernels (Delta Response)')
# Split legend into two columns
axes[1].legend(kernel_lines, kernel_labels, ncol=2, loc='upper right', fontsize='small')
axes[1].grid(True)
fig.show()
