import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
matplotlib.use('Agg')  # Use Agg backend for better compatibility

from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, sosfreqz, sosfilt
from pathlib import Path
import mne

# --- 2) Butterworth bandpass filter (1–8 Hz, order 2 each stage) ---
low_cut = 1
high_cut = 8
order = 2
fs = 512

# Bandpass with lowpass & highpass order=2 each
sos = butter(order, [low_cut, high_cut], btype='bandpass', fs=fs, output='sos')

# --- 3) Visualize the filter kernel (impulse response) ---
impulse_duration = 0.8  # seconds
impulse_samples = int(impulse_duration * fs)
t = np.linspace(-impulse_duration/2, impulse_duration/2, impulse_samples)
sigma = 0.01  # standard deviation in seconds
impulse = np.exp(-t**2 / (2 * sigma**2))
impulse /= impulse.max()  # Normalize area to 1
kernel0 = sosfiltfilt(sos, impulse)

# Create MNE RawArray for impulse
info_impulse = mne.create_info(ch_names=['impulse'], sfreq=fs, ch_types=['eeg'])
impulse_raw = mne.io.RawArray(impulse.reshape(1, -1), info_impulse)

# Apply filter to impulse
filtered_impulse = impulse_raw.copy().filter(
    l_freq=low_cut,
    h_freq=high_cut,
    **{'method': 'iir', 'iir_params': {
        "ftype": "cheby2",
        "order": 4,
        "rs": 20
        }
    }
)

# Get the filtered impulse response (kernel)
kernel1 = filtered_impulse.get_data().flatten()
time_axis = np.arange(len(kernel1)) / fs
time_axis_centered = time_axis - impulse_duration/2  # Center around 0

# --- Butterworth kernel using MNE ---
filtered_impulse_butter = impulse_raw.copy().filter(
    l_freq=low_cut,
    h_freq=high_cut,
    method='iir',
    iir_params={
        "ftype": "butter",
        "order": order,
    }
)
kernel_mne_butter = filtered_impulse_butter.get_data().flatten()

# --- Plotting ---
fig = plt.figure(figsize=(10, 6), tight_layout=True)

ax1 = plt.subplot(2, 1, 1)
ax1.plot(time_axis_centered, impulse, label='Impulse', color='black', linestyle='-')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")
ax1.grid(visible=True)
ax1.legend()

ax2 = plt.subplot(2, 1, 2)
ax2.plot(time_axis_centered, kernel0, label='Butter (scipy)', color='orange', linestyle='--')
ax2.plot(time_axis_centered, kernel1, label='Cheby (MNE)', color='blue')
ax2.plot(time_axis_centered, kernel_mne_butter, label='Butter (MNE)', color='green', linestyle=':')
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Amplitude")
ax2.grid(visible=True)
ax2.legend()

figure_path = Path(rf"figures/analysis/trying_filter/kernels_{low_cut}_{high_cut}.png")
if not figure_path.exists():
    figure_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)