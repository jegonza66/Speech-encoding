from pathlib import Path
import numpy as np
import mne

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib
matplotlib.use('Agg')  


from utils.general_functions import load_pickle
from utils.processing import band_freq, subsample
import config

# ========================= Filter comparison
eeg_path = Path(r'data\EEG\S21\s21-1-Trial1-Deci-Filter-Trim-ICA-Pruned.set')
band = 'All'

l_freq, h_freq = band_freq(band)

raw = mne.io.read_raw_eeglab(
    input_fname=eeg_path, 
    preload=True
)
eeg_dic = {}        
if band:
    eeg_dic['MNE 1.9'] = raw.copy().filter(
        l_freq=l_freq, 
        h_freq=h_freq, 
        phase='minimum'
    )
    eeg_dic['MNE 1.6'] = raw.copy().filter(
        l_freq=l_freq, 
        h_freq=h_freq, 
        phase='minimum-half'
    )
    iir_params = {
    "ftype": "cheby2",       # Filter type: Chebyshev Type II
    "order": 4,              # Filter order
    "rs": 20,                # Stopband attenuation (dB)
    }
    eeg_dic['Di Lib.'] = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params
    )

for eeg_name in eeg_dic:
    eeg_dic[eeg_name] = eeg_dic[eeg_name].get_data().T*1e6  
    eeg_dic[eeg_name] = subsample(
        x=eeg_dic[eeg_name], 
        step=int(raw.info.get("sfreq")/ config.sr)
    )

# ========================= Filter kernels/impulse responses
fig_kernels, axes_kernels = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(10, 8), 
    dpi=600, 
    # sharex=True,
    tight_layout=True
)
axes_kernels = axes_kernels

# Create impulse signal
impulse_duration = 6.0  # seconds
impulse_samples = int(impulse_duration * raw.info['sfreq'])
impulse = np.zeros(impulse_samples)
impulse[impulse_samples//2] = 1.0  # Impulse at the center

# Create MNE RawArray for impulse
info_impulse = mne.create_info(ch_names=['impulse'], sfreq=raw.info['sfreq'], ch_types=['eeg'])
impulse_raw = mne.io.RawArray(impulse.reshape(1, -1), info_impulse)

# Apply each filter to the impulse and plot the kernel
filter_configs = [
    ('Di Lib.', {'method': 'iir', 'iir_params': {
        "ftype": "cheby2",
        "order": 4,
        "rs": 20
    }}),
    ('MNE 1.6', {'phase': 'minimum-half'}),
    ('MNE 1.9', {'phase': 'minimum'}),
]

for i, (filter_name, filter_params) in enumerate(filter_configs):
    # Apply filter to impulse
    filtered_impulse = impulse_raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        **filter_params
    )
    
    # Get the filtered impulse response (kernel)
    kernel = filtered_impulse.get_data().flatten()
    kernel_transform = np.log(np.abs(np.fft.rfft(kernel))/len(kernel)*2)
    freq = np.fft.fftfreq(len(kernel), d=1/raw.info['sfreq'])[:len(kernel)//2 + 1]
    
    # axes_kernels[3].plot(freq[:-2], kernel_transform[:-2], label=filter_name, color=f'C{i}', lw=1.5)
    # axes_kernels[3].legend(loc='upper right', fontsize=10, frameon=False)
    # axes_kernels[3].set_xlim(0,12)
    # Create time axis
    time_axis = np.arange(len(kernel)) / raw.info['sfreq']
    time_axis_centered = time_axis - impulse_duration/2  # Center around 0
    
    # Plot kernel
    ax = axes_kernels[i]
    ax.plot(time_axis_centered * 1000, kernel, 'b-', linewidth=1.5, label=f'{filter_name} kernel')
    ax.plot(kernel, 'b-', linewidth=1.5, label=f'{filter_name} kernel')
    
    ax.set_title(f'Filter Kernel: {filter_name}')
    ax.set_ylabel('Amplitude')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.7)
    ax.axvline(0, color='r', lw=0.5, ls='--', alpha=0.7, label='Impulse position')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    ax.set_xticks(np.arange(-200, 700, 100))
    
    # Set x-label only for bottom subplot
    if i == 2:
        ax.set_xlabel('Time (ms)')
    
    # Zoom in to see the kernel better (adjust window as needed)
    window_ms = 600  # Show ±200ms around the impulse
    ax.set_xlim(-window_ms, window_ms)

# Add overall title
fig_kernels.suptitle(f'Filter Impulse Responses ({band} band: {l_freq}-{h_freq} Hz)', fontsize=14, y=0.98)

# Save kernels figure
fig_kernels.savefig(
    rf'figures\analysis\filter_descrapancy_mne\filter_kernels_{band}.png', 
    dpi=600, 
    bbox_inches='tight'
)

# Plot the three versions OUTPUT
fig, axes = plt.subplots(
    nrows=4, 
    ncols=1, 
    figsize=(8, 6), 
    dpi=600, 
    sharex=True, 
    sharey=True,
    tight_layout=True
)
axes = axes.flatten()
stop_data = 2000
ax = axes[0]
eeg_name = 'Raw EEG'
eeg_data = raw.get_data().T*1e6
ax.plot(
    np.arange(eeg_data.shape[0])[:stop_data] / config.sr, 
    eeg_data.mean(axis=1)[:stop_data], 
    lw=0.5,
    label=f'{eeg_name} (mean)',
    color='black'
)
ax.set_title(eeg_name)
ax.set_ylabel('Amplitude (μV)')
ax.set_xlabel('')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.grid(True)

for i, (eeg_name, eeg_data) in enumerate(eeg_dic.items()):
    ax = axes[i+1]
    ax.plot(
        np.arange(eeg_data.shape[0])[:stop_data] / config.sr, 
        eeg_data.mean(axis=1)[:stop_data], 
        lw=0.5,
        label=f'{eeg_name} (mean)',
        color='black'
    )
    ax.set_title(eeg_name)
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlabel('Time (s)') if i == 2 else ax.set_xlabel('')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.grid(True)

fig.savefig(
    rf'figures\analysis\filter_descrapancy_mne\filter_comparison.png', 
    dpi=600, 
    bbox_inches='tight'
)

# ========================= Weights comparison
trfs_path_cheby = Path(r'output_cheby\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\Theta\Spectrogram\total_weights_per_subject.pkl')
trfs_path_mne16 = Path(r'output_mne16\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\Theta\Spectrogram\total_weights_per_subject.pkl')
trfs_path_mne19 = Path(r'output_mne19\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\Theta\Spectrogram\total_weights_per_subject.pkl')

trfs_cheby = load_pickle(
    path=trfs_path_cheby
)['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times
trfs_mne16 = load_pickle(
    path=trfs_path_mne16
)['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times
trfs_mne19 = load_pickle(
    path=trfs_path_mne19
)['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times

# Plot the three versions
fig, axes = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(5, 6), 
    dpi=600, 
    sharex=True, 
    sharey=True,
    tight_layout=True
)  
axes = axes.flatten()

for i, (trfs_name, trfs_data) in enumerate(zip(['Di Lib.', 'MNE 1.6', 'MNE 1.9'], [trfs_cheby, trfs_mne16, trfs_mne19])):
    evoked = mne.EvokedArray(
        data=trfs_data, 
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
        units='mTRFs (U.A)',
        axes=axes[i],
    )
    
    # Eliminar la etiqueta "Nave"
    for txt in fig.findobj(mtext.Text):
        if "ave" in txt.get_text():
                txt.remove()
            
    # axes[i].plot(
    #     config.times*1e3, #ms
    #     evoked._data.mean(axis=0),
    #     zorder=130,
    #     linewidth=1.2,
    #     label='mean',
    #     color='black'
    # )
    axes[i].set_xlabel('Time (ms)') if i >= 2 else axes[i].set_xlabel('')
    axes[i].set_title(trfs_name)
    axes[i].grid(True)
    
    axes[i].legend(
        loc='upper right', 
        fontsize=12, 
        frameon=False
    )
fig.savefig(
    rf'figures\analysis\filter_descrapancy_mne\trfs_comparison.png', 
)

# ========================= Unified comparison plot
fig_unified, axes_unified = plt.subplots(
    nrows=3, 
    ncols=2, 
    figsize=(16, 8), 
    dpi=600, 
    tight_layout=True
)

# Left column: Filter comparison
stop_data = 500
for i, eeg_name in enumerate(['Di Lib.', 'MNE 1.6', 'MNE 1.9']):
    eeg_data = eeg_dic[eeg_name]
    ax = axes_unified[i, 0]
    ax.plot(
        np.arange(eeg_data.shape[0])[:stop_data] / config.sr, 
        eeg_data.mean(axis=1)[:stop_data], 
        lw=0.5,
        label=f'{eeg_name} (mean)',
        color='black'
    )
    ax.set_title(f'Filter: {eeg_name}')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlabel('Time (s)') if i == 2 else ax.set_xlabel('')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.grid(True)
    ax.legend(
        loc='upper right', 
        fontsize=10, 
        frameon=False
    )

# Right column: TRFs comparison
for i, (trfs_name, trfs_data) in enumerate(zip(['Di Lib.', 'MNE 1.6', 'MNE 1.9'], [trfs_cheby, trfs_mne16, trfs_mne19])):
    ax = axes_unified[i, 1]
    
    evoked = mne.EvokedArray(
        data=trfs_data, 
        info=config.info_mne
    )     
    evoked.shift_time(
        config.times[0], 
        relative=True
    )
    
    # Plot individual channels with spatial colors
    evoked_plot = evoked.plot(
        scalings={'eeg':1},
        zorder='std',
        time_unit='ms',
        show=False,
        spatial_colors=True,
        units='mTRFs (U.A)',
        axes=ax,
    )
    
    # Remove "ave" labels
    for txt in fig_unified.findobj(mtext.Text):
        if "ave" in txt.get_text():
            txt.remove()
            
    # Plot mean across channels
    ax.plot(
        config.times*1e3, #ms
        evoked._data.mean(axis=0),
        zorder=130,
        linewidth=1.2,
        label='mean',
        color='black'
    )
    
    ax.set_xlabel('Time (ms)') if i == 2 else ax.set_xlabel('')
    ax.set_title(f'TRFs: {trfs_name}')
    ax.grid(True)
    
    ax.legend(
        loc='upper right', 
        fontsize=10, 
        frameon=False
    )

# Add overall title
fig_unified.suptitle('Filter and TRFs Comparison Across Methods', fontsize=16, y=0.98)

# Save unified figure
fig_unified.savefig(
    rf'figures\analysis\filter_descrapancy_mne\unified_comparison.png', 
    dpi=600, 
    bbox_inches='tight'
)

plt.close('all')  # Close all figures to free memory

# ========================= Shuffle EEG data in 800ms windows

def shuffle_eeg_windows(raw_data, window_duration_ms=800, sfreq=512):
    """
    Shuffle EEG data in windows of specified duration
    
    Parameters:
    - raw_data: MNE Raw object
    - window_duration_ms: Window duration in milliseconds
    - sfreq: Sampling frequency
    
    Returns:
    - shuffled_raw: MNE Raw object with shuffled windows
    """
    # Get data
    data = raw_data.get_data()  # Shape: (n_channels, n_samples)
    n_channels, n_samples = data.shape
    
    # Calculate window size in samples
    window_samples = int(window_duration_ms * sfreq / 1000)
    
    # Calculate number of complete windows
    n_windows = n_samples // window_samples
    remainder_samples = n_samples % window_samples
    
    print(f"    Total samples: {n_samples}")
    print(f"    Window size: {window_samples} samples ({window_duration_ms} ms)")
    print(f"    Complete windows: {n_windows}")
    print(f"    Remainder samples: {remainder_samples}")
    
    # Create list of all windows (including remainder if exists)
    windows = []
    
    # Add complete windows
    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = (i + 1) * window_samples
        windows.append(data[:, start_idx:end_idx])
    
    # Add remainder window if exists
    if remainder_samples > 0:
        windows.append(data[:, n_windows * window_samples:])
    
    # Shuffle the windows
    np.random.shuffle(windows)
    
    # Concatenate shuffled windows
    shuffled_data = np.concatenate(windows, axis=1)
    
    # Create new Raw object with shuffled data
    shuffled_raw = mne.io.RawArray(
        data=shuffled_data,
        info=raw_data.info.copy()
    )
    
    return shuffled_raw

# Create base directory for shuffled data
shuffled_data_base = Path(r'data\EEG_SHUFFLED_DATA')
shuffled_data_base.mkdir(exist_ok=True)

# Get all subject folders from original EEG data
eeg_data_path = Path(r'data\EEG')
subject_folders = [f for f in eeg_data_path.iterdir() if f.is_dir()]

print(f"Found {len(subject_folders)} subject folders: {[f.name for f in subject_folders]}")

# Process each subject and trial
for subject_folder in subject_folders:
    subject_name = subject_folder.name
    print(f"Processing subject: {subject_name}")
    
    # Create subject folder in shuffled data
    shuffled_subject_path = shuffled_data_base / subject_name
    shuffled_subject_path.mkdir(exist_ok=True)
    
    # Find all .set files for this subject
    set_files = list(subject_folder.glob("*.set"))
    print(f"  Found {len(set_files)} .set files")
    
    for set_file in set_files:
        print(f"  Shuffling data for: {set_file.name}")
        
        # Load original file
        original_raw = mne.io.read_raw_eeglab(
            input_fname=set_file,  # Load each individual file
            preload=True
        )
        
        print(f"    Original duration: {original_raw.n_times / original_raw.info['sfreq']:.2f} seconds")
        print(f"    Original samples: {original_raw.n_times}")
        print(f"    Sampling frequency: {original_raw.info['sfreq']} Hz")
        
        # Shuffle data in 800ms windows
        shuffled_raw = shuffle_eeg_windows(
            raw_data=original_raw,
            window_duration_ms=800,
            sfreq=original_raw.info['sfreq']
        )
        
        # Create output filename
        output_filename = set_file.stem + "_eeg.fif"
        output_path = shuffled_subject_path / output_filename
        
        # Save shuffled data as .fif format (more efficient)
        shuffled_raw.save(
            fname=str(output_path),
            overwrite=True
        )
        
        print(f"    Saved: {output_filename}")
        print(f"    Shuffled duration: {shuffled_raw.n_times / shuffled_raw.info['sfreq']:.2f} seconds")
        print(f"    Shuffled samples: {shuffled_raw.n_times}")
        print()

print("\nEEG data shuffling completed!")
print(f"All shuffled files saved to: {shuffled_data_base}")

# Verify the structure matches
print(f"\nVerification - Original vs Shuffled structure:")
for subject_folder in subject_folders:
    original_files = len(list(subject_folder.glob("*.set")))
    shuffled_files = len(list((shuffled_data_base / subject_folder.name).glob("*.fif")))
    print(f"{subject_folder.name}: Original={original_files}, Shuffled={shuffled_files}")

# ========================= Plot comparison of original vs shuffled data
print("\nCreating comparison plot...")

# Load one example for comparison
example_file = list(subject_folders[0].glob("*.set"))[0]
original_example = mne.io.read_raw_eeglab(input_fname=example_file, preload=True)
shuffled_example_path = shuffled_data_base / subject_folders[0].name / (example_file.stem + "_eeg.fif")
shuffled_example = mne.io.read_raw_fif(shuffled_example_path, preload=True)

# Plot comparison
fig_comp, axes_comp = plt.subplots(2, 1, figsize=(15, 8), dpi=300)

# Plot settings
plot_duration = 10  # seconds to plot
plot_samples = min(int(plot_duration * original_example.info['sfreq']), original_example.n_times)
time_axis = np.arange(plot_samples) / original_example.info['sfreq']

# Original data
axes_comp[0].plot(
    time_axis,
    (original_example.get_data()[:, :plot_samples] * 1e6).mean(axis=0),
    'b-', linewidth=0.8, label='Original EEG (mean across channels)'
)
axes_comp[0].set_title(f'Original EEG Data - {example_file.name}')
axes_comp[0].set_ylabel('Amplitude (µV)')
axes_comp[0].grid(True, alpha=0.3)
axes_comp[0].legend()
axes_comp[0].axhline(0, color='k', lw=0.5, ls='--', alpha=0.7)

# Shuffled data
axes_comp[1].plot(
    time_axis,
    (shuffled_example.get_data()[:, :plot_samples] * 1e6).mean(axis=0),
    'r-', linewidth=0.8, label='Shuffled EEG (800ms windows, mean across channels)'
)
axes_comp[1].set_title('Shuffled EEG Data (800ms windows)')
axes_comp[1].set_xlabel('Time (s)')
axes_comp[1].set_ylabel('Amplitude (µV)')
axes_comp[1].grid(True, alpha=0.3)
axes_comp[1].legend()
axes_comp[1].axhline(0, color='k', lw=0.5, ls='--', alpha=0.7)

# Add vertical lines to show 800ms windows in shuffled data
for i in range(int(plot_duration * 1000 / 800)):
    window_time = i * 0.8
    if window_time <= plot_duration:
        axes_comp[1].axvline(window_time, color='g', lw=1, ls=':', alpha=0.5)

plt.tight_layout()

# Save comparison plot
fig_comp.savefig(
    rf'figures\analysis\filter_descrapancy_mne\original_vs_shuffled_comparison.png',
    dpi=600,
    bbox_inches='tight'
)

print(f"Comparison plot saved to: figures\\analysis\\filter_descrapancy_mne\\original_vs_shuffled_comparison.png")

plt.close('all')


# ========================= Reverse audios time
import scipy.io.wavfile as wavfile
import pandas as pd

wavs_path = Path(r'data\wavs')
reversed_wavs_path = Path(r'data\wavs_reversed')
reversed_wavs_path.mkdir(exist_ok=True, parents=True)

# Get all subject folders from original WAV data
subject_folders = [f for f in wavs_path.iterdir() if f.is_dir() and f.name!='Sin separar canales']

# Copy files in reverse order
for i, subject_folder in enumerate(subject_folders):
    subject_name = subject_folder.name
    print(f"Processing subject: {subject_name}")
    
    # Create subject folder in reversed data
    reversed_subject_path = reversed_wavs_path / subject_folders[i].name
    reversed_subject_path.mkdir(exist_ok=True)
    
    # Find all .wav files for this subject
    wav_files = list(subject_folder.glob("*.wav"))
    print(f"  Found {len(wav_files)} .wav files")
    
    for wav_file in wav_files:
        print(f"  Copying: {wav_file.name}")
        sr, wav = wavfile.read(filename=wav_file)  
        
        # Save reversed audio
        wav_reversed = wav[::-1]
        wavfile.write(
            filename=reversed_subject_path / wav_file.name, 
            rate=sr, 
            data=wav_reversed
        )
        
# Now reverse phrases to identify moments of interest
phrases_path = Path(r'data\phrases')
phrases_reversed_path = Path(r'data\phrases_reversed')
phrases_reversed_path.mkdir(exist_ok=True, parents=True)

subject_folders = [f for f in phrases_path.iterdir() if f.is_dir()]
for subject_folder in subject_folders:
    
    phrases_files = list(subject_folder.glob("*.phrases"))
    
    for phrases_file in phrases_files:

        # Read phrases into pandas.DataFrame
        table = pd.read_table(
            phrases_file, 
            header=None, 
            sep="\t"
        )
        max_time = table[1].max()
        table_reversed = pd.DataFrame(table.sort_index(ascending=False).values)
        table_reversed[1] = (max_time - table[0].values[::-1])
        table_reversed[0] = (max_time - table[1].values[::-1])
        
        Path(phrases_reversed_path / phrases_file.parent.stem).mkdir(
            exist_ok=True, 
            parents=True
        )
        table_reversed.to_csv(
            str(phrases_reversed_path / phrases_file.parent.stem / phrases_file.stem) + ".phrases", 
            sep="\t", 
            header=False, 
            index=False
        )
