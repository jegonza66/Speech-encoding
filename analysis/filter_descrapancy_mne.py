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
for band in ['Broad', 'Theta', 'Beta', 'Alpha']:
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
    # axes = axes.flatten()
    axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
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
        rf'figures\analysis\filter_descrapancy_mne\filter_comparison_{band}.png', 
        dpi=600, 
        bbox_inches='tight'
    )

    # TODO faltaría correr nuevamente con los filtros. No lo hago por espacio
    # ========================= Weights comparison
    trfs_path_cheby = Path(rf'output\mtrf-ridge\External\weights\stims_Standarize_EEG_Standarize\distinct_alpha\tmin-0.2_tmax0.6\{band}\Envelope\total_weights_per_subject.pkl')
    # trfs_path_mne16 = Path(rf'output_mne16\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\{band}\Envelope\total_weights_per_subject.pkl')
    # trfs_path_mne19 = Path(rf'output_mne19\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\{band}\Envelope\total_weights_per_subject.pkl')

    trfs_cheby = load_pickle(
        path=trfs_path_cheby
    )['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times
    # trfs_mne16 = load_pickle(
    #     path=trfs_path_mne16
    # )['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times
    # trfs_mne19 = load_pickle(
    #     path=trfs_path_mne19
    # )['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times

    # Plot the three versions
    fig, axes = plt.subplots(
        # nrows=3, 
        nrows=1, 
        ncols=1, 
        figsize=(5, 6), 
        dpi=600, 
        sharex=True, 
        sharey=True,
        tight_layout=True
    )  
    axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()

    # for i, (trfs_name, trfs_data) in enumerate(zip(['Di Lib.', 'MNE 1.6', 'MNE 1.9'], [trfs_cheby, trfs_mne16, trfs_mne19])):
    for i, (trfs_name, trfs_data) in enumerate(zip(['Di Lib.'], [trfs_cheby])):
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
        rf'figures\analysis\filter_descrapancy_mne\trfs_comparison_{band}.png', 
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
    # for i, eeg_name in enumerate(['Di Lib.', 'MNE 1.6', 'MNE 1.9']):
    for i, eeg_name in enumerate(['Di Lib.']):
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
    # for i, (trfs_name, trfs_data) in enumerate(zip(['Di Lib.', 'MNE 1.6', 'MNE 1.9'], [trfs_cheby, trfs_mne16, trfs_mne19])):
    for i, (trfs_name, trfs_data) in enumerate(zip(['Di Lib.'], [trfs_cheby])):
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
        rf'figures\analysis\filter_descrapancy_mne\unified_comparison_{band}.png', 
        dpi=600, 
        bbox_inches='tight'
    )

    plt.close('all')  # Close all figures to free memory

# # # ################# # Trying filters of DILIBERTO 2023 paper
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility

# from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, sosfreqz, sosfilt
# import mne

# # --- 2) Butterworth bandpass filter (1–8 Hz, order 2 each stage) ---
# low_cut = 1
# high_cut = 8
# order = 2
# fs = 512

# # Bandpass with lowpass & highpass order=2 each
# sos = butter(order, [low_cut, high_cut], btype='bandpass', fs=fs, output='sos')

# # --- 3) Visualize the filter kernel (impulse response) ---
# impulse_duration = 0.8  # seconds
# impulse_samples = int(impulse_duration * fs)
# t = np.linspace(-impulse_duration/2, impulse_duration/2, impulse_samples)
# sigma = 0.01  # standard deviation in seconds
# impulse = np.exp(-t**2 / (2 * sigma**2))
# impulse /= impulse.max()  # Normalize area to 1
# kernel0 = sosfiltfilt(sos, impulse)

# # Create MNE RawArray for impulse
# info_impulse = mne.create_info(ch_names=['impulse'], sfreq=fs, ch_types=['eeg'])
# impulse_raw = mne.io.RawArray(impulse.reshape(1, -1), info_impulse)

# # Apply filter to impulse
# filtered_impulse = impulse_raw.copy().filter(
#     l_freq=low_cut,
#     h_freq=high_cut,
#     **{'method': 'iir', 'iir_params': {
#         "ftype": "cheby2",
#         "order": 4,
#         "rs": 20
#         }
#     }
# )

# # Get the filtered impulse response (kernel)
# kernel1 = filtered_impulse.get_data().flatten()
# time_axis = np.arange(len(kernel1)) / fs
# time_axis_centered = time_axis - impulse_duration/2  # Center around 0

# # --- Butterworth kernel using MNE ---
# filtered_impulse_butter = impulse_raw.copy().filter(
#     l_freq=low_cut,
#     h_freq=high_cut,
#     method='iir',
#     iir_params={
#         "ftype": "butter",
#         "order": order,
#     }
# )
# kernel_mne_butter = filtered_impulse_butter.get_data().flatten()

# # --- Plotting ---
# plt.figure(figsize=(10, 6))

# ax1 = plt.subplot(2, 1, 1)
# ax1.plot(time_axis_centered, impulse, label='Impulse', color='black', linestyle='-')
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Amplitude")
# ax1.grid(visible=True)
# ax1.legend()

# ax2 = plt.subplot(2, 1, 2)
# ax2.plot(time_axis_centered, kernel0, label='Butter (scipy)', color='orange', linestyle='--')
# ax2.plot(time_axis_centered, kernel1, label='Cheby (MNE)', color='blue')
# ax2.plot(time_axis_centered, kernel_mne_butter, label='Butter (MNE)', color='green', linestyle=':')
# ax2.set_xlabel("Time [s]")
# ax2.set_ylabel("Amplitude")
# ax2.grid(visible=True)
# ax2.legend()


# plt.tight_layout()

# plt.show()