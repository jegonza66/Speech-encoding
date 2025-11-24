from pathlib import Path
import numpy as np
import mne

import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import rc
import scienceplots
import matplotlib

plt.style.use(['science'])
rc('text', usetex=True)


from utils.general_functions import load_pickle
from utils.processing import band_freq
from load import main_parallel as main_load
from validation import main as main_validation
from main import main as main_main
import config

EEG_PATH = Path(r'data\EEG\S21\s21-1-Trial1-Deci-Filter-Trim-ICA-Pruned.set')
FIG_SAVE_PATH = Path(r'figures\analysis\filter_descrapancy_mne')
AUXILIARY_FILES_PATH = FIG_SAVE_PATH / 'auxiliary_files'
AUXILIARY_FILES_PATH.mkdir(parents=True, exist_ok=True)
SAME_VALIDATION_SUBJECTS = True
EEG_PLOT_STOP_DATA = 2000
FILTER_CONFIGS = [
    ('No filter', {}),
    ('Di Lib.', {
        'method': 'iir', 
        'iir_params': {
            "ftype": "cheby2",
            "order": 4,
            "rs": 20
        },
        'phase': 'zero'}),
    ('MNE 1.6', {'phase': 'minimum-half'}),
    ('MNE 1.9', {'phase': 'minimum'}),
    ('MNE 1.9-zero phase', {'phase': 'zero'})
]

for band in ['Broad', 'Delta', 'Theta', 'Alpha', 'Beta']:
    l_freq, h_freq = band_freq(band)
    raw = mne.io.read_raw_eeglab(
        input_fname=EEG_PATH, 
        preload=True
    )
    eeg_dic = {}        
    for filter_name, filter_params in FILTER_CONFIGS:
        eeg_dic[filter_name] = raw.copy()
        if filter_name != 'No filter':
            eeg_dic[filter_name] = raw.filter(
                l_freq=l_freq, 
                h_freq=h_freq, 
                **filter_params
            )
        eeg_dic[filter_name] = eeg_dic[filter_name].resample(
            sfreq=config.sr, 
            npad=0, 
            window='hamming', 
            method='fft'
        ).get_data().T*1e6  

    # ========================= Filter kernels/impulse responses
    fig_kernels, axes_kernels = plt.subplots(
        nrows=len(eeg_dic), 
        ncols=1, 
        figsize=(10, 8), 
        dpi=600, 
        sharex=True,
        constrained_layout=True
    )
    # Add overall title
    fig_kernels.suptitle(f'Filter Impulse Responses ({band} band: {l_freq}-{h_freq} Hz)', fontsize=14)
    
    # Create impulse signal
    impulse_duration = config.tmax-config.tmin  # seconds
    impulse = np.zeros(len(config.times))
    impulse[config.times==0] = 1.0  # Impulse at 0 seconds

    # Create MNE RawArray for impulse
    info_impulse = mne.create_info(
        ch_names=['impulse'], sfreq=config.sr, ch_types=['eeg']
    )
    impulse_raw = mne.io.RawArray(
        impulse.reshape(1, -1), info_impulse
    )

    # Apply filter to impulse and plot
    for i, (filter_name, filter_params) in enumerate(FILTER_CONFIGS):
        if filter_name == 'No filter':
            filtered_impulse = impulse_raw.copy()
        else:
            filtered_impulse = impulse_raw.copy().filter(
                l_freq=l_freq,
                h_freq=h_freq,
                **filter_params
            )
        
        # Get the filtered impulse response (kernel)
        kernel = filtered_impulse.get_data().flatten()
        
        # Plot kernel
        axes_kernels[i].plot(config.times * 1e3, kernel, 'b-', linewidth=1.5, label=f'{filter_name} kernel')
        
        axes_kernels[i].set_title(f'Filter Kernel: {filter_name}')
        axes_kernels[i].set_ylabel('Amplitude')
        axes_kernels[i].axhline(0, color='k', lw=0.5, ls='--', alpha=0.7)
        axes_kernels[i].axvline(0, color='r', lw=0.5, ls='--', alpha=0.7, label='Impulse position')
        axes_kernels[i].grid(True, alpha=0.3)
        axes_kernels[i].legend(fontsize=10, frameon=False)
        
        # Set x-label only for bottom subplot
        if i == len(eeg_dic):
            axes_kernels[i].set_xlabel('Time (ms)')

    # Save kernels figure
    fig_kernels.savefig(
        FIG_SAVE_PATH / f'filter_kernels_{band}.png', 
        dpi=600, 
        # bbox_inches='tight'
    )

    # Plot the three versions OUTPUT
    fig, axes = plt.subplots(
        nrows=len(eeg_dic), 
        ncols=1, 
        figsize=(8, 6), 
        dpi=600, 
        sharex=True, 
        sharey=True,
        constrained_layout=True
    )
    # Add overall title
    fig.suptitle(f'EEG Signal Filter Comparison ({band} band: {l_freq}-{h_freq} Hz)', fontsize=14)
    
    for i, (eeg_name, eeg_data) in enumerate(eeg_dic.items()):
        # eeg_data = eeg_data.get_data().T*1e6
        axes[i].plot(
            np.arange(eeg_data.shape[0])[:EEG_PLOT_STOP_DATA] / config.sr, 
            eeg_data.mean(axis=1)[:EEG_PLOT_STOP_DATA], 
            lw=0.5,
            label=f'{eeg_name} (mean)',
            color='black'
        )
        axes[i].set_title(eeg_name)
        axes[i].set_ylabel('Amplitude (μV)')
        axes[i].set_xlabel('Time (s)') if i == len(eeg_dic) else axes[i].set_xlabel('')
        axes[i].axhline(0, color='k', lw=0.5, ls='--')
        axes[i].grid(True)

    fig.savefig(
        FIG_SAVE_PATH / f'filter_EEG_{band}.png', 
        dpi=600, 
        # bbox_inches='tight'
    )
    trfs_filters = {}
    correlations_filters = {}
    for i, (filter_name, filter_params) in enumerate(FILTER_CONFIGS):
        if SAME_VALIDATION_SUBJECTS:
            save_path_corr = AUXILIARY_FILES_PATH / 'SAME_VAL' / f'EEG_{band}_{filter_name.replace(" ", "_")}_corr.npy'
            save_path_trf = AUXILIARY_FILES_PATH / 'SAME_VAL' / f'EEG_{band}_{filter_name.replace(" ", "_")}_trf.npy'
        else:
            save_path_corr = AUXILIARY_FILES_PATH / 'DIFF_VAL' / f'EEG_{band}_{filter_name.replace(" ", "_")}_corr.npy'
            save_path_trf = AUXILIARY_FILES_PATH / 'DIFF_VAL' / f'EEG_{band}_{filter_name.replace(" ", "_")}_trf.npy'
        if save_path_corr.exists() and save_path_trf.exists():
            correlations = np.load(save_path_corr)
            correlations_filters[filter_name] = correlations
            trf = np.load(save_path_trf)
            trfs_filters[filter_name] = trf
            continue  
        else:
            load_results = main_load(
                situations=['External'],
                stimuli=['Envelope'],
                bands=['Unfiltered'] if filter_name == 'No filter' else [band],
                # bands=['Broad'],
                filter_eeg=filter_params
            )
            # results['External'][band]['Envelope'][21][0]
            validation_results = main_validation(
                load_results=load_results,
                save_results=False
            )
            main_results = main_main(
                load_results=load_results,
                validation_results=validation_results,
                save_results=False,
                same_validation_subjects=SAME_VALIDATION_SUBJECTS,
            )
            correlations = main_results['External'][
                band if filter_name != 'No filter' else 'Unfiltered'
                ]['Envelope']['average_correlation_subjects']
            trf = main_results['External'][
                band if filter_name != 'No filter' else 'Unfiltered'
                ]['Envelope']['average_weights_subjects']
            save_path_corr.parent.mkdir(parents=True, exist_ok=True)
            save_path_trf.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path_corr, correlations)
            np.save(save_path_trf, trf[:, :, 0, :].mean(axis=0))
            correlations_filters[filter_name] = correlations
            trfs_filters[filter_name] = trf[:, :, 0, :].mean(axis=0)
    
    # Add DiLiberto TRF
    if band in ['Broad', 'Theta']:
        band_di = 'All' if band == 'Broad' else 'Theta'
        trfs_di = load_pickle(
            path=rf"output\mtrf-ridge\DiLib\weights\{band_di}\Envelope\total_weights_per_subject.pkl"
        )['average_weights_subjects'][:, :, 0, :].mean(axis=0)
        correlations_di = load_pickle(
            path=rf"output\mtrf-ridge\DiLib\correlations\{band_di}\Envelope\Envelope.pkl"
        )['average_correlation_subjects']
        trfs_filters['Di Liberto TRF'] = trfs_di
        correlations_filters['Di Liberto TRF'] = correlations_di
    
    # Plot all versions
    fig, axes = plt.subplots(
        nrows=len(trfs_filters), 
        ncols=1, 
        figsize=(10, 8), 
        dpi=600, 
        sharex=True, 
        constrained_layout=True
    )  
    for i, (trfs_name, trfs_data) in enumerate(trfs_filters.items()):
        correlation_mean = correlations_filters[trfs_name].mean()
        correlation_std = correlations_filters[trfs_name].mean(axis=1).std(ddof=1)/np.sqrt(correlations_filters[trfs_name].shape[0])
        trfs_name = rf'{trfs_name} ($\rho$={correlation_mean:.3f}±{correlation_std:.3f})'
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
            units='mTRFs (U.A)',
            axes=axes[i],
        )
        
        # Eliminar la etiqueta "Nave"
        for txt in fig.findobj(mtext.Text):
            if "ave" in txt.get_text():
                    txt.remove()
                
        axes[i].plot(
            config.times*1e3, #ms
            evoked._data.mean(axis=0),
            zorder=130,
            linewidth=1.2,
            label='mean',
            color='black'
        )
        axes[i].set_xlabel('Time (ms)') if i == len(trfs_filters) else axes[i].set_xlabel('')
        axes[i].set_title(trfs_name)
        axes[i].grid(True)
        
        axes[i].legend(
            loc='upper right', 
            fontsize=12, 
            frameon=False
        )
    fig.savefig(
        FIG_SAVE_PATH / f'filter_TRF_{band}_same_val.png' if SAME_VALIDATION_SUBJECTS else FIG_SAVE_PATH / f'filter_TRF_{band}_diff_val.png',
        dpi=600, 
    )
    print('Figures saved in:', FIG_SAVE_PATH)