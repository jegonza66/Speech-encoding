import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rc
import scienceplots
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
plt.style.use(['science'])
rc('text', usetex=True)

import numpy as np
np.random.seed(42)
import mne
import os

from utils.general_functions import load_pickle
from sklearn.model_selection import KFold
from model_implementations import fold_model
import config
from scipy.signal.windows import gaussian

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

# # Remove "future" information 
hypothetical_kernel = hypothetical_kernel[config.times > 0]
# width_samples = 12
# center_idx = np.where(config.delays == 0)[0][0]
# hypothetical_kernel = np.zeros(config.delays.shape)
# gauss = gaussian(M=width_samples, std=width_samples/3)
# start = max(center_idx - width_samples // 2, 0)
# end = start + width_samples
# hypothetical_kernel[start:end] = gauss[:hypothetical_kernel[start:end].shape[0]]
# hypothetical_kernel /= hypothetical_kernel.sum()  # Normalize area to 1
band, filter_stimulus  = 'Theta', False
filter_configs = [
    ('dilib', {'method': 'iir', 'iir_params': {
        "ftype": "cheby2",
        "order": 4,
        "rs": 20
    }}),
    # ('mne16', {'phase': 'minimum-half'}),
    # ('mne19', {'phase': 'minimum'}),
    # ('base', None),
]
(experiment, filter_params) = filter_configs[0]
band_ranges = [(1,8), (2,8), (3,8), (4,8), (1,9), (1,10), (1,11), (1,12), (1,14), (1,15)]
# band_ranges = [(1, 40)]

total_TRFS = []
# Make artificial data
for band_range in band_ranges:
    l_freq, r_freq = band_range
    band = str(band_range[0]) + '-' + str(band_range[1])
    print(f"Creating artificial data for {experiment} experiment with {band} Hz band")

    if filter_stimulus:
        eeg_save_path = rf"saves_simulated\{experiment}_filtered_stim\preprocessed_data\All\tmin-0.2_tmax0.6\EEG\{band}\Causal"
        stimuli_save_path = rf"saves_simulated\{experiment}_filtered_stim\preprocessed_data\All\tmin-0.2_tmax0.6\Envelope"
    else:
        eeg_save_path = rf"saves_simulated\{experiment}\preprocessed_data\All\tmin-0.2_tmax0.6\EEG\{band}\Causal"
        stimuli_save_path = rf"saves_simulated\{experiment}\preprocessed_data\All\tmin-0.2_tmax0.6\Envelope"
    
    os.makedirs(eeg_save_path, exist_ok=True)
    os.makedirs(stimuli_save_path, exist_ok=True)

    original_stimulus = load_pickle(
        path=rf"saves\preprocessed_data\All\tmin-0.2_tmax0.6\Envelope\Sesion{21}.pkl"
    )
    n_samples = original_stimulus[0].shape[0]
    
    stimulus = np.zeros(
        shape=(n_samples, 1),
        dtype=np.float64
    ) 
    
    # Possible percentages of events over the total number of samples
    percentage_of_events = (0.10, 0.40)  
    number_of_events = np.random.randint(
        low=int(n_samples*percentage_of_events[0]), 
        high=int(n_samples*percentage_of_events[1])
    )
    random_indexes = np.random.choice(
        a=range(n_samples), 
        size=number_of_events, 
        replace=False
    )
    stimulus[random_indexes, :] = 1.0
    
    # Filter stimulus
    if filter_stimulus and filter_params is not None:
        stimulus = mne.filter.filter_data(
            data=stimulus.T, 
            sfreq=config.sr, 
            l_freq=l_freq,
            h_freq=r_freq,
            verbose=False,
            **filter_params
        ).T
        
    # Now convolve to get 1 channel EEG signal
    eeg_signal = np.convolve(
        a=stimulus[:, 0], 
        v=hypothetical_kernel[:], # use weights (reverse order as TRFs)
        mode='full'
    )[:stimulus.shape[0]]

    # Add some noise to different channels
    eeg_signal = np.tile(
        A=eeg_signal, 
        reps=(128, 1)
    ).T  
    
    noise = np.random.normal(
        loc=0, 
        scale=0.01, 
        size=eeg_signal.shape
    )
    
    eeg_signal += noise
    
    # Make it mne to apply filtering
    eeg_signal = mne.io.RawArray(
        data=eeg_signal.T, 
        info=config.info_mne,
        verbose=False
    )

    # # First apply broadband filter
    # eeg_signal = eeg_signal.filter(
    #     l_freq=1,
    #     h_freq=40,
    #     verbose=False
    # )
    
    # Then apply specific filter
    if filter_params is not None:
        eeg_signal = eeg_signal.filter(
            l_freq=l_freq,
            h_freq=r_freq,
            verbose=False,
            **filter_params
        )
    eeg_signal = eeg_signal.get_data().T
    
    weights_per_fold = np.zeros((config.n_folds, 128, 1, len(config.delays)), dtype=np.float32)
    correlation_per_channel = np.zeros((config.n_folds, 128))

    # Variable to store all channel's p-value
    topo_pvalues_corr_per_fold = np.zeros((config.n_folds, 128))

    # Set alpha for specific subject
    alpha = 40000

    # Make the Kfold test
    kf_test = KFold(config.n_folds, shuffle=False)
    
    # Run folds
    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(eeg_signal)):
        # Store model output
        output = fold_model(
            fold=fold,
            alpha=alpha,
            stims=stimulus,
            eeg=eeg_signal,
            statistical_test=config.statistical_test,
            relevant_indexes=None,
            train_indexes=train_indexes,
            test_indexes=test_indexes,
            path_null=None,
            validation=False,
            subject=1,                              
            session=21
        )
        fold, weights_per_fold[fold], correlation_per_channel[fold], _ = output[:4]
    
    empty_mask = np.array([np.all(weight == 0) for weight in weights_per_fold])
    if empty_mask.any():
        empty_fold_indices = np.where(empty_mask)[0]
        weights_per_fold[empty_mask] = np.nan
            
    # import IPython; IPython.embed()
    average_weights = np.nanmean(weights_per_fold, axis=0) # info['nchan'], np.sum(n_feats), len(delays)
    average_weights = np.nan_to_num(average_weights).mean(axis=0).mean(axis=0)
    total_TRFS.append(average_weights)       
    # # Take average correlation and RMSE between folds of all channels
    # average_correlation = np.nanmean(correlation_per_channel, axis=0)
    # average_correlation = np.nan_to_num(average_correlation)
    
fig, axes = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(8, 6),
    sharex=True,
    tight_layout=True
)

# First axis: TRFs for Different Bands
colors = plt.cm.berlin(np.linspace(0, 1, len(total_TRFS)))
trf_lines = []
trf_labels = []
for i, (trf, band_range) in enumerate(zip(total_TRFS, band_ranges)):
    line, = axes[0].plot(
        config.times*1e3, 
        trf/trf.max(), 
        color=colors[i], 
        label=f'Band {band_range}'
    )
    trf_lines.append(line)
    trf_labels.append(f'Band {band_range}')
axes[0].set_ylabel('TRF Amplitude')
axes[0].set_title('TRFs for Different Bands')
# Split legend into two columns
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

kernel_lines = []
kernel_labels = []
for i, band_range in enumerate(band_ranges):
    l_freq, r_freq = band_range
    filtered_impulse = impulse_raw.copy().filter(
            l_freq=l_freq,
            h_freq=r_freq,
            **filter_params
        )
    kernel = filtered_impulse.get_data().flatten()
    line, = axes[1].plot(config.times*1e3, kernel/kernel.max(), color=colors[i], label=f'Band {band_range}')
    kernel_lines.append(line)
    kernel_labels.append(f'Band {band_range}')

axes[1].set_ylabel('Filter Kernel')
axes[1].set_title('Filter Kernels (Delta Response)')
# Split legend into two columns
axes[1].legend(kernel_lines, kernel_labels, ncol=2, loc='upper right', fontsize='small')
axes[1].grid(True)

# Third axis: Hypothetical kernel
hypothetical_kernel = load_pickle(
    path=rf'output\DiLib\weights\All\Envelope\total_weights_per_subject.pkl'
)['average_weights_subjects'].mean(axis=0).mean(axis=1)[selection, :].mean(axis=0)

axes[2].plot(config.times*1e3, hypothetical_kernel, color='black')
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Hypothetical Kernel')
axes[2].set_title('Hypothetical Kernel')
axes[2].grid(True)

os.makedirs(rf'figures\analysis\simulations', exist_ok=True)
fig.savefig(
    rf'figures\analysis\simulations\different_bands.png',
    dpi=600
)
fig.show()

# ===== CONCLUSIONES
# 0. El ajuste de las TRFs, si se filtra el EEG, da como resultado la convolución de la verdadera TRF con el kernel del filtro, 
# como es de esperarse, ya que se trata de un modelo lineal.# Por lo cual, hay que ser cauteloso al interpretar la forma de las
# curvas. 
# Para los datos de DiLiberto se observa que la TRF ajusta el ERP esperado, mientras que nuestros datos se ajustan mucho al filtro.
# Nuestra verdadera TRF (ajustada sin filtro) es una gaussiana centrada en 0.