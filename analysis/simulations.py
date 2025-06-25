import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import numpy as np
np.random.seed(42)
import mne
import os

# from ..utils.general_functions import load_pickle
from utils.general_functions import load_pickle, dump_pickle
from utils.processing import band_freq
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

# Remove "future" information 
hypothetical_kernel = hypothetical_kernel[config.times > 0]
band  = 'All'
filter_configs = [
    ('dilib', {'method': 'iir', 'iir_params': {
        "ftype": "cheby2",
        "order": 4,
        "rs": 20
    }}),
    ('mne16', {'phase': 'minimum-half'}),
    ('mne19', {'phase': 'minimum'}),
    ('base', None),
]

# Make artificial data
for i, (experiment, filter_params) in enumerate(filter_configs):
    print(f"Creating artificial data for {experiment} experiment with {band} band")
    eeg_save_path = rf"saves_simulated\{experiment}\preprocessed_data\All\tmin-0.2_tmax0.6\EEG\{band}\Causal"
    stimuli_save_path = rf"saves_simulated\{experiment}\preprocessed_data\All\tmin-0.2_tmax0.6\Envelope"
    os.makedirs(eeg_save_path, exist_ok=True)
    os.makedirs(stimuli_save_path, exist_ok=True)
    stimuli = {session:[] for session in config.sessions}
    eeg_signals = {session:[] for session in config.sessions}
    for session in tqdm(config.sessions, desc="Creating artificial data", total=len(config.sessions)):
        for subject in [0, 1]:
            original_stimulus = load_pickle(
                path=rf"saves\preprocessed_data\All\tmin-0.2_tmax0.6\Envelope\Sesion{session}.pkl"
            )
            n_samples = original_stimulus[subject].shape[0]
            
            stimulus = np.zeros(
                shape=(n_samples, 1)
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
            stimulus[random_indexes, :] = 1
            stimuli[session].append(stimulus)
            
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
                scale=0.35, 
                size=eeg_signal.shape
            )
            
            eeg_signal += noise
            
            # Make it mne to apply filtering
            eeg_signal = mne.io.RawArray(
                data=eeg_signal.T, 
                info=config.info_mne,
                verbose=False
            )

            # First apply broadband filter
            eeg_signal = eeg_signal.filter(
                l_freq=1,
                h_freq=40,
                verbose=False
            )
            
            # Then apply specific filter
            if filter_params is not None:
                l_freq, r_freq = band_freq(band)
                eeg_signal = eeg_signal.filter(
                    l_freq=l_freq,
                    h_freq=r_freq,
                    verbose=False,
                    **filter_params
                )
            eeg_signal = eeg_signal.get_data().T

            eeg_signals[session].append(eeg_signal)

    # Save simulated data
    for session in tqdm(config.sessions, desc="Saving simulated data", total=len(config.sessions)):
        dump_pickle(
            path=os.path.join(eeg_save_path, rf'Sesion{session}.pkl'),
            obj=eeg_signals[session],
            rewrite=True
        )
        dump_pickle(
            path=os.path.join(stimuli_save_path, rf'Sesion{session}.pkl'),
            obj=stimuli[session],
            rewrite=True
        )
        
# Print names of experiments
print("Experiments with simulated data:")
for experiment in filter_configs:
    print(experiment[0])