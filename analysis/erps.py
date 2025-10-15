"""
Make ERPs paradigm

⦁	ERPs a cambio de turno
 	--> Hacer en formato Gridsearch filtrado en bandas
 	--> Tomar (-1, .25) s respecto al onset del cambio de turno
 	--> ERP_hold - ERP_switch


"""
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import mne
mne.set_log_level(verbose='CRITICAL')

from utils.processing import band_freq
from utils.general_functions import load_pickle
import config

save_path = Path(f"figures/analysis/turn_taking_ERPs/")
save_path.mkdir(parents=True, exist_ok=True)
base_path = Path(rf"data")

for l_freq_eeg, h_freq_eeg in [(1,4), (1,8), (1,15), (4,8), (8,15), (15,30)]:
    erps = []
    erps_hold = []
    for session in tqdm(config.sessions, desc=f"Processing Band {l_freq_eeg}-{h_freq_eeg} Hz", total=len(config.sessions)):
        for channel in [1, 2]:

            # Load EEG data
            eeg_session_dir = base_path / f"EEG/S{session}"
            trials_path = [
                trial for trial in eeg_session_dir.iterdir() 
                if trial.suffix == '.set' and f"-{channel}-" in trial.name
            ]
            eeg = []
            turn_onsets = []
            hold_onsets = []
            
            samples_info_path = rf"saves\preprocessed_data\tmin-0.2_tmax0.6\samples_info\External\samples_info_{session}.pkl"
            channel_inter = 1 if channel == 2 else 2
            trial_lengths = load_pickle(
                path = samples_info_path
            )['trial_lengths'+str(channel_inter)]

            for trial, trial_path in enumerate(trials_path):
                raw = mne.io.read_raw_eeglab(
                    input_fname=trial_path, 
                    preload=True
                )
                raw = raw.filter(
                    l_freq=l_freq_eeg,
                    h_freq=h_freq_eeg,
                    method="iir",
                    iir_params={
                    "ftype": "cheby2",       # Filter type: Chebyshev Type II # También "ftype": "butter",
                    "order": 4,              # Filter order # Cuyo caso order 2
                    "rs": 20,                # Stopband attenuation (dB)
                    }
                )
                # raw = raw.resample(
                #     sfreq=config.sr, 
                #     npad=0, 
                #     window='hamming', 
                #     method='fft'
                # )
                raw = raw.get_data().T*1e6 
                # from IPython import embed; embed() 
                # raw = raw[:int(trial_lengths[trial-1]*512/128)] # trim to original length
                
                # Get onset times
                turn_path = base_path / "turns/switches_external"/ f"sess_{session}_trial_{trial+1:02}_ch_{channel}.json"
                turn_path_holds = base_path / "turns/holds_external"/ f"sess_{session}_trial_{trial+1:02}_ch_{channel}.json"
                offset = sum(trial_lengths[:trial])/128
                
                with open(turn_path, 'r') as f:
                    turn_times = json.load(f)
                for turn in turn_times:
                    turn_onsets.append(
                        int((turn['ipu1_start_time'] + offset)*512)
                    )
                with open(turn_path_holds, 'r') as f:
                    hold_times = json.load(f)
                for hold in hold_times:
                    hold_onsets.append(
                        int((hold['ipu1_start_time'] + offset)*512)
                    )
                eeg.append(raw)
            eeg = np.concatenate(eeg, axis=0)
            
            # # Load corresponding stimulus
            # envelope_session_dir = Path(rf"saves\preprocessed_data\tmin-0.2_tmax0.6\Envelope\Sesion{session}.pkl")
            # envelope = load_pickle(
            #     path = envelope_session_dir
            # )[0].reshape(-1)+load_pickle(
            #     path = envelope_session_dir
            # )[1].reshape(-1)
            # from IPython import embed; embed()
            
            # from scipy.signal import savgol_filter
            # envelope_smooth = savgol_filter(envelope, window_length=200, polyorder=3)
            # env_mean = envelope_smooth.mean()
            # env_std = envelope_smooth.std()
            # threshold = env_mean + 1 * env_std

            # # Z-score normalization
            # z_envelope = (envelope_smooth - env_mean) / env_std

            # # Parámetros robustos
            # min_event_duration = int(0.2 * config.sr)  # 200 ms
            # refractory_period = int(0.250 * config.sr)    # 250 ms

            # onsets = []
            # i = 1
            # while i < len(z_envelope):
            #     if z_envelope[i-1] < 2 and z_envelope[i] >= 2:
            #         # Busca el final del evento
            #         start = i
            #         while i < len(z_envelope) and z_envelope[i] >= 2:
            #             i += 1
            #         end = i
            #         # Solo cuenta si dura lo suficiente
            #         if end - start >= min_event_duration:
            #             onsets.append(start)
            #         # Aplica periodo refractario
            #         i += refractory_period
            #     else:
            #         i += 1
            # onsets = np.array(onsets)
            
            # # Plot envelope and onset to check validity #QUEDA MUUY MASO MENOS
            # plt.figure(figsize=(15, 5))
            # time = np.arange(envelope.shape[0])/config.sr
            # # plt.plot(time, envelope, label='Envelope', alpha=0.2)
            # plt.plot(time, envelope, label='Envelope (smoothed)', alpha=0.8)
            # plt.vlines(np.cumsum(time[trial_lengths]), 
            #            ymin=envelope.min(), ymax=envelope.max(), color='orange', linestyle='--', label='Annotated starts')
            # plt.scatter(time[turn_onsets], envelope[turn_onsets], color='red', label='Onsets (2σ)')
            # plt.title(f'Session {session} Channel {channel} - Envelope with Onsets')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude (µV)')
            # plt.legend()
            # plt.show()
            
            # Extract epochs
            # times = -(config.times)[::-1] # --> se ve un pico en menos -400ms (que será?)
            # times = config.times
            
            # Extract epochs
            times = np.arange(-1, 1.25 + 1/512, 1/512)
            epochs = np.zeros(
                (len(turn_onsets), len(times), eeg.shape[1]) # (n_epochs, n_times, n_channels
            ) * np.nan
            epochs_hold = np.zeros(
                (len(hold_onsets), len(times), eeg.shape[1]) # (n_epochs, n_times, n_channels
            ) * np.nan
            
            valid_epochs = []
            for idx, onset in enumerate(turn_onsets):
                # start = onset + int(times[0] * 512) -1
                # end = onset + int(times[-1] * 512) + 1
                start = onset + int(times[0] * 512) - 1
                end = start + len(times)
                if start >= 0 and end <= eeg.shape[0]:
                    epochs[idx] = eeg[start:end]
                    valid_epochs.append(idx)
            epochs = epochs[valid_epochs]
            valid_epochs = []
            for idx, onset in enumerate(hold_onsets):
                start = onset + int(times[0] * 512) - 1
                end = start + len(times)
                if start >= 0 and end <= eeg.shape[0]:
                    epochs_hold[idx] = eeg[start:end]
                    valid_epochs.append(idx)
            epochs_hold = epochs_hold[valid_epochs]

            print(f"\n\tExtracted {len(epochs)} valid epochs for Session {session} Channel {channel}\n")
            
            # Extract ERP
            erp = np.nanmean(np.nanmean(epochs, axis=0), axis=1)
            erps.append(erp)
            erp_hold = np.nanmean(np.nanmean(epochs_hold, axis=0), axis=1)
            erps_hold.append(erp_hold)

    # Average across sessions and channels
    erp = np.nanmean(
        np.array(erps), axis=0
    )
    erp_hold = np.nanmean(
        np.array(erps_hold), axis=0
    )

    # Plot ERP
    plt.figure(figsize=(5, 5))
    plt.plot(times[::-1], erp-erp_hold, label='ERP', alpha=0.8)

    plt.title(f'Session {session} Channel {channel} - ERP - Band {l_freq_eeg}, {h_freq_eeg} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')

    plt.grid(visible=True)
    plt.legend()

    plt.savefig(save_path / rf"ERP_{l_freq_eeg}_{h_freq_eeg}.png", dpi=300)
    # plt.show(block=False)