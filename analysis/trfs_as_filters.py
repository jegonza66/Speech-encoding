from utils.general_functions import load_pickle
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.use('TkAgg')  
# matplotlib.use('Qt5Agg')


from scipy.fft import fft
import numpy as np
# import Path
import mne
import config
stimuli = [
    "Envelope",
    "Pitch-Log-Raw",
    "Spectrogram",
    "Phonological",
]
bands = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta1",
    "Beta2",
    "All"
]

trfs_stim_path = lambda stim: lambda band: rf'output\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\{band}\{stim}\total_weights_per_subject.pkl'
for stim in stimuli:
    trfs_path = trfs_stim_path(stim)

    # Create plots
    fig, axes = plt.subplots(
        ncols=2, 
        nrows=2, 
        figsize=(12, 8),
        tight_layout=True
        )
    fig.suptitle(f'TRFs and their FFTs for {stim}', fontsize=16)

    for b, band in enumerate(bands):
        trfs = load_pickle(path=trfs_path(band))['average_weights_subjects']
        trf = trfs.mean(axis=0).mean(axis=0).mean(axis=0)

        # Compute the FFT of the TRF
        trf_fft = fft(trf)
        trf_magnitude = np.abs(trf_fft)
        trf_phase = np.angle(trf_fft)

        # Create frequency axis
        n_samples = len(trf)
        sample_rate = 1 / (config.times[1] - config.times[0])  
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)

        # Only keep positive frequencies for plotting
        n_pos = n_samples // 2
        freqs_pos = freqs[:n_pos]
        trf_magnitude_pos = trf_magnitude[:n_pos]
        trf_phase_pos = trf_phase[:n_pos]

        # Plot original TRF
        axes[0, 0].plot(config.times, trf, label=f'{band}', color=f'C{b}')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Original TRF')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        

        # Plot magnitude spectrum
        axes[0, 1].plot(freqs_pos, trf_magnitude_pos, label=f'{band}', color=f'C{b}')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Magnitude Spectrum')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # Plot phase spectrum
        axes[1, 0].plot(freqs_pos, trf_phase_pos, label=f'{band}', color=f'C{b}')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Phase (radians)')
        axes[1, 0].set_title('Phase Spectrum')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # Plot magnitude spectrum in dB
        axes[1, 1].plot(freqs_pos, 20 * np.log10(trf_magnitude_pos + 1e-12), label=f'{band}', color=f'C{b}')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].set_title('Magnitude Spectrum (dB)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
    fig.show()

    # TRF comparison
    fig, axes = plt.subplots(
        ncols=2,
        nrows=len(bands),
        figsize=(8, 8),
        tight_layout=True
    )
    fig.suptitle(f'TRFs and their FFTs for {stim}', fontsize=16)
    for b, band in enumerate(bands):
        trfs = load_pickle(path=trfs_path(band))['average_weights_subjects']
        trf = trfs.mean(axis=0).mean(axis=0).mean(axis=0)

        # Compute the FFT of the TRF
        trf_fft = fft(trf)
        trf_magnitude = np.abs(trf_fft)
        trf_phase = np.angle(trf_fft)

        # Create frequency axis
        n_samples = len(trf)
        sample_rate = 1 / (config.times[1] - config.times[0])  
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)

        # Only keep positive frequencies for plotting
        n_pos = n_samples // 2
        freqs_pos = freqs[:n_pos]
        trf_magnitude_pos = trf_magnitude[:n_pos]
        trf_phase_pos = trf_phase[:n_pos]

        # Plot original TRF
        axes[b, 0].plot(config.times, trf, label=f'{band}', color=f'C{b}')
        axes[b, 0].set_xlabel('Time (s)')
        axes[b, 0].set_ylabel('Amplitude')
        axes[b, 0].set_title(f'Original TRF - {band}')
        axes[b, 0].grid(True)
        axes[b, 0].legend()
        
        # Plot magnitude spectrum
        axes[b, 1].plot(freqs_pos, trf_magnitude_pos, label=f'{band}', color=f'C{b}')
        axes[b, 1].set_xlabel('Frequency (Hz)')
        axes[b, 1].set_ylabel('Magnitude')
        axes[b, 1].set_title(f'Magnitude Spectrum - {band}')
        axes[b, 1].grid(True)
        axes[b, 1].legend()
        
    fig.show()