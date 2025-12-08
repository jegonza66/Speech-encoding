"""
This script is design to catch audio being infiltrated during EEG recording.
With this purpose cross-correlation between audio signal and preprocessed EEG
was computed for different lags.
"""
import matplotlib.pyplot as plt
from scipy import signal as sgn
from scipy.io import wavfile
from pathlib import Path
from scipy import signal 
import numpy as np
import mne

from utils.general_functions import load_pickle
from utils.processing import custom_resample
from utils.load_utils import get_trials

import config

def diagnose_crosstalk(session:int,eeg_data, envelope, info_mne, sr, noise_window_sec=2.0):
    """
    Cuantifica la contaminación por audio en el EEG mediante:
    1. Correlación de Pearson Instantánea (Topografía e Histograma).
    2. Coherencia Espectral.
    3. Correlación Cruzada Normalizada con Test de Z-Score.
    
    Args:
        eeg_data: (n_samples, n_channels) numpy array
        envelope: (n_samples, 1) numpy array
        info_mne: objeto mne.Info
        sr: sampling rate (int)
        noise_window_sec: segundos a partir de los cuales se considera "ruido de fondo" para el Z-score.
    """
    # Asegurar dimensiones
    if envelope.ndim == 1: envelope = envelope.reshape(-1, 1)
    n_channels = eeg_data.shape[1]
    
    # --- 1. Correlación de Pearson Instantánea (Lag 0) ---
    correlations = np.zeros(n_channels)
    for ch in range(n_channels):
        correlations[ch] = np.corrcoef(eeg_data[:, ch], envelope.flatten())[0, 1]
    
    max_corr = np.max(np.abs(correlations))
    mean_corr = np.mean(np.abs(correlations))
    
    print(f"\n--- DIAGNÓSTICO DE CROSSTALK ---")
    print(f"Correlación Máxima Absoluta (Lag 0): {max_corr:.4f}")
    print(f"Correlación Promedio Absoluta (Lag 0): {mean_corr:.4f}")
    
    if max_corr > 0.1:
        print("ALERTA ROJA: Crosstalk masivo o movimiento correlacionado muy fuerte.")
    elif max_corr > 0.05:
        print("ALERTA NARANJA: Probable fuga eléctrica o artefacto muscular.")
    else:
        print("VERDE: Niveles bajos, podría ser ruido espurio.")

    # --- 2. Varianza Explicada ---
    r_squared = correlations ** 2
    print(fr"Varianza del EEG explicada por el Audio (Max Channel): {np.max(r_squared)*100:.2f}\%")

    # --- 3. Coherencia Espectral (Canal Peor) ---
    worst_ch_idx = np.argmax(np.abs(correlations))
    f, Cxy = signal.coherence(
        eeg_data[:, worst_ch_idx], 
        envelope.flatten(), fs=sr, nperseg=sr*2
    )
    
    # --- 4. Correlación Cruzada Promedio y Z-Score ---
    # Usamos el promedio de canales para ver el efecto global
    eeg_mean = eeg_data.mean(axis=1)
    
    # Centrar señales
    eeg_to_correlate = eeg_mean - eeg_mean.mean()
    envelope_to_correlate = envelope.flatten() - envelope.flatten().mean()
    
    # Normalización (N-1 para estimador insesgado)
    n = len(eeg_to_correlate)
    std_eeg = np.std(eeg_to_correlate, ddof=1)
    std_env = np.std(envelope_to_correlate, ddof=1)
    norm_factor = std_eeg * std_env * (n - 1)
    
    # Calcular Cross-Correlation completa normalizada
    cross_correlation = signal.correlate(
            eeg_to_correlate,
            envelope_to_correlate, 
            mode='full'
        ) / norm_factor
    
    lags = np.arange(-(n - 1), n) / sr

    # --- CÁLCULO DE Z-SCORE ---
    # Valor exacto en Lag 0
    lag_0_idx = np.searchsorted(lags, 0)
    val_lag_0 = cross_correlation[lag_0_idx]
    
    # Definir zona de ruido (lags lejanos > noise_window_sec)
    mask_noise = np.abs(lags) > noise_window_sec
    noise_vals = cross_correlation[mask_noise]
    
    noise_mean = np.mean(noise_vals)
    noise_std = np.std(noise_vals)
    
    # Z-Score: ¿Cuántas desviaciones estándar se aleja el 0 del ruido de fondo?
    z_score = (val_lag_0 - noise_mean) / noise_std
    
    print(f"Z-Score del Lag 0 (vs Fondo > {noise_window_sec}s): {z_score:.2f}")
    if abs(z_score) > 5:
        print("SIGNIFICANCIA: El pico en 0 es estadísticamente significativo (distinto del ruido).")
    else:
        print("SIGNIFICANCIA: El pico en 0 es indistinguible del ruido de fondo.")

    # --- PLOTS ---
    fig, ax = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(f'Sesión {session} - Diagnóstico de Crosstalk Audio-EEG', fontsize=16)
    
    # A. Topomap
    im, _ = mne.viz.plot_topomap(
        correlations, info_mne, axes=ax[0], show=False, cmap='RdBu_r', vlim=(-max_corr, max_corr)
    )
    ax[0].set_title('Topografía (Lag 0)')
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    
    # B. Histograma
    ax[1].hist(correlations, bins=20, color='skyblue', edgecolor='black')
    ax[1].set_title('Hist. Correlaciones')
    ax[1].set_xlabel('Pearson r')
    
    # C. Coherencia
    ax[2].plot(f, Cxy)
    ax[2].set_title(f'Coherencia (Ch_M :{worst_ch_idx})')
    ax[2].set_xlabel('Frecuencia (Hz)')
    ax[2].grid(True)
    ax[2].set_xlim(0, 40)
    
    # D. Correlación Cruzada con Z-Score
    ax[3].plot(lags, cross_correlation, label='Cross-Corr', linewidth=1.5)
    
    # Marcar Lag 0
    ax[3].scatter(0, val_lag_0, color='red', zorder=5, label=f'Lag 0 (Z={z_score:.1f})')
    
    # Marcar umbrales de ruido (3 sigmas)
    ax[3].axhline(noise_mean + 3*noise_std, color='orange', linestyle='--', alpha=0.7, label=r'Ruido (3$\sigma$)')
    ax[3].axhline(noise_mean - 3*noise_std, color='orange', linestyle='--', alpha=0.7)
    
    ax[3].set_title('Correlación Cruzada Promedio')
    ax[3].set_xlim(-10, 10) # Zoom a +/- 10 segundos
    ax[3].set_xlabel('Lag (s)')
    ax[3].set_ylabel('Correlación Normalizada (r)')
    ax[3].grid(True, alpha=0.3)
    ax[3].legend(loc='upper right', fontsize='small')
    
    plt.show()
    
    return correlations
for session in config.sessions:
    envelope = load_pickle(
        rf"saves\preprocessed_data\tmin-0.2_tmax0.6\Envelope\Sesion{session}.pkl"
    )

    envelope = envelope[1]# + envelope[1]
    # audio = np.load(rf"data\resampled_audio_cache\session_{session}_channel_{2}.npy")
    # audio = signal.resample_poly(audio, up=1, down=4, axis=0, padtype='mean')
    
    # start_sample_target = []
    # end_sample_target = []
    # current_sample = 0
    # for trial in get_trials(session=session):
    #     wav_path_trial = rf"data\wavs\S{session}\s{session}.objects.{trial:02d}.channel2.wav"
    #     sr_wav, wav_temp = wavfile.read(wav_path_trial, mmap=True) 
    #     trial_time = wav_temp.shape[0]/sr_wav
    #     len_target = int(round(trial_time * 128))
    #     start_sample_target.append(current_sample)
    #     end_sample_target.append(current_sample + len_target)
    #     current_sample += len_target

    # audio = np.concatenate(
    #     [
    #         audio[start:end]
    #         for start, end in zip(start_sample_target, end_sample_target)
    #     ],
    #     axis=0   
    # )
    # eeg = load_pickle(
    #     rf"saves\preprocessed_data\tmin-0.2_tmax0.6\EEG\Broad\Sesion{session}.pkl"
    # )[0]
    # minimum_len = min(eeg.shape[0], audio.shape[0])
    # eeg = eeg[:minimum_len, :]
    # audio = audio[:minimum_len]


    # eeg = load_pickle(
    #     rf"saves\preprocessed_data\tmin-0.2_tmax0.6\EEG\Broad\Sesion{session}.pkl"
    # )[0]
    # minimum_len = min(eeg.shape[0], envelope.shape[0])
    # eeg = eeg[:minimum_len, :]
    # diagnose_crosstalk(
    #     session=session,
    #     eeg_data=eeg,
    #     envelope=envelope,
    #     info_mne=config.info_mne,
    #     sr=config.sr
    # )
    total_audio = []
    for trial in get_trials(session=session):
        wav_path_trial=rf"data\wavs\S{session}\s{session}.objects.{trial:02d}.channel{2}.wav"
        sr_wav, wav_temp = wavfile.read(wav_path_trial)
        total_audio.append(wav_temp)
    total_audio = np.concatenate(total_audio, axis=0) # 2572.8496875 s
    audio = custom_resample(
        array=total_audio,
        original_sr=sr_wav,
        target_sr=1024,
        axis=0,
        padtype='mean'
    ) # 2572.8515625
    # A. High-pass (0.1 Hz, Order 16896)
    b_hp = sgn.firwin(
        numtaps=16896 + 1, 
        cutoff=0.1, 
        fs=1024, 
        pass_zero=False, 
        window='hamming'
    )
    # B. Low-pass (100 Hz, Order 100)
    b_lp = sgn.firwin(
        numtaps=100 + 1, 
        cutoff=100, 
        fs=1024, 
        pass_zero=True, 
        window='hamming'
    )
    # C. Notch (49-51 Hz, Order 3380) -> Band-stop
    b_notch = sgn.firwin(
        numtaps=3380 + 1, 
        cutoff=[49, 51], 
        fs=1024, 
        pass_zero=True, # Band-stop (pasa extremos, corta centro)
        window='hamming'
    )

    audio = sgn.filtfilt(b_hp, 1.0, audio)
    audio = sgn.filtfilt(b_lp, 1.0, audio)
    audio = sgn.filtfilt(b_notch, 1.0, audio)

    audio = custom_resample(
        array=audio,
        original_sr=1024,
        target_sr=config.sr,
        axis=0,
        padtype='mean'
    )
    total_eeg = []
    for trial in get_trials(session=session):
        eeg_path = rf"data\EEG\S{session}\s{session}-{1}-Trial{trial}-Deci-Filter-Trim-ICA-Pruned.set"
        raw = mne.io.read_raw_eeglab(eeg_path, preload=True).get_data().T
        total_eeg.append(raw)
    
    eeg = np.concatenate(total_eeg, axis=0) 
    eeg = custom_resample(
        array=eeg,
        original_sr=512,
        target_sr=config.sr,
        axis=0,
        padtype='mean'
    )# 2573.8046875 s
    print(eeg.shape[0]/config.sr-audio.size/config.sr)
    minimum_len = min(eeg.shape[0], audio.shape[0])
    audio = audio[:minimum_len]
    eeg = eeg[:minimum_len, :]
    diagnose_crosstalk(
        session=session,
        eeg_data=eeg,
        envelope=audio,
        info_mne=config.info_mne,
        sr=config.sr
    )