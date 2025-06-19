from pathlib import Path

import scipy.io.wavfile as wavfile
from IPython import embed
import pandas as pd
import numpy as np

from utils.general_functions import load_pickle
from load import SessionData
import config

def labeling(
    trial:int, 
    channel:int,
    session:int,
    )->np.ndarray:
    """
    Gives an array with speaking channel: 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)

    Parameters
    ----------
    trial : int
        Number of trial
    channel : int
        Channel of audio signal

    Returns
    -------
    np.ndarray
        Speaking channels by sample, matching EEG sample rate and almost matching its length
    """
    phrases_path = Path(f"data/phrases/S{session}/")
    
    # Read phrases into pandas.DataFrame
    speaker_path = phrases_path / f's{session}.objects.{trial:02d}.channel{channel}.phrases'
    speaker_table = pd.read_table(
        speaker_path, 
        header=None, 
        sep="\t"
    )

    # Replace and '#' by ''. And then all text by 1 and silences by 0 
    speaker_table.iloc[:, 2] = (
        speaker_table.iloc[:, 2].replace("#", "").apply(len) > 0
    ).apply(int)
    
    # Same with listener
    listener_channel = (channel - 3) * -1
    listener_path = phrases_path / f's{session}.objects.{trial:02d}.channel{listener_channel}.phrases'
    listener_table = pd.read_table(
        listener_path, 
        header=None, 
        sep="\t"
    )

    # Replace and '#' by ''. And then all text by 1 and silences by 0
    listener_table.iloc[:, 2] = (
        listener_table.iloc[:, 2].replace("#", "").apply(len) > 0
    ).apply(int)
    
    # Get speaker and listener time ranges
    listener_time_ranges = listener_table[listener_table[2]==1][[0,1]].values
    speaker_time_ranges = speaker_table[speaker_table[2]==1][[0,1]].values
    
    overlaps = []
    
    for speaker_start, speaker_end in speaker_time_ranges:
        for listener_start, listener_end in listener_time_ranges:
            # Find overlap
            overlap_start = max(speaker_start, listener_start)
            overlap_end = min(speaker_end, listener_end)
            
            # Check if there's actual overlap
            if overlap_start < overlap_end:
                overlaps.append([overlap_start, overlap_end])
    overlaps = np.array(overlaps) if overlaps else np.array([]).reshape(0, 2)

    return speaker_time_ranges, overlaps
    
situations = [
    # 'External',
    # 'External_Silence_10',
    # 'External_Silence_50',
    # 'External_Silence_100',
    'External_BS',
]

wavs_path = Path(r'data\wavs')
wav_directories = [
    d for d in wavs_path.iterdir()\
    if d.is_dir() and d.name.startswith('S') and d.name[1:].isdigit()
]
for situation in situations:
    for session, wav_dir_path in zip(config.sessions, wav_directories):
        wav_trials = list(wav_dir_path.glob('*.wav'))
        
        if not situation.endswith('_BS'):
            audio_to_concatenate1 = []
            audio_to_concatenate2 = []
            for tr, wav_trial in enumerate(wav_trials):
                if wav_trial.name.endswith('1.wav'):
                    speaker_range, overlap = labeling(
                        trial=int(tr/2+1),
                        channel=1,
                        session=session
                    )
                    sampling_f, audio = wavfile.read(wav_trial)
                    time = np.arange(len(audio))/sampling_f
                    
                    speaker_audio_segments = []
                    for start_time, end_time in speaker_range:
                        mask = (time >= start_time) & (time <= end_time)
                        segment = audio[mask]
                        if len(segment) > 0:
                            speaker_audio_segments.append(segment)

                    audio_to_concatenate1.append(
                        np.concatenate(speaker_audio_segments)    
                    )
                else:
                    speaker_range, overlap = labeling(
                        trial=int(tr/2+1),
                        channel=2,
                        session=session
                    )
                    time = np.arange(len(audio))/sampling_f
                    
                    speaker_audio_segments = []
                    for start_time, end_time in speaker_range:
                        mask = (time >= start_time) & (time <= end_time)
                        segment = audio[mask]
                        if len(segment) > 0:
                            speaker_audio_segments.append(segment)

                    audio_to_concatenate2.append(
                        np.concatenate(speaker_audio_segments)    
                    )  
            audio_to_concatenate1 = np.concatenate(audio_to_concatenate1)
            audio_to_concatenate2 = np.concatenate(audio_to_concatenate2)
            
            # Save audio files
            # embed()
            # wavfile.write(
            #     Path('analysis') / f's{session}_{situation}_1.wav',
            #     sampling_f, 
            #     audio_to_concatenate1*int(2)
            # )
        else:
            audio_to_concatenate1 = []
            audio_to_concatenate2 = []
            for tr, wav_trial in enumerate(wav_trials):
                if tr%2 == 0:
                    speaker_range, overlap = labeling(
                        trial=int(tr/2+1),
                        channel=1,
                        session=session
                    )
                    sampling_f, audio = wavfile.read(wav_trial)
                    time = np.arange(len(audio))/sampling_f
                    
                    overlap_audio_segments = []
                    for start_time, end_time in overlap:
                        mask = (time >= start_time) & (time <= end_time)
                        segment = audio[mask]
                        if len(segment) > 0:
                            overlap_audio_segments.append(segment)

                    audio_to_concatenate1.append(
                        np.concatenate(overlap_audio_segments)    
                    )
                else:
                    speaker_range, overlap = labeling(
                        trial=int(tr/2+1),
                        channel=2,
                        session=session
                    )
                    sampling_f, audio = wavfile.read(wav_trial)
                    time = np.arange(len(audio))/sampling_f
                    
                    overlap_audio_segments = []
                    for start_time, end_time in overlap:
                        mask = (time >= start_time) & (time <= end_time)
                        segment = audio[mask]
                        if len(segment) > 0:
                            overlap_audio_segments.append(segment)

                    audio_to_concatenate2.append(
                        np.concatenate(overlap_audio_segments)    
                    )
            audio_to_concatenate1 = np.concatenate(audio_to_concatenate1)
            audio_to_concatenate2 = np.concatenate(audio_to_concatenate2)
            embed()
            wavfile.write(
                Path('analysis') / f's{session}_{situation}.wav',
                sampling_f, 
                (audio_to_concatenate1+audio_to_concatenate2)*int(4)
            )