"""
This script is used to find turn-taking cues in the hearing condition: holds and switches.

A switch is a change in speaker, where the current speaker stops talking and then the interlocutor carry on. 
It is requested that a period of at least 100 ms occurs between the end of the speaker turn.

A hold is a period of silence or non-speech where a speaker is expected to take their turn but does not.
It's also requested that a period of at least 100 ms occurs between intraturns.

Other definitions:
External_BS/Internal_BS: samples where both subjects are speaking
External: samples where interlocutor is speaking
Internal: samples where the locutor is speaking
"""
from pathlib import Path
from tqdm import tqdm
import numpy as np, json

from utils.general_functions import load_pickle
from utils.processing import shifted_matrix
import config

for session in tqdm(config.sessions, desc=f"Processing turns for External condition", total=len(config.sessions)):
# for session in [21]:
    for ch_speaker in [0, 1]:
    # for ch_speaker in [0]:
        # Get relevant indexes for each subject
        samples_info = load_pickle(
            path=f"saves/preprocessed_data/tmin-0.2_tmax0.6/samples_info/External/samples_info_{session}.pkl"
        )
        trial_lengths = samples_info[f'trial_lengths{ch_speaker+1}'] # has length of trials + 1 (0 at start)
        keep_indexes = samples_info[f'keep_indexes{ch_speaker+1}']

        ch_interlocutor = 1 if ch_speaker == 0 else 0
        trial_lengths_interlocutor = samples_info[f'trial_lengths{ch_interlocutor+1}']
        if trial_lengths!=trial_lengths_interlocutor:
            raise ValueError("Trial lengths for speaker and interlocutor do not match.")
        keep_indexes_interlocutor = samples_info[f'keep_indexes{ch_interlocutor+1}']

        for trial, trial_length in enumerate(trial_lengths):
            if trial==0:
                continue
            
            def get_hearing_indexes(trial_start, trial_length, keep_indexes):
                original_indexes = np.arange(trial_length) + trial_start
                hearing_indexes = np.zeros_like(original_indexes)
                local_keep_index = [idx - trial_start for idx in keep_indexes if trial_start <= idx < trial_start + trial_length]
                hearing_indexes[local_keep_index] = 1
                for d in config.delays:
                    shifted_indexes = np.array(local_keep_index) + d
                    valid_shifted = shifted_indexes[(shifted_indexes >= 0) & (shifted_indexes < original_indexes.shape[0])]
                    if valid_shifted.size == 0:
                        continue
                    hearing_indexes[valid_shifted] = 1
                    
                # Identify starting time of each hearing time
                hearing_starts = (np.diff(hearing_indexes) > 0).nonzero()[0] + 1
                hearing_ends = (np.diff(hearing_indexes) < 0).nonzero()[0] + 1
                
                # To prevent last utterance without ending 
                if hearing_starts.shape[0] > hearing_ends.shape[0]:
                    hearing_starts = hearing_starts[:hearing_ends.shape[0]]
                # To prevent first utterance without start
                elif hearing_ends.shape[0] > hearing_starts.shape[0]:
                    hearing_ends = hearing_ends[1:]
                return original_indexes, hearing_starts/config.sr, hearing_ends/config.sr
            
            original_indexes, hearing_starts_time, hearing_ends_time = get_hearing_indexes(
                sum(trial_lengths[:trial]), trial_length, keep_indexes
            )
            _, speaking_starts_time, speaking_ends_time = get_hearing_indexes(
                sum(trial_lengths[:trial]), trial_length, keep_indexes_interlocutor
            )

            # Find switches (take diff between end of speaker turn and next utterance to classify.
            # If 100 ms between utterance with no overlap and the utterance corresponds to interlocutor classify as 
            # switch. If 100 ms between utterance with no overlap and the utt correspons to speaker classify as hold.
            # If overlap between utterances or not 100 ms between utterance is still the same turn)
            prev_turn_start_s, switches_t_start, switches_t_end = [], [], []
            prev_turn_start_h, holds_t_start, holds_t_end = [], [], []
            for l, (start, end) in enumerate(zip(hearing_starts_time, hearing_ends_time), start=0):
                # For the last value check if interlocutor speaks with interval of .1s
                if (l+1) == len(hearing_starts_time):
                    if any(speaking_starts_time - end >= 0.1):
                        first_speak = speaking_starts_time[(speaking_starts_time - end >= 0.1)][0]
                        end_first_speak = speaking_ends_time[(first_speak<=speaking_ends_time)][0]

                        # If so, next turn needs to be at least 400 ms to not qualify as backchannel
                        if (end_first_speak - first_speak) >= .4:  
                            prev_turn_start_s.append(start)
                            switches_t_start.append(end)
                            switches_t_end.append(first_speak)
                        else:
                            continue
                    else:
                        continue
                # For the rest of these values
                elif (l+1) <= len(hearing_starts_time):
                    # Check if there are any possible holds or switches
                    if (hearing_starts_time[l+1]-end) >= .1:
                        # If there are possible interlocutor turns
                        if any(speaking_starts_time - end >= .1):
                            first_speak = speaking_starts_time[(speaking_starts_time >= end)][0]
                            if first_speak-end<.1:
                                continue
                            end_first_speak = speaking_ends_time[(first_speak<=speaking_ends_time)][0]
                            
                            # Check if is a turn
                            if first_speak < hearing_starts_time[l+1]:
                                # If so, next turn needs to be at least 400 ms to not qualify as backchannel
                                if (end_first_speak - first_speak) >= .4: 
                                    prev_turn_start_s.append(start)
                                    switches_t_start.append(end)
                                    switches_t_end.append(first_speak)
                                else:
                                    continue
                            #If not a switch must be a hold, just if next turn is at least 400 ms to not qualify as backchannel
                            elif hearing_starts_time[l+1] - end >= .4:  
                                prev_turn_start_h.append(start)
                                holds_t_start.append(end)
                                holds_t_end.append(hearing_starts_time[l+1])
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                    
            # Save the data in a json
            json_path = Path(rf"data\turns\switches_external\sess_{session}_trial_{trial:02}_ch_{ch_speaker+1}.json")
            ch_interlocutor = ch_speaker+1 
            speaker = 2 if ch_interlocutor == 1 else 1
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump([{
                    "speaker": speaker,
                    "interlocutor": ch_interlocutor,
                    "ipu1_start_time": prev_start,
                    "ipu1_end_time": start,
                    "ipu2_start_time": end
                } for prev_start, start, end in zip(prev_turn_start_s,switches_t_start, switches_t_end)], f, indent=2)

            json_path = Path(rf"data\turns\holds_external\sess_{session}_trial_{trial:02}_ch_{ch_speaker+1}.json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump([{
                    "speaker": speaker,
                    "interlocutor": ch_interlocutor,
                    "ipu1_start_time": prev_start,
                    "ipu1_end_time": start,
                    "ipu2_start_time": end
                } for prev_start, start, end in zip(prev_turn_start_h, holds_t_start, holds_t_end)], f, indent=2)

# [
#   {
#     "speaker": 1,
#     "interlocutor": 2,
#     "ipu1_start_time": 10.88,
#     "ipu1_end_time": 12.201478
#   },
#   {
#     "speaker": 1,
#     "interlocutor": 2,
#     "ipu1_start_time": 37.279514,
#     "ipu1_end_time": 38.306368
#   },
#   {
#     "speaker": 1,
#     "interlocutor": 2,
#     "ipu1_start_time": 60.581474,
#     "ipu1_end_time": 61.06697
#   },
#   {
#     "speaker": 1,
#     "interlocutor": 2,
#     "ipu1_start_time": 64.107184,
#     "ipu1_end_time": 65.204067
#   }
# ]