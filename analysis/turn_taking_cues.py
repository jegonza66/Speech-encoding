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
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np, json

from utils.general_functions import load_pickle
from utils.processing import shifted_matrix
import config

switch_turn_table = {
    "session": [],
    "channel": [],
    "start_time": [],
    "end_time": []
}
hold_turn_table = switch_turn_table.copy()

sessions_dic = {s: {0: None, 1: None} for s in config.sessions}
for session in tqdm(sessions_dic, desc=f"Processing turns for External condition", total=len(sessions_dic)):
    for ch in [0, 1]:
        # Get relevant indexes for each subject
        samples_info = load_pickle(
            path=f"saves/preprocessed_data/External/tmin-0.2_tmax0.6/samples_info/samples_info_{session}.pkl"
        )
        trial_lengths = samples_info[f'trial_lengths{ch+1}']
        keep_indexes = samples_info[f'keep_indexes{ch+1}']

        for k, trial_length in enumerate(trial_lengths):
            if k==0:
                continue
            trial_start = sum(trial_lengths[:k])
            original_indexes = np.arange(trial_length) + trial_start
            hearing_indexes = np.zeros_like(original_indexes)

            # # Filter keep_indexes to only those within the current trial range
            # keep_index = [idx for idx in keep_indexes if trial_start <= idx < trial_start + trial_length]
            
            # # Map global keep_index to local trial indices
            # local_keep_index = [idx - trial_start for idx in keep_index if trial_start <= idx < trial_start + trial_length]
            
            # Map global keep_indexes to local trial indices within the current trial range
            local_keep_index = [idx - trial_start for idx in keep_indexes if trial_start <= idx < trial_start + trial_length]
            hearing_indexes[local_keep_index] = 1
            
            # Sum 1 windows of delays = [-26, ...,  0, ..., 77] surrounding indexes to keep
            for d in config.delays:
                shifted_indexes = np.array(local_keep_index) + d
                
                # Only keep indexes within bounds
                valid_shifted = shifted_indexes[(shifted_indexes >= 0) & (shifted_indexes < original_indexes.shape[0])]
                if valid_shifted.size == 0:
                   continue
                hearing_indexes[valid_shifted] = 1
            
            # Identify starting time of each hearing time
            hearing_starts = (np.diff(hearing_indexes) < 0).nonzero()[0] + 1
            
            hearing_starts_time = hearing_starts/config.sr

            # trial_original_indexes = []
            # trial_mask_keep = []
            # trial_stimulus = []
            
            # # Segment data according to trial lengths
            # for l, length in enumerate(trial_lengths):
            #     start, end = sum(trial_lengths[:l]), sum(trial_lengths[:l+1])
            #     if (start == end):
            #         continue
            #     trial_original_indexes.append(original_indexes[start:end])
            #     trial_mask_keep.append(mask_keep[start:end])
            #     trial_stimulus.append(stimulus[start:end])

            # stimuli[condition][session][ch] = {
            #     "original_indexes": original_indexes,
            #     "mask_keep": mask_keep,
            #     "stimulus": stimulus,
            #     "trial_original_indexes": trial_original_indexes,
            #     "trial_mask_keep": trial_mask_keep,
            #     "trial_stimulus": trial_stimulus
            # }
