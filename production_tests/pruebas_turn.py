import os, numpy as np, pandas as pd
phrases_path = 'Datos/phrases/'
sesion, trial, channel = 21, 1, 1
subject_2_channel = (channel - 3) * -1
sr = 128

# First channel ---> channel
subject_1 = os.path.join(phrases_path, f'S{sesion}', f's{sesion}.objects.{trial:02d}.channel{channel}.phrases')
phrases_1 = pd.read_table(subject_1, header=None, sep="\t")

# Replace and '#' by ''. And then all text by 1 and silences by 0 
phrases_1.iloc[:, 2] = (phrases_1.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)

# Take difference in time and multiply it by sample rate in order to match envelope length (almost, miss by a sample or two)
samples = np.round((phrases_1[1] - phrases_1[0]) * sr).astype("int")
subject_1 = np.repeat(phrases_1.iloc[:, 2], samples).ravel()

# Same with subject_2
ubi_subject_2 = os.path.join(phrases_path, f'S{sesion}', f's{sesion}.objects.{trial:02d}.channel{subject_2_channel}.phrases')
phrases_2 = pd.read_table(ubi_subject_2, header=None, sep="\t")
phrases_2.iloc[:, 2] = (phrases_2.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
samples = np.round((phrases_2[1] - phrases_2[0]) * sr).astype("int")
subject_2 = np.repeat(phrases_2.iloc[:, 2], samples).ravel()

# If there are differences in length, corrects them with 0-padding
diff = len(subject_1) - len(subject_2)
if diff > 0:
    subject_2 = np.concatenate([subject_2, np.repeat(0, diff)])
elif diff < 0:
    subject_1 = np.concatenate([subject_1, np.repeat(0, np.abs(diff))])
    
# Filter to where each subject starts talking
filter_turn_1 = np.concatenate((np.array([0]), np.diff(subject_1)))!=0
filter_turn_2 = np.concatenate((np.array([0]), np.diff(subject_2)))!=0

# Number of turns per subject
n_turns_1 = filter_turn_1.sum()
n_turns_2 = filter_turn_2.sum()

# Calculate overlap signal (ones both speaking, 0 otherwise) and gaps (both in silence)
sum_signal = subject_1 + subject_2
overlap_signal = ((sum_signal)==2).astype(int)
silence_signal = ((sum_signal)==0).astype(int)

filter_turn_overlap = np.concatenate((np.array([0]), np.diff(overlap_signal)))!=0
filter_turn_silence = np.concatenate((np.array([0]), np.diff(silence_signal)))!=0


prueba = np.array([0,0,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,1])
(np.concatenate((np.array([0]), np.diff(prueba)))!=0).astype(int)

prueba2 = np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,1])
(np.concatenate((np.array([0]), np.diff(prueba2)))!=0).astype(int)








# subject_1.astype(bool)&subject_2.astype(bool)
np.concatenate((np.array([0]), np.diff(subject_1)))!=0