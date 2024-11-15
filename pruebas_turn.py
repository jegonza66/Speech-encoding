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
filter_1 = np.concatenate((np.array([0]), np.diff(subject_1)))!=0
filter_2 = np.concatenate((np.array([0]), np.diff(subject_2)))!=0

# Number of turns
np.sum()

diferen = []
for i in range(prueba.shape[0]-1):
    diferen.append(prueba[i+1] - prueba[i])

np.roll(prueba, shift=1)-prueba

# change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
# segments = np.split(arr, change_indices)