import pandas as pd, os, textgrids, numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal as sgn

# Mistakes folder
mistake_folder = os.path.normpath(path='Datos/errores/')

# La sesion con más errores es la 25 --> 48 errores, 27 en canal 1 y 21 en canal 2. La señal del que habla, pero podemos implementarlo con las 4 variantes
session, trial, channel = 25, 6, 2

# Read envelope
wav = wavfile.read(f'Datos/wavs/S{session}/s{session}.objects.{str(trial).zfill(2)}.channel{channel}.wav')[1]
wav = wav.astype("float")
envelope = np.abs(sgn.hilbert(wav))
window_size, stride = int(16e3/128), int(16e3/128)
envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
envelope =  envelope.reshape(-1, 1)

# def f_mistakes(self, envelope:np.ndarray):

#     # Read phrases to identify time of error inside phrases time
#     phrases = pd.read_table(f'Datos/phrases/S{session}/s{session}.objects.{str(trial).zfill(2)}.channel{channel}.phrases', header=None, sep="\t")
#     start_time, end_time = phrases[0].iloc[0], phrases[1].iloc[-1]
#     phrases_time = np.arange(start_time, end_time, 1/128)

#     # Identify start and end of error within mistake
#     mistake_code = {
#                     'A':0, # articulatorio
#                     'L':1, # léxico
#                     'D':2 # discursivo
#                     }
#     mistake_signal = np.zeros(shape=(len(phrases_time), 3))

#     textgrids_path = os.path.join(mistake_folder, f'filtered_session{session}_trial{trial}_channel{channel}.TextGrid')
#     if os.path.isfile(textgrids_path):
#         # Read textgrid        
#         grid = textgrids.TextGrid(textgrids_path)[f'canal {channel}']
        
#         # Identify onset, offset and type of mistake
#         mistake_taggs, mistake_count = np.unique([el.text.split('Palabra del error: ')[1] for el in grid], return_counts=True)
#         mistakes = {mistake:{'start':None, 'end':None, 'type':None} for mistake in mistake_taggs}
#         mistake_taggs = np.repeat(mistake_taggs, mistake_count)

#         for item, mistake in zip(grid, mistake_taggs):
#             # Identify time_intervals and mistake type
#             mistakes[mistake]['type'] = mistake_code[item.text.split('Etiqueta: ')[1][0]]
#             if int(item.text[0])==1:
#                 mistakes[mistake]['start'] = item.xpos
#             elif int(item.text[0])==2:
#                 mistakes[mistake]['end'] = item.xpos
#             # else:
#             #     nextword_start.append(item.xpos)

#         # Fill mistake_signal
#         for mistake in mistakes:
#             onset_filter = mistakes[mistake]['start']<=phrases_time
#             offset_filter = phrases_time<=mistakes[mistake]['end']
            
#             mistake_signal[onset_filter&offset_filter, mistakes[mistake]['type']] = np.ones(shape=np.sum(onset_filter&offset_filter))
            
#     # Match length of mistake signal with envelope
#     difference = len(mistake_signal)-len(envelope)

#     if difference>0:
#         mistake_signal = mistake_signal[:-difference]
#     elif difference<0:
#         mistake_signal = np.concatenate((mistake_signal, np.zeros(shape=(np.abs(difference), 3))))

#     return mistake_signal


# filtered_mistake_folder = os.path.join(mistake_folder,'Filtrados')
# filtered_mistake_output = os.path.normpath('Datos/mistakes/')
# for j, file in enumerate([f for f in os.listdir(filtered_mistake_folder) if f.endswith('.TextGrid')]):
#     # Get the filename and open it to extract the chanel number
#     textgrids_path = os.path.normpath(os.path.join(filtered_mistake_folder, file))
#     file.split('s')[1][:2]
#     session, trial = int(file.split('s')[1][:2]), int(file.split('objects_')[1][:2])


#     grid = textgrids.TextGrid(textgrids_path)
#     canales = list(grid.keys())
#     if len(canales)==2:
#         # Pop channel 2
#         _ = grid.pop(canales[1])
#         # Save channel 1 with new name
#         new_name_file = os.path.join(filtered_mistake_output, f'filtered_session{session}_trial{str(trial).zfill(2)}_channel{1}.TextGrid')
#         grid.write(new_name_file)
#         # Re-open in ANSI, re-save in UTF-8
#         with open(new_name_file, 'r', encoding='ansi') as f:
#             data = f.read()
#         with open(new_name_file, 'w', encoding='utf-8') as f:
#             f.write(data)
#         # Read the file again to get second channel
#         grid = textgrids.TextGrid(textgrids_path)
#         _ = grid.pop(canales[0])
#         # Save channel 2 with new name
#         new_name_file = os.path.join(filtered_mistake_output, f'filtered_session{session}_trial{str(trial).zfill(2)}_channel{2}.TextGrid')
#         grid.write(new_name_file)
#         # Re-open in ANSI, re-save in UTF-8
#         with open(new_name_file, 'r', encoding='ansi') as f:
#             data = f.read()
#         with open(new_name_file, 'w', encoding='utf-8') as f:
#             f.write(data)
#     else:
#         channel = int(canales[0].split('canal ')[1])
#         # Save channel 2 with new name
#         new_name_file = os.path.join(filtered_mistake_output, f'filtered_session{session}_trial{str(trial).zfill(2)}_channel{channel}.TextGrid')
#         grid.write(new_name_file)
#         # Re-open in ANSI, re-save in UTF-8
#         with open(new_name_file, 'r', encoding='ansi') as f:
#             data = f.read()
#         with open(new_name_file, 'w', encoding='utf-8') as f:
#             f.write(data)

def filter_repetitions(df, column, min_repetitions=3):
    # Extract the column as a numpy array
    indexes = df.index.values
    arr = df[column].values

    # Find change points where the value in the column changes
    change_indixes = np.where(arr[:-1] != arr[1:])[0] + 1
    
    # Split the array into segments of contiguous values
    segments = np.split(indexes, change_indixes)
    
    # Filter segments that have min_repetitions or more repetitions
    filtered_segments = [segment[-min_repetitions:] for segment in segments if len(segment) >= min_repetitions]
    filtered_array = np.concatenate(filtered_segments)
    
    filtered_df = df.iloc[filtered_array]
    # filtered_df = filtered_df[filtered_df['Datos Palabra Candidato']!='No esta exacto']
    # filtered_df = filtered_df[filtered_df['Datos Palabra Candidato']!='Falta dato']
    # arr = filtered_df['error seleccionado'].values
    return filtered_df

# Read csv
control_words = pd.read_csv(os.path.join(mistake_folder, 'score_con_palabras.csv'), header=0, delimiter=';')

# Filter unprecised data #TODO filtramos datos que no tienen el timestamp exacto
control_words = control_words[control_words['Datos Palabra Candidato']!='No esta exacto']
control_words = control_words[control_words['Datos Palabra Candidato']!='Falta dato']

# Re initialize index
control_words = control_words.reset_index(drop=True)

# TODO: no matchean los datos de los controles y de los errores, filtro los que no matchean
malas = [] #745
buenas = [] #702
drop_index = []
session_control = []
trial_control = []
channel_control = []
for i in control_words.index:
    control = control_words.iloc[i]
    session = int(control['File Control'].split('s')[1][:2])
    trial = int(control['File Control'].split('objects_')[1][:2])
    channel = int(control['File Control'].split(", '")[1][:1])
    
    session_e = int(control['file error'].split('s')[1][:2])
    trial_e = int(control['file error'].split('objects_')[1][:2])
    channel_e = int(control['file error'].split(", '")[1][:1])
    
    
    # TODO channel y session coinciden siempre, chequeado
    if trial!=trial_e:
        malas.append([(session, trial, channel), (session_e, trial_e, channel_e)])
        drop_index.append(i)
    else:
        buenas.append([(session, trial, channel), (session_e, trial_e, channel_e)])
        session_control.append(session)
        trial_control.append(trial)
        channel_control.append(channel)
        
# Drop bad indexes and re initialize index
control_words = control_words.drop(index=drop_index)
control_words['session'] = session_control
control_words['trial'] = trial_control
control_words['channel'] = channel_control
control_words = control_words.reset_index(drop=True)


# #TODOnumero de controles y cantidad de elementos con esa cantidad
# arr = control_words['error seleccionado'].values
# change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
# segments = np.split(arr, change_indices)
# np.unique([len(segment) for segment in segments], return_counts=True) #after filtering occur it may be segments with multiples of 

# Filter repetitions to get exactly the length wanted for all controls
control_words = filter_repetitions(control_words, 'error seleccionado', min_repetitions=4)
control_words['error seleccionado'].head(16)

# Re initialize index
control_words = control_words.reset_index(drop=True)

# Redefine score normalizing it
max_score = control_words['score total (más bajo mejor)'].max()
control_words['score normalizado'] = 1-control_words['score total (más bajo mejor)']/max_score

# Check wether all errors have controls
mistake_files = [(int(file.split('session')[1][:2]),int(file.split('trial')[1][:2]),int(file.split('channel')[1][:1])) for file in os.listdir('Datos\\mistakes')]
# datos_file = [(int(file.split('s')[1][:2]), int(file.split('objects_')[1][:2])) for file in os.listdir('Datos\\errores\\Filtrados\\')]
control_files = []

for i in control_words.index:
    control = control_words.iloc[i]

    # Datos del archivo
    control_files.append((control['session'], control['trial'], control['channel']))
    
    # Contenido
    info_control = control['info control seleccionado'].replace('(','').replace(')','').split(', ')
    comienzo, final, score = float(info_control[0]), float(info_control[1]), -float(control['score normalizado'])

# No son conjuntos disjuntos
len(set(mistake_files).symmetric_difference(set(control_files)))