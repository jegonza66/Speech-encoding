import pandas as pd, os, textgrids, numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
from collections import OrderedDict, namedtuple
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

# # ERROR

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

# # CONTROL
# def filter_repetitions(df, column, min_repetitions=3):
#     # Extract the column as a numpy array
#     indexes = df.index.values
#     arr = df[column].values

#     # Find change points where the value in the column changes
#     change_indixes = np.where(arr[:-1] != arr[1:])[0] + 1
    
#     # Split the array into segments of contiguous values
#     segments = np.split(indexes, change_indixes)
    
#     # Filter segments that have min_repetitions or more repetitions
#     filtered_segments = [segment[-min_repetitions:] for segment in segments if len(segment) >= min_repetitions]
#     filtered_array = np.concatenate(filtered_segments)
    
#     filtered_df = df.iloc[filtered_array]
#     # filtered_df = filtered_df[filtered_df['Datos Palabra Candidato']!='No esta exacto']
#     # filtered_df = filtered_df[filtered_df['Datos Palabra Candidato']!='Falta dato']
#     # arr = filtered_df['error seleccionado'].values
#     return filtered_df

# # Read csv
# control_words = pd.read_csv(os.path.join(mistake_folder, 'score_con_palabras.csv'), header=0, delimiter=';')

# # Filter unprecised data #TODO filtramos datos que no tienen el timestamp exacto
# control_words = control_words[control_words['Datos Palabra Candidato']!='No esta exacto']
# control_words = control_words[control_words['Datos Palabra Candidato']!='Falta dato']

# # Re initialize index
# control_words = control_words.reset_index(drop=True)

# # # TODO: no matchean los datos de los controles y de los errores, filtro los que no matchean
# # malas = [] #745
# # buenas = [] #702
# # drop_index = []
# session_control = []
# trial_control = []
# channel_control = []
# error_type = []

# for i in control_words.index:
#     # Get the row
#     control = control_words.iloc[i]
    
#     # Get file info of control
#     session = int(control['File Control'].split('s')[1][:2])
#     trial = int(control['File Control'].split('objects_')[1][:2])
#     channel = int(control['File Control'].split(", '")[1][:1])
#     session_control.append(session)
#     trial_control.append(trial)
#     channel_control.append(channel)
    
#     # Get file info of error
#     session_e = int(control['file error'].split('s')[1][:2])
#     trial_e = int(control['file error'].split('objects_')[1][:2])
#     channel_e = int(control['file error'].split(", '")[1][:1])
#     start_e = float(control['info  error seleccionado'].split("(")[1].split(', ')[0])
    
#     # Identify the type of error    
#     path_to_error = f'Datos/mistakes/filtered_session{session_e}_trial{str(trial_e).zfill(2)}_channel{channel_e}.TextGrid'
#     grid = textgrids.TextGrid(path_to_error)[f"canal {int(path_to_error.split('channel')[1][0])}"]
#     for el in grid:
#         if int(el.text[:1])==1:
#             if el.xpos==start_e:
#                 error_type.append(el.text.split('Etiqueta: ')[1][0])
        
# # Drop bad indexes and re initialize index
# control_words['session'] = session_control
# control_words['channel'] = channel_control
# control_words['trial'] = trial_control
# control_words['error type'] = error_type
# control_words = control_words.reset_index(drop=True)

# #TODOnumero de controles y cantidad de elementos con esa cantidad
# arr = control_words['error seleccionado'].values
# change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
# segments = np.split(arr, change_indices)
# np.unique([len(segment) for segment in segments], return_counts=True) #after filtering occur it may be segments with multiples of 

# # Filter repetitions to get exactly the length wanted for all controls
# control_words = filter_repetitions(control_words, 'error seleccionado', min_repetitions=4)
# control_words['error seleccionado'].head(16)

# # Re initialize index
# control_words = control_words.reset_index(drop=True)

# # Redefine score normalizing it
# max_score = control_words['score total (más bajo mejor)'].max()
# control_words['score normalizado'] = 1-control_words['score total (más bajo mejor)']/max_score

# # Output folder
# mistakes_control = 'Datos/mistakes_control'

# for i in control_words.index:
#     # Get the row information
#     control = control_words.iloc[i]
#     new_filename = os.path.join(mistakes_control, f'filtered_session{control["session"]}_trial{str(control["trial"]).zfill(2)}_channel{control["channel"]}.TextGrid')

#     # Open a textgrid as example to base the new grid
#     path_to_error = f'Datos/mistakes/filtered_session{29}_trial{str(25).zfill(2)}_channel{1}.TextGrid'
#     grid = textgrids.TextGrid(path_to_error)

#     # Modify it according to control information
#     grid[f"canal {control['channel']}"] = grid.pop(f"canal 1")
    
#     # Leave just two points in the grid (that will be control start and end points)
#     for j in range(len(grid[f"canal {control['channel']}"])-2):
#         _ = grid[f"canal {control['channel']}"].pop(j)
    
#     # Modify start 
#     text = f"1 Etiqueta: {control['error type']}, Posicion en IPU: no está anotado del todo claro, Longitud IPU: no está anotado del todo claro, Sesión + trial: s{control['session']}_objects_{trial}.TextGrid, Canal: {control['channel']}, Palabra del error: {control['control seleccionado']}"
#     xpos = float(control['info control seleccionado'].replace('(', '').split(', ')[0])
#     Point = namedtuple('Point', ['text', 'xpos'])
#     grid[f"canal {control['channel']}"][0] = Point(text, xpos)
    
#     # Modify end
#     text = f"2 Etiqueta: {control['error type']}, Posicion en IPU: no está anotado del todo claro, Longitud IPU: no está anotado del todo claro, Sesión + trial: s{control['session']}_objects_{trial}.TextGrid, Canal: {control['channel']}, Palabra del error: {control['control seleccionado']}"
#     xpos = float(control['info control seleccionado'].replace('(', '').split(', ')[1])
#     Point = namedtuple('Point', ['text', 'xpos'])
#     grid[f"canal {control['channel']}"][1] = Point(text, xpos)
    
#     # Save it
#     grid.write(new_filename)
#     # Re-open in ANSI, re-save in UTF-8
#     with open(new_filename, 'r', encoding='ansi') as f:
#         data = f.read()
#     with open(new_filename, 'w', encoding='utf-8') as f:
#         f.write(data)

# # # Check wether all errors have controls
# # mistake_files = [(int(file.split('session')[1][:2]),int(file.split('trial')[1][:2]),int(file.split('channel')[1][:1])) for file in os.listdir('Datos\\mistakes')]
# # # datos_file = [(int(file.split('s')[1][:2]), int(file.split('objects_')[1][:2])) for file in os.listdir('Datos\\errores\\Filtrados\\')]
# # control_files = []

# # for i in control_words.index:
# #     control = control_words.iloc[i]

# #     # Datos del archivo
# #     control_files.append((control['session'], control['trial'], control['channel']))
    
# #     # Contenido
# #     info_control = control['info control seleccionado'].replace('(','').replace(')','').split(', ')
# #     comienzo, final, score = float(info_control[0]), float(info_control[1]), -float(control['score normalizado'])

# # # No son conjuntos disjuntos


