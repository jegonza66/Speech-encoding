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

def f_mistakes(self, envelope:np.ndarray):

    # Read phrases to identify time of error inside phrases time
    phrases = pd.read_table(f'Datos/phrases/S{session}/s{session}.objects.{str(trial).zfill(2)}.channel{channel}.phrases', header=None, sep="\t")
    start_time, end_time = phrases[0].iloc[0], phrases[1].iloc[-1]
    phrases_time = np.arange(start_time, end_time, 1/128)

    # Identify start and end of error within mistake
    mistake_code = {
                    'A':0, # articulatorio
                    'L':1, # léxico
                    'D':2 # discursivo
                    }
    mistake_signal = np.zeros(shape=(len(phrases_time), 3))

    textgrids_path = os.path.join(mistake_folder, f'filtered_session{session}_trial{trial}_channel{channel}.TextGrid')
    if os.path.isfile(textgrids_path):
        # Read textgrid        
        grid = textgrids.TextGrid(textgrids_path)[f'canal {channel}']
        
        # Identify onset, offset and type of mistake
        mistake_taggs, mistake_count = np.unique([el.text.split('Palabra del error: ')[1] for el in grid], return_counts=True)
        mistakes = {mistake:{'start':None, 'end':None, 'type':None} for mistake in mistake_taggs}
        mistake_taggs = np.repeat(mistake_taggs, mistake_count)

        for item, mistake in zip(grid, mistake_taggs):
            # Identify time_intervals and mistake type
            mistakes[mistake]['type'] = mistake_code[item.text.split('Etiqueta: ')[1][0]]
            if int(item.text[0])==1:
                mistakes[mistake]['start'] = item.xpos
            elif int(item.text[0])==2:
                mistakes[mistake]['end'] = item.xpos
            # else:
            #     nextword_start.append(item.xpos)

        # Fill mistake_signal
        for mistake in mistakes:
            onset_filter = mistakes[mistake]['start']<=phrases_time
            offset_filter = phrases_time<=mistakes[mistake]['end']
            
            mistake_signal[onset_filter&offset_filter, mistakes[mistake]['type']] = np.ones(shape=np.sum(onset_filter&offset_filter))
            
    # Match length of mistake signal with envelope
    difference = len(mistake_signal)-len(envelope)

    if difference>0:
        mistake_signal = mistake_signal[:-difference]
    elif difference<0:
        mistake_signal = np.concatenate((mistake_signal, np.zeros(shape=(np.abs(difference), 3))))

    return mistake_signal


# filtered_mistake_folder = os.path.join(mistake_folder,'Filtrados')

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
#         new_name_file = os.path.join(filtered_mistake_folder, f'filtered_session{session}_trial{trial}_channel{1}.TextGrid')
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
#         new_name_file = os.path.join(filtered_mistake_folder, f'filtered_session{session}_trial{trial}_channel{2}.TextGrid')
#         grid.write(new_name_file)
#         # Re-open in ANSI, re-save in UTF-8
#         with open(new_name_file, 'r', encoding='ansi') as f:
#             data = f.read()
#         with open(new_name_file, 'w', encoding='utf-8') as f:
#             f.write(data)
#     else:
#         channel = int(canales[0].split('canal ')[1])
#         # Save channel 2 with new name
#         new_name_file = os.path.join(filtered_mistake_folder, f'filtered_session{session}_trial{trial}_channel{channel}.TextGrid')
#         grid.write(new_name_file)
#         # Re-open in ANSI, re-save in UTF-8
#         with open(new_name_file, 'r', encoding='ansi') as f:
#             data = f.read()
#         with open(new_name_file, 'w', encoding='utf-8') as f:
#             f.write(data)

control_words = pd.read_csv(os.path.join(mistake_folder, 'score_con_palabras.csv'), header=0, delimiter=';')

# Filter unprecised data #TODO filtramos datos que no tienen el timestamp exacto
control_words = control_words[control_words['Datos Palabra Candidato']!='No esta exacto']
control_words = control_words[control_words['Datos Palabra Candidato']!='Falta dato']

arr = control_words['error seleccionado'].values
arr = filtered_df['error seleccionado'].values
change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
segments = np.split(arr, change_indices)
np.unique([len(segment) for segment in segments], return_counts=True)


##HACERLO POR LOS INDICES DEL DF CON ENUMERATE Y PONER DE SCORE 1-normalizado(score actuak)

def filter_repetitions(df, column, min_repetitions=4):
    # Extract the column as a numpy array
    arr = df[column].values
    
    # Find change points where the value in the column changes
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    
    df[column].iloc[change_indices]
    
    # Split the array into segments of contiguous values
    segments = np.split(arr, change_indices)
    
    # Filter segments that have min_repetitions or more repetitions
    filtered_segments = [segment[-min_repetitions:] for segment in segments if len(segment) >= min_repetitions]
    filtered_array = np.concatenate(filtered_segments)
    
    filtered_df = df[np.isin(arr, filtered_array)]
    return filtered_df
# Example usage
filtered_df = filter_repetitions(control_words, 'error seleccionado')
print(filtered_df)

# # Create a list of tuples with the element and its contiguous count
# [(segment[0], len(segment)) for segment in segments]

    

elements_of_tuple = [el.replace('(', '').replace(')', '').replace("'","") for el in control_words['Datos Palabra Candidato'][0].split(', ')]
mistake_tuple = tuple([float(el) if i<2 else el for i, el in enumerate(elements_of_tuple)])