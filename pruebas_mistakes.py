import pandas as pd, os, textgrids

# Mistakes folder
mistake_folder = os.path.normpath(path='C:/Users/User/repos/Speech-encoding/Datos/mistakes/')
data_info = pd.read_csv(filepath_or_buffer=os.path.join(mistake_folder, 'Filtrado_anotaciones.csv'), header=0, delimiter=';')

# La sesion con más errores es la 25 --> 48 errores, 27 en canal 1 y 21 en canal 2. La señal del que habla, pero podemos implementarlo con las 4 variantes
filtered_mistake_folder = os.path.join(mistake_folder,'Filtrados/')

# # Rename filepath in order to include the channel
# for file in [f for f in os.listdir(filtered_mistake_folder) if f.endswith('.TextGrid')]:
#     # Get the filename and open it to extract de chanel number
#     textgrids_path = os.path.normpath(os.path.join(filtered_mistake_folder, file))
#     grid = textgrids.TextGrid(textgrids_path)
#     channel_number = int(grid[list(grid.keys())[0]][0].text.split('Canal: ')[1][0])
    
#     # Define renamed file
#     session = int(file.split('s')[1][:2])
#     obj = int(file.split('_')[2][:2])
#     new_name_file = f'filtered_session{session}_object{obj}_channel{channel_number}.TextGrid'

#     # Rename the file to include this information in its name
#     os.rename(textgrids_path, os.path.normpath(os.path.join(filtered_mistake_folder, new_name_file)))

# Dictionary to store data
session_mistakes = {m:{} for m in [21,22,23,24,25,26,27,28,29,30]}
for file in [f for f in os.listdir(filtered_mistake_folder) if f.endswith('.TextGrid')]:
    session = int(file.split('session')[1][:2])
    if session==25:
        # # Define phrases filename
        # obj, channel = int(file.split('object')[1][0]), int(file.split('channel')[1][0])
        # phrases_fname = f"Datos/phrases/S{25}/s{25}.objects.{obj:02d}.channel{channel}.phrases"
        
        # # Get phrases and total time
        # phrases = pd.read_table(phrases_fname, header=None, sep="\t")
        # trial_tmax = phrases[1].iloc[-1]

        # Load grid with the data of the mistake
        grid = textgrids.TextGrid(os.path.normpath(os.path.join(filtered_mistake_folder, file)))
        
        # Points are tuples objects of textgrid: in this case the first element is text and the second the temporal position
        list_of_points = grid[list(grid.keys())[0]]
        
        # Store data of the mistakes in the session
        for i, point in enumerate(list_of_points):
            text, time = point
            mistakes = {}
            for element in text.split(','):
                key, value = element.split(':')
                
                # Modify the structure of the tagg of moment
                if 'Etiqueta' in key:
                    new_key, new_value = 'Etiqueta', (int(key.split(' ')[0]),value)

                # Filter unwanted blanck space
                else:
                    new_key = ' '.join([w for w in key.split(' ') if w!=''])
                    new_value = ' '.join([w for w in value.split(' ') if w!=''])

                mistakes[new_key] = new_value
            session_mistakes[session][i] = mistakes
        
        # Now iterates again to get onset of mistaken word (1) and the start of the closest word (3)
        for mistake in session_mistakes[session]:
            location, type_of_mistake = session_mistakes[session][mistake]['Etiqueta']

            # Find closest word of mistake
            if location==1:
                tagg_of_end, dummie_j = 1, 0
                while tagg_of_end!=3:
                    tagg_of_end = session_mistakes[session][mistake + dummie_j]['Etiqueta'][0]
                    dummie_j += 1
                # Store mistake and end of mistake
                
                print(mistake, mistake+dummie_j)
                


        # # Extend on more phoneme of silence till end of trial 
        # labels.append("")
        # times.append((ph.xmin, trial_tmax))
        # samples.append(np.round((trial_tmax - ph.xmax) * self.sr).astype("int"))

        # # If use envelope amplitude to make continuous stimuli: the total number of samples must match the samples use for stimuli
        # diferencia = np.sum(samples) - len(envelope)

        # if diferencia > 0:
        #     # Making the way back checking when does the number of samples of the ith phoneme exceed diferencia
        #     for ith_phoneme in [-i-1 for i in range(len(samples))]:
        #         if diferencia > samples[ith_phoneme]:
        #             diferencia -= samples[ith_phoneme]
        #             samples[ith_phoneme] = 0
        #         # When samples is greater than the difference, takes the remaining samples to match the envelope
        #         else:
        #             samples[ith_phoneme] -= diferencia
        #             break
        # elif diferencia < 0:
        #     # In this case, the last silence is prolonged
        #     samples[-1] -= diferencia
        
        # # Make a list with phoneme labels tha already are in the known set
        # updated_taggs = exp_info_labels + [ph for ph in np.unique(labels) if ph not in exp_info_labels]

        # # Repeat each label the number of times it was sampled
        # phonemes_tgrid = np.repeat(labels, samples)
        
        # # Make empty array of phonemes
        # phonemes = np.zeros(shape = (np.sum(samples), len(updated_taggs)))
        
        # # Match phoneme with kind
        # if kind.startswith('Phonemes-Envelope'):
        #     for i, tagg in enumerate(phonemes_tgrid):
        #         phonemes[i, updated_taggs.index(tagg)] = envelope[i]
        # elif kind.startswith('Phonemes-Discrete'):
        #     for i, tagg in enumerate(phonemes_tgrid):
        #         phonemes[i, updated_taggs.index(tagg)] = 1
        # elif kind.startswith('Phonemes-Onset'):
        #     # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
        #     phonemes_onset = [phonemes_tgrid[0]]
        #     for i in range(1, len(phonemes_tgrid)):
        #         if phonemes_tgrid[i] == phonemes_tgrid[i-1]:
        #             phonemes_onset.append(0)
        #         else:
        #             phonemes_onset.append(phonemes_tgrid[i])
        #     # Match phoneme with envelope
        #     for i, tagg in enumerate(phonemes_onset):
        #         if tagg!=0:
        #             phonemes[i, updated_taggs.index(tagg)] = 1
        # return phonemes