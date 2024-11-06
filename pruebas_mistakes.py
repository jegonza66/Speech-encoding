import pandas as pd, os, textgrids

# Mistakes folder
mistake_folder = os.path.normpath(path='Datos/mistakes/')

# La sesion con más errores es la 25 --> 48 errores, 27 en canal 1 y 21 en canal 2. La señal del que habla, pero podemos implementarlo con las 4 variantes










































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