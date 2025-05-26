from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

import config
from funciones import dump_pickle


number_of_ipus = {session: {'ch1':[], 'ch2':[]} for session in config.sessions}
len_of_ipus = {session: {'ch1':[], 'ch2':[]} for session in config.sessions}

data = pd.read_csv(r'Datos\turns\turn_table.csv')

for session in config.sessions:
    session_data = data[data['session_number']==session]
    number_of_ipus1 = []
    number_of_ipus2 = []
    len_of_ipus1 = []
    len_of_ipus2 = []
    
    
    trials_number_str = [trial.split('objects.')[1] for trial in sorted(session_data['trial_id'].unique())]
    for trial in sorted(session_data['trial_id'].unique()):
        trial_data = session_data[session_data['trial_id']==trial]
        trial_data_sorted = trial_data.sort_values(by=['ipu1_start_time'], inplace=False)[['ipu1_start_time', 'ipu1_end_time', 'ipu2_start_time', 'ipu2_end_time', 'speaker1', 'speaker2', 'tt_label']]
        
        ipus_ch1 = np.unique(
                np.concatenate(
                [
                    trial_data_sorted[['ipu1_start_time', 'ipu1_end_time']][trial_data_sorted['speaker1']=='channel1'].values,
                    trial_data_sorted[['ipu2_start_time', 'ipu2_end_time']][trial_data_sorted['speaker2']=='channel1'].values 
                ],
                axis=0,   
                ), 
                axis=0
                )
        ipus_ch2 = np.unique(
                np.concatenate(
                [
                    trial_data_sorted[['ipu1_start_time', 'ipu1_end_time']][trial_data_sorted['speaker1']=='channel2'].values,
                    trial_data_sorted[['ipu2_start_time', 'ipu2_end_time']][trial_data_sorted['speaker2']=='channel2'].values 
                ],
                axis=0,   
                ), 
                axis=0
                )
        number_of_ipus1.append(ipus_ch1.shape[0])
        number_of_ipus2.append(ipus_ch2.shape[0])
        
        len_of_ipus1.append(np.mean(ipus_ch2[:,1]-ipus_ch2[:,0]))
        len_of_ipus2.append(np.mean(ipus_ch1[:,1]-ipus_ch1[:,0]))
        
        
        # fig = plt.figure(
        #     figsize=(6,4),
        #     tight_layout=True
        #     )
        # time = np.arange(0, trial_data_sorted['ipu2_end_time'].max(), 0.00001)
        # ax = plt.subplot(2,1,1)
        # for ipu in ipus_ch1:
        #     ax.fill_between(
        #         time,
        #         0,
        #         1,
        #         where=(time >= ipu[0]) & (time <= ipu[1]),
        #         color='blue',
        #         # label='Speaker 1'           
        #         alpha=0.5,
        #         )
        # ax.set_title('Speaker 1')
        # ax2 = plt.subplot(2,1,2)
        # for ipu in ipus_ch2:
        #     ax2.fill_between(
        #         time,
        #         0,
        #         1,
        #         where=(time >= ipu[0]) & (time <= ipu[1]),
        #         color='orange',
        #         # label='Speaker 1'           
        #         alpha=0.5,
        #         )
        # ax2.set_title('Speaker 2')
        # fig.show()
    number_of_ipus[session]['ch1'] = number_of_ipus1
    number_of_ipus[session]['ch2'] = number_of_ipus2
    len_of_ipus[session]['ch1'] = len_of_ipus1
    len_of_ipus[session]['ch2'] = len_of_ipus2
        
leader_according_number_of_ipus = {
    session: {
        'lead':np.zeros(len(number_of_ipus[session]['ch1']), dtype=int),
        'indexes':np.zeros(len(number_of_ipus[session]['ch1']), dtype=int)
        }
    for session in config.sessions
    }
for session in config.sessions:
    number_of_trials = len(leader_according_number_of_ipus[session]['lead'])
    for trial in range(number_of_trials):
        if number_of_ipus[session]['ch1'][trial] > number_of_ipus[session]['ch2'][trial]:
            leader_according_number_of_ipus[session]['lead'][trial] = 1
        else:
            leader_according_number_of_ipus[session]['lead'][trial] = 2
    indexes = np.arange(number_of_trials)
    indexes[(leader_according_number_of_ipus[session]['lead']==1)] = np.arange((leader_according_number_of_ipus[session]['lead']==1).sum())
    indexes[(leader_according_number_of_ipus[session]['lead']==2)] = np.arange((leader_according_number_of_ipus[session]['lead']==2).sum())
    leader_according_number_of_ipus[session]['indexes'] = indexes
dump_pickle(path=r'Datos/turns/leader_according_number_of_ipus.pkl', obj=leader_according_number_of_ipus, rewrite=True)

leader_according_len_of_ipus = {
    session: {
        'lead':np.zeros(len(len_of_ipus[session]['ch1']), dtype=int),
        'indexes':np.zeros(len(len_of_ipus[session]['ch1']), dtype=int)
        }
    for session in config.sessions
    }
for session in config.sessions:
    number_of_trials = len(leader_according_len_of_ipus[session]['lead'])
    for trial in range(number_of_trials):
        if len_of_ipus[session]['ch1'][trial] > len_of_ipus[session]['ch2'][trial]:
            leader_according_len_of_ipus[session]['lead'][trial] = 1
        else:
            leader_according_len_of_ipus[session]['lead'][trial] = 2
    indexes = np.arange(number_of_trials)
    indexes[(leader_according_len_of_ipus[session]['lead']==1)] = np.arange((leader_according_len_of_ipus[session]['lead']==1).sum())
    indexes[(leader_according_len_of_ipus[session]['lead']==2)] = np.arange((leader_according_len_of_ipus[session]['lead']==2).sum())
    leader_according_len_of_ipus[session]['indexes'] = indexes
dump_pickle(path=r'Datos/turns/leader_according_len_of_ipus.pkl', obj=leader_according_len_of_ipus, rewrite=True)


# total_number_of_ipus = {
#     session:{'ch1' : sum(number_of_ipus[session]['ch1']),
#      'ch2' : sum(number_of_ipus[session]['ch2'])
#     } for session in config.sessions}

# for session in config.sessions:
    
#     plt.figure(figsize=(10,5))
#     plt.title('Number of IPUs per trial')
#     plt.xlabel('Trial')
#     plt.ylabel('Number of IPUs')
#     plt.plot(
#         np.arange(1, len(number_of_ipus[session]['ch1'])+1),
#         number_of_ipus[session]['ch1'],
#         label=f'Session {session} - Channel 1',
#         )
#     plt.plot(
#         np.arange(1, len(number_of_ipus[session]['ch2'])+1),
#         number_of_ipus[session]['ch2'],
#         label=f'Session {session} - Channel 2',
#         )
#     plt.legend()
#     plt.show(block=False)
# 0.0	3.109072	#
# 3.109072	3.838978	el mimo
# 3.838978	5.189837	#
# 5.189837	7.78501	arriba del búho a la derecha de la oreja con
# 7.78501	8.215539	#
# 8.215539	10.373356	espacio entre ambos está bien a

def calcular_proporcion_habla(filepath):
    total_tiempo = 0.0
    tiempo_hablado = 0.0

    with open(filepath, encoding='utf-8') as f:
        for linea in f:
            if linea.strip() == "" or linea.startswith("//"):
                continue
            partes = linea.strip().split('\t')
            if len(partes) != 3:
                continue
            inicio, fin, texto = partes
            inicio, fin = float(inicio), float(fin)
            duracion = fin - inicio
            total_tiempo += duracion
            if texto.strip() != "#":
                tiempo_hablado += duracion

    if total_tiempo == 0:
        return 0
    return tiempo_hablado, total_tiempo

# Ejemplo para todos los archivos .phrases en una carpeta
tiempo_hablado = {session:{'ch1':None, 'ch2':None} for session in config.sessions}

for session in config.sessions:
    archivos = glob.glob(fr"Datos/phrases/S{session}/*.phrases")
    archivos1 = [archivo for archivo in archivos if archivo.endswith("1.phrases")]
    archivos2 = [archivo for archivo in archivos if archivo.endswith("2.phrases")]

    tiempos_hablado1 = []
    tiempos_hablado2 = []
    tiempos_totales = []
    for archivo1, archivo2 in zip(archivos1, archivos2):
        tiempo_hablado1, total_tiempo = calcular_proporcion_habla(archivo1)
        tiempo_hablado2, _ = calcular_proporcion_habla(archivo2)
        tiempos_hablado1.append(tiempo_hablado1)
        tiempos_hablado2.append(tiempo_hablado2)
        tiempos_totales.append(total_tiempo)
        
    tiempo_hablado[session]['ch1'] = sum(tiempos_hablado1)/sum(tiempos_totales)
    tiempo_hablado[session]['ch2'] = sum(tiempos_hablado2)/sum(tiempos_totales) 

        # print(
        #     f"Sesión {archivo1.split('/S')[1][:2]}\n \t"
        #     f"1: {proporcion1:.2%}",
        #     f"2: {proporcion2:.2%}\n",
        #     f'El participante 1 habló {proporcion1/proporcion2:.2f} veces más',
        #     )

        
# tiempo_hablado
# total_number_of_ipus

# # Extraer listas de proporciones de habla y número total de IPUs por canal y sesión
# tiempo_hablado_ch1 = []
# tiempo_hablado_ch2 = []
# ipus_ch1 = []
# ipus_ch2 = []

# for session in config.sessions:
#     tiempo_hablado_ch1.append(tiempo_hablado[session]['ch1'])
#     tiempo_hablado_ch2.append(tiempo_hablado[session]['ch2'])
#     ipus_ch1.append(total_number_of_ipus[session]['ch1'])
#     ipus_ch2.append(total_number_of_ipus[session]['ch2'])

# # Calcular correlación de Pearson para cada canal
# corr_ch1, pval_ch1 = pearsonr(tiempo_hablado_ch1, ipus_ch1)
# corr_ch2, pval_ch2 = pearsonr(tiempo_hablado_ch2, ipus_ch2)

# print(f"Canal 1: r = {corr_ch1:.3f}, p = {pval_ch1:.3g}")
# print(f"Canal 2: r = {corr_ch2:.3f}, p = {pval_ch2:.3g}")