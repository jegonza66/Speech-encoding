import time
# Standard libraries
from datetime import datetime
import os, numpy as np

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from funciones import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from load import load_data
import config, plot

# Load data by subject, EEG and info
situation, sesion, stim, band = 'External', 21, 'Phones-Discrete-Phonet', 'Theta'
preprocessed_data_path = f'saves/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/'

sujeto_1, sujeto_2, samples_info = load_data(
                                sesion=sesion,
                                stim=stim,
                                band=band,
                                sr=config.sr,
                                delays=config.delays,
                                preprocessed_data_path=preprocessed_data_path,
                                praat_executable_path=config.praat_executable_path,
                                situation=situation
                                )
eeg_sujeto_1, eeg_sujeto_2, info = sujeto_1['EEG'], sujeto_2['EEG'], sujeto_1['info']


# Load stimuli by subject (i.e: concatenated stimuli features)
stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')])
n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

relevant_indexes = samples_info['keep_indexes1'].copy()
from processing import shifted_matrix, shifted_matrix_2

times_shifted = []
times_shifted_base = []
for n in range(100):
    if n % 10 == 0:
        print(f"Iteration {n} of 100")
    
    t_0 = time.time()
    design_matrix = shifted_matrix(
                stims_sujeto_1, 
                delays=config.delays, 
                indices_to_keep=relevant_indexes,
                use_gpu=True,
                output_torch=True
                )
    times_shifted.append(time.time() - t_0)
    del design_matrix
    t_0 = time.time()
    design_matrix2 = shifted_matrix_2(
                stims_sujeto_1, 
                delays=config.delays, 
                use_gpu=True,
                indices_to_keep=relevant_indexes
                )
    times_shifted_base.append(time.time() - t_0)
    del design_matrix2

print(rf"Time to compute shifted matrix: $({np.mean(times_shifted)} ± {np.std(times_shifted)})$ seconds")
print(rf"Time to compute shifted matrix base: $({np.mean(times_shifted_base)} ± {np.std(times_shifted_base)})$ seconds")

design_matrix = shifted_matrix(
                np.array([[1,2,3,4],[5,6,7,8]]).T, 
                delays=[-2, -1, 0, 1, 2], 
                use_gpu=True,
                indices_to_keep=None
                )
import torch

def toeplitz(c: torch.Tensor, delays: torch.Tensor = None) -> torch.Tensor:
    """
    Crea una matriz de Toeplitz a partir de la primera columna `c`
    y, opcionalmente, la primera fila `r`. Si no se da `r`, se asume
    que la matriz es simétrica: r = c[0], c[1], ..., c[-1].

    Parámetros:
    -----------
    c : Tensor de forma (n,)
        Primera columna de la matriz.
    r : Tensor de forma (m,), opcional
        Primera fila de la matriz.

    Devuelve:
    ---------
    Tensor de forma (n, m)
        Matriz de Toeplitz.
    """
    # rellenar con ceros
    r = torch.cat([c[:1], torch.zeros(len(c) - 1)])
    
    # Construcción del índice de diferencia
    i = torch.arange(len(c)).unsqueeze(1)
    j = torch.arange(len(r)).unsqueeze(0)
    idx = i - j
    print(i)
    print(j)
    print(idx)

    # Creamos un vector extendido con los valores necesarios
    # r[0], r[1], ..., r[-1], c[1], ..., c[-1]
    vals = torch.cat([r.flip(0), c[1:]])

    # Desplazamos los índices para que estén en rango positivo
    return vals[idx + len(r) - 1]


c = torch.tensor([1., 2., 3., 4.])
delays = torch.tensor([-1, 0,1,2])


toeplitz(c, delays)

def shifted_matrix_fast(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Versión eficiente usando `unfold`, sin construcciones explícitas.
    Agrega ceros a la izquierda para emular desplazamientos.
    """
    x_pad = torch.nn.functional.pad(x, (n - 1, 0))  # (T + n - 1,)
    return x_pad.unfold(0, n, 1)  # (T, n)

shifted_matrix_fast(c, 2)