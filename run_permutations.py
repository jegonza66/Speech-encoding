# Standard libraries
import os, numpy as np
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from funciones import load_pickle, dump_pickle, dict_to_csv
from simulations import simulation_mtrf
from load import load_data

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542

start_time = datetime.now()

# ==========
# PARAMETERS
# ==========

# Run setup
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]

# EEG sample rate
sr = 128

# Run times
tmin, tmax = -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# Stimuli, EEG frecuency band and dialogue situation
stimuli = ['Pitch-Log-Raw', 'Spectrogram', 'Deltas','Phonemes-Discrete-Manual']
# stimuli = ['Envelope', 'Phonological', 'Deltas', 'Phonemes-Discrete-Manual','Pitch-Log-Raw','Spectrogram']
stimuli = ['Phonemes-Discrete-Manual_Pitch-Log-Raw_Envelope', 'Phonemes-Discrete-Manual_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw']

bands = ['Theta']#['Alpha', 'Beta1', 'All']
situation = 'External'

# Model and normalization of input
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'
model = 'mtrf'

# Preset alpha (penalization parameter)
alpha_correlation_limit = 0.01
default_alpha = 1000
alphas_fname = f'saves/alphas/alphas_corr{alpha_correlation_limit}.pkl'
try:
    alphas = load_pickle(path=alphas_fname)
except:
    print('\n\nAlphas file not found.\n\n')

# Number of folds and iterations
iterations = 200
n_folds = 5

# if model == 'Decoding':
#     t_lags_fname = f'saves/mtrf/Decoding_t_lag/{situation}/correlations/tmin{tmin}_tmax{tmax}/Max_t_lags.pkl'
#     try:
#         t_lags = load_pickle(path=t_lags_fname)
#     except:
#         print('\n\nMean_Correlations file not found\n\n')
#         t_lags = {}

# ============
# RUN ANALYSIS
# ============
just_load_data = False

for band in bands:
    for stim in stimuli:
        ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
        stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)

        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + situation+'\n',f'Time interval: ({tmin},{tmax})s\n','\n===========================\n')
        
        # Relevant paths
        preprocessed_data_path = f'saves/preprocessed_data/{situation}/tmin{tmin}_tmax{tmax}/{band}/'
        path_null = f'saves/{model}/{situation}/null/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
        praat_executable_path=r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe" #r"C:\Program Files\Praat\Praat.exe"

        # Iterate over sessions
        for sesion in sesiones:
            print(f'\n------->\tStart of session {sesion}\n')
            
            # Load data by subject, EEG and info
            sujeto_1, sujeto_2, samples_info = load_data(sesion=sesion,
                                                         stim=stim,
                                                         band=band,
                                                         sr=sr,
                                                         delays=delays,
                                                         preprocessed_data_path=preprocessed_data_path,
                                                         praat_executable_path=praat_executable_path,
                                                         situation=situation,
                                                         silence_threshold=0.03)
            eeg_sujeto_1, eeg_sujeto_2, info = sujeto_1['EEG'], sujeto_2['EEG'], sujeto_1['info']

            if just_load_data:
                continue

            # Load stimuli by subject (i.e: concatenated stimuli features)
            stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')]) 
            stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])
            n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
            delayed_length_per_stimuli = [n_feat*len(delays) for n_feat in n_feats]

            # Get relevant indexes
            relevant_indexes_1 = samples_info['keep_indexes1'].copy()
            relevant_indexes_2 = samples_info['keep_indexes2'].copy()

            # Initialize empty variables to store relevant data of each fold 
            null_weights_per_fold = np.zeros((n_folds, iterations, info['nchan'], np.sum(n_feats), len(delays)), dtype=np.float16)
            null_correlation_per_channel_per_fold = np.zeros((n_folds, iterations, info['nchan']))
            null_errors_per_fold = np.zeros((n_folds, iterations, info['nchan']))

            # if model == 'decoding':
            #     null_patterns = np.zeros((n_folds, iterations, info['nchan'], sum(delayed_length_per_stimuli)), dtype=np.float16)

            # Run model for each subject
            for sujeto, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                print(f'\n\t······  Running permutations for Subject {sujeto}\n')
                
                # Set alpha for specific subject
                try:
                    alpha = alphas[band][stim][sesion][sujeto]
                except:
                    alpha = default_alpha
            
                # Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
                kf_test = KFold(n_folds, shuffle=False)
                
                # Keep relevant indexes for eeg
                relevant_eeg = eeg[relevant_indexes]

                for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                    print(f'\n\t······  [{fold+1}/{n_folds}]')

                    # Determine wether to run the model in parallel or not
                    n_jobs=-1 if sum(n_feats)>1 else 1

                    # Run permutations # TODO parallelize it
                    null_weights_per_fold, null_correlation_per_channel_per_fold, null_errors_per_fold = simulation_mtrf(
                        iterations=iterations,
                        fold=fold,
                        stims=stims,
                        eeg=eeg,
                        sr=sr,
                        tmin=tmin,
                        tmax=tmax,
                        relevant_indexes=relevant_indexes,
                        alpha=alpha,
                        train_indexes=train_indexes,
                        test_indexes=test_indexes,
                        stims_preprocess=stims_preprocess,
                        eeg_preprocess=eeg_preprocess,
                        null_correlation=null_correlation_per_channel_per_fold,
                        null_weights=null_weights_per_fold,
                        null_errors=null_errors_per_fold, 
                        n_jobs=n_jobs)
                    # if model == 'decoding':
                    #     t_lag = np.where(times == t_lags[band])[0][0]
                    #     Fake_Model = mtrf_models.mne_mtrf_decoding(tmin, tmax, sr, info, alpha, t_lag)
                    #     Pesos_fake, Patterns_fake, Correlaciones_fake, Errores_fake = \
                    #         processing.simular_iteraciones_decoding(Fake_Model, iteraciones, sesion, sujeto, fold,
                    #                                                dstims_train_val, eeg_train_val, dstims_test,
                    #                                                eeg_test, Pesos_fake, Patterns_fake, Correlaciones_fake,
                    #                                                Errores_fake)

                # Save permutations
                os.makedirs(path_null, exist_ok=True)
                dump_pickle(path=path_null+ f'Corr_Rmse_fake_Sesion{sesion}_Sujeto{sujeto}.pkl',
                            obj=(null_correlation_per_channel_per_fold, null_errors_per_fold),
                            rewrite=True)
                dump_pickle(path=path_null+ f'Pesos_fake_Sesion{sesion}_Sujeto{sujeto}.pkl',
                            obj=null_weights_per_fold.mean(axis=0),
                            rewrite=True)

                # if model == 'Decoding':
                #     dump_pickle(path=path_null + f'Patterns_fake_Sesion{sesion}_Sujeto{sujeto}.pkl',
                #                       obj=Patterns_fake.mean(axis=0),
                #                       rewrite=True)
                print(f'\n\t······  Run permutations for Subject {sujeto}\n')

# Get run time            
run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
text = f'PARAMETERS  \nPermutation Model: ' + model +f'\nBands: {bands}'+'\nStimuli: ' + f'{stimuli}'+'\nStatus: ' +situation+f'\nTime interval: ({tmin},{tmax})s'
if just_load_data:
    text += '\n\n\tJUST LOADING DATA'
text += f'\n\n\t\t RUN TIME \n\n\t\t{run_time} hours'
print(text)

# Dump metadata
metadata_path = f'saves/log/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
os.makedirs(metadata_path, exist_ok=True)
metadata = {'stimuli':stimuli, 'bands':bands, 'situation':situation, 
            'sesiones':sesiones, 'sr':sr, 'tmin':tmin, 'tmax':tmax, 
            'stims_preprocess':stims_preprocess, 'eeg_preprocess':eeg_preprocess, 'model':model, 
            'n_folds':n_folds, 'default_alpha':default_alpha,
            'preprocessed_data_path':preprocessed_data_path,
            'path_null':path_null,'metadata_path': metadata_path,
            'n_folds':n_folds, 'n_iterations':iterations,
            'date': str(datetime.now()), 'run_time':str(run_time), 'just_loading_data':just_load_data}
dict_to_csv(path=metadata_path+'metadata.csv', 
            obj=metadata,
            rewrite=True)

# Send text to telegram bot
mensaje_tel(api_token=api_token,chat_id=chat_id, mensaje=text)