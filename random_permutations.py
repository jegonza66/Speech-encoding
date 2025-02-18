# Standard libraries
import os, numpy as np
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from funciones import load_pickle, dump_pickle, dict_to_csv, Suppress_print
from model_implementations import simulation_mtrf
from load import load_data
import config

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542

# ============
# RUN ANALYSIS
# ============
for situation in config.situations:
    start_time = datetime.now()
    for band in config.bands:
        for stim in config.stimuli:
            ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
            stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)

            # Update
            print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + config.model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Condition: ' + situation+'\n',f'Time interval: ({config.tmin},{config.tmax})s\n','\n===========================\n')
            
            # Relevant paths
            preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/')
            path_null = f'saves/{config.model}/{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            path_validation = f'saves/{config.model}/{situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')

            # Iterate over sessions
            for sesion in config.sesiones:
                print(f'\n------->\tStart of session {sesion}\n')
                
                # Load data by subject, EEG and info
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

                if config.just_load_data:
                    continue

                # Load stimuli by subject (i.e: concatenated stimuli features)
                stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')]) 
                stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])
                n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
                delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

                # Get relevant indexes
                relevant_indexes_1 = samples_info['keep_indexes1'].copy()
                relevant_indexes_2 = samples_info['keep_indexes2'].copy()

                # Initialize empty variables to store relevant data of each fold 
                null_weights_per_fold = np.zeros((config.n_folds, config.random_permutations, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
                null_correlation_per_channel_per_fold = np.zeros((config.n_folds, config.random_permutations, info['nchan']))
                null_errors_per_fold = np.zeros((config.n_folds, config.random_permutations, info['nchan']))

                # Run model for each subject
                for sujeto, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                # for sujeto, eeg, stims, relevant_indexes in zip([2], [eeg_sujeto_2], [stims_sujeto_2], [relevant_indexes_2]):
                    print(f'\n\t······  Running permutations for Subject {sujeto}\n')
                    
                    # Set alpha for specific subject
                    if config.set_alpha is None:
                        try:
                            alphas = load_pickle(path=alphas_path)
                            alpha = alphas[sesion][sujeto]
                        except:
                            alpha = config.default_alpha
                    else:
                        alpha = config.set_alpha
                
                    # Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
                    kf_test = KFold(config.n_folds, shuffle=False)
                    
                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]

                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        print(f'\n\t······  [{fold+1}/{config.n_folds}]')

                        # Run permutations 
                        null_weights_per_fold[fold], null_correlation_per_channel_per_fold[fold], null_errors_per_fold[fold] = fold_model(
                            fold=fold,
                            alpha=np.float32(alpha),#TODO adapt inside
                            stims=stims,
                            eeg=eeg,
                            relevant_indexes=relevant_indexes,
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,
                            validation=False,
                            shuffle=True,
                            statistical_test=False,
                            )                        
                    # Save permutations
                    os.makedirs(path_null, exist_ok=True)
                    dump_pickle(
                                path=path_null+ f'null_metrics_ses_{sesion}_sub_{sujeto}_{config.random_permutations}.pkl',
                                obj={
                                    'null_correlation_per_channel_per_fold':null_correlation_per_channel_per_fold, 
                                    'null_errors_per_fold':null_errors_per_fold
                                    },
                                rewrite=True
                                )
                    dump_pickle(
                                path=path_null+ f'null_weights_ses_{sesion}_sub_{sujeto}_{config.random_permutations}.pkl',
                                obj=null_weights_per_fold.mean(axis=0),
                                rewrite=True
                                )
                    print(f'\n\t······  Run permutations for Subject {sujeto}\n')

    # Get run time
    run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
    text = f'\n\n\t\t\tPARAMETERS  \n\n\tModel: ' + config.model +f'\n\tBands: {config.bands}'+'\n\tStimuli: ' + f'{config.stimuli}'+'\n\tCondition: ' +situation+f'\n\tTime interval: ({config.tmin},{config.tmax})s'+f'\n\tSessions: {config.sesiones}'
    if config.just_load_data:
        text += '\n\n\t\t\tJUST LOADING DATA'
    text += '\n\n\t\trandom_permutations.py'
    text += f'\n\n\t\t\tRUN TIME:{run_time}'

    # Dump metadata
    metadata_path = f'saves/log/permutations_{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
    os.makedirs(metadata_path, exist_ok=True)
    metadata = {
            name: getattr(config, name) for name in dir(config) 
            if (not name.startswith("__")) and (not callable(getattr(config, name)) and (name not in ['phonemes_to_ipa','ordered_phonemes']))
            }
    dict_to_csv(
                path=metadata_path+'metadata.csv',
                obj=metadata,
                rewrite=True
                )

    # Send text to telegram bot
    with Suppress_print():
        mensaje_tel(api_token=api_token,chat_id=chat_id, mensaje=text)
    print(text)
