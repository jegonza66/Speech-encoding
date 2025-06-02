# Standard libraries
import numpy as np, os
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold
from tqdm import tqdm 

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from model_implementations import fold_model
from utils.plot import hyperparameter_selection
from load import load_data
import config

# Notofication bot
from utils.notification_telegram import tel_message, generate_completion_message
from telegram_config import API_TOKEN, CHAT_ID

     
# ============
# RUN ANALYSIS
# ============

# Start execution
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
            figures_path = os.path.normpath(f'figures/{config.model}_trace/{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}')
            
            path_validation = f'saves/{config.model}/{situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')
            
            # Try to access alphas
            try:
                alphas = load_pickle(path=alphas_path)
            except:
                alphas = {s: {} for s in config.sessions} 
        
            # Iterate over sessions
            for session in config.sessions:
                print(f'\n\n------->\tStart of session {session}\n')

                # Load data by subject, EEG and info
                subject_1, subject_2, samples_info = load_data(
                                                session=session,
                                                stim=stim,
                                                band=band,
                                                sr=config.sr,
                                                delays=config.delays,
                                                preprocessed_data_path=preprocessed_data_path,
                                                praat_executable_path=config.praat_executable_path,
                                                situation=situation
                                                )
                eeg_subject_1, eeg_subject_2, info = subject_1['EEG'], subject_2['EEG'], subject_1['info']
                
                if config.just_load_data:
                    continue

                # Load stimuli by subject (i.e: concatenated stimuli features)
                stims_subject_1 = np.hstack([subject_1[stimulus] for stimulus in stim.split('_')]) 
                stims_subject_2 = np.hstack([subject_2[stimulus] for stimulus in stim.split('_')])

                n_feats = [subject_1[stimulus].shape[1] for stimulus in stim.split('_')]
                delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

                # Get relevant indexes
                relevant_indexes_1 = samples_info['keep_indexes1'].copy()
                relevant_indexes_2 = samples_info['keep_indexes2'].copy()

                # Run model for each subject
                for subject, eeg, stims, relevant_indexes in zip((1, 2), (eeg_subject_1, eeg_subject_2), (stims_subject_1, stims_subject_2), (relevant_indexes_1, relevant_indexes_2)):
                    print(f'\n\n\t······  Running model for Subject {subject}\n')

                    # Take some metrics for each alpha
                    correlations = np.zeros(len(config.alphas_swept))
                    correlations_std = np.zeros(len(config.alphas_swept))
                    
                    # Make sweep 
                    correlation_per_channel = np.zeros((config.n_folds, len(config.alphas_swept)))

                    # Make the Kfold test
                    kf_test = KFold(config.n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]
                    
                    # Run folds 
                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        correlation_per_channel[fold] = fold_model(
                            fold=fold,
                            alpha=config.alphas_swept,
                            stims=stims,
                            eeg=eeg,
                            relevant_indexes=relevant_indexes,
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,  
                            validation=True
                            )     

                    # Calculate mean correlation and std
                    correlations = np.nan_to_num(np.nanmean(correlation_per_channel, axis=0))
                    correlations_std = np.nan_to_num(np.nanstd(correlation_per_channel, axis=0))
                    
                    # Find all indexes where the relative difference between the correlation and its maximum is within corr_limit_percent
                    relative_difference = abs((correlations.max() - correlations)/correlations.max())
                    good_indexes_range = np.where(relative_difference < config.val_correlation_limit_percentage)[0]

                    # Get the very last one, because the greater the alpha, the smoothest the signal gets
                    alpha_subject = config.alphas_swept[int(good_indexes_range[-1])]
                    
                    # Make the alpha selection process plot
                    hyperparameter_selection(
                                            alphas_swept=config.alphas_swept,
                                            correlations=correlations, 
                                            correlations_std=correlations_std, 
                                            alpha_subject=alpha_subject,
                                            correlation_limit_percentage=config.val_correlation_limit_percentage, 
                                            session=session, 
                                            subject=subject, 
                                            stim=stim, 
                                            band=band, 
                                            save_path=figures_path, 
                                            save=config.save_figures, 
                                            no_figures=config.no_figures
                                            )

                    # Update dictionary
                    alphas[session][subject] = alpha_subject

                    # Save results
                    os.makedirs(name=path_validation, exist_ok=True)
                    if config.save_alphas:
                        dump_pickle(path=alphas_path, obj=alphas, rewrite=True)
                    
                # Print the progress of the iteration
                iteration_percentage(txt=f'\n------->\tEnd of session {session}\n', i=config.sessions.index(session), length_of_iterator=len(config.sessions))

    # Get run time            
    run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
    text = f'PARAMETERS  \nModel: ' + config.model +f'\nBands: {config.bands}'+'\nStimuli: ' + f'{config.stimuli}'+'\nCondition: ' +situation+f'\nTime interval: ({config.tmin},{config.tmax})s'
    if config.just_load_data:
        text += '\n\n\tJUST LOADING DATA'
    else:
        text += f'\n\n\tvalidation.py'
    text += f'\n\n\t\t RUN TIME \n\n\t\t{run_time} hours'
    print(text)

    # Dump metadata
    metadata_path = f'saves/log/validation_{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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
        mensaje_tel(api_token=api_token, chat_id=chat_id, mensaje=text)
