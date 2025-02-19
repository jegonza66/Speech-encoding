# Standard libraries
import numpy as np, os
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold
from tqdm import tqdm 

# Modules
from funciones import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from model_implementations import fold_model
from plot import hyperparameter_selection
from processing import shifted_matrix_2
from load import load_data
import config

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542
     
# ============
# RUN ANALYSIS
# ============
for situation in config.situations:    
    # Start execution
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
                alphas = {s: {} for s in config.sesiones} 
        
            # Iterate over sessions
            for sesion in config.sesiones:
                print(f'\n\n------->\tStart of session {sesion}\n')

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

                # Run model for each subject
                for subject, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                    print(f'\n\n\t······  Running model for Subject {subject}\n')

                    # Take some metrics for each alpha
                    correlations = np.zeros(len(config.alphas_swept))
                    correlations_std = np.zeros(len(config.alphas_swept))
                    
                    if config.precomputed_design_matrix:
                        os.makedirs('temporal', exist_ok=True)
                        design_matrix = shifted_matrix_2(
                            stims, 
                            delays=config.delays, 
                            use_gpu=config.use_gpu,
                            indices_to_keep=relevant_indexes
                            )
                        precomputed_design_matrix_path = os.path.join('temporal', 'design_matrix.pkl')
                        dump_pickle(
                            path=precomputed_design_matrix_path, 
                            obj=design_matrix, 
                            rewrite=True, 
                            verbose=True
                        )
                    else:
                        precomputed_design_matrix_path = None
                    
                    # Make sweep
                    for i_alpha, alpha in tqdm(enumerate(config.alphas_swept), total=len(config.alphas_swept), desc='Sweeping progress'):
                        weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
                        correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
                        rmse_per_channel = np.zeros((config.n_folds, info['nchan']))

                        # Make the Kfold test
                        kf_test = KFold(config.n_folds, shuffle=False)

                        # Keep relevant indexes for eeg
                        relevant_eeg = eeg[relevant_indexes]
                        
                        # Run folds 
                        k_models_output = []
                        for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                            k_models_output.append(
                                            fold_model(
                                            fold=fold,
                                            alpha=alpha,
                                            stims=stims,
                                            eeg=eeg,
                                            relevant_indexes=relevant_indexes,
                                            train_indexes=train_indexes,
                                            test_indexes=test_indexes,  
                                            validation=True,
                                            precomputed_design_matrix_path=precomputed_design_matrix_path
                                            ) 
                                            )     
                        # Unpack model outputs  
                        for fold, weights, correlation_matrix, root_mean_square_error in k_models_output:
                            weights_per_fold[fold] = weights
                            correlation_per_channel[fold] = correlation_matrix
                            rmse_per_channel[fold] = root_mean_square_error
                        
                        # Calculate mean correlation and std
                        correlations[i_alpha] = np.nan_to_num(np.nanmean(correlation_per_channel))
                        correlations_std[i_alpha] = np.nan_to_num(np.nanstd(correlation_per_channel))
                    
                    if config.precomputed_design_matrix:
                        os.remove(precomputed_design_matrix_path) 
                    
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
                                            session=sesion, subject=subject, 
                                            stim=stim, 
                                            band=band, 
                                            save_path=figures_path, 
                                            save=config.save_figures, 
                                            no_figures=config.no_figures
                                            )

                    # Update dictionary
                    alphas[sesion][subject] = alpha_subject

                    # Save results
                    os.makedirs(name=path_validation, exist_ok=True)
                    if config.save_alphas:
                        dump_pickle(path=alphas_path, obj=alphas, rewrite=True)
                    
                # Print the progress of the iteration
                iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=config.sesiones.index(sesion), length_of_iterator=len(config.sesiones))

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
