# Standard libraries
import numpy as np, os
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

# Modules
from funciones import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from mtrf_models import Receptive_field_adaptation
from model_parallelization import parallel_fold_model
from plot import hyperparameter_selection
from load import load_data
import config

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542
     
# ============
# RUN ANALYSIS
# ============
start_time = datetime.now()
for band in config.bands:
    for stim in config.stimuli:
        ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
        stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)
        
        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + config.model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + config.situation+'\n',f'Time interval: ({config.tmin},{config.tmax})s\n','\n===========================\n')
        
        # Relevant paths
        preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/{config.situation}/tmin{config.tmin}_tmax{config.tmax}/')
        figures_path = os.path.normpath(f'figures/{config.model}_trace/{config.situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}')
        
        path_validation = f'saves/{config.model}/{config.situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
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
                                                        situation=config.situation
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
                standarized_betas = np.zeros(len(config.alphas_swept))
                errors = np.zeros(len(config.alphas_swept))
                correlations = np.zeros(len(config.alphas_swept))
                correlations_std = np.zeros(len(config.alphas_swept))
                
                # Make sweep
                for i_alpha, alpha in enumerate(config.alphas_swept):
                    weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
                    correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
                    rmse_per_channel = np.zeros((config.n_folds, info['nchan']))

                    # Make the Kfold test
                    kf_test = KFold(config.n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]
                    
                    # Run folds simultaneously
                    # results = Parallel(n_jobs=-1, verbose=0)(delayed(parallel_fold_model)(
                    #                                                                 fold=fold,
                    #                                                                 alpha=alpha,
                    #                                                                 stims=stims,
                    #                                                                 eeg=eeg,
                    #                                                                 relevant_indexes=relevant_indexes,
                    #                                                                 train_indexes=train_indexes,
                    #                                                                 test_indexes=test_indexes,                              
                    #                                                                 ) for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg))
                    #                                         )
                    results=[]
                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        results.append(parallel_fold_model(
                                        fold=fold,
                                        alpha=alpha,
                                        stims=stims,
                                        eeg=eeg,
                                        relevant_indexes=relevant_indexes,
                                        train_indexes=train_indexes,
                                        test_indexes=test_indexes,                              
                                        ) 
                                    )       
                    for fold, weights, correlation_matrix, root_mean_square_error in results:
                        weights_per_fold[fold] = weights
                        correlation_per_channel[fold] = correlation_matrix
                        rmse_per_channel[fold] = root_mean_square_error
                    
                    # Calculate mean correlation and std
                    correlations[i_alpha] = np.nan_to_num(np.nanmean(correlation_per_channel))
                    correlations_std[i_alpha] = np.nan_to_num(np.nanstd(correlation_per_channel))
                    print(f'\r·················· Sweeping progress  {int((i_alpha + 1) * 100 / config.steps)}% ··················', end='')
                print('\n')
                
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
text = f'PARAMETERS  \nModel: ' + config.model +f'\nBands: {config.bands}'+'\nStimuli: ' + f'{config.stimuli}'+'\nStatus: ' +config.situation+f'\nTime interval: ({config.tmin},{config.tmax})s'
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

# def parallel_alpha_search(
#     i_alpha:float, alpha:float, stims:np.ndarray, eeg:np.ndarray,
#     relevant_indexes:np.ndarray, info:dict, n_feats:list
#     ) -> tuple:
#     """
#     Performs parallel alpha search for model validation.
    
#     Parameters
#     ----------
#     i_alpha : float
#         Index of the current alpha value in the sweep.
#     alpha : float
#         Regularization parameter for the model.
#     stims : np.ndarray
#         Stimuli data.
#     eeg : np.ndarray
#         EEG data.
#     relevant_indexes : np.ndarray
#         Indexes of relevant data points.
#     info : dict
#         Dictionary containing information about the data.
#     n_feats : list
#         List containing the number of features for each stimulus.
    
#     Returns
#     -------
#     tuple
#         A tuple containing the index of the alpha, mean correlation, and standard deviation of the correlation.
#     """
#     weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
#     correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
#     rmse_per_channel = np.zeros((config.n_folds, info['nchan']))

#     # Make the Kfold test
#     kf_test = KFold(config.n_folds, shuffle=False)

#     # Keep relevant indexes for eeg
#     relevant_eeg = eeg[relevant_indexes]

#     for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
#         # print(f'\n\t\t······  [{fold+1}/{n_folds}]')

#         # Determine wether to run the model in parallel or not
#         # n_jobs=-1 if sum(n_feats)>1 else 1
        
#         # Implement mne model
#         mtrf = Receptive_field_adaptation(
#                                         tmin=config.tmin, 
#                                         tmax=config.tmax, 
#                                         sample_rate=config.sr, 
#                                         alpha=alpha, 
#                                         relevant_indexes=np.array(relevant_indexes),
#                                         train_indexes=train_indexes, 
#                                         test_indexes=test_indexes, 
#                                         stims_preprocess=config.stims_preprocess, 
#                                         eeg_preprocess=config.eeg_preprocess,
#                                         fit_intercept=False,
#                                         # n_jobs=n_jobs, 
#                                         n_jobs=-1,
#                                         estimator=config.estimator,
#                                         validation=True
#                                         )
        
#         # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
#         mtrf.fit(stims, eeg)
        
#         # Get weights coefficients shape n_chans, feats, delays
#         weights_per_fold[fold] = mtrf.coefs
        
#         # Predict and save
#         predicted, eeg_val = mtrf.predict(stims)
#         if (predicted==0).all():
#             print(f'\n\t\tFold {fold+1}/{config.n_folds} prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.')

#         # Calculates and saves correlation of each channel
#         try:
#             correlation_matrix = np.array([np.corrcoef(eeg_val[:, j], predicted[:, j])[0,1] for j in range(eeg_val.shape[1])])
#         except RuntimeWarning:
#             correlation_matrix = np.zeros(eeg_val.shape[1])
#         correlation_per_channel[fold] = correlation_matrix

#         # Calculates and saves root mean square error of each channel
#         root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_val), 2).mean(0)))
#         rmse_per_channel[fold] = root_mean_square_error

#     return i_alpha, np.nan_to_num(np.nanmean(correlation_per_channel)), np.nan_to_num(np.nanstd(correlation_per_channel))