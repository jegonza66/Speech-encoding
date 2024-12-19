# Standard libraries
from datetime import datetime
import os
import numpy as np
import warnings

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from funciones import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from mtrf_models import Receptive_field_adaptation
from load import load_data
from processing import tfce
from setup import exp_info
import config, plot

# Notification bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542

# ============
# RUN ANALYSIS
# ============
start_time = datetime.now()

for band in config.bands:
    for stim in config.stimuli:
        sorted_stimuli, sorted_bands = sorted(stim.split('_')), sorted(band.split('_'))
        stim, band = '_'.join(sorted_stimuli), '_'.join(sorted_bands)

        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + config.model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + config.situation+'\n',f'Time interval: ({config.tmin},{config.tmax})s\n','\n===========================\n')

        # Relevant paths
        save_results_path = f'saves/{config.model}/{config.situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/{band}/'
        preprocessed_data_path = f'saves/preprocessed_data/{config.situation}/tmin{config.tmin}_tmax{config.tmax}/'
        path_weights = f'saves/{config.model}/{config.situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
        path_null = f'saves/{config.model}/{config.situation}/null/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
        path_figures = f'figures/{config.model}/{config.situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
        prat_executable_path = r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe" #r"C:\Program Files\Praat\Praat.exe"#
        alphas_directory = os.path.normpath(f'saves/alphas/{config.situation}/stims_{config.stims_preprocess}/EEG_{config.eeg_preprocess}//tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/')
        alphas_path = os.path.join(alphas_directory, f'corr_limit_{config.correlation_limit_percentage}.pkl')
        path_TFCE = f'saves/{config.model}/{config.situation}/TFCE/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/'

        # Make lists to store relevant data across sobjects
        average_weights_subjects = []
        average_correlation_subjects = []
        average_rmse_subjects = []
        pvalues_corr_subjects = []
        pvalues_rmse_subjects = []
        repeated_good_correlation_channels_subjects = []
        repeated_good_rmse_channels_subjects = []
        phonemes_occurrences = {sesion:{} for sesion in config.sesiones}

        # Store total number of subjects (18) to save figures and results just in this case
        total_number_of_subjects = 0

        # Iterate over sessions
        for sesion in config.sesiones:
            print(f'\n------->\tStart of session {sesion}\n')

            # Load data by subject, EEG and info
            sujeto_1, sujeto_2, samples_info = load_data(sesion=sesion,
                                                         stim=stim,
                                                         band=band,
                                                         sr=config.sr,
                                                         delays=config.delays,
                                                         preprocessed_data_path=preprocessed_data_path,
                                                         praat_executable_path=prat_executable_path,
                                                         situation=config.situation,
                                                         silence_threshold=0.03
                                                         )
            eeg_sujeto_1, eeg_sujeto_2, info = sujeto_1['EEG'], sujeto_2['EEG'], sujeto_1['info']

            if config.just_load_data:
                continue

            # Load stimuli by subject (i.e: concatenated stimuli features)
            stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')])
            stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])
            n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
            delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

            # Store phonemes ocurrences to make boxplot
            for stimulus in stim.split('_'):
                if stimulus.startswith('Phonemes'):
                    # Change to 1's every value that isn't 0. In this way the method works for every kind
                    matrix_1 = sujeto_1[stimulus].copy()
                    matrix_1[matrix_1!=0.] = 1
                    matrix_2 = sujeto_2[stimulus].copy()
                    matrix_2[matrix_2!=0.] = 1
                    matrix = matrix_1 + matrix_2

                    # Identify the phonemes
                    phonemes = exp_info()
                    if stimulus.endswith('Manual'):
                        phonemes = phonemes.ph_labels_man
                    elif stimulus.endswith('Phonet'):
                        phonemes = [el if el!='<p:>' else '' for el in phonemes.ph_labels_phonet]
                    else:
                        phonemes = phonemes.ph_labels
                    phonemes_occurrences[sesion][stimulus] = {'phonemes':phonemes, 'count':np.sum(matrix, axis=0)}
                else:
                    pass

            # Get relevant indexes
            relevant_indexes_1 = samples_info['keep_indexes1'].copy()
            relevant_indexes_2 = samples_info['keep_indexes2'].copy()

            # Initialize empty variables to store relevant data of each fold
            weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float16)
            correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
            rmse_per_channel = np.zeros((config.n_folds, info['nchan']))

            # Variable to store all channel's p-value
            topo_pvalues_corr = np.zeros((config.n_folds, info['nchan']))
            topo_pvalues_rmse = np.zeros((config.n_folds, info['nchan']))

            # Variable to store p-value of significant channels
            proba_correlation_per_channel = np.ones((config.n_folds, info['nchan']))
            proba_rmse_per_channel = np.ones((config.n_folds, info['nchan']))

            # Variable to store significant channels
            repeated_good_correlation_channels = np.zeros(info['nchan'])
            repeated_good_rmse_channels = np.zeros(info['nchan'])

            # Run model for each subject
            for sujeto, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                print(f'\n\t······  Running model for Subject {sujeto}\n')

                # Set alpha for specific subject
                if config.set_alpha is None:
                    try:
                        alphas = load_pickle(path=alphas_path)
                        alpha = alphas[sesion][sujeto]
                    except:
                        alpha = config.default_alpha
                else:
                    alpha = config.set_alpha

                # Make the Kfold test
                kf_test = KFold(config.n_folds, shuffle=False)

                # Keep relevant indexes for eeg
                relevant_eeg = eeg[relevant_indexes]
                for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                    print(f'\n\t······  [{fold+1}/{config.n_folds}]')

                    # Determine wether to run the model in parallel or not
                    n_jobs=-1 if sum(n_feats)>1 else 1

                    # Implement mne model
                    mtrf = Receptive_field_adaptation(
                                                    tmin=config.tmin,
                                                    tmax=config.tmax,
                                                    sample_rate=config.sr,
                                                    alpha=alpha,
                                                    relevant_indexes=np.array(relevant_indexes),
                                                    train_indexes=train_indexes,
                                                    test_indexes=test_indexes,
                                                    stims_preprocess=config.stims_preprocess,
                                                    eeg_preprocess=config.eeg_preprocess,
                                                    fit_intercept=False,
                                                    n_jobs=n_jobs,
                                                    estimator=config.estimator
                                                    )

                    # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
                    mtrf.fit(stims, eeg)

                    # Get weights coefficients shape n_chans, feats, delays
                    weights_per_fold[fold] = mtrf.coefs

                    # Predict and save
                    predicted, eeg_test = mtrf.predict(stims)
                    if (predicted==0).all():
                        print(f'\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\t\tFold {fold+1}/{config.n_folds} prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    
                    # Calculates and saves correlation of each channel
                    # warnings.filterwarnings("ignore", category=RuntimeWarning) # avoid runtime error dividing per zero, this is caught later
                    try:
                        correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
                    except RuntimeWarning:
                        correlation_matrix = np.zeros(eeg_test.shape[1])
                    correlation_per_channel[fold] = correlation_matrix

                    # Calculates and saves root mean square error of each channel
                    root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                    rmse_per_channel[fold] = root_mean_square_error

                    # Perform statistical test
                    if config.statistical_test:
                        null_correlation_per_channel, null_errors = load_pickle(path=path_null + f'Corr_Rmse_fake_Sesion{sesion}_Sujeto{sujeto}.pkl')
                        iterations =  null_correlation_per_channel.shape[1]

                        # Correlation and RMSE
                        null_correlation_matrix = null_correlation_per_channel[fold]
                        null_root_mean_square_error = null_errors[fold]

                        # p-values for both tests
                        p_corr = ((null_correlation_matrix > correlation_matrix).sum(0) + 1) / (iterations + 1)
                        p_rmse = ((null_root_mean_square_error < root_mean_square_error).sum(0) + 1) / (iterations + 1)

                        # Threshold
                        proba_correlation_per_channel[fold][p_corr < config.umbral] = p_corr[p_corr < config.umbral]
                        proba_rmse_per_channel[fold][p_rmse < config.umbral] = p_rmse[p_rmse < config.umbral]

                        # p-value topographic distribution
                        topo_pvalues_corr[fold] = p_corr
                        topo_pvalues_rmse[fold] = p_rmse
                
                print(f'\n\t······  Run model\n')

                # Take average weights, avoiding folds fill entirely with zeros
                for k, weight in enumerate(weights_per_fold):
                    if (weight==0).all():
                        weights_per_fold[k] = np.full(shape=weight.shape, fill_value=np.nan)
                        print(
                            f'\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>\n'
                            f'\t\tFold {k+1}/{config.n_folds} weights are empty\n'
                            f'\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>'
                        )
                        
                average_weights = np.nanmean(weights_per_fold, axis=0) # info['nchan'], np.sum(n_feats), len(delays)
                average_weights = np.nan_to_num(average_weights)
                                
                # Take average correlation and RMSE between folds of all channels
                average_correlation = np.nanmean(correlation_per_channel, axis=0)
                average_correlation = np.nan_to_num(average_correlation)
                average_rmse = rmse_per_channel.mean(axis=0)

                # Channels that pass the tests
                corr_good_channel_indexes = []
                rmse_good_channel_indexes = []

                if config.statistical_test:
                    # Correlation and RMSE of channels that pass the test
                    corr_good_channel_indexes, = np.where(np.all((proba_correlation_per_channel < 1), axis=0))
                    rmse_good_channel_indexes, = np.where(np.all((proba_rmse_per_channel < 1), axis=0))

                    # Saves passing channels by subject
                    repeated_good_correlation_channels[corr_good_channel_indexes] += 1
                    repeated_good_rmse_channels[rmse_good_channel_indexes] += 1

                    # Plot shadows
                    plot.null_correlation_vs_correlation_good_channels(display_interactive_mode=config.display_interactive_mode, session=sesion, subject=sujeto,
                                              save_path=path_figures, good_channels_indexes=corr_good_channel_indexes, average_correlation=average_correlation,
                                              save=config.save_figures, correlation_per_channel=correlation_per_channel,
                                              null_correlation_per_channel=null_correlation_per_channel, no_figures=config.no_figures)

                # Adapt to yield average p-values
                topo_pval_corr_sujeto = topo_pvalues_corr.mean(axis=0)
                topo_pval_rmse_sujeto = topo_pvalues_rmse.mean(axis=0)

                # Plot head topomap across al channel for correlation and rmse
                plot.topomap(good_channels_indexes=corr_good_channel_indexes, average_coefficient=average_correlation, info=info,
                             coefficient_name='Correlation', save=config.save_figures, display_interactive_mode=config.display_interactive_mode,
                             save_path=path_figures, subject=sujeto, session=sesion, no_figures=config.no_figures)
                plot.topomap(good_channels_indexes=rmse_good_channel_indexes, average_coefficient=average_rmse, info=info,
                             coefficient_name='RMSE', save=config.save_figures, display_interactive_mode=config.display_interactive_mode,
                             save_path=path_figures, subject=sujeto, session=sesion, no_figures=config.no_figures)

                # Plot weights
                plot.channel_weights(info=info, save=config.save_figures, save_path=path_figures, average_correlation=average_correlation,
                                     average_rmse=average_rmse, best_alpha=alpha, average_weights=average_weights, times=config.times,
                                     n_feats=n_feats, stim=stim, session=sesion, subject=sujeto, hierarchical_clustering=config.hierarchical_clustering,
                                     display_interactive_mode=config.display_interactive_mode, no_figures=config.no_figures)

                # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                average_weights_subjects.append(average_weights)
                average_correlation_subjects.append(average_correlation)
                average_rmse_subjects.append(average_rmse)
                pvalues_corr_subjects.append(topo_pval_corr_sujeto)
                pvalues_rmse_subjects.append(topo_pval_rmse_sujeto)
                repeated_good_correlation_channels_subjects.append(repeated_good_correlation_channels)
                repeated_good_rmse_channels_subjects.append(repeated_good_rmse_channels)

                # Update the number of subjects
                total_number_of_subjects+=1

            # Print the progress of the iteration
            iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=config.sesiones.index(sesion), length_of_iterator=len(config.sesiones))

            # del average_weights, average_rmse, average_correlation, correlation_per_channel, rmse_per_channel, correlation_matrix, root_mean_square_error,\
            #     eeg_test, eeg, stims, stims_sujeto_1, stims_sujeto_2, sujeto_1, sujeto_2, eeg_sujeto_1, eeg_sujeto_2

        if config.just_load_data:
            continue

        # Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
        average_weights_subjects = np.stack(average_weights_subjects, axis=0)
        average_correlation_subjects = np.stack(average_correlation_subjects , axis=0)
        average_rmse_subjects = np.stack(average_rmse_subjects , axis=0)
        pvalues_corr_subjects = np.stack(pvalues_corr_subjects , axis=0)
        pvalues_rmse_subjects = np.stack(pvalues_rmse_subjects , axis=0)
        repeated_good_correlation_channels_subjects = np.stack(repeated_good_correlation_channels_subjects , axis=0)
        repeated_good_rmse_channels_subjects = np.stack(repeated_good_rmse_channels_subjects , axis=0)

        # Save results
        if config.save_results and total_number_of_subjects==18:
            os.makedirs(save_results_path, exist_ok=True)
            os.makedirs(path_weights, exist_ok=True)
            dump_pickle(path=save_results_path+f'{stim}.pkl',
                        obj={'average_correlation_subjects':average_correlation_subjects,
                            'repeated_good_correlation_channels_subjects':repeated_good_correlation_channels_subjects},
                        rewrite=True,
                        verbose=True)
            dump_pickle(path=path_weights+'total_weights_per_subject.pkl',
                        obj={'average_weights_subjects':average_weights_subjects},
                        rewrite=True)

        # Plot phoneme ocurrences
        # if np.array([bool(d) for d in phonemes_occurrences.values()]).any():
        #     plot.phonemes_occurrences(occurrences=phonemes_occurrences, save_path=path_figures, save=save_figures, no_figures=config.no_figures)

        # Plot average results only if all subjects are analyzed
        no_figures=True if (total_number_of_subjects!=18) else no_figures

        # Plot average topomap across each subject
        plot.average_topomap(average_coefficient_subjects=average_rmse_subjects, stim=stim, info=info, display_interactive_mode=config.display_interactive_mode,
                             save=config.save_figures, save_path=path_figures, coefficient_name='RMSE', no_figures=config.no_figures)
        plot.average_topomap(average_coefficient_subjects=average_correlation_subjects, stim=stim, display_interactive_mode=config.display_interactive_mode,
                             info=info, save=config.save_figures, save_path=path_figures, coefficient_name='Correlation', test_result=False, no_figures=config.no_figures) # USING ZERO METHOD PRATT

        # Plot topomap with relevant times
        plot.topo_map_relevant_times(average_weights_subjects=average_weights_subjects, info=info, n_feats=n_feats, band=band, stim=stim, times=config.times,
                                sample_rate=config.sr, save_path=path_figures, save=config.save_figures, display_interactive_mode=config.display_interactive_mode, no_figures=config.no_figures)

        # Plot channel-wise correlation topomap
        plot.channel_wise_correlation_topomap(average_weights_subjects=average_weights_subjects, info=info, stim=stim, save=config.save_figures,
                                              save_path=path_figures, display_interactive_mode=config.display_interactive_mode, no_figures=config.no_figures)

        # Plot weights
        plot.average_regression_weights(average_weights_subjects=average_weights_subjects, info=info, save=config.save_figures, save_path=path_figures, hierarchical_clustering=config.hierarchical_clustering,
                                        times=config.times, n_feats=n_feats, stim=stim, display_interactive_mode=config.display_interactive_mode, no_figures=config.no_figures)

        # Plot correlation matrix between subjects
        plot.correlation_matrix_subjects(average_weights_subjects=average_weights_subjects, stim=stim, n_feats=n_feats, save=config.save_figures,
                                         save_path=path_figures, display_interactive_mode=config.display_interactive_mode, no_figures=config.no_figures)

        if config.statistical_test:
            # Plot topomap of average p-values across all subject
            plot.topo_average_pval(pvalues_coefficient_subjects=pvalues_corr_subjects, info=info, display_interactive_mode=config.display_interactive_mode,
                                   save=config.save_figures, save_path=path_figures, coefficient_name='correlation', no_figures=config.no_figures)
            plot.topo_average_pval(pvalues_coefficient_subjects=pvalues_rmse_subjects, info=info, display_interactive_mode=config.display_interactive_mode,
                                   save=config.save_figures, save_path=path_figures, coefficient_name='RMSE', no_figures=config.no_figures)

            # Plot topomap of sum of repeated channels across all subject
            plot.topo_repeated_channels(repeated_good_coefficients_channels_subjects=repeated_good_correlation_channels_subjects,
                                        info=info, display_interactive_mode=config.display_interactive_mode, save=config.save_figures,
                                        save_path=path_figures, coefficient_name='correlation', no_figures=config.no_figures)
            plot.topo_repeated_channels(repeated_good_coefficients_channels_subjects=repeated_good_rmse_channels_subjects,
                                        info=info, display_interactive_mode=config.display_interactive_mode, save=config.save_figures,
                                        save_path=path_figures, coefficient_name='RMSE', no_figures=config.no_figures)
        if config.perform_tfce:
            del average_weights, average_rmse, average_correlation, correlation_per_channel, rmse_per_channel, correlation_matrix, root_mean_square_error,\
                eeg_test, eeg, stims, stims_sujeto_1, stims_sujeto_2, sujeto_1, sujeto_2, eeg_sujeto_1, eeg_sujeto_2, predicted
            try:
                print("\nLoading TFCE data")
                tvalue_tfce, pvalue_tfce = load_pickle(path=os.path.join(path_TFCE, band, stim + f'_{config.n_permutations}.pkl'))
                print('Succesfull load')
            except:
                print("\nLoad fail", f"\nComputing TFCE: {config.n_permutations} permutations.")

                # Compute TFCE to get p-value
                tvalue_tfce, pvalue_tfce = tfce(
                                                average_weights_subjects=average_weights_subjects, # (n_subjects, n_chan, n_feats, n_delays)
                                                stimulus=stim,
                                                n_jobs=config.number_of_jobs,
                                                n_permutations=config.n_permutations,
                                                verbose_tfce=True
                                                )

                # Save TFCE
                os.makedirs(os.path.join(path_TFCE, band), exist_ok=True)
                dump_pickle(path=os.path.join(path_TFCE, band, stimulus + f'_{config.n_permutations}.pkl'), obj=(tvalue_tfce, pvalue_tfce), rewrite=True)

            # Plot t and p values
            plot.plot_pvalue_tfce(average_weights_subjects=average_weights_subjects, pvalue=pvalue_tfce, times=config.times, stim=stim,
                                  n_feats=n_feats, info=info, significance=config.significance, save_path=path_figures, display_interactive_mode=config.display_interactive_mode,
                                  save=config.save_figures, no_figures=config.no_figures)


# Get run time
run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
text = f'\n\n\t\t\tPARAMETERS  \n\n\tModel: ' + config.model +f'\n\tBands: {config.bands}'+'\n\tStimuli: ' + f'{config.stimuli}'+'\n\tStatus: ' +config.situation+f'\n\tTime interval: ({config.tmin},{config.tmax})s'+f'\n\tNumber of subjects analyzed: {total_number_of_subjects}. \n\tSessions: {config.sesiones}'
if config.just_load_data:
    text += '\n\n\t\t\tJUST LOADING DATA'
text += f'\n\n\t\t\tRUN TIME:{run_time}'

# Dump metadata
metadata_path = f'saves/log/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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
