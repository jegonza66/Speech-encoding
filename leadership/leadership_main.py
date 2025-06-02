# Standard libraries
from datetime import datetime
import os, numpy as np

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from model_implementations import fold_model
from utils.processing import tfce, subsampling_indexes_to_minimum
from leadership.leadership_load import load_data
import config, utils.plot as plot

# Notification bot
from utils.notification_telegram import tel_message, generate_completion_message
from telegram_config import API_TOKEN, CHAT_ID


# ============
# RUN ANALYSIS
# ============
for situation in ['External']:
    start_time = datetime.now()
    for band in config.bands:
        for stim in config.stimuli:
            sorted_stimuli, sorted_bands = sorted(stim.split('_')), sorted(band.split('_'))
            stim, band = '_'.join(sorted_stimuli), '_'.join(sorted_bands)

            # Update
            print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + config.model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Condition: ' + situation+'\n',f'Time interval: ({config.tmin},{config.tmax})s\n','\n===========================\n')

            # Relevant paths
            save_results_path = f'leadership/saves/{config.model}/{situation}/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}/correlations/tmin{config.tmin}_tmax{config.tmax}/{band}/'
            preprocessed_data_path = f'leadership/saves/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/'
            path_weights = f'leadership/saves/{config.model}/{situation}/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            path_null = f'leadership/saves/{config.model}/{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            path_figures_leader = f'figures/leadership/leader/{config.model}/{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}/{band}/{stim}/'
            path_figures_follower = f'figures/leadership/follower/{config.model}/{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}/{band}/{stim}/'
            
            if config.leadership_criterion_path == r'data/turns/leader_according_number_of_ipus.pkl':
                edition_index = save_results_path.index('leadership/') + len('leadership/') 
                edition_index_figs = path_figures_leader.index('/leadership/') + len('/leadership/')
                
                # Insert criterion word in edition index
                save_results_path = save_results_path[:edition_index] + 'criterion' + '/' + save_results_path[edition_index:]  
                preprocessed_data_path = preprocessed_data_path[:edition_index] + 'criterion' + '/' + preprocessed_data_path[edition_index:]  
                path_weights = path_weights[:edition_index] + 'criterion' + '/' + path_weights[edition_index:]  
                path_null = path_null[:edition_index] + 'criterion' + '/' + path_null[edition_index:]  
                path_figures_leader = path_figures_leader[:edition_index_figs] + 'criterion' + '/' + path_figures_leader[edition_index_figs:]  
                path_figures_follower = path_figures_follower[:edition_index_figs] + 'criterion' + '/' + path_figures_follower[edition_index_figs:]  
            
            elif config.leadership_criterion_path == r'data/turns/leader_according_len_of_ipus.pkl':
                edition_index = save_results_path.index('leadership/') + len('leadership/') 
                edition_index_figs = path_figures_leader.index('/leadership/') + len('/leadership/')
                
                # Insert criterion word in edition index
                save_results_path = save_results_path[:edition_index] + 'criterion2' + '/' + save_results_path[edition_index:]  
                preprocessed_data_path = preprocessed_data_path[:edition_index] + 'criterion2' + '/' + preprocessed_data_path[edition_index:]  
                path_weights = path_weights[:edition_index] + 'criterion2' + '/' + path_weights[edition_index:]  
                path_null = path_null[:edition_index] + 'criterion2' + '/' + path_null[edition_index:]  
                path_figures_leader = path_figures_leader[:edition_index_figs] + 'criterion2' + '/' + path_figures_leader[edition_index_figs:]  
                path_figures_follower = path_figures_follower[:edition_index_figs] + 'criterion2' + '/' + path_figures_follower[edition_index_figs:]  

            if config.external_validation:
                path_validation = f'saves/{config.model}/External/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            else:
                path_validation = f'saves/{config.model}/{situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')

            # Make lists to store relevant data across sobjects
            average_weights_subjects_leader = []
            average_correlation_subjects_leader = []
            average_rmse_subjects_leader = []
            pvalues_corr_subjects_leader = []
            pvalues_rmse_subjects_leader = []
            repeated_good_correlation_channels_subjects_leader = []
            repeated_good_rmse_channels_subjects_leader = []

            average_weights_subjects_follower = []
            average_correlation_subjects_follower = []
            average_rmse_subjects_follower = []
            pvalues_corr_subjects_follower = []
            pvalues_rmse_subjects_follower = []
            repeated_good_correlation_channels_subjects_follower = []
            repeated_good_rmse_channels_subjects_follower = []

            # Store total number of subjects (18) to save figures and results just in this case
            total_number_of_subjects_leader = 0
            total_number_of_subjects_follower = 0

            # Iterate over sessions
            for session in config.sessions:
                print(f'\n------->\tStart of session {session}\n')

                # Load data by subject, EEG and info
                leader_1, leader_2, follower_1, follower_2, samples_info = load_data(
                                                session=session,
                                                stim=stim,
                                                band=band,
                                                sr=config.sr,
                                                delays=config.delays,
                                                preprocessed_data_path=preprocessed_data_path,
                                                praat_executable_path=config.praat_executable_path,
                                                situation=situation
                                                )
                # eeg_leader_1, eeg_leader_2, eeg_follower_1, eeg_follower_2, info = leader_1['EEG'], leader_2['EEG'], follower_1['EEG'], follower_2['EEG'], leader_1['info']
                eeg_leader_1, eeg_leader_2, eeg_follower_1, eeg_follower_2, info = leader_1['EEG'], leader_2['EEG'], follower_1['EEG'], follower_2['EEG'], leader_1['info']
                

                if config.just_load_data:
                    continue

                # Load stimuli by subject (i.e: concatenated stimuli features)
                stims_follower_1 = np.hstack([follower_1[stimulus] for stimulus in stim.split('_')])
                stims_leader_1 = np.hstack([leader_1[stimulus] for stimulus in stim.split('_')])

                stims_follower_2 = np.hstack([follower_2[stimulus] for stimulus in stim.split('_')])
                stims_leader_2 = np.hstack([leader_2[stimulus] for stimulus in stim.split('_')])

                n_feats = [leader_1[stimulus].shape[1] for stimulus in stim.split('_')]
                delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

                # Get relevant indexes and match lengths
                relevant_indexes_leader_1, relevant_indexes_follower_1, relevant_indexes_leader_2, relevant_indexes_follower_2 = subsampling_indexes_to_minimum(
                    samples_info=samples_info,
                    tollerance=0.1,
                    kind=config.leadership_kind_of_subsampling,
                    seed=42
                    )

                # Run model for each subject
                for sujeto, eeg, stims, relevant_indexes in zip((1, 2, 3, 4), (eeg_leader_1, eeg_leader_2, eeg_follower_1, eeg_follower_2), (stims_leader_1, stims_leader_2, stims_follower_1, stims_follower_2), (relevant_indexes_leader_1, relevant_indexes_leader_2, relevant_indexes_follower_1, relevant_indexes_follower_2)):
                    path_figures = path_figures_leader if sujeto in [1, 2] else path_figures_follower

                    # Initialize empty variables to store relevant data of each fold
                    weights_per_fold = np.zeros((config.n_folds, info['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float32)
                    correlation_per_channel = np.zeros((config.n_folds, info['nchan']))
                    rmse_per_channel = np.zeros((config.n_folds, info['nchan']))

                    # Variable to store all channel's p-value
                    topo_pvalues_corr_per_fold = np.zeros((config.n_folds, info['nchan']))
                    topo_pvalues_rmse_per_fold = np.zeros((config.n_folds, info['nchan']))

                    # Variable to store p-value of significant channels
                    proba_correlation_per_channel = np.ones((config.n_folds, info['nchan']))
                    proba_rmse_per_channel = np.ones((config.n_folds, info['nchan']))
                    print(f'\n\t······  Running model for Subject {sujeto}\n')

                    # Set alpha for specific subject
                    if config.set_alpha is None:
                        try:
                            alphas = load_pickle(path=alphas_path)
                            suj = 1 if sujeto in [1,3] else 2
                            alpha = alphas[session][suj]
                        except:
                            alpha = config.default_alpha
                    else:
                        alpha = config.set_alpha

                    # Make the Kfold test
                    kf_test = KFold(config.n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]

                    # Run folds
                    k_models_output = []
                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        print(f'\n\t······  [{fold+1}/{config.n_folds}]\t-->\t α:{alpha}')
                        k_models_output.append(
                                        fold_model(
                                            fold=fold,
                                            alpha=alpha,
                                            stims=stims,
                                            eeg=eeg,
                                            relevant_indexes=relevant_indexes,
                                            train_indexes=train_indexes,
                                            test_indexes=test_indexes,
                                            validation=False,
                                            statistical_test=config.statistical_test,
                                            path_null=path_null,
                                            session=session,
                                            subject=sujeto,
                                            )
                                        )
                    # Store model output
                    for output_k in k_models_output:
                        fold, weights, correlation_matrix, root_mean_square_error = output_k[:4]

                        # Update weights and metrics per fold
                        weights_per_fold[fold] = weights
                        correlation_per_channel[fold] = correlation_matrix
                        rmse_per_channel[fold] = root_mean_square_error

                        if config.statistical_test:
                            # p_corr, p_rmse, significant_corr_count, significant_rmse_count, null_correlation_per_channel = output_k[4:]
                            p_corr, p_rmse, null_correlation_per_channel = output_k[4:]

                            # p-values for significant channels (the rest are ones, i.e: not significant)
                            proba_correlation_per_channel[fold][p_corr < config.significance_threshold] = p_corr[p_corr < config.significance_threshold]
                            proba_rmse_per_channel[fold][p_rmse < config.significance_threshold] = p_rmse[p_rmse < config.significance_threshold]

                            # all p-values for topographic distribution across channels
                            topo_pvalues_corr_per_fold[fold] = p_corr
                            topo_pvalues_rmse_per_fold[fold] = p_rmse

                            # Estimate power
                            # power_correlation_per_channel[fold] = significant_corr_count / (config.power_n_bootstrap_samples * eeg.shape[1])
                            # power_rmse_per_channel[fold] = significant_rmse_count / (config.power_n_bootstrap_samples * eeg.shape[1])

                    print(f'\n\t······  Run model\n')

                    # Take average weights, avoiding folds entirely filled with zeros
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

                    # Channels that passed the tests
                    corr_good_channel_indexes = []
                    rmse_good_channel_indexes = []

                    # Variable to store significant channels
                    repeated_good_correlation_channels = np.zeros(info['nchan'])
                    repeated_good_rmse_channels = np.zeros(info['nchan'])

                    if config.statistical_test:
                        # Find good indexes by checking where all folds (at the same time) are significant
                        try:
                            corr_good_channel_indexes, = np.where(
                                                        np.all((proba_correlation_per_channel < 1), axis=0)
                                                        )
                            rmse_good_channel_indexes, = np.where(
                                                        np.all((proba_rmse_per_channel < 1), axis=0)
                                                        )
                        except:
                            corr_good_channel_indexes = []
                            rmse_good_channel_indexes = []
                            print('No significant channels found')

                        # Saves passing channels by subject
                        repeated_good_correlation_channels[corr_good_channel_indexes] += 1 # binary array with ones where significant
                        repeated_good_rmse_channels[rmse_good_channel_indexes] += 1

                        # # Plot shadows for each subject
                        # plot.null_correlation_vs_correlation_good_channels(
                        #     display_interactive_mode=config.display_interactive_mode,
                        #     session=session,
                        #     subject=sujeto,
                        #     save_path=path_figures,
                        #     good_channels_indexes=corr_good_channel_indexes,
                        #     correlation_per_channel=correlation_per_channel,
                        #     null_correlation_per_channel=null_correlation_per_channel,
                        #     # power_correlation=power_correlation_per_channel.mean(),
                        #     # power_rmse=power_rmse_per_channel.mean(),
                        #     save=config.save_figures,
                        #     no_figures=config.no_figures
                        #     )

                    # Avergae p-values across all folds
                    topo_pval_corr_sujeto = topo_pvalues_corr_per_fold.mean(axis=0)
                    topo_pval_rmse_sujeto = topo_pvalues_rmse_per_fold.mean(axis=0)

                    # Plot head topomap across al channel for correlation and rmse
                    plot.topomap(
                        good_channels_indexes=corr_good_channel_indexes,
                        average_coefficient=average_correlation,
                        info=info,
                        coefficient_name='Correlation',
                        save=config.save_figures,
                        display_interactive_mode=config.display_interactive_mode,
                        save_path=path_figures,
                        subject=sujeto,
                        session=session,
                        no_figures=config.no_figures
                        )
                    plot.topomap(
                        good_channels_indexes=rmse_good_channel_indexes,
                        average_coefficient=average_rmse,
                        info=info,
                        coefficient_name='RMSE',
                        save=config.save_figures,
                        display_interactive_mode=config.display_interactive_mode,
                        save_path=path_figures,
                        subject=sujeto,
                        session=session,
                        no_figures=config.no_figures #TODO: remove all config. parameters and put them in plot module
                        )

                    # Plot weights
                    plot.channel_weights(
                        info=info,
                        save=config.save_figures,
                        save_path=path_figures,
                        average_correlation=average_correlation,
                        average_rmse=average_rmse,
                        best_alpha=alpha,
                        average_weights=average_weights,
                        times=config.times,
                        n_feats=n_feats,
                        stim=stim,
                        session=session,
                        subject=sujeto,
                        hierarchical_clustering=config.hierarchical_clustering,
                        display_interactive_mode=config.display_interactive_mode,
                        no_figures=config.no_figures
                        )
                    if sujeto in [1,2]:
                        # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                        average_weights_subjects_leader.append(average_weights)
                        average_correlation_subjects_leader.append(average_correlation)
                        average_rmse_subjects_leader.append(average_rmse)
                        pvalues_corr_subjects_leader.append(topo_pval_corr_sujeto)
                        pvalues_rmse_subjects_leader.append(topo_pval_rmse_sujeto)
                        repeated_good_correlation_channels_subjects_leader.append(repeated_good_correlation_channels)
                        repeated_good_rmse_channels_subjects_leader.append(repeated_good_rmse_channels)

                        # Update the number of subjects
                        total_number_of_subjects_leader+=1
                    else:
                        # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                        average_weights_subjects_follower.append(average_weights)
                        average_correlation_subjects_follower.append(average_correlation)
                        average_rmse_subjects_follower.append(average_rmse)
                        pvalues_corr_subjects_follower.append(topo_pval_corr_sujeto)
                        pvalues_rmse_subjects_follower.append(topo_pval_rmse_sujeto)
                        repeated_good_correlation_channels_subjects_follower.append(repeated_good_correlation_channels)
                        repeated_good_rmse_channels_subjects_follower.append(repeated_good_rmse_channels)

                        # Update the number of subjects
                        total_number_of_subjects_follower+=1


                # Print the progress of the iteration
                iteration_percentage(txt=f'\n------->\tEnd of session {session}\n', i=config.sessions.index(session), length_of_iterator=len(config.sessions))

            if config.just_load_data:
                continue

            # Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
            average_weights_subjects_leader = np.stack(average_weights_subjects_leader, axis=0) # n_subj, n_chans, n_feats, n_delays
            average_correlation_subjects_leader = np.stack(average_correlation_subjects_leader, axis=0) # n_subj, n_chans
            average_rmse_subjects_leader = np.stack(average_rmse_subjects_leader, axis=0) # n_subj, n_chans
            pvalues_corr_subjects_leader = np.stack(pvalues_corr_subjects_leader, axis=0) # n_subj, n_chans
            pvalues_rmse_subjects_leader = np.stack(pvalues_rmse_subjects_leader, axis=0) # n_subj, n_chans
            repeated_good_correlation_channels_subjects_leader = np.stack(repeated_good_correlation_channels_subjects_leader, axis=0) # n_subj, n_chans
            repeated_good_rmse_channels_subjects_leader = np.stack(repeated_good_rmse_channels_subjects_leader, axis=0) # n_subj, n_chans

            # Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
            average_weights_subjects_follower = np.stack(average_weights_subjects_follower, axis=0) # n_subj, n_chans, n_feats, n_delays
            average_correlation_subjects_follower = np.stack(average_correlation_subjects_follower, axis=0) # n_subj, n_chans
            average_rmse_subjects_follower = np.stack(average_rmse_subjects_follower, axis=0) # n_subj, n_chans
            pvalues_corr_subjects_follower = np.stack(pvalues_corr_subjects_follower, axis=0) # n_subj, n_chans
            pvalues_rmse_subjects_follower = np.stack(pvalues_rmse_subjects_follower, axis=0) # n_subj, n_chans
            repeated_good_correlation_channels_subjects_follower = np.stack(repeated_good_correlation_channels_subjects_follower, axis=0) # n_subj, n_chans
            repeated_good_rmse_channels_subjects_follower = np.stack(repeated_good_rmse_channels_subjects_follower, axis=0) # n_subj, n_chans

            # Save results
            if config.save_results and total_number_of_subjects_leader==18 and total_number_of_subjects_follower==18:
                os.makedirs(save_results_path, exist_ok=True)
                os.makedirs(path_weights, exist_ok=True)
                dump_pickle(
                        path=save_results_path+f'{stim}.pkl',
                        obj={'average_correlation_subjects_leader':average_correlation_subjects_leader, 'average_correlation_subjects_follower':average_correlation_subjects_follower},
                        rewrite=True,
                        verbose=True
                        )
                if np.sum(repeated_good_correlation_channels_subjects_leader)!=0 and np.sum(repeated_good_correlation_channels_subjects_follower)!=0:
                    dump_pickle(
                            path=save_results_path+f'{stim}_significant_channels.pkl',
                            obj={'significant_channels_leader':repeated_good_correlation_channels_subjects_leader, 'significant_channels_follower':repeated_good_correlation_channels_subjects_follower},
                            rewrite=True,
                            verbose=True
                            )
                dump_pickle(
                        path=path_weights+'total_weights_per_subject.pkl',
                        obj={'average_weights_subjects_leader':average_weights_subjects_leader, 'average_weights_subjects_follower':average_weights_subjects_follower},
                        rewrite=True
                        )

            # Plot average results only if all subjects are analyzed
            config.no_figures=True if (total_number_of_subjects_leader!=18 and total_number_of_subjects_follower!=18) else config.no_figures
            for average_weights_subjects, average_correlation_subjects, average_rmse_subjects, path_figures in zip([average_weights_subjects_leader, average_weights_subjects_follower], [average_correlation_subjects_leader, average_correlation_subjects_follower], [average_rmse_subjects_leader, average_rmse_subjects_follower],[path_figures_leader, path_figures_follower]):
                # Plot average topomap metrics across each subject
                plot.average_topomap(
                    average_coefficient_subjects=average_rmse_subjects,
                    stim=stim,
                    info=info,
                    display_interactive_mode=config.display_interactive_mode,
                    save=config.save_figures,
                    save_path=path_figures,
                    coefficient_name='RMSE',
                    no_figures=config.no_figures
                    )
                plot.average_topomap(
                    average_coefficient_subjects=average_correlation_subjects,
                    stim=stim,
                    display_interactive_mode=config.display_interactive_mode,
                    info=info,
                    save=config.save_figures,
                    save_path=path_figures,
                    coefficient_name='Correlation',
                    test_result=False,
                    no_figures=config.no_figures
                    )

                # Plot topomap with relevant times
                plot.topo_map_relevant_times(
                    average_weights_subjects=average_weights_subjects,
                    info=info,
                    n_feats=n_feats,
                    band=band,
                    stim=stim,
                    times=config.times,
                    sample_rate=config.sr,
                    save_path=path_figures,
                    save=config.save_figures,
                    display_interactive_mode=config.display_interactive_mode,
                    no_figures=config.no_figures
                    )

                # Plot channel-wise correlation topomap
                plot.channel_wise_correlation_topomap(
                    average_weights_subjects=average_weights_subjects,
                    info=info,
                    stim=stim,
                    save=config.save_figures,
                    save_path=path_figures,
                    display_interactive_mode=config.display_interactive_mode,
                    no_figures=config.no_figures
                    )

                # Plot weights
                plot.average_regression_weights(
                    average_weights_subjects=average_weights_subjects,
                    info=info,
                    save=config.save_figures,
                    save_path=path_figures,
                    hierarchical_clustering=config.hierarchical_clustering,
                    times=config.times,
                    n_feats=n_feats,
                    stim=stim,
                    display_interactive_mode=config.display_interactive_mode,
                    no_figures=config.no_figures
                    )

                # Plot correlation matrix between subjects
                plot.correlation_matrix_subjects(
                    average_weights_subjects=average_weights_subjects,
                    stim=stim,
                    n_feats=n_feats,
                    save=config.save_figures,
                    save_path=path_figures,
                    display_interactive_mode=config.display_interactive_mode,
                    no_figures=config.no_figures
                    )

    # Get run time
    run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
    text = f'\n\n\t\t\tPARAMETERS  \n\n\tModel: ' + config.model +f'\n\tBands: {config.bands}'+'\n\tStimuli: ' + f'{config.stimuli}'+'\n\tCondition: ' +situation+f'\n\tTime interval: ({config.tmin},{config.tmax})s'+f'\n\tNumber of subjects analyzed: {total_number_of_subjects_leader}. \n\tSessions: {config.sessions}'
    if config.just_load_data:
        text += '\n\n\t\t\tJUST LOADING DATA'
    text += '\n\n\t\t\tmain.py'
    text += f'\n\n\t\t\tRUN TIME:{run_time}'

    # Dump metadata
    metadata_path = f'saves/log/main_{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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
    
if __name__=='__main__':
    # ===========================
    # ANALYSIS: # TODO comentar la diferencia en pesos, está interesante. Pareciera ser que los pesos de los líderes tienen más orden que los de los seguidores.
    # Notar también la diferencia en la latencia, por ej. para Phonological.

    import matplotlib.pyplot as plt, pandas as pd, numpy as np, seaborn as sns, os
    from scipy.stats import wilcoxon
    import config
    from utils.general_functions import load_pickle, dump_pickle
    import mne

    # Fz electrode
    # electrodes = [config.info_mne['ch_names'].index(f'C{i}') for i in [20,21,22,12,25]]
    # ===================================================================
    
    # Comparación de pesos promedios entre dimensiones, sujetos y canales
    figs_subsampling_path = f'figures/leadership/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}'
    config.leadership_criterion_path = r'data/turns/leader_according_len_of_ipus.pkl'
    config.leadership_criterion_path = r'data/turns/leader_according_number_of_ipus.pkl'
    config.leadership_criterion_path = None
    if config.leadership_criterion_path  == r'data/turns/leader_according_number_of_ipus.pkl':
        edition_index_figs = figs_subsampling_path.index('/leadership/') + len('/leadership/')
        figs_subsampling_path = figs_subsampling_path[:edition_index_figs] + 'criterion' + '/' + figs_subsampling_path[edition_index_figs:]  
        mtrfs_path = lambda stimulus, band: fr"leadership\criterion\saves\mtrf_ridge_torch\External\{config.leadership_kind_of_subsampling}\tollerance_{config.tollerance}\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\{band}\{stimulus}\total_weights_per_subject.pkl"
        correlation_path = lambda stimulus, band: fr"leadership\criterion\saves\mtrf_ridge_torch\External\{config.leadership_kind_of_subsampling}\tollerance_{config.tollerance}\correlations\tmin-0.2_tmax0.6\{band}\{stimulus}.pkl"
    elif config.leadership_criterion_path == r'data/turns/leader_according_number_of_ipus.pkl':
        edition_index_figs = figs_subsampling_path.index('/leadership/') + len('/leadership/')
        figs_subsampling_path = figs_subsampling_path[:edition_index_figs] + 'criterion2' + '/' + figs_subsampling_path[edition_index_figs:]  
        mtrfs_path = lambda stimulus, band: fr"leadership\criterion2\saves\mtrf_ridge_torch\External\{config.leadership_kind_of_subsampling}\tollerance_{config.tollerance}\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\{band}\{stimulus}\total_weights_per_subject.pkl"
        correlation_path = lambda stimulus, band: fr"leadership\criterion2\saves\mtrf_ridge_torch\External\{config.leadership_kind_of_subsampling}\tollerance_{config.tollerance}\correlations\tmin-0.2_tmax0.6\{band}\{stimulus}.pkl"
    else:
        mtrfs_path = lambda stimulus, band: fr"leadership\saves\mtrf_ridge_torch\External\{config.leadership_kind_of_subsampling}\tollerance_{config.tollerance}\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\{band}\{stimulus}\total_weights_per_subject.pkl"
        correlation_path = lambda stimulus, band: fr"leadership\saves\mtrf_ridge_torch\External\{config.leadership_kind_of_subsampling}\tollerance_{config.tollerance}\correlations\tmin-0.2_tmax0.6\{band}\{stimulus}.pkl"
    os.makedirs(figs_subsampling_path, exist_ok=True)

    band = 'Theta'
    stimuli = [
        'Envelope',
        # 'Pitch-Log-Raw',
        'Spectrogram',
        # 'Phonological',
        'Phonemes-Frequency-Phonet', 
    ]

    # Plot results
    fig, axes = plt.subplots(
        nrows=4,
        ncols=len(stimuli),
        figsize=(2.5*len(stimuli), 8),
        layout='constrained'
    )
    fig.suptitle(f'{band}', fontsize=14)

    for i, stimulus in enumerate(stimuli):
        # Load data
        mtrfs = load_pickle(path=mtrfs_path(stimulus, band))
        correlations = load_pickle(path=correlation_path(stimulus, band))
        data = pd.DataFrame({
            'average_correlation_subjects_follower': correlations['average_correlation_subjects_follower'].mean(axis=1),
            'average_correlation_subjects_leader': correlations['average_correlation_subjects_leader'].mean(axis=1),
            # 'average_correlation_subjects_follower': correlations['average_correlation_subjects_follower'][:, electrodes].mean(axis=1),
            # 'average_correlation_subjects_leader': correlations['average_correlation_subjects_leader'][:, electrodes].mean(axis=1),
        })
        colors_leadership = {
            'average_correlation_subjects_follower': '#BD164F',
            'average_correlation_subjects_leader': '#1567A3'
        }

        # Column title
        axes[0, i].set_title(stimulus, fontsize=12)

        # ============
        # Averge mTRFs
        axes[0, i].plot(
            config.times*1e3,
            mtrfs['average_weights_subjects_follower'].mean(axis=(0,1,2)),
            # mtrfs['average_weights_subjects_follower'][:,electrodes,:,:].mean(axis=(0,1,2)),
            label='Follower',
            color=colors_leadership['average_correlation_subjects_follower']
            )
        axes[0, i].plot(
            config.times*1e3,
            mtrfs['average_weights_subjects_leader'].mean(axis=(0,1,2)),
            # mtrfs['average_weights_subjects_leader'][:,electrodes,:,:].mean(axis=(0,1,2)),
            label='Leader',
            color=colors_leadership['average_correlation_subjects_leader']
            )
        
        if i == 0:
            axes[0, i].set_ylabel('Amplitude (U.A)')
        if i == len(stimuli) // 2:
            axes[0, i].legend()
        axes[0, i].grid(True)
        axes[0, i].set_xlabel('Time (ms)')

        # ===================
        # Correlation boxplot
        stat, p_val = wilcoxon(
            data['average_correlation_subjects_follower'],
            data['average_correlation_subjects_leader'],
            alternative='two-sided'
        )
        sns.boxplot(data=data, palette=colors_leadership, ax=axes[1, i])
        sns.stripplot(data=data, color='black', alpha=0.5, ax=axes[1, i])
        axes[1, i].text(
            0.5, 0.15, f'p = {p_val:.2e}',
            ha='center', va='center',
            transform=axes[1, i].transAxes,
            fontsize=10, color='black'
        )
        # axes[1, i].set_xticklabels(['Follower', 'Leader']) if i == 0 else ['', ''])
        axes[1, i].set_xticklabels(['Follower', 'Leader'])
        axes[1, i].grid(True)
        if i == 0: axes[1, i].set_ylabel('Average correlation')

        # =====================
        # Topo correlation diff
        norm = correlations['average_correlation_subjects_follower'].mean()
        diff_corr = (correlations['average_correlation_subjects_follower'].mean(axis=0)
                    - correlations['average_correlation_subjects_leader'].mean(axis=0)) / norm

        im = mne.viz.plot_topomap(
            data=diff_corr,
            pos=config.info_mne,
            axes=axes[2, i],
            show=False,
            sphere=0.07,
            cmap='RdBu_r',
            vlim=(-diff_corr.max(), diff_corr.max())
            )
        # label_topo = 'Normalized\ncorrelation diff' if i == 0 else ''
        label_topo = 'Normalized\ncorrelation diff (by follower)'
        
        fig.colorbar(
            im[0],
            ax=axes[2, i],
            shrink=0.85,
            label=label_topo,
            orientation='horizontal',
            boundaries=np.linspace(-diff_corr.max().round(2), diff_corr.max().round(2), 100),
            ticks=np.linspace(-diff_corr.max(), diff_corr.max(), 5).round(2)
            )
        
        # =====================
        # Statistical sig. topo
        significance = 0.05
        diff_corr_stat = (correlations['average_correlation_subjects_follower']
                        - correlations['average_correlation_subjects_leader']) / norm
        
        _, p_val_vec = wilcoxon(diff_corr_stat, alternative='two-sided')
        p_val_vec[p_val_vec > significance] = 1
        log_pval = -np.log10(p_val_vec)

        im = mne.viz.plot_topomap(
            data=log_pval,
            pos=config.info_mne,
            axes=axes[3, i],
            show=False,
            sphere=0.07,
            cmap='inferno',
            vlim=(log_pval.min(), log_pval.max())
            # log_pval.mean(axis=0).min(), log_pval.mean(axis=0).max()
            )
        # label_topo_pval = r'$-\log_{10}(p)$' if i == 0 else ''
        
        label_topo_pval = r'$-\log_{10}(p)$' 
        fig.colorbar(
            im[0],
            ax=axes[3, i],
            shrink=0.85,
            label=label_topo_pval,
            orientation='horizontal',
            boundaries=np.linspace(log_pval.min().round(2), log_pval.max().round(2), 100),
            ticks=np.linspace(log_pval.min(), log_pval.max(), 5).round(2)
            )
    fig.show()
    # fig.savefig(f'figures/leadership/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}/difference.png')


    # =====================
    # Population statistics
    statistics = {
        'leader1':[],
        'leader2':[],
        'follower1':[],
        'follower2':[],
    }
    import os
    root_path= r'leadership\saves\preprocessed_data\External\tmin-0.2_tmax0.6\samples_info'
    
    for session, archive in enumerate(os.listdir(root_path)):
        session, archive
        data = load_pickle(os.path.join(root_path, archive))
        shifted_indexes_leader1, shifted_indexes_follower1, shifted_indexes_leader2, shifted_indexes_follower2 = subsampling_indexes_to_minimum(samples_info=data, tollerance=0.1, kind=config.leadership_kind_of_subsampling, seed=42)
        
        statistics['leader1'].append(len(shifted_indexes_leader1))
        statistics['follower1'].append(len(shifted_indexes_follower1))
        statistics['leader2'].append(len(shifted_indexes_leader2))
        statistics['follower2'].append(len(shifted_indexes_follower2))
        
    df_stats = pd.DataFrame(statistics)
    df_stats_long = df_stats.melt(var_name='Role', value_name='Length')

    # Graficar
    plt.figure(figsize=(8, 5))
    custom_order = ['leader1', 'follower1', 'leader2', 'follower2']
    sns.boxplot(data=df_stats_long, x='Role', y='Length', order=custom_order, palette='Set2')
    sns.stripplot(data=df_stats_long, x='Role', y='Length', order=custom_order, color='black', alpha=0.5)

    plt.title('Número de muestras por condición')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'figures/leadership/{config.leadership_kind_of_subsampling}/tollerance_{config.tollerance}/statistics.png')