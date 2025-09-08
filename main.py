# Standard libraries
from datetime import datetime
import os, numpy as np

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv
from utils.general_functions import iteration_percentage
from model_implementations import fold_model
from utils.processing import tfce 
from load import load_data
import get_general_plots
import config

# Notification bot
from utils.notification_telegram import tel_message, generate_completion_message
from utils.telegram_config import API_TOKEN, CHAT_ID

# Command line and logging
from utils.from_commands import create_dynamic_parser, apply_args_to_config
from utils.logs import setup_logger

# Use it
parser = create_dynamic_parser()
args = parser.parse_args()
apply_args_to_config(args)
if config.set_alpha:
    print(f"WARNING: ALPHA IS BEING FORCE TO {config.set_alpha}")
    
# Initialize logger
logger = setup_logger(
    name='main',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR if config.LOG_TO_FILE else None,
    level=config.LOG_LEVEL
)

# ============
# RUN ANALYSIS
# ============
for situation in config.situations:
    start_time = datetime.now()
    stimulus_runtimes = {}
    
    for band in config.bands:
        for stim in config.stimuli:
            stim_start_time = datetime.now()
            sorted_stimuli, sorted_bands = sorted(stim.split('_')), sorted(band.split('_'))
            stim, band = '_'.join(sorted_stimuli), '_'.join(sorted_bands)

            # Update
            logger.info(
                '\n===========================\n'
                '\tPARAMETERS\n\n'
                f'Model: {config.model}-{config.solver}\n'
                f'Band: {band}\n'
                f'Stimulus: {stim}\n'
                f'Condition: {situation}\n'
                f'Time interval: ({config.tmin},{config.tmax})s\n'
                '\n===========================\n'
            )

            # Relevant paths
            if config.same_validation_subjects:
                if config.external_validation:
                    path_weights = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_null = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_figures = f'{config.figures_dir}/{config.model}-{config.solver}/External-{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_TFCE = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/TFCE/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/'
                    save_results_path = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/correlations/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/'
                else:
                    path_weights = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_null = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_figures = f'{config.figures_dir}/{config.model}-{config.solver}/{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_TFCE = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/TFCE/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/'
                    save_results_path = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/correlations/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/'
            else:
                if config.external_validation:
                    path_weights = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_null = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_figures = f'{config.figures_dir}/{config.model}-{config.solver}/External-{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_TFCE = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/TFCE/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/'
                    save_results_path = f'{config.output_dir}/{config.model}-{config.solver}/External-{situation}/correlations/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/'
                else:
                    path_weights = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_null = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_figures = f'{config.figures_dir}/{config.model}-{config.solver}/{situation}/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                    path_TFCE = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/TFCE/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/'
                    save_results_path = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/correlations/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/'

            preprocessed_data_path = f'{config.saves_dir}/preprocessed_data/tmin{config.tmin}_tmax{config.tmax}/'
            if config.external_validation:
                path_validation = f'{config.output_dir}/{config.model}-{config.solver}/External/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            else:
                path_validation = f'{config.output_dir}/{config.model}-{config.solver}/{situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')


            # Make lists to store relevant data across sobjects
            repeated_good_correlation_channels_subjects = []
            repeated_good_rmse_channels_subjects = []
            null_correlation_per_channel_subjects = []
            null_rmse_per_channel_subjects = []
            correlation_per_channel_subjects = []
            average_correlation_subjects = []
            average_weights_subjects = []
            average_rmse_subjects = []
            pvalues_corr_subjects = []
            pvalues_rmse_subjects = []
            alphas_subjects = []

            # Store total number of subjects (18) to save figures and results just in this case
            total_number_of_subjects = 0

            # from IPython import embed; embed()
            if config.same_validation_subjects:
                alphas_total = []
                alphas = load_pickle(path=alphas_path)
                for session in config.sessions:
                    for subject in [1, 2]:
                        alphas_total.append(alphas[session][subject])
                alphas_total = np.array(alphas_total)
                config.set_alpha = 10**(np.median(np.log10(alphas_total)))
                logger.info(f'Setting alpha to {config.set_alpha} for all subjects')
            else:
                ...
            
            # Iterate over sessions
            for session in config.sessions:
                print(f'\n-------> Start of session {session}\n')

                # Load data by subject, EEG and info
                subject_1, subject_2, samples_info = load_data(
                    preprocessed_data_path=preprocessed_data_path,
                    situation=situation,
                    session=session,
                    stimuli=stim,
                    band=band
                )
                eeg_subject_1, eeg_subject_2, info = subject_1['EEG'], subject_2['EEG'], subject_1['info']

                if config.just_load_data:
                    continue

                # Load stimuli by subject (i.e: concatenated stimuli features)
                n_feats = [subject_1[stimulus].shape[1] for stimulus in stim.split('_')]
                delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]
                stims_subject_1 = np.hstack([subject_1[stimulus] for stimulus in stim.split('_')])
                stims_subject_2 = np.hstack([subject_2[stimulus] for stimulus in stim.split('_')])

                # Get relevant indexes
                relevant_indexes_1 = samples_info['keep_indexes1'].copy()
                relevant_indexes_2 = samples_info['keep_indexes2'].copy()

                # Run model for each subject
                for subject, eeg, stims, relevant_indexes in zip(
                    (1, 2), 
                    (eeg_subject_1, eeg_subject_2), 
                    (stims_subject_1, stims_subject_2), 
                    (relevant_indexes_1, relevant_indexes_2)
                    ):
                    print(f'\n\t······  Running model for Subject {subject}\n')
                    
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

                    # Set alpha for specific subject
                    if config.set_alpha is None:
                        try:
                            alphas = load_pickle(path=alphas_path)
                            alpha = alphas[session][subject]
                        except:
                            alpha = config.default_alpha
                    else:
                        alpha = config.set_alpha
                    alphas_subjects.append(alpha)

                    # Make the Kfold test
                    kf_test = KFold(config.n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]
                    
                    # Run folds
                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        logger.debug(f'\n\t······  [{fold+1}/{config.n_folds}]\t-->\t α:{alpha:.2f}')

                        # Store model output
                        output = fold_model(
                            fold=fold,
                            alpha=alpha,
                            stims=stims,
                            eeg=eeg,
                            statistical_test=config.statistical_test,
                            relevant_indexes=relevant_indexes,
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,
                            path_null=path_null,
                            validation=False,
                            subject=subject,                              
                            session=session
                        )
                        # Update weights and metrics per fold
                        fold, weights_per_fold[fold], correlation_per_channel[fold], rmse_per_channel[fold] = output[:4]
                        
                        # If statistical test is performed, get p-values and null correlation
                        if config.statistical_test:
                            p_corr, p_rmse, null_correlation_per_channel, null_rmse_per_channel = output[4:]

                            # p-values for significant channels (the rest are ones, i.e: not significant)
                            proba_correlation_per_channel[fold][p_corr < config.significance_threshold] = p_corr[p_corr < config.significance_threshold]
                            proba_rmse_per_channel[fold][p_rmse < config.significance_threshold] = p_rmse[p_rmse < config.significance_threshold]
                            
                            # all p-values for topographic distribution across channels
                            topo_pvalues_corr_per_fold[fold] = p_corr
                            topo_pvalues_rmse_per_fold[fold] = p_rmse

                    # Take average weights, avoiding folds entirely filled with zeros
                    empty_mask = np.array([np.all(weight == 0) for weight in weights_per_fold])
                    if empty_mask.any():
                        empty_fold_indices = np.where(empty_mask)[0]
                        weights_per_fold[empty_mask] = np.nan
                        logger.warning(f'\n\t\t{">" * 26}\n'
                            f'\t\tFolds {", ".join(map(str, empty_fold_indices + 1))} out of {config.n_folds} are empty\n'
                            f'\t\t{">" * 26}')
                            
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

                    # Find good indexes by checking where all folds (at the same time) are significant
                    if config.statistical_test: 
                        corr_good_channel_indexes, = np.where(
                            np.all((proba_correlation_per_channel < 1), axis=0)
                        )
                        rmse_good_channel_indexes, = np.where(
                            np.all((proba_rmse_per_channel < 1), axis=0)
                        )
                        if len(corr_good_channel_indexes) == 0:
                            logger.warning('No significant channels found (correlation)')   
                            corr_good_channel_indexes = []
                        if len(rmse_good_channel_indexes) == 0:
                            logger.warning('No significant channels found (RMSE)')   
                            rmse_good_channel_indexes = []

                    # Avergae p-values across all folds
                    topo_pval_corr_subject = topo_pvalues_corr_per_fold.mean(axis=0)
                    topo_pval_rmse_subject = topo_pvalues_rmse_per_fold.mean(axis=0)

                    # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                    if config.statistical_test:
                        null_correlation_per_channel_subjects.append(null_correlation_per_channel)  
                        null_rmse_per_channel_subjects.append(null_rmse_per_channel)
                    else: 
                        null_correlation_per_channel_subjects.append(
                            np.zeros((config.n_folds, info['nchan'])) # Null correlation is zeros
                            )
                        null_rmse_per_channel_subjects.append(
                            np.zeros((config.n_folds, info['nchan'])) # Null RMSE is zeros
                            )
                    repeated_good_correlation_channels_subjects.append(corr_good_channel_indexes)
                    repeated_good_rmse_channels_subjects.append(rmse_good_channel_indexes)
                    correlation_per_channel_subjects.append(correlation_per_channel)
                    average_correlation_subjects.append(average_correlation)
                    pvalues_corr_subjects.append(topo_pval_corr_subject)
                    pvalues_rmse_subjects.append(topo_pval_rmse_subject)
                    average_weights_subjects.append(average_weights)
                    average_rmse_subjects.append(average_rmse)

                    # Update the number of subjects
                    total_number_of_subjects+=1

                # Print the progress of the iteration
                iteration_percentage(
                    txt=f'\n-------> End of session {session}\n', 
                    i=config.sessions.index(session), 
                    length_of_iterator=len(config.sessions),
                    # logger=logger
                )

            if config.just_load_data:
                continue
            
            # Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
            average_correlation_subjects = np.stack(average_correlation_subjects , axis=0) # n_subj, n_chans
            average_weights_subjects = np.stack(average_weights_subjects, axis=0) # n_subj, n_chans, n_feats, n_delays
            average_rmse_subjects = np.stack(average_rmse_subjects , axis=0) # n_subj, n_chans
            pvalues_corr_subjects = np.stack(pvalues_corr_subjects , axis=0) # n_subj, n_chans
            pvalues_rmse_subjects = np.stack(pvalues_rmse_subjects , axis=0) # n_subj, n_chans

            # Save results
            if config.save_results and total_number_of_subjects==18:
                os.makedirs(save_results_path, exist_ok=True)
                os.makedirs(path_weights, exist_ok=True)
                
                # Correlation
                dump_pickle(
                    path=save_results_path+f'{stim}.pkl',
                    obj={'average_correlation_subjects':average_correlation_subjects},
                    rewrite=True,
                    verbose=True
                )
                # Significant channels
                has_significant_channels = any(len(channels) > 0 for channels in repeated_good_correlation_channels_subjects)
                if has_significant_channels:
                    dump_pickle(
                    path=save_results_path+f'{stim}_significant_channels.pkl',
                    obj={'significant_channels':repeated_good_correlation_channels_subjects},
                    rewrite=True,
                    verbose=True
                )
                # Weights
                dump_pickle(
                    path=path_weights+'total_weights_per_subject.pkl',
                    obj={'average_weights_subjects':average_weights_subjects},
                    rewrite=True
                )
                            
            if config.perform_tfce:
                try:
                    logger.info("Loading TFCE data")
                    tvalue_tfce, pvalue_tfce = load_pickle(
                        path=os.path.join(path_TFCE, band, stim + f'_{config.n_permutations}.pkl')
                        )
                    logger.info('Successful load ✓')
                except:
                    logger.warning("Load fail")
                    logger.info(f"Computing TFCE: {config.n_permutations} permutations.")

                    # Compute TFCE to get p-value
                    tvalue_tfce, pvalue_tfce = tfce(
                        average_weights_subjects=average_weights_subjects, # (n_subjects, n_chan, n_feats, n_delays)
                        stimulus=stim,
                        n_permutations=config.n_permutations,
                        n_jobs=config.number_of_jobs,
                        verbose_tfce=True
                    )

                    # Save TFCE
                    os.makedirs(os.path.join(path_TFCE, band), exist_ok=True)
                    dump_pickle(
                        path=os.path.join(path_TFCE, band, stim + f'_{config.n_permutations}.pkl'), 
                        obj=(tvalue_tfce, pvalue_tfce), rewrite=True
                        )
        
            # Calculate runtime for this stimulus
            stim_runtime = datetime.now().replace(microsecond=0) - stim_start_time.replace(microsecond=0)
            stimulus_runtimes[f"{band}_{stim}"] = stim_runtime
            
            # Print stimulus completion time
            logger.info(
                f"\n\t{'='*40}\n"
                f"\t✅ STIMULUS COMPLETED: {band}_{stim}\n\n"
                f"\t⏱️  Runtime: {stim_runtime}\n"
                f"\t📊 Subjects processed: {total_number_of_subjects}\n"
                f"\t{'='*40}\n"
            )
            
            if not config.no_figures:
                logger.info("🎨 Iniciando generación de gráficos...")
                get_general_plots.main(
                    repeated_good_correlation_channels_subjects=repeated_good_correlation_channels_subjects,
                    null_correlation_per_channel_subjects=null_correlation_per_channel_subjects,
                    repeated_good_rmse_channels_subjects=repeated_good_rmse_channels_subjects,
                    correlation_per_channel_subjects=correlation_per_channel_subjects,
                    average_correlation_subjects=average_correlation_subjects,
                    average_weights_subjects=average_weights_subjects,
                    total_number_of_subjects=total_number_of_subjects,
                    average_rmse_subjects=average_rmse_subjects,
                    pvalues_corr_subjects=pvalues_corr_subjects,
                    pvalues_rmse_subjects=pvalues_rmse_subjects,
                    alphas_subjects=alphas_subjects,
                    path_figures=path_figures,
                    pvalue_tfce=pvalue_tfce if config.perform_tfce else None,
                    n_feats=n_feats,
                    band=band,
                    stim=stim,
                    same_validation_subjects=config.same_validation_subjects
                )
    # Get total run time
    total_runtime = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
    
    # Generate completion message
    text = generate_completion_message(
        situation=situation,
        total_number_of_subjects=total_number_of_subjects,
        stimulus_runtimes=stimulus_runtimes,
        total_runtime=str(total_runtime),
        save_path=path_weights,
        fig_path=path_figures
    )
    # Send text to telegram bot
    tel_message(
        api_token=API_TOKEN,
        chat_id=CHAT_ID, 
        message=text,
        caption='Run finished'
        )

    # Dump metadata
    metadata_path = f'saves/log/main/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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
    
    # Print the completion message
    logger.info(text)