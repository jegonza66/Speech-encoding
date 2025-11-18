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

# Initialize logger
logger_main = setup_logger(
    name='main',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR if config.LOG_TO_FILE else None,
    level=config.LOG_LEVEL,
    console_output=True
)

# Use it
if __name__ == "__main__":
    parser = create_dynamic_parser()
    args = parser.parse_args()
    apply_args_to_config(args, logger=logger_main)

# ============
# RUN ANALYSIS
# ============
def main(
    situations = config.situations,
    sessions = config.sessions,
    stimuli = config.stimuli,
    bands = config.bands,
    stims_preprocess = config.stims_preprocess,
    eeg_preprocess = config.eeg_preprocess,
    solver = config.solver,
    model = config.model, 
    delays = config.delays,
    tmax = config.tmax,
    tmin = config.tmin,
    figures_dir = config.figures_dir,
    output_dir = config.output_dir,
    saves_dir = config.saves_dir,
    val_correlation_limit_percentage = config.val_correlation_limit_percentage,
    n_folds = config.n_folds,
    just_load_data = config.just_load_data,
    save_results = config.save_results,
    no_figures = config.no_figures,
    same_validation_subjects = config.same_validation_subjects,
    external_validation = config.external_validation,
    default_alpha = config.default_alpha,
    set_alpha = config.set_alpha,
    info_mne = config.info_mne,
    statistical_test = config.statistical_test,
    significance_threshold = config.significance_threshold,
    perform_tfce = config.perform_tfce,
    n_permutations = config.n_permutations,
    number_of_jobs = config.number_of_jobs,
    logger_main=logger_main,
    ROI:bool=config.ROI,
    validation_results:dict=None,
    load_results:dict=None
):
    if set_alpha:
        logger_main.warning(f"\n\n\tWARNING: ALPHA IS BEING FORCE TO {set_alpha}\n\n")
    
    if validation_results is not None:
        logger_main.info("Using provided validation results")
        ROI=False
        situations = list(validation_results.keys())
        bands = validation_results[situations[0]].keys()
        stimuli = validation_results[situations[0]][list(bands)[0]].keys()
    
    total_results = {
        situation: {
            band: {
                stim: None for stim in stimuli
            } for band in bands
        } for situation in situations
    }
    for situation in situations:
        start_time = datetime.now()
        stimulus_runtimes = {}
        
        for band in bands:
            for stim in stimuli:
                stim_start_time = datetime.now()
                sorted_stimuli, sorted_bands = sorted(stim.split('_')), sorted(band.split('_'))
                stim, band = '_'.join(sorted_stimuli), '_'.join(sorted_bands)

                # Update
                logger_main.info(
                    '\n\t\t===========================\n'
                    '\t\t\tPARAMETERS\n\n'
                    f'\t\tModel: {model}-{solver}\n'
                    f'\t\tBand: {band}\n'
                    f'\t\tStimulus: {stim}\n'
                    f'\t\tCondition: {situation}\n'
                    f'\t\tTime interval: ({tmin},{tmax})s\n'
                    '\n\t\t===========================\n'
                )

                # Relevant paths
                if same_validation_subjects:
                    if external_validation:
                        path_weights = f'{output_dir}/{model}-{solver}/External-{situation}/weights/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_null = f'{output_dir}/{model}-{solver}/External-{situation}/null_model/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_figures = f'{figures_dir}/{model}-{solver}/External-{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_TFCE = f'{output_dir}/{model}-{solver}/External-{situation}/TFCE/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/'
                        save_results_path = f'{output_dir}/{model}-{solver}/External-{situation}/correlations/same_alpha/tmin{tmin}_tmax{tmax}/{band}/'
                    else:
                        path_weights = f'{output_dir}/{model}-{solver}/{situation}/weights/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_null = f'{output_dir}/{model}-{solver}/{situation}/null_model/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_figures = f'{figures_dir}/{model}-{solver}/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_TFCE = f'{output_dir}/{model}-{solver}/{situation}/TFCE/stims_{stims_preprocess}_EEG_{eeg_preprocess}/same_alpha/tmin{tmin}_tmax{tmax}/'
                        save_results_path = f'{output_dir}/{model}-{solver}/{situation}/correlations/same_alpha/tmin{tmin}_tmax{tmax}/{band}/'
                else:
                    if external_validation:
                        path_weights = f'{output_dir}/{model}-{solver}/External-{situation}/weights/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_null = f'{output_dir}/{model}-{solver}/External-{situation}/null_model/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_figures = f'{figures_dir}/{model}-{solver}/External-{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_TFCE = f'{output_dir}/{model}-{solver}/External-{situation}/TFCE/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/'
                        save_results_path = f'{output_dir}/{model}-{solver}/External-{situation}/correlations/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/'
                    else:
                        path_weights = f'{output_dir}/{model}-{solver}/{situation}/weights/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_null = f'{output_dir}/{model}-{solver}/{situation}/null_model/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_figures = f'{figures_dir}/{model}-{solver}/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                        path_TFCE = f'{output_dir}/{model}-{solver}/{situation}/TFCE/stims_{stims_preprocess}_EEG_{eeg_preprocess}/distinct_alpha/tmin{tmin}_tmax{tmax}/'
                        save_results_path = f'{output_dir}/{model}-{solver}/{situation}/correlations/distinct_alpha/tmin{tmin}_tmax{tmax}/{band}/'

                preprocessed_data_path = f'{saves_dir}/preprocessed_data/tmin{tmin}_tmax{tmax}/'
                if external_validation:
                    path_validation = f'{output_dir}/{model}-{solver}/External/validation/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                else:
                    path_validation = f'{output_dir}/{model}-{solver}/{situation}/validation/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                if ROI:
                    path_validation = path_validation.replace(f'EEG', 'ROI')
                alphas_path = os.path.join(path_validation, f'corr_limit_{val_correlation_limit_percentage}.pkl')


                # Make lists to store relevant data across sobjects
                repeated_good_correlation_channels_subjects = []
                null_correlation_per_channel_subjects = []
                correlation_per_channel_subjects = []
                average_correlation_subjects = []
                average_weights_subjects = []
                pvalues_corr_subjects = []
                alphas_subjects = []

                # Store total number of subjects (18) to save figures and results just in this case
                total_number_of_subjects = 0

                # from IPython import embed; embed()
                if same_validation_subjects and set_alpha is None:
                    alphas_total = []
                    if validation_results is not None:
                        for session in sessions:
                            for subject in [1, 2]:
                                alphas_total.append(validation_results[situation][band][stim][session][subject])
                    else:
                        alphas = load_pickle(path=alphas_path)
                        for session in sessions:
                            for subject in [1, 2]:
                                alphas_total.append(alphas[session][subject])
                    alphas_total = np.array(alphas_total)
                    set_alpha = 10**(np.median(np.log10(alphas_total)))
                    logger_main.info(f'Setting alpha to {set_alpha} for all subjects')
                else:
                    ...
                
                # Iterate over sessions
                for session in sessions:
                    logger_main.info(f'\n-------> Start of session {session}\n')

                    # Load data by subject, EEG and info
                    if load_results:
                        subject_1, subject_2, samples_info = load_results[situation][band][stim][session]
                    else:
                        subject_1, subject_2, samples_info = load_data(
                            preprocessed_data_path=preprocessed_data_path,
                            situation=situation,
                            session=session,
                            stimuli=stim,
                            band=band,
                            save_results=save_results,
                            logger=logger_main
                        )          
                    if ROI:
                        ROI_labels = load_pickle(
                            path=rf'data\ROIs\S{session}\s{session}-1-roi-signals-bilateral-labels.pkl'
                        )
                        eeg_subject_1 = load_pickle(
                            path=rf'data\ROIs\S{session}\s{session}-1-roi-signals-bilateral.pkl'
                        )
                        eeg_subject_2 = load_pickle(
                            path=rf'data\ROIs\S{session}\s{session}-2-roi-signals-bilateral.pkl'
                        )
                        relevant_indexes_1 = np.arange(eeg_subject_1.shape[0])
                        relevant_indexes_2 = np.arange(eeg_subject_2.shape[0])
                        logger_main.info("Using ROI data for EEG")
                    else:
                        eeg_subject_1, eeg_subject_2 = subject_1['EEG'], subject_2['EEG']
                        relevant_indexes_1 = samples_info['keep_indexes1'].copy()
                        relevant_indexes_2 = samples_info['keep_indexes2'].copy()

                    if just_load_data:
                        continue

                    # Load stimuli by subject (i.e: concatenated stimuli features)
                    n_feats = [subject_1[stimulus].shape[1] for stimulus in stim.split('_')]
                    stims_subject_1 = np.hstack([subject_1[stimulus] for stimulus in stim.split('_')])
                    stims_subject_2 = np.hstack([subject_2[stimulus] for stimulus in stim.split('_')])

                    # Run model for each subject
                    for subject, eeg, stims, relevant_indexes in zip(
                        (1, 2), 
                        (eeg_subject_1, eeg_subject_2), 
                        (stims_subject_1, stims_subject_2), 
                        (relevant_indexes_1, relevant_indexes_2)
                        ):
                        logger_main.info(f'\n\t······  Running model for Subject {subject}\n')
                        
                        # Initialize empty variables to store relevant data of each fold
                        weights_per_fold = np.zeros((n_folds, info_mne['nchan'], np.sum(n_feats), len(delays)), dtype=np.float32)
                        correlation_per_channel = np.zeros((n_folds, info_mne['nchan']))

                        # Variable to store all channel's p-value
                        topo_pvalues_corr_per_fold = np.zeros((n_folds, info_mne['nchan']))

                        # Variable to store p-value of significant channels
                        proba_correlation_per_channel = np.ones((n_folds, info_mne['nchan']))

                        # Set alpha for specific subject
                        if validation_results is not None and set_alpha is None:
                            alpha = validation_results[situation][band][stim][session][subject]
                        else:
                            if set_alpha is None:
                                try:
                                    alphas = load_pickle(path=alphas_path)
                                    alpha = alphas[session][subject]
                                except:
                                    alpha = default_alpha
                            else:
                                alpha = set_alpha
                        alphas_subjects.append(alpha)

                        # Make the Kfold test
                        kf_test = KFold(n_folds, shuffle=False)

                        # Keep relevant indexes for eeg
                        relevant_eeg = eeg[relevant_indexes]
                        
                        # Run folds
                        for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                            logger_main.debug(f'\n\t······  [{fold+1}/{n_folds}]\t-->\t α:{alpha:.2f}')

                            # Store model output
                            if not statistical_test:
                                fold, weights_per_fold[fold], correlation_per_channel[fold] = fold_model(
                                    fold=fold,
                                    alpha=alpha,
                                    stims=stims,
                                    eeg=eeg,
                                    statistical_test=statistical_test,
                                    relevant_indexes=relevant_indexes,
                                    train_indexes=train_indexes,
                                    test_indexes=test_indexes,
                                    path_null=path_null,
                                    validation=False,
                                    subject=subject,                              
                                    session=session,
                                    logger=logger_main
                                )
                            else:
                                fold, weights_per_fold[fold], correlation_per_channel[fold], p_corr, null_correlation_per_channel= fold_model(
                                    fold=fold,
                                    alpha=alpha,
                                    stims=stims,
                                    eeg=eeg,
                                    statistical_test=statistical_test,
                                    relevant_indexes=relevant_indexes,
                                    train_indexes=train_indexes,
                                    test_indexes=test_indexes,
                                    path_null=path_null,
                                    validation=False,
                                    subject=subject,                              
                                    session=session,
                                    logger=logger_main
                                )

                                # p-values for significant channels (the rest are ones, i.e: not significant)
                                proba_correlation_per_channel[fold][p_corr < significance_threshold] = p_corr[p_corr < significance_threshold]
                                
                                # all p-values for topographic distribution across channels
                                topo_pvalues_corr_per_fold[fold] = p_corr

                        # Take average weights, avoiding folds entirely filled with zeros
                        empty_mask = np.array([np.all(weight == 0) for weight in weights_per_fold])
                        if empty_mask.any():
                            empty_fold_indices = np.where(empty_mask)[0]
                            weights_per_fold[empty_mask] = np.nan
                            logger_main.warning(f'\n\t\t{">" * 26}\n'
                                f'\t\tFolds {", ".join(map(str, empty_fold_indices + 1))} out of {n_folds} are empty\n'
                                f'\t\t{">" * 26}')
                                
                        average_weights = np.nanmean(weights_per_fold, axis=0) # info_mne['nchan'], np.sum(n_feats), len(delays)
                        average_weights = np.nan_to_num(average_weights)
                                        
                        # Take average correlation 
                        average_correlation = np.nanmean(correlation_per_channel, axis=0)
                        average_correlation = np.nan_to_num(average_correlation)

                        # Channels that passed the tests
                        corr_good_channel_indexes = []
                    
                        # Variable to store significant channels
                        repeated_good_correlation_channels = np.zeros(info_mne['nchan'])

                        # Find good indexes by checking where all folds (at the same time) are significant
                        if statistical_test: 
                            corr_good_channel_indexes, = np.where(
                                np.all((proba_correlation_per_channel < 1), axis=0)
                            )
                            if len(corr_good_channel_indexes) == 0:
                                logger_main.warning('No significant channels found (correlation)')   
                                corr_good_channel_indexes = []

                        # Avergae p-values across all folds
                        topo_pval_corr_subject = topo_pvalues_corr_per_fold.mean(axis=0)

                        # Saves average correlation and weights between folds of each channel of each subject to take average above subjects channels
                        if statistical_test:
                            null_correlation_per_channel_subjects.append(null_correlation_per_channel)  
                        else: 
                            null_correlation_per_channel_subjects.append(
                                np.zeros((n_folds, info_mne['nchan'])) # Null correlation is zeros
                                )
                        repeated_good_correlation_channels_subjects.append(corr_good_channel_indexes)
                        correlation_per_channel_subjects.append(correlation_per_channel)
                        average_correlation_subjects.append(average_correlation)
                        pvalues_corr_subjects.append(topo_pval_corr_subject)
                        average_weights_subjects.append(average_weights)

                        # Update the number of subjects
                        total_number_of_subjects+=1

                    # Print the progress of the iteration
                    iteration_percentage(
                        txt=f'\n-------> End of session {session}\n', 
                        i=sessions.index(session), 
                        length_of_iterator=len(sessions),
                        logger=logger_main
                    )

                if just_load_data:
                    continue
                
                # Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
                average_correlation_subjects = np.stack(average_correlation_subjects , axis=0) # n_subj, n_chans
                average_weights_subjects = np.stack(average_weights_subjects, axis=0) # n_subj, n_chans, n_feats, n_delays
                pvalues_corr_subjects = np.stack(pvalues_corr_subjects , axis=0) # n_subj, n_chans

                # Save results
                if save_results and total_number_of_subjects==18:
                    if ROI:
                        save_results_path += 'ROI/'
                        path_weights += 'ROI/'
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
                
                # Store total results
                total_results[situation][band][stim] = {
                    'average_correlation_subjects': average_correlation_subjects,
                    'average_weights_subjects': average_weights_subjects,
                }

                if perform_tfce:
                    if ROI:
                        path_TFCE += 'ROI/'
                    try:
                        logger_main.info("Loading TFCE data")
                        tvalue_tfce, pvalue_tfce = load_pickle(
                            path=os.path.join(path_TFCE, band, stim + f'_{n_permutations}.pkl')
                            )
                        logger_main.info('Successful load ✓')
                    except:
                        logger_main.warning("Load fail")
                        logger_main.info(f"Computing TFCE: {n_permutations} permutations.")

                        # Compute TFCE to get p-value
                        tvalue_tfce, pvalue_tfce = tfce(
                            average_weights_subjects=average_weights_subjects, # (n_subjects, n_chan, n_feats, n_delays)
                            stimulus=stim,
                            n_permutations=n_permutations,
                            n_jobs=number_of_jobs,
                            verbose_tfce=True
                        )

                        # Save TFCE
                        os.makedirs(os.path.join(path_TFCE, band), exist_ok=True)
                        dump_pickle(
                            path=os.path.join(path_TFCE, band, stim + f'_{n_permutations}.pkl'), 
                            obj=(tvalue_tfce, pvalue_tfce), rewrite=True
                            )
            
                # Calculate runtime for this stimulus
                stim_runtime = datetime.now().replace(microsecond=0) - stim_start_time.replace(microsecond=0)
                stimulus_runtimes[f"{band}_{stim}"] = stim_runtime
                
                # Print stimulus completion time
                logger_main.info(
                    f"\n\n\t{'='*40}\n"
                    f"\t✅ STIMULUS COMPLETED: {band}_{stim}\n\n"
                    f"\t⏱️  Runtime: {stim_runtime}\n"
                    f"\t📊 Subjects processed: {total_number_of_subjects}\n"
                    f"\t{'='*40}\n"
                )
                
                if not no_figures:
                    logger_main.info("🎨 Iniciando generación de gráficos...")
                    get_general_plots.main(
                        repeated_good_correlation_channels_subjects=repeated_good_correlation_channels_subjects,
                        null_correlation_per_channel_subjects=null_correlation_per_channel_subjects,
                        correlation_per_channel_subjects=correlation_per_channel_subjects,
                        average_correlation_subjects=average_correlation_subjects,
                        average_weights_subjects=average_weights_subjects,
                        total_number_of_subjects=total_number_of_subjects,
                        pvalues_corr_subjects=pvalues_corr_subjects,
                        alphas_subjects=alphas_subjects,
                        path_figures=path_figures,
                        pvalue_tfce=pvalue_tfce if perform_tfce else None,
                        n_feats=n_feats,
                        band=band,
                        stim=stim,
                        same_validation_subjects=same_validation_subjects,
                        ROI=ROI,
                        ROI_labels=ROI_labels if ROI else None,
                        logger=logger_main
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
            caption='Run finished',
            logger=logger_main
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
        logger_main.info(text)
    return total_results


if __name__ == "__main__":
    results = main()
    # print(load_pickle(r"output\mtrf-ridge\External\correlations\distinct_alpha\tmin-0.2_tmax0.6\Broad\ROI\Envelope.pkl")['average_correlation_subjects'].mean())