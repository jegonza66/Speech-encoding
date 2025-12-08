# Standard libraries
import numpy as np, os
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv, iteration_percentage
from model_implementations import fold_model
from utils.plot import hyperparameter_selection
from load import load_data
import config

# Notification bot
from utils.notification_telegram import tel_message, generate_completion_message
from utils.telegram_config import API_TOKEN, CHAT_ID

# Command line and logging
from utils.from_commands import create_dynamic_parser, apply_args_to_config
from utils.logs import setup_logger

# Initialize logger
logger_val = setup_logger(
    name='validation',
    log_to_file=config.LOG_TO_FILE,
    log_dir=os.path.join(config.LOG_DIR, datetime.now().strftime('%Y-%m-%d--%H-%M-%S') + '_validation.log') if config.LOG_TO_FILE else None,
    level=config.LOG_LEVEL
)

# Use it
if __name__ == "__main__":
    parser = create_dynamic_parser()
    args = parser.parse_args()
    apply_args_to_config(args, logger=logger_val)

# ============
# RUN ANALYSIS
# ============

# Start execution
def main(
    situations = config.situations,
    sessions = config.sessions,
    stimuli = config.stimuli,
    bands = config.bands,
    stims_preprocess = config.stims_preprocess,
    eeg_preprocess = config.eeg_preprocess,
    solver = config.solver,
    model = config.model, 
    times = config.times,
    tmax = config.tmax,
    tmin = config.tmin,
    figures_dir = config.figures_dir,
    output_dir = config.output_dir,
    saves_dir = config.saves_dir,
    val_correlation_limit_percentage = config.val_correlation_limit_percentage,
    alphas_swept = config.alphas_swept,
    n_folds = config.n_folds,
    save_results = config.save_results,
    save_figures = config.save_figures,
    recompute = True,
    no_figures = config.no_figures,
    logger_val=logger_val,
    ROI:bool = config.ROI,
    load_results:dict=None
):
    if load_results is not None:
        situations = list(load_results.keys())
        bands = list(load_results[situations[0]].keys())
        stimuli = list(load_results[situations[0]][bands[0]].keys())
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
                ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
                stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)
                
                # Update
                logger_val.info(
                    '\n===========================\n'
                    '\tPARAMETERS\n\n'
                    f'Model: {model}-{solver}\n'
                    f'Band: {band}\n'
                    f'Stimulus: {stim}\n'
                    f'Condition: {situation}\n'
                    f'Time interval: ({tmin},{tmax})s\n'
                    '\n===========================\n'
                )
                
                # Relevant paths
                preprocessed_data_path = os.path.normpath(f'{saves_dir}/preprocessed_data/tmin{tmin}_tmax{tmax}/')
                figures_path = os.path.normpath(f'{figures_dir}/{model}-{solver}_trace/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}')
                path_validation = f'{output_dir}/{model}-{solver}/{situation}/validation/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
                if ROI:
                    figures_path += '/ROI'
                    path_validation = path_validation.replace('EEG', 'ROI')
                alphas_path = os.path.join(path_validation, f'corr_limit_{val_correlation_limit_percentage}.pkl')
                
                # Try to access alphas
                try:
                    if recompute:
                        alphas = {s: {} for s in sessions}
                    else:
                        alphas = load_pickle(path=alphas_path)
                except:
                    alphas = {s: {} for s in sessions} 
            
                # Iterate over sessions
                for session in sessions:
                    logger_val.info(f'\n\n------->\tStart of session {session}\n')

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
                            logger=logger_val
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
                        logger_val.info("Using ROI data for EEG")
                    else:
                        eeg_subject_1, eeg_subject_2 = subject_1['EEG'], subject_2['EEG']
                        relevant_indexes_1 = samples_info['keep_indexes1'].copy()
                        relevant_indexes_2 = samples_info['keep_indexes2'].copy()
                    
                    # Load stimuli by subject (i.e: concatenated stimuli features)
                    stims_subject_1 = np.hstack([subject_1[stimulus] for stimulus in stim.split('_')]) 
                    stims_subject_2 = np.hstack([subject_2[stimulus] for stimulus in stim.split('_')])

                    # Run model for each subject
                    for subject, eeg, stims, relevant_indexes in zip((1, 2), (eeg_subject_1, eeg_subject_2), (stims_subject_1, stims_subject_2), (relevant_indexes_1, relevant_indexes_2)):
                        logger_val.info(f'\n\n\t······  Running model for Subject {subject}\n')
                        if alphas[session].get(subject) is not None:
                            logger_val.info(f'\n\t······  Skipping Subject {subject}, already computed alpha: {alphas[session][subject]}\n')
                            continue
                        # Make sweep 
                        correlations_per_fold = np.zeros(
                            (n_folds, len(alphas_swept))
                        )
                        trfs_per_fold = np.zeros(
                            (n_folds, len(alphas_swept), len(times))
                        )
                        
                        # Make sweep 
                        correlations_per_fold_train = np.zeros(
                            (n_folds, len(alphas_swept))
                        )

                        # Make the Kfold test
                        kf_test = KFold(n_folds, shuffle=False)


                        # Keep relevant indexes for eeg
                        if relevant_indexes is not None:
                            relevant_eeg = eeg[relevant_indexes]
                        else:
                            relevant_eeg = eeg
                        
                        # Run folds 
                        for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                            logger_val.debug(f'\n\t······  [{fold+1}/{n_folds}]\t-->\t Validation fold')
                            trfs_per_fold[fold], correlations_per_fold[fold], correlations_per_fold_train[fold]  = fold_model(
                                relevant_indexes=relevant_indexes,
                                train_indexes=train_indexes,
                                alpha=alphas_swept,
                                test_indexes=test_indexes,  
                                validation=True,
                                stims=stims,
                                fold=fold,
                                eeg=eeg,
                                logger=logger_val,
                                solver=solver
                            )     

                        # Calculate mean correlation, and std
                        correlations = np.nan_to_num(np.nanmean(correlations_per_fold, axis=0))
                        correlations_std = np.nan_to_num(np.nanstd(correlations_per_fold, axis=0))
                        
                        # Same for training
                        correlations_train = np.nan_to_num(np.nanmean(correlations_per_fold_train, axis=0))
                        
                        # Calculate mean TRFs
                        trfs = np.nanmean(trfs_per_fold, axis=0)
                        
                        # Find all indexes where the relative difference between the correlation and its maximum is within corr_limit_percent
                        relative_difference = abs((correlations.max() - correlations)/correlations.max())
                        good_indexes_range = np.where(relative_difference < val_correlation_limit_percentage)[0]

                        # Get the very last one, because the greater the alpha, the smoothest the signal gets
                        alpha_subject = alphas_swept[int(good_indexes_range[-1])]
                        
                        # Make the alpha selection process plot
                        hyperparameter_selection(
                            alphas_swept=alphas_swept,
                            correlations=correlations, 
                            correlations_std=correlations_std,
                            correlations_train=correlations_train, 
                            trfs=trfs,
                            alpha_subject=alpha_subject,
                            correlation_limit_percentage=val_correlation_limit_percentage, 
                            session=session, 
                            subject=subject, 
                            stim=stim, 
                            band=band, 
                            save_path=figures_path, 
                            save=save_figures, 
                            no_figures=no_figures
                        )

                        # Update dictionary
                        alphas[session][subject] = alpha_subject

                        # Save/store results
                        os.makedirs(name=path_validation, exist_ok=True)
                        if save_results:
                            dump_pickle(path=alphas_path, obj=alphas, rewrite=True)
                    total_results[situation][band][stim] = alphas.copy()
                        
                    # Print the progress of the iteration
                    iteration_percentage(
                        txt=f'\n------->\tEnd of session {session}\n', 
                        i=sessions.index(session), 
                        length_of_iterator=len(sessions),
                        logger=logger_val
                    )

                # Calculate runtime for this stimulus
                stim_runtime = datetime.now().replace(microsecond=0) - stim_start_time.replace(microsecond=0)
                stimulus_runtimes[f"{band}_{stim}"] = stim_runtime
                
                # Print stimulus completion time
                logger_val.info(
                    f"\n\t{'='*40}\n"
                    f"\t✅ STIMULUS COMPLETED: {band}_{stim}\n\n"
                    f"\t⏱️  Runtime: {stim_runtime}\n"
                    f"\t📊 Subjects processed: {len(sessions) * 2}\n"
                    f"\t{'='*40}\n"
                )

        # Get total run time            
        total_runtime = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
        
        # Generate completion message
        text = generate_completion_message(
            situation=situation,
            total_number_of_subjects=len(sessions) * 2,
            stimulus_runtimes=stimulus_runtimes,
            total_runtime=str(total_runtime),
            save_path=path_validation,
            fig_path=figures_path
        )
        
        # Send text to telegram bot
        tel_message(
            api_token=API_TOKEN,
            chat_id=CHAT_ID, 
            message=text,
            caption='Validation Analysis Completed',
            logger=logger_val
        )

        # Dump metadata
        metadata_path = f'saves/log/validation/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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
        logger_val.info(text)
    return total_results

if __name__=='__main__':
    results = main()