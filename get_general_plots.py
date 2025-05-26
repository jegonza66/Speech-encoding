# Standard libraries
from typing import Union
import numpy as np

# Specific libraries
import fire

# Modules
import config
import utils.plot as plot


def main(
    band: str,
    stim: str,
    n_feats: int,
    path_figures: str,
    total_number_of_subjects: int,
    alphas_subjects: list, # n_subj
    average_weights_subjects: np.ndarray,  # n_subj, n_chans, n_feats, n_delays
    average_rmse_subjects: np.ndarray, # n_subj, n_chans
    pvalues_corr_subjects: np.ndarray, # n_subj, n_chans
    pvalues_rmse_subjects: np.ndarray, # n_subj, n_chans
    average_correlation_subjects: np.ndarray, # n_subj, n_chans
    repeated_good_rmse_channels_subjects: np.ndarray, # n_subj, n_chans
    repeated_good_correlation_channels_subjects: np.ndarray, # n_subj, n_chans
    correlation_per_channel_subjects: np.ndarray, # n_subj, n_chans
    null_correlation_per_channel_subjects: np.ndarray, # n_chans
    pvalue_tfce: Union[None, np.ndarray] = None, # n_chans
    )->None:
    """
    Main function to generate general plots for the project.
    
    Returns:
        str: Path to the generated plot.
    """
    import IPython 
    IPython.embed()
    for session in config.sessions:
        for subject in range(average_weights_subjects.shape[0]):
            
            if config.statistical_test:
                # Plot shadows for each subject
                plot.null_correlation_vs_correlation_good_channels(
                    display_interactive_mode=config.display_interactive_mode, 
                    session=session, 
                    subject=subject,
                    save_path=path_figures, 
                    good_channels_indexes=repeated_good_correlation_channels_subjects[subject], 
                    correlation_per_channel=correlation_per_channel_subjects[subject],
                    null_correlation_per_channel=null_correlation_per_channel_subjects[subject], 
                    # power_correlation=power_correlation_per_channel.mean(),
                    # power_rmse=power_rmse_per_channel.mean(),
                    save=config.save_figures, 
                    no_figures=config.no_figures
                    )
            
            # Plot head topomap across al channel for correlation and rmse
            plot.topomap(
                good_channels_indexes=repeated_good_correlation_channels_subjects[subject], 
                average_coefficient=average_correlation_subjects[subject], 
                info=config.info_mne,
                coefficient_name='Correlation', 
                save=config.save_figures, 
                display_interactive_mode=config.display_interactive_mode,
                save_path=path_figures, 
                subject=subject, 
                session=session, 
                no_figures=config.no_figures
                )
            plot.topomap(
                good_channels_indexes=repeated_good_rmse_channels_subjects[subject], 
                average_coefficient=average_rmse_subjects[subject], 
                info=config.info_mne,
                coefficient_name='RMSE', 
                save=config.save_figures, 
                display_interactive_mode=config.display_interactive_mode,
                save_path=path_figures, 
                subject=subject, 
                session=session, 
                no_figures=config.no_figures #TODO: remove all config. parameters and put them in plot module
                )

            # Plot weights
            plot.channel_weights(
                info=config.info_mne, 
                save=config.save_figures, 
                save_path=path_figures, 
                average_correlation=average_correlation_subjects[subject],
                average_rmse=average_rmse_subjects[subject], 
                best_alpha=alphas_subjects[subject], 
                average_weights=average_weights_subjects[subject], 
                times=config.times,
                n_feats=n_feats, 
                stim=stim, 
                session=session, 
                subject=subject, 
                hierarchical_clustering=config.hierarchical_clustering,
                display_interactive_mode=config.display_interactive_mode, 
                no_figures=config.no_figures
                )
    
    # Plot average results only if all subjects are analyzed
    config.no_figures=True if (total_number_of_subjects!=18) else config.no_figures

    # Plot average topomap metrics across each subject
    plot.average_topomap(
        average_coefficient_subjects=average_rmse_subjects, 
        stim=stim, 
        info=config.info_mne, 
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
        info=config.info_mne, 
        save=config.save_figures, 
        save_path=path_figures,
        coefficient_name='Correlation', 
        test_result=False, 
        no_figures=config.no_figures
        ) 

    # Plot topomap with relevant times
    plot.topo_map_relevant_times(
        average_weights_subjects=average_weights_subjects, 
        info=config.info_mne, 
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
        info=config.info_mne,
        stim=stim, 
        save=config.save_figures,
        save_path=path_figures, 
        display_interactive_mode=config.display_interactive_mode, 
        no_figures=config.no_figures
        )

    # Plot weights
    plot.average_regression_weights(
        average_weights_subjects=average_weights_subjects, 
        info=config.info_mne, 
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

    if config.statistical_test:
        # Plot topomap of average p-values across all subject
        plot.topo_average_pval(
            pvalues_coefficient_subjects=pvalues_corr_subjects, 
            info=config.info_mne, 
            display_interactive_mode=config.display_interactive_mode,
            save=config.save_figures, 
            save_path=path_figures,
            coefficient_name='correlation', 
            no_figures=config.no_figures
            )
        plot.topo_average_pval(
            pvalues_coefficient_subjects=pvalues_rmse_subjects, 
            info=config.info_mne, 
            display_interactive_mode=config.display_interactive_mode,
            save=config.save_figures, 
            save_path=path_figures, 
            coefficient_name='RMSE', 
            no_figures=config.no_figures
            )

        # Plot topomap of sum of repeated channels across all subject
        plot.topo_repeated_channels(
            repeated_good_coefficients_channels_subjects=repeated_good_correlation_channels_subjects,
            info=config.info_mne, 
            display_interactive_mode=config.display_interactive_mode, 
            save=config.save_figures,
            save_path=path_figures,
            coefficient_name='correlation',
            no_figures=config.no_figures
            )
        plot.topo_repeated_channels(
            repeated_good_coefficients_channels_subjects=repeated_good_rmse_channels_subjects,
            info=config.info_mne, 
            display_interactive_mode=config.display_interactive_mode, 
            save=config.save_figures,
            save_path=path_figures,
            coefficient_name='RMSE',
            no_figures=config.no_figures
            )
        if config.perform_tfce:
            # Plot t and p values
            plot.plot_pvalue_tfce(
                average_weights_subjects=average_weights_subjects, 
                pvalue=pvalue_tfce, 
                times=config.times, 
                stim=stim,
                n_feats=n_feats, 
                info=config.info_mne, 
                significance=config.significance, 
                save_path=path_figures, 
                display_interactive_mode=config.display_interactive_mode,
                save=config.save_figures, 
                no_figures=config.no_figures
                )
    
    
if __name__ == "__main__":
    fire.Fire(
        main
    )