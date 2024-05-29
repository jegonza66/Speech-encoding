# Standard libraries
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, seaborn as sn, mne, scipy.signal as sgn, warnings

# Specific libraries
from scipy.stats import wilcoxon#, pearsonr
from statannot import add_stat_annotation
import librosa

# Default size is 10 pts, the scalings (10pts*scale) are:
#'xx-small':0.579,'x-small':0.694,'small':0.833,'medium':1.0,'large':1.200,'x-large':1.440,'xx-large':1.728,None:1.0}
import matplotlib.pylab as pylab
import matplotlib.ticker as tkr
params = {'legend.fontsize': 'x-large',
          'legend.title_fontsize': 'x-large',
          'figure.figsize': (8, 6),
          'figure.titlesize': 'xx-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Modules
import Funciones, setup
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")
warnings.filterwarnings("ignore", message="More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.")
exp_info = setup.exp_info()

# TODO HALF CHECK
def null_correlation_vs_correlation_good_channels(good_channels_indexes:np.ndarray,
                         average_correlation:np.ndarray, 
                         save_path:str,
                         correlation_per_channel:np.ndarray, 
                         null_correlation_per_channel:np.ndarray,
                         save:bool=False, 
                         display_interactive_mode:bool=False, 
                         session:int=21, 
                         subject:int=1):
    """_summary_

    Parameters
    ----------
    good_channels_indexes : np.ndarray
        _description_
    average_correlation : np.ndarray
        _description_
    save_path : str
        _description_
    correlation_per_channel : np.ndarray
        _description_
    null_correlation_per_channel : np.ndarray
        _description_
    save : bool, optional
        _description_, by default False
    display_interactive_mode : bool, optional
        _description_, by default False
    session : int, optional
        _description_, by default 21
    subject : int, optional
        _description_, by default 1
    """
    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Define minimum and maximum of null correlations
    null_correlation_per_channel_min = null_correlation_per_channel.min(axis=1).min(axis=0)
    null_correlation_per_channel_max = null_correlation_per_channel.max(axis=1).max(axis=0)

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7), layout='tight')
    fig.suptitle(f'Session {session} - Subject {subject}')

    # Graph average correlation
    ax.plot(average_correlation, '.', color='C0', label="Full mean of correlations among folds")
    if len(good_channels_indexes): 
        ax.plot(good_channels_indexes, average_correlation[good_channels_indexes], '*', color='C1', label="Mean of correlations among folds (Test passed)")

    # Add shadow between min and max
    ax.fill_between(x=np.arange(len(average_correlation)), 
                    y1=correlation_per_channel.min(axis=0),
                    y2=correlation_per_channel.max(axis=0), 
                    alpha=0.5,
                    label='Correlation distribution (Real data)')
    ax.fill_between(x=np.arange(len(average_correlation)), 
                    y1=null_correlation_per_channel_min,
                    y2=null_correlation_per_channel_max, 
                    alpha=0.5,
                    label='Correlation distribution (Random data)')
    
    # Graph properties
    ax.grid(visible=True)
    ax.set(xlim=[-1, 129],
           xlabel='Channels',
           ylabel='Correlation')
    ax.legend(loc="lower right")

    # If there are no good channels
    if not len(good_channels_indexes): 
        plt.text(64, 
                 np.max(abs(correlation_per_channel))/2, 
                 "No surviving channels", 
                 size='xx-large', 
                 ha='center')

    # Wether graph is saved
    if save:
        save_path_graficos = save_path + 'correlation_vs_null_correlation/'
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + f'session{session}_subject{subject}.png')

# TODO CHECK DESCRIPTION
def lateralized_channels(info:mne.io.meas_info.Info, 
                         save_path:str, 
                         channels_right:list=['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3'], 
                         channels_left:list=['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3'], 
                         display_interactive_mode:bool=False, 
                         save:bool=True):
    """Make a topomap showing masked channels for lateralization comparisson

    Parameters
    ----------
    info : mne.io.meas_info.Info
        _description_
    save_path : str
        _description_
    channels_right : list, optional
        _description_, by default ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3']
    channels_left : list, optional
        _description_, by default ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3']
    display_interactive_mode : bool, optional
        _description_, by default False
    save : bool, optional
        _description_, by default True
    """
    
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Lateralization comparison
    if channels_right is None:
        channels_right = ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3']
    if channels_left is None:
        channels_left = ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3']

    # Create mask
    mask = [i in channels_right + channels_left for i in info['ch_names']]
    
    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.title('Masked channels for lateralization comparisson')
    
    # Make topomap
    mne.viz.plot_topomap(data=np.zeros(info['nchan']),
                         pos=info, 
                         show=display_interactive_mode, 
                         sphere=0.07, 
                         mask=np.array(mask),
                         mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=12), 
                         axes=ax)
    # Save figure
    if save:
        save_path += 'lateralization/'
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + f'masked_left_vs_right_chs_{len(channels_right)}_channels.png')
        fig.savefig(save_path + f'masked_left_vs_right_chs_{len(channels_right)}_channels.svg')

#TODO CHECK DESCRIPTION
def topomap(good_channels_indexes:np.ndarray,
            average_coefficient:np.ndarray, 
            info:mne.io.meas_info.Info,
            coefficient_name:str, 
            save:bool, 
            save_path:str, 
            display_interactive_mode:bool=False,
            session:int=21, 
            subject:int=1):
    """Make topographic plot of brain with heat-like map for given coefficient

    Parameters
    ----------
    good_channels_indexes : np.ndarray
        _description_
    average_coefficient : np.ndarray
        _description_
    info : mne.io.meas_info.Info
        _description_
    coefficient_name : str
        _description_
    save : bool
        _description_
    save_path : str
        _description_
    display_interactive_mode : bool, optional
        _description_, by default False
    session : int, optional
        _description_, by default 21
    subject : int, optional
        _description_, by default 1
    """

    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Plot head correlation
    if len(good_channels_indexes):
        # Create figure and title
        fig, axs = plt.subplots(nrows=1, ncols=2, layout='tight')
        plt.suptitle(f"Session{session} Subject{subject}\n{coefficient_name} = ({average_coefficient.mean():.3f}" +r'$\pm$'+ f"{average_coefficient.std():.3f})")
        
        # Make topomap
        im = mne.viz.plot_topomap(data=average_coefficient, 
                                  pos=info, 
                                  axes=axs[0], 
                                  show=False, 
                                  sphere=0.07, 
                                  cmap='Greys', 
                                  vmin=average_coefficient.min(), 
                                  vmax=average_coefficient.max()
                                  )
        
        # Mask for good channels
        mask = np.array([i in good_channels_indexes for i in range(info['nchan'])])
        im2 = mne.viz.plot_topomap(data=np.zeros(info['nchan']), 
                                   pos=info, 
                                   axes=axs[1], 
                                   show=False, 
                                   sphere=0.07,
                                   mask=mask, 
                                   mask_params=dict(marker='o', markerfacecolor='g', markeredgecolor='k', linewidth=0, markersize=4)
                                   )
        # Make plot
        plt.colorbar(im[0], 
                     ax=[axs[0], axs[1]],
                     shrink=0.85, 
                     label=coefficient_name, 
                     orientation='horizontal',
                     boundaries=np.linspace(average_coefficient.min().round(decimals=3), average_coefficient.max().round(decimals=3), 100),
                     ticks=np.linspace(average_coefficient.min(), average_coefficient.max(), 9).round(decimals=3)
                     )

    else:
        # Create figure and title
        fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
        plt.suptitle(f"Session{session} Subject{subject}\n{coefficient_name} = ({average_coefficient.mean():.3f}" +r'$\pm$'+ f"{average_coefficient.std():.3f})")
        
        # Make topomap
        im = mne.viz.plot_topomap(data=average_coefficient, 
                                  pos=info, 
                                  axes=ax, 
                                  show=False, 
                                  sphere=0.07, 
                                  cmap='Greys',
                                  vmin=average_coefficient.min(), 
                                  vmax=average_coefficient.max()
                                  )
        plt.colorbar(im[0], 
                     ax=ax, 
                     shrink=0.85, 
                     label=coefficient_name, 
                     orientation='horizontal',
                     boundaries=np.linspace(average_coefficient.min().round(decimals=3), average_coefficient.max().round(decimals=3), 100),
                     ticks=np.linspace(average_coefficient.min(), average_coefficient.max(), 9).round(decimals=3)
                     )
    if save:
        save_path_topo = save_path + 'cabezas_topomap/'
        os.makedirs(save_path_topo, exist_ok=True)
        fig.savefig(save_path_topo + f'{coefficient_name.lower()}_topomap_session_{session}_subject_{subject}.png')

#TODO CHECK DESCRIPTION
def average_topomap(average_coefficient_subjects:np.ndarray, 
                          info:mne.io.meas_info.Info,
                          save:bool, 
                          save_path:str, 
                          coefficient_name:str, 
                          number_of_lat_channels:int=12,
                          display_interactive_mode:bool=False,
                          test_result:bool=False):
    """_summary_

    Parameters
    ----------
    average_coefficient_subjects : np.ndarray
        _description_
    info : mne.io.meas_info.Info
        _description_
    save : bool
        _description_
    save_path : str
        _description_
    coefficient_name : str
        _description_
    number_of_lat_channels : int, optional
        _description_, by default 12
    display_interactive_mode : bool, optional
        _description_, by default False
    fontsize : int, optional
        _description_, by default 19
    test_result : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean over all subjects
    mean_average_coefficient = average_coefficient_subjects.mean(axis=0)

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.suptitle(f'{coefficient_name} = ({mean_average_coefficient.mean():.3f}'+r'$\pm$'+f'{mean_average_coefficient.std():.3f})')

    # Make topomap
    im = mne.viz.plot_topomap(data=mean_average_coefficient, 
                              pos=info, 
                              cmap='OrRd',
                              vmin=mean_average_coefficient.min(), 
                              vmax=mean_average_coefficient.max(),
                              show=False, 
                              sphere=0.07, 
                              axes=ax)
    
    # colorbar = plt.colorbar(im[0], shrink=0.85, orientation='horizontal', label=coefficient_name)
    # # colorbar.ax.tick_params(labelsize=fontsize)
    plt.colorbar(im[0],
                 ax=ax, 
                 shrink=0.85,
                 label=coefficient_name,
                 orientation='horizontal',
                 boundaries=np.linspace(mean_average_coefficient.min().round(decimals=3), mean_average_coefficient.max().round(decimals=3), 100),
                 ticks=np.linspace(mean_average_coefficient.min(), mean_average_coefficient.max(), 9).round(decimals=3)
                 )
    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + f'average_{coefficient_name.lower()}_topomap.png')

    # Make Lateralization comparison
    if coefficient_name == 'Correlation':
        # Relevant channels
        all_channels_right = ['B27','B28','B29','B30','B31','B32','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16']
        all_channels_left = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','C24','C25','C26','C27','C28','C29','C30','C31','C32']
        sorted_chs_right = sorted([i for i in info['ch_names'] if i in all_channels_right])
        sorted_chs_left = sorted([i for i in info['ch_names'] if i in all_channels_left])

        # Correlation of relevant channels
        mask_right = [i in all_channels_right for i in info['ch_names']]
        mask_left = [i in all_channels_left for i in info['ch_names']]
        corr_right = np.sort(mean_average_coefficient[mask_right])
        corr_left = np.sort(mean_average_coefficient[mask_left])

        # Lateralization channels to use: currently working with last 
        if number_of_lat_channels:
            corr_right = corr_right[-number_of_lat_channels:]
            corr_left = corr_left[-number_of_lat_channels:]
            sorted_chs_right = sorted_chs_right[-number_of_lat_channels:]
            sorted_chs_left = sorted_chs_left[-number_of_lat_channels:]
        
        # Make figure and data to plot
        fig = plt.figure(layout='tight')
        data = pd.DataFrame({'Left': corr_left, 'Right': corr_right})
        
        # Make boxplot and swarmplot
        ax = sn.boxplot(data=data, width=0.35)
        for patch in ax.artists:
            r, g, b, alpha = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .8))
        sn.swarmplot(data=data, color=".25")

        # Figure properties
        ax.set_ylabel('Correlation')

        # Disable print
        import sys
        sys.stdout = open(os.devnull, 'w')
        add_stat_annotation(ax, 
                            data=data, 
                            box_pairs=[('Left', 'Right')],
                            test='Wilcoxon', 
                            text_format='full', 
                            loc='outside', 
                            fontsize='xx-large', 
                            verbose=0)
        # Restore printing
        sys.stdout = sys.__stdout__
        
        # Make Wilcoxon test for comparisson
        test_results = wilcoxon(data['Left'], data['Right'])

        if save:
            save_path_lat = save_path + 'lateralization/'
            os.makedirs(save_path_lat, exist_ok=True)
            fig.savefig(save_path_lat + f'left_vs_right_{coefficient_name.lower()}_{len(sorted_chs_right)}_channels.svg')
            fig.savefig(save_path_lat + f'left_vs_right_{coefficient_name.lower()}_{len(sorted_chs_right)}_channels.png')

        lateralized_channels(info=info, 
                             channels_right=sorted_chs_right, 
                             channels_left=sorted_chs_left, 
                             save_path=save_path,
                             display_interactive_mode=display_interactive_mode,
                             save=save)
    else:
        test_results = None
    if test_result:
        return test_results

#TODO CHECK DESCRIPTION
def topo_average_pval(pvalues_coefficient_subjects:np.ndarray, 
              info:mne.io.meas_info.Info, 
              save:bool, 
              save_path:str, 
              coefficient_name:str,
              display_interactive_mode:bool=False):
    """_summary_

    Parameters
    ----------
    pvalues_coefficient_subjects : np.ndarray
        _description_
    info : mne.io.meas_info.Info
        _description_
    save : bool
        _description_
    save_path : str
        _description_
    coefficient_name : str
        _description_
    display_interactive_mode : bool, optional
        _description_, by default False
    fontsize : int, optional
        _description_, by default 19
    """
    
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean across all subjects
    topo_pval = pvalues_coefficient_subjects.mean(axis=0)

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.suptitle(f"Mean p-values - {coefficient_name}")
    plt.title(f'Mean: {topo_pval.mean():.3f}'+r'$\pm$'+f'{topo_pval.std():.3f}')

    # Make topomap
    im = mne.viz.plot_topomap(data=topo_pval, 
                              pos=info, 
                              cmap='OrRd',
                              vmin=0, 
                              vmax=topo_pval.max(),
                              show=False, 
                              sphere=0.07,
                              axes=ax)
    # And colorbar
    plt.colorbar(im[0], 
                 shrink=0.85, 
                 orientation='vertical',
                 label='p-value')

    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + f'p-value_topo_{coefficient_name.lower()}.png')
        fig.savefig(save_path + f'p-value_topo_{coefficient_name.lower()}.svg')

#TODO CHECK DESCRIPTION
def topo_repeated_channels(repeated_good_coefficients_channels_subjects:np.ndarray, 
                      info:mne.io.meas_info.Info, 
                      save:bool, 
                      save_path:str, 
                      coefficient_name:str, 
                      display_interactive_mode:bool=False):
    """_summary_

    Parameters
    ----------
    repeated_good_coefficients_channels_subjects : np.ndarrays
        _description_
    info : mne.io.meas_info.Info
        _description_
    save : bool
        _description_
    save_path : str
        _description_
    coefficient_name : str
        _description_
    display_interactive_mode : bool, optional
        _description_, by default False
    """
    
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean across all subjects
    sum_of_repeated_chan = repeated_good_coefficients_channels_subjects.sum(axis=0)
    
    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.suptitle(f"Channels passing 5 test per subject - {coefficient_name}")
    plt.title(f'Mean: {sum_of_repeated_chan.mean():.3f}' +r'$\pm$'+ f'{sum_of_repeated_chan.std():.3f}')

    # Make topomap
    im = mne.viz.plot_topomap(data=sum_of_repeated_chan, 
                              pos=info, 
                              cmap='OrRd',
                              vmin=0, 
                              vmax=18,
                              show=False, 
                              sphere=0.07, 
                              axes=ax)
    # And colorbar
    plt.colorbar(im[0], 
                 shrink=0.85, 
                 orientation='vertical', 
                 label='Number of subjects passed')

    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + f'topo_repeated_channels_{coefficient_name.lower()}.png')
        fig.savefig(save_path + f'topo_repeated_channels_{coefficient_name.lower()}.svg')

#TODO CHECK DESCRIPTION
def topo_map_relevant_times(average_weights_subjects:np.ndarray, 
                            info:mne.io.meas_info.Info,
                            n_feats:list, 
                            band:str, 
                            stim:str, 
                            times:np.ndarray, 
                            sample_rate:int, 
                            save_path:int, 
                            save:bool=True, 
                            display_interactive_mode:bool=False):
    """_summary_

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        _description_
    info : mne.io.meas_info.Info
        _description_
    n_feats : list
        _description_
    band : str
        _description_
    stim : str
        _description_
    times : np.ndarray
        _description_
    sample_rate : int
        _description_
    save_path : int
        _description_
    save : bool, optional
        _description_, by default True
    display_interactive_mode : bool, optional
        _description_, by default False
    fontsize : int, optional
        _description_, by default 19
    """
    # Relevant parameters
    stimuli = stim.split('_')
    
    # Take mean across all subjects
    average_weights = average_weights_subjects.mean(axis=0)

    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing to get corresponding features of given attribute
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat
        weights_across_channels = average_weights[:,index_slice[0]:index_slice[1],:].mean(axis=1)
        mean_weights = weights_across_channels.mean(axis=0)
        
        # Find relevant indexes of average weights across channels
        relevant_indexes, _ = sgn.find_peaks(np.abs(mean_weights), height=np.abs(mean_weights).max()*.3)
        positive_relevant_indexes = np.array([i for i in relevant_indexes if (i/sample_rate + times[0]) >= 0])

        # Keep just positive times (s)
        relevant_times = np.array([i/sample_rate + times[0] for i in positive_relevant_indexes])
        
        if len(positive_relevant_indexes)==0:
            break

        # Turn on/off interactive mode
        plt.close()
        if display_interactive_mode:
            plt.ion()
        else:
            plt.ioff()

        # Create color map for each time
        blues_map = plt.cm.get_cmap('Blues').reversed()
        reds_map = plt.cm.get_cmap('Reds').reversed()
        cmaps = [reds_map if mean_weights[i] > 0 else blues_map for i in positive_relevant_indexes]

        # Create figure and title
        fig, axs = plt.subplots(figsize=(4*len(cmaps), 4), ncols=len(cmaps), layout='tight', sharey=True)
        fig.suptitle(f'Mean weight among subjects - {feat} - {band} band', fontsize='xx-large')
        for j in range(len(positive_relevant_indexes)):
            if len(cmaps)>1:
                ax = axs[j]
            else:
                ax = axs

            # Make topomap
            ax.set_title(f'{int(relevant_times[j]*1000)} ms')
            chan_weight_j = weights_across_channels[:, j].flatten()
            im = mne.viz.plot_topomap(data=chan_weight_j, 
                                      pos=info, 
                                      axes=ax,
                                      show=False,
                                      sphere=0.07, cmap=cmaps[j],
                                      vmin=chan_weight_j.min().round(3),
                                      vmax=chan_weight_j.max().round(3))
            
            # Configure colorbar
            f = lambda x: round(x, -int(np.floor(np.log10(abs(x)))))
            cbar = plt.colorbar(im[0], # TODO PROBLEMA
                         ax=ax,
                         orientation='vertical',
                         shrink=0.6,
                         aspect=15,
                         boundaries=[f(x) for x in np.linspace(chan_weight_j.min(), chan_weight_j.max(), 100)],
                         ticks=[f(x) for x in np.linspace(chan_weight_j.min(),chan_weight_j.max(), 4)])
            # cbar.formatter.set_powerlimits((-2, 2))
            # cbar.ax.xaxis.get_offset_text().set_position((.5,.5))

            if j==len(positive_relevant_indexes)-1:
                cbar.ax.set_ylabel('Weights')
        plt.figtext(x=.05, y=.05, s='Red is reserved for positive peaks, blue for negative ones', fontdict={'weight':'light'})
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path + 'relevant_times.png')
            fig.savefig(save_path + 'relevant_times.svg')

#TODO CHECK DESCRIPTION
def channel_wise_correlation_topomap(average_weights_subjects:np.ndarray, 
                                     info:mne.io.meas_info.Info, 
                                     save:bool, 
                                     save_path:str,
                                     display_interactive_mode:bool=False):
    """_summary_

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        _description_
    info : mne.io.meas_info.Info
        _description_
    save : bool
        _description_
    save_path : str
        _description_
    display_interactive_mode : bool, optional
        _description_, by default False
    fontsize : int, optional
        _description_, by default 19
    """
    # Relevant parameters
    n_subjects, n_chan, _, _ = average_weights_subjects.shape
    average_weights = average_weights_subjects.mean(axis=2)
    correlation_matrices = np.zeros(shape=(n_chan, n_subjects, n_subjects))

    # Calculate correlation betweem subjects
    for channel in range(n_chan):
        matrix = average_weights[:,channel,:] # TODO HAVE ONE MORE DIMENSION
        correlation_matrices[channel] = np.corrcoef(matrix)

    # Correlacion por canal
    absolute_correlation_per_channel = np.zeros(n_chan)
    for channel in range(n_chan):
        channel_corr_values = correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)]
        absolute_correlation_per_channel[channel] = np.mean(np.abs(channel_corr_values))

    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    fig.suptitle('Channel-wise mTRFs similarity')

    # Make topomap
    im = mne.viz.plot_topomap(data=absolute_correlation_per_channel, 
                              pos=info, 
                              axes=ax, 
                              show=False, 
                              sphere=0.07,
                              cmap='Greens', 
                              vmin=absolute_correlation_per_channel.min(),
                              vmax=absolute_correlation_per_channel.max())
    
    # Make colorbar
    cbar = plt.colorbar(im[0], 
                        ax=ax, 
                        shrink=0.85, 
                        orientation='vertical', 
                        label='Correlation')
    
    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + 'channelwise_correlation_topo.png')
        fig.savefig(save_path + 'channelwise_correlation_topo.svg')

#TODO CHECK DESCRIPTION
def channel_weights(info:mne.io.meas_info.Info,
                    save:bool, 
                    save_path:str, 
                    average_correlation:np.ndarray, 
                    average_rmse:np.ndarray, 
                    best_alpha:float, 
                    average_weights:np.ndarray, 
                    times:np.ndarray, 
                    n_feats:list, 
                    stim:str,
                    display_interactive_mode:bool=False,
                    session:int=21, 
                    subject:int=1, 
                    fontsize:int=13):
    """Plot weights of features as an evoked response. If multidimensional features are used, a colormesh is used.

    Parameters
    ----------
    info : mne.io.meas_info.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        Whether to store the figure
    save_path : str
        Path to store the figure
    average_correlation : np.ndarray
        Average correlation across each channel
    average_rmse : np.ndarray
        Average RMSE across each channel
    best_alpha : float
        Alpha used to implement the model
    average_weights : np.ndarray
        Average weights used by the model across folds. Its shape should be n_chan, n_feats, n_delays
    times : np.ndarray
        Time matching with corresponding delays
    n_feats : list
        Number of features within each attribute
    stim : str
        Stimuli used in the model
    display_interactive_mode : bool, optional
        Whether to activate interactive mode, by default False
    session : int, optional
        Number of session, by default 21
    fontsize : int, optional
        Fontsize of labels, by default 13
    """
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Get best correlation and rmse
    best_correlation = average_correlation.max()
    best_rmse = average_rmse.max()
    stimuli = stim.split('_')
    
    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=len(stimuli), figsize=(int(8*(len(stimuli))), 8), layout='tight')
    ax = np.array([[ax]]) if len(stimuli)==1 else ax.reshape(1, len(stimuli))

    fig.suptitle(f'Session {session} - Subject {subject} - Mcorr: {best_correlation:.2f} - Mrmse: {best_rmse:.2f} - '+ r'$\alpha$'+f': {best_alpha:.2f}')

    # Iterate over all stimuli
    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing of relevant features
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat

        # Add color mesh plot for atributes with more than one feature
        if feat.startswith('Phonemes'):
            # Take mean across all phonemes
            average_by_phoneme = average_weights[:, index_slice[0]:index_slice[1], :].mean(axis=0)

            # Create colormesh figure
            im = ax[0,i_feat].pcolormesh(times*1000,
                            np.arange(n_feat), 
                            average_by_phoneme, 
                            cmap='jet', 
                            shading='auto')

            # Set figure configuration
            ax[0,i_feat].tick_params(axis='both', labelsize='medium') # Change labelsize because there are too many phonemes

            if feat.endswith('Manual'):
                ax[0,i_feat].set(xlabel='Time (ms)', ylabel='Phonemes', yticks=np.arange(n_feat), ytickslabel=exp_info.ph_labels_man)
            else:
                ax[0,i_feat].set(xlabel='Time (ms)', ylabel='Phonemes', yticks=np.arange(n_feat), ytickslabel=exp_info.ph_labels)

            fig.colorbar(im, 
                         ax=ax[0,i_feat], 
                         orientation='horizontal', 
                         shrink=1, 
                         label='Amplitude (a.u)', 
                         aspect=15)
        elif feat.startswith('Spectro'):
            # Take mean across all band frequencies
            average_by_band = average_weights[:, index_slice[0]:index_slice[1], :].mean(axis=0)

            # Create colormesh figure
            im = ax[0,i_feat].pcolormesh(times * 1000, 
                            np.arange(n_feat), 
                            average_by_band, 
                            cmap='jet', 
                            shading='auto',
                            vmin=-average_by_band.max(),
                            vmax=average_by_band.max())

            # Set figure configuration
            bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
            ax[0,i_feat].set(xlabel='Time (ms)', 
                             yticklabels=[int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)],
                             ylabel='Frecuency (Hz)',
                             yticks=np.arange(0, n_feat, 2))
            # Configure colorbar
            fig.colorbar(im,
                         ax=ax[0,i_feat], 
                         orientation='horizontal', 
                         shrink=1, 
                         label='Amplitude (a.u.)', 
                         aspect=15)
        else:
            # Create evoked response as graph of weights averaged across all feats
            weights = average_weights[:, index_slice[0]:index_slice[1], :].mean(axis=1)
            evoked = mne.EvokedArray(data=weights, info=info)
        
            # Relabel time 0
            evoked.shift_time(times[0], relative=True)
            
            # Plot
            evoked.plot(
                scalings={'eeg':1}, 
                zorder='std', 
                time_unit='ms',
                show=False, 
                spatial_colors=True, 
                # unit=False, 
                units='mTRF (a.u.)',
                axes=ax[0,i_feat], #axes=ax, 
                gfp=False)

            # Add mean of all channels
            ax[0,i_feat].plot(
                times * 1000, #ms
                evoked._data.mean(0), 
                'k--', 
                label='Mean', 
                zorder=130, 
                linewidth=2)
            
            # Graph properties
            ax[0,i_feat].legend()
            ax[0,i_feat].grid(visible=True)
            ax[0,i_feat].set_title(f'{feat}')
    
    if save:
        save_path_graficos = save_path + 'individual_weights/'
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + f'session_{session}_subject_{subject}.png')

#TODO CHECK DESCRIPTION
def average_regression_weights(average_weights_subjects:np.ndarray, 
                               info:mne.io.meas_info.Info,
                               save:bool, 
                               save_path:str,
                               times:np.ndarray,
                               n_feats:list,
                               stim:str,
                               display_interactive_mode:bool=False,
                               colormesh_form:bool=False):
    """Plot average weights of features as an evoked response. If colormesh_form is passed, a colormesh graph is performed in case of multifeature attribute are used.

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        _description_
    info : mne.io.meas_info.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        Whether to store the figure
    save_path : str
        Path to store the figure
    times : np.ndarray
        Time matching with corresponding delays
    n_feats : list
        Number of features within each attribute
    stim : str
        Stimuli used in the model
    display_interactive_mode : bool, optional
        Whether to activate interactive mode, by default False
    fontsize : int, optional
        Fontsize of labels, by default 13
    colormesh_form : bool, optional
        Whether to add graph with colormesh (only for multifeature attributes), by default False
    """
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean over all subjects
    mean_average_weights_subjects = average_weights_subjects.mean(axis=0)
    stimuli = stim.split('_')

    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Create figure and title
        if colormesh_form:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 12), layout='tight', sharex=True)
            ax, ax1 = ax[0], ax[1]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
        fig.suptitle(f'{feat}')
        
        # Make slicing of relevant features
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat

        # Create evoked response as graph of weights averaged across all feats and all 
        weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=1)
        evoked = mne.EvokedArray(data=weights, info=info)
    
        # Relabel time 0
        evoked.shift_time(times[0], relative=True)
        
        # Plot
        evoked.plot(
            scalings={'eeg':1}, 
            zorder='std', 
            time_unit='ms',
            show=False, 
            spatial_colors=True, 
            # unit=False, 
            units='mTRF (a.u.)',
            axes=ax,
            gfp=False)

        # Add mean of all channels
        ax.plot(
            times * 1000, #ms
            evoked._data.mean(0), 
            'k--', 
            label='Mean', 
            zorder=130, 
            linewidth=2)
        
        # Graph properties
        ax.legend()
        ax.grid(visible=True)

        if colormesh_form:
            if feat.startswith('Spectro'):
                # Take mean across all band frequencies
                average_by_band = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=0)

                # Create colormesh figure
                im = ax1.pcolormesh(times * 1000, 
                                np.arange(n_feat), 
                                average_by_band, 
                                cmap='jet', 
                                shading='auto',
                                vmin=-average_by_band.max(),
                                vmax=average_by_band.max())

                # Set figure configuration
                bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
                ax1.set(ylabel='Frecuency (Hz)', yticks=np.arange(0, n_feat, 2), yticklabels=[int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)])

                # Configure colorbar
                fig.colorbar(im, 
                            ax=ax1, 
                            orientation='horizontal', 
                            shrink=1, 
                            label='Amplitude (a.u.)', 
                            aspect=15)
            elif feat.startswith('Phonemes'):
                # Take mean across all phonemes
                average_by_phoneme = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=0)

                # Create colormesh figure
                im = ax1.pcolormesh(times * 1000,
                                np.arange(n_feat), 
                                average_by_phoneme, 
                                cmap='jet', 
                                shading='auto')

                # Set figure configuration
                if feat.endswith('Manual'):
                    ax1.set(ylabel='Phonemes', yticks=np.arange(n_feat), yticklabels=exp_info.ph_labels_man)
                else:
                    ax1.set(ylabel='Phonemes', yticks=np.arange(n_feat), yticklabels=exp_info.ph_labels)
                
                ax1.tick_params(axis='both', labelsize='medium') # Change labelsize because there are too many phonemes
                
                # Make color bar
                fig.colorbar(im,
                             ax=ax1,
                             orientation='horizontal',
                             label='Amplitude (a.u.)',
                             shrink=1,
                             aspect=20)

        if save:
            os.makedirs(save_path, exist_ok=True)
            if colormesh_form:
                fig.savefig(save_path + f'average_weights_mesh_{feat.lower()}.svg')
                fig.savefig(save_path + f'average_weights_mesh_{feat.lower()}.png')
            else:
                fig.savefig(save_path + f'average_weights_{feat.lower()}.svg')
                fig.savefig(save_path + f'average_weights_{feat.lower()}.png')

#TODO CHECK DESCRIPTION E Y LABEL
def correlation_matrix_subjects(average_weights_subjects:np.ndarray, 
                             stim:str, 
                             n_feats:list, 
                             save:bool, 
                             save_path:str,
                             display_interactive_mode:bool=False):
    """_summary_

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        _description_
    stim : str
        _description_
    n_feats : list
        _description_
    save : bool
        _description_
    save_path : str
        _description_
    display_interactive_mode : bool, optional
        _description_, by default False
    fontsize : int, optional
        _description_, by default 19
    """
    # Relevant parameters
    stimuli = stim.split('_')
    n_subjects, n_chan, _, n_delays = average_weights_subjects.shape

    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing to get corresponding features of given feat
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat
        weights_across_features = average_weights_subjects[:,:,index_slice[0]:index_slice[1],:].mean(axis=2)

        # To store correlation matrix of each channel
        correlation_matrices_of_each_channel = np.zeros(shape=(n_chan, n_subjects, n_subjects)) 
        for channel in range(n_chan):
            matrix = weights_across_features[:,channel,:] 
            correlation_matrices_of_each_channel[channel] = np.corrcoef(matrix)

        # Get correlation of each subject against mean
        correlation_of_channel_vs_average_chann = np.zeros(shape=(n_chan, n_subjects)) 
        mean_across_subjects = weights_across_features.mean(axis=0) # nchans, ndelays
        for chan in range(n_chan):
            for subject in range(n_subjects):
                delays_per_subject_per_channel = weights_across_features[subject, chan, :]
                correlation_of_channel_vs_average_chann[chan] = np.corrcoef(delays_per_subject_per_channel, mean_across_subjects)[0,1]

        # Take average across all channels
        correlation_matrix = correlation_matrices_of_each_channel.mean(axis=0)
        correlation_of_channel_vs_average = correlation_of_channel_vs_average_chann.mean(axis=0)

        # Change diagonal for values of last row. 
        for i in range(n_subjects):
            correlation_matrix[i, i] = correlation_of_channel_vs_average[i]

        subject_names = np.arange(1, n_subjects+1).tolist()

        # Make mask for lower triangle of correlation matrix (this is a symmetric matrix)
        mask = np.ones_like(correlation_matrix)
        mask[np.tril_indices_from(mask)] = False

        # Take average
        correlation_mean, correlation_std = np.mean(np.abs(correlation_of_channel_vs_average)), np.std(np.abs(correlation_of_channel_vs_average))

        fig, (ax, cax) = plt.subplots(nrows=2, figsize=(16, 16), gridspec_kw={"height_ratios": [1, 0.05]}, layout='tight')
        fig.suptitle(f'Similarity among subject\'s {feat} TRFs - Mean: ({correlation_mean:.2f}'+r'$\pm$'+f'{correlation_std:.2f})', fontsize=19)
        sn.heatmap(correlation_matrix, 
                   mask=mask, 
                   cmap="coolwarm", 
                   fmt='.2f', 
                   ax=ax,
                   annot=True, 
                   center=0, 
                   xticklabels=True, 
                   annot_kws={"size": 12},
                   cbar=False)

        ax.set_yticklabels(['Subjects mean'] + subject_names[1:], rotation=35, fontsize='xx-large')
        ax.set_xticklabels(subject_names[:-1] + ['Subjects mean'], rotation=35, fontsize='xx-large')

        # Make colorbar
        sn.despine(right=True, left=True, bottom=True, top=True)
        cbar = plt.colorbar(ax.get_children()[0], 
                            cax=cax, 
                            orientation="horizontal")
        cbar.set_label('Correlation', fontsize='xx-large')
        cbar.ax.tick_params(labelsize='xx-large')
        # Save data
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path + f'TRF_correlation_matrix_{feat}.png')
            fig.savefig(save_path + f'TRF_correlation_matrix_{feat}.svg')

#TODO CHECK DESCRIPTION E Y LABEL
def plot_t_p_tfce(t:np.ndarray,
                  p:np.ndarray, 
                  trf_subjects_shape:tuple, 
                  band:str, 
                  stim:str, 
                  pval_tresh:float, 
                  save_path:str, 
                  display_interactive_mode:bool=False, 
                  save=True):

    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Create figure and title
    fig = plt.figure(figsize=(16, 6), layout='tight')
    gs = fig.add_gridspec(1, 5)
    axes = [fig.add_subplot(gs[0, :3], projection="3d"), fig.add_subplot(gs[0, 3])]

    # T surface plot
    x, y = np.mgrid[0:trf_subjects_shape[1], 0:trf_subjects_shape[2]]
    surf = axes[0].plot_surface(x,
                                y,
                                np.reshape(t, (trf_subjects_shape[1], trf_subjects_shape[2])),
                                rstride=1,
                                cstride=1,
                                linewidth=0,
                                cmap="viridis")
    axes[0].set(xticks=[], 
                yticks=[], 
                zticks=[], 
                xlim=[0, trf_subjects_shape[1] - 1], 
                ylim=[0, trf_subjects_shape[2] - 1])
    axes[0].view_init(30, 15)

    # Make colorbar
    cbar = plt.colorbar(ax=axes[0],
                        shrink=0.5,
                        orientation="horizontal",
                        label='T-value',
                        fraction=0.05,
                        pad=0.025,
                        mappable=surf)
    cbar.ax.get_xaxis().set_label_coords(0.5, -2)
    
    # Set title
    if not display_interactive_mode:
        axes[0].set(title='TFCE')
        axes[0].title.set_weight("bold")

    if pval_tresh:
        # Mask p-values over threshold
        p[p > pval_tresh] = 1
    
    # Probability plot
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (trf_subjects_shape[1], trf_subjects_shape[2])))
    img = axes[1].imshow(use_p,  # TODO hay que flipar?
                         cmap="inferno", 
                         interpolation="nearest", 
                         aspect='auto')
    axes[1].set(xticks=[], yticks=[])

    # Make colorbar
    cbar = plt.colorbar(ax=axes[1],
                        shrink=1,
                        orientation="horizontal",
                        label=r"$-\log_{10}(p)$",
                        fraction=0.05,
                        pad=0.025,
                        mappable=img)
    cbar.ax.get_xaxis().set_label_coords(0.5, -3)
    
    # Use mel frequencies for centers
    bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
    axes[1].set(yticks=np.arange(0, 16, 2)+1,
                yticklabels=[int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)])

    if display_interactive_mode:
        text = fig.suptitle('TFCE')
        text.set_weight("bold")
        plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
        mne.viz.utils.plt_show()
    
    # Save figures
    if save:
        save_path += 'TFCE/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'{band}_{stim}_{pval_tresh}.png')
        plt.savefig(save_path + f'{band}_{stim}_{pval_tresh}.svg')

#TODO CHECK DESCRIPTIONdef plot_p_tfce(p:np.ndarray,
                times:np.ndarray, 
                trf_subjects_shape:tuple, 
                band:str, 
                stim:str, 
                pval_tresh:float, 
                save_path:str, 
                display_interactive_mode:bool=False, 
                save:bool=True,
                fontsize:int=10):

    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()
    
    fig, ax = plt.subplots(layout='tight')

    # Mask p-values over threshold
    p[p > pval_tresh] = 1
    
    # Probability plot
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (trf_subjects_shape[1], trf_subjects_shape[2])))
    img = ax.pcolormesh(times*1000, 
                        np.arange(trf_subjects_shape[1]), 
                        use_p,#np.flip(use_p, axis=0), # TODO hay que flipar?
                        cmap="inferno", 
                        shading='auto', 
                        vmin=use_p.min(), 
                        vmax=use_p.max())

    # Configure plot
    bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
    ax.set(yticks=np.arange(0, 16, 2),
           yticklabels=[int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)])
    
    # Make colorbar
    cbar = plt.colorbar(ax=ax, 
                        shrink=1, 
                        orientation="vertical", 
                        label=r"$-\log_{10}(p)$",
                        fraction=0.05, 
                        pad=0.025, 
                        mappable=img)
    cbar.ax.get_xaxis().set_label_coords(0.5, -3)

    if display_interactive_mode:
        text = fig.suptitle('')
        text.set_weight("bold")
        plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
        mne.viz.utils.plt_show()

    # Save figures
    if save:
        save_path += 'TFCE/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'pval_{band}_{stim}_{pval_tresh}.png')
        plt.savefig(save_path + f'pval_{band}_{stim}_{pval_tresh}.svg')

#TODO CHECK DESCRIPTION E Y LABEL
def plot_trf_tfce(average_weights_subjects:np.ndarray, 
                  p:np.ndarray, 
                  times:np.ndarray,
                  trf_subjects_shape:tuple,
                  save_path:str, 
                  band:str, 
                  stim:str, 
                  n_permutations:int,  
                  pval_trhesh:float, 
                  display_interactive_mode:bool=False, 
                  save:bool=True):
    
    # Turn on/off interactive mode
    plt.close()
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Define relevant parameters
    spectrogram_weights_bands = average_weights_subjects.mean(axis=0).mean(axis=0)
    
    # Create figure and title
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 6), layout='tight')
    fig.suptitle(f'{stim} - {band}')

    # Make colormesh
    im = axs[0].pcolormesh(times * 1000, 
                           np.arange(16), 
                           spectrogram_weights_bands, 
                           cmap='jet',
                           vmin=-spectrogram_weights_bands.max(), 
                           vmax=spectrogram_weights_bands.max(),
                           shading='auto')
        
    # Configure axis
    bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
    axs[0].set(xlabel='Time (ms)', 
               ylabel='Frequency (Hz)',
               xticks=np.arange(-100, 700, 100),
               yticks=np.arange(0, 16, 2),
               yticklabels=[int(bands_center[i]) for i in np.arange(0, 16, 2)])

    # And colorbar
    cbar = fig.colorbar(im, 
                        ax=axs[0], 
                        orientation='horizontal', 
                        label='mTRF amplitude (a.u.)',
                        shrink=0.7)

    # Mask p-values over threshold
    p[p>pval_trhesh] = 1

    # Plot probabilities using a colormesh
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (trf_subjects_shape[1], trf_subjects_shape[2])))
    img = axs[1].pcolormesh(times * 1000, 
                            np.arange(trf_subjects_shape[1]), 
                            use_p, #np.flip(use_p, axis=0), 
                            cmap="inferno", 
                            shading='auto',
                            vmin=use_p.min(), 
                            vmax=use_p.max())

    # Configure axis
    axs[1].set(xlabel='Time (ms)')

    # Plot color bar    
    cbar = fig.colorbar(ax=axs[1], 
                        orientation="horizontal", 
                        label=r"$-\log_{10}(p)$",
                        mappable=img, 
                        shrink=0.7)
    
    if display_interactive_mode:
        text = fig.suptitle('')
        text.set_weight("bold")
        plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
        mne.viz.utils.plt_show()

    # Save figures
    if save:
        save_path += 'TFCE/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'trf_tfce_{pval_trhesh}_{n_permutations}.png')
        plt.savefig(save_path + f'trf_tfce_{pval_trhesh}_{n_permutations}.svg')



##############################################################

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def corr_sujeto_decoding(sesion, sujeto, Valores_promedio, display_interactive_mode, name, Save, Run_graficos_path):
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    data = pd.DataFrame({name: Valores_promedio})
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    sn.violinplot(data=data, ax=ax)
    plt.ylim([-0.2, 1])
    plt.ylabel(name)
    plt.title('{}:{:.3f} +/- {:.3f}'.format(name, np.mean(Valores_promedio), np.std(Valores_promedio), fontsize=19))

    if Save:
        save_path_cabezas = Run_graficos_path + 'Corr_sujetos/'
        try:
            os.makedirs(save_path_cabezas)
        except:
            pass
        fig.savefig(save_path_cabezas + '{}_Sesion{}_Sujeto{}.png'.format(name, sesion, sujeto))


def Plot_PSD(sesion, sujeto, Band, situacion, display_interactive_mode, Save, save_path, info, data, fmin=0, fmax=40):
    psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(data, info['sfreq'], fmin, fmax)

    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    fig.suptitle('Sesion {} - Sujeto {} - Situacion {} - Band {}'.format(sesion, sujeto, situacion, Band))

    evoked = mne.EvokedArray(psds_welch_mean, info)
    evoked.times = freqs_mean
    evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s',
                show=False, spatial_colors=True, unit=False, units='w', axes=ax)
    ax.set_xlabel('Frequency [Hz]')
    ax.grid()

    if Save:
        save_path_graficos = 'gráficos/PSD/Zoom/{}/{}/'.format(save_path, Band)
        os.makedirs(save_path_graficos, exist_ok=True)
        plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.png'.format(sesion, sujeto, Band))
        plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.svg'.format(sesion, sujeto, Band))


def violin_plot_decoding(Correlaciones_totales_sujetos, display_interactive_mode, Save, Run_graficos_path, title):

    data = pd.DataFrame({title: Correlaciones_totales_sujetos.ravel()})
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    sn.violinplot(data=data, ax=ax)
    plt.ylim([-0.2, 1])
    plt.ylabel(title)
    plt.title('{}:{:.3f} +/- {:.3f}'.format(title, np.mean(Correlaciones_totales_sujetos),
                                                     np.std(Correlaciones_totales_sujetos), fontsize=19))

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + '{}_promedio.svg'.format(title))
        fig.savefig(save_path_graficos + '{}_promedio.png'.format(title))

    return Correlaciones_totales_sujetos.mean(), Correlaciones_totales_sujetos.std()


def Cabezas_3d(Correlaciones_totales_sujetos, info, display_interactive_mode, Save, Run_graficos_path, title):
    Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)

    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    sample_data_folder = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(sample_data_folder, 'subjects')
    sample_data_trans_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                          'sample_audvis_raw-trans.fif')

    evoked = mne.EvokedArray(np.array([Correlaciones_promedio,]).transpose(), info)
    field_map = mne.make_field_map(evoked, trans=sample_data_trans_file,
                                   subject='sample', subjects_dir=subjects_dir, ch_type='eeg',
                                   meg_surf='head')

    fig = evoked.plot_field(field_map, time=0)
    xy, im = mne.viz.snapshot_brain_montage(fig, info)
    # mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title('Correlation', size='large')
    ax.imshow(im)

    if Save:
        try:
            os.makedirs(Run_graficos_path)
        except:
            pass
        fig.savefig(Run_graficos_path + '{}.svg'.format(title))
        fig.savefig(Run_graficos_path + '{}.png'.format(title))

    return Correlaciones_promedio.mean(), Correlaciones_promedio.std()


def PSD_boxplot(psd_pred_correlations, psd_rand_correlations, display_interactive_mode, Save, Run_graficos_path):
    psd_rand_correlations = Funciones.flatten_list(psd_rand_correlations)

    data = pd.DataFrame({'Prediction': psd_pred_correlations, 'Random': psd_rand_correlations})
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    sn.violinplot(data=data, ax=ax)
    plt.ylim([-0.2, 1])
    plt.ylabel('Correlation')
    plt.title('Prediction Correlation:{:.2f} +/- {:.2f}\n'
              'Random Correlation:{:.2f} +/- {:.2f}'.format(np.mean(psd_pred_correlations), np.std(psd_pred_correlations),
                                                            np.mean(psd_rand_correlations), np.std(psd_rand_correlations)))
    add_stat_annotation(ax, data=data, box_pairs=[(('Prediction'), ('Random'))],
                        test='t-test_ind', text_format='full', loc='inside', verbose=2)

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'PSD Boxplot.png')
        fig.savefig(save_path_graficos + 'PSD Boxplot.svg')


def weights_ERP(Pesos_totales_sujetos_todos_canales, info, times, display_interactive_mode,
                Save, Run_graficos_path, Len_Estimulos, stim, decorrelation_times=None):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0).transpose()

    # Ploteo pesos y cabezas
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    Stims_Order = stim.split('_')

    Cant_Estimulos = len(Len_Estimulos)
    for j in range(Cant_Estimulos):
        Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)].mean(0)

        evoked = mne.EvokedArray(
            np.flip(Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)], axis=1), info)
        evoked.shift_time(-times[0], relative=True)

        fig, ax = plt.subplots(figsize=(15, 5))
        fig.suptitle('{}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim), fontsize=23)
        evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
                    show=False, spatial_colors=True, unit=True, units='W', axes=ax)

        ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
        if times[0] < 0:
            # ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-stimulus')
            ax.axvline(x=0, ymin=0, ymax=1, color='grey')
        if decorrelation_times:
            # ax.vlines(-np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
            #           color='red', label='Decorrelation time')
            ax.axvspan(-np.mean(decorrelation_times), 0, alpha=0.4, color='red', label=' Mean decorrelation time')
            # ax.axvspan(-np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
            #            -np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
            #            alpha=0.4, color='red', label='Decorrelation time std.')

        ax.xaxis.label.set_size(23)
        ax.yaxis.label.set_size(23)
        ax.tick_params(axis='both', labelsize=23)
        ax.grid()
        ax.legend(fontsize=15, loc='lower right')

        fig.tight_layout()

        if Save:
            os.makedirs(Run_graficos_path, exist_ok=True)
            fig.savefig(
                Run_graficos_path + 'Regression_Weights_{}.svg'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))
            fig.savefig(
                Run_graficos_path + 'Regression_Weights_{}.png'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))


def decoding_t_lags(Correlaciones_totales_sujetos, times, Band, display_interactive_mode, Save, Run_graficos_path):
    Corr_time_sub = Correlaciones_totales_sujetos.mean(0)
    mean_time_corr = np.flip(Corr_time_sub.mean(1))
    std_time_corr = np.flip(Corr_time_sub.std(1))

    plot_times = -np.flip(times)

    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # get max correlation t_lag
    max_t_lag = np.argmax(mean_time_corr)

    fig, ax = plt.subplots()
    plt.plot(plot_times, mean_time_corr)
    plt.title('{}'.format(Band))
    plt.fill_between(plot_times, mean_time_corr - std_time_corr/2, mean_time_corr + std_time_corr/2, alpha=.5)
    plt.vlines(plot_times[max_t_lag], ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed', color='k',
               label='Max. correlation delay: {:.2f}s'.format(plot_times[max_t_lag]))
    plt.xlabel('Time lag [s]')
    plt.ylabel('Correlation')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(axis='both', labelsize=15)
    plt.grid()
    plt.legend()

    if Save:
        os.makedirs(Run_graficos_path, exist_ok=True)
        fig.savefig(Run_graficos_path + 'Correlation_time_lags_{}.svg'.format(Band))
        fig.savefig(Run_graficos_path + 'Correlation_time_lags_{}.png'.format(Band))


def Brain_sync(data, Band, info, display_interactive_mode, Save, graficos_save_path, total_subjects=18, sesion=None, sujeto=None):

    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    if data.shape == (total_subjects, info['nchan'], info['nchan']):
        data_ch = data.mean(0)
    elif data.shape == (info['nchan'], info['nchan']):
        data_ch = data

    plt.figure(figsize=(10, 8))
    plt.title('Inter Brain Phase Synchornization - {}'.format(Band), fontsize=14)
    plt.imshow(data_ch)
    plt.xticks(np.arange(0, info['nchan'], 4), labels=info['ch_names'][0:-1:4], rotation=45)
    plt.yticks(np.arange(0, info['nchan'], 4), labels=info['ch_names'][0:-1:4])
    plt.ylabel('Speaker', fontsize=13)
    plt.xlabel('Listener', fontsize=13)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if data.shape == (total_subjects, info['nchan'], info['nchan']):
            plt.savefig(graficos_save_path + 'Inter Brain sync - {}.png'.format(Band))
            plt.savefig(graficos_save_path + 'Inter Brain sync - {}.svg'.format(Band))
        elif data.shape == (info['nchan'], info['nchan']):
            plt.savefig(graficos_save_path + 'Inter Brain sync - Sesion{}_Sujeto{}.png'.format(sesion, sujeto))
            plt.savefig(graficos_save_path + 'Inter Brain sync - Sesion{}_Sujeto{}.svg'.format(sesion, sujeto))



def ch_heatmap_topo(total_data, info, delays, times, display_interactive_mode, Save, graficos_save_path, title, total_subjects=18,
                    sesion=None, sujeto=None, fontsize=14):

    if total_data.shape == (info['nchan'], len(delays)):
        phase_sync_ch = total_data
    elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
        phase_sync_ch = total_data.mean(0)

    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(figsize=(9, 5), nrows=2, ncols=2, gridspec_kw={'width_ratios': [2, 1]})

    # Remove axes of column 2
    for ax_col in axs[:, 1]:
        ax_col.remove()

    # Add one axis in column
    ax = fig.add_subplot(1, 3, (3, 3))

    # Plot topo
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)
    max_pahse_sync = phase_sync_ch[:, max_t_lag]

    # ax.set_title('Mean = {:.3f} +/- {:.3f}'.format(max_pahse_sync.mean(), max_pahse_sync.std()))
    im = mne.viz.plot_topomap(max_pahse_sync, info, cmap='Reds',
                              vmin=max_pahse_sync.min(),
                              vmax=max_pahse_sync.max(),
                              show=False, sphere=0.07, axes=ax)
    cb = plt.colorbar(im[0], shrink=1, orientation='horizontal')
    cb.set_label('r')


    # Invert times for PLV plot
    phase_sync_ch = np.flip(phase_sync_ch)
    phase_sync_std = phase_sync_ch.std(0)
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)

    times_plot = np.flip(-times)

    im = axs[0, 0].pcolormesh(times_plot * 1000, np.arange(info['nchan']), phase_sync_ch, shading='auto')
    axs[0, 0].set_ylabel('Channels')
    axs[0, 0].set_xticks([])

    cbar = plt.colorbar(im, orientation='vertical', ax=axs[0, 0])
    cbar.set_label('PLV')

    axs[1, 0].plot(times_plot * 1000, phase_sync)
    axs[1, 0].fill_between(times_plot * 1000, phase_sync - phase_sync_std / 2, phase_sync + phase_sync_std / 2, alpha=.5)
    # axs[1, 0].set_ylim([0, 0.2])
    axs[1, 0].vlines(times_plot[max_t_lag] * 1000, axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], linestyle='dashed', color='k',
                label='Max: {}ms'.format(int(times_plot[max_t_lag] * 1000)))
    axs[1, 0].set_xlabel('Time lag [ms]')
    axs[1, 0].set_ylabel('Mean {}'.format(title))
    # axs2.tick_params(axis='both', labelsize=12)
    axs[1, 0].set_xlim([times_plot[0] * 1000, times_plot[-1] * 1000])
    axs[1, 0].grid()
    axs[1, 0].legend()

    fig.tight_layout()

    # Change axis 0 to match axis 1 width after adding colorbar
    ax0_box = axs[0, 0].get_position().bounds
    ax1_box = axs[1, 0].get_position().bounds
    ax1_new_box = (ax1_box[0], ax1_box[1], ax0_box[2], ax1_box[3])
    axs[1, 0].set_position(ax1_new_box)

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if total_data.shape == (info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.png'.format(title, sesion, sujeto))
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.svg'.format(title, sesion, sujeto))
        elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}.png'.format(title))
            plt.savefig(graficos_save_path + 't_lags_{}.svg'.format(title))







# ## VIEJAS NO SE USAN

# def Plot_instantes_interes(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, display_interactive_mode_figure_instantes,
#                            Save_figure_instantes, Run_graficos_path, Cant_Estimulos, Stims_Order, stim,
#                            Autocorrelation_value=0.1):
#     # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
#     Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
#     Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0).transpose()

#     # Ploteo pesos y cabezas
#     if Display_figure_instantes:
#         plt.ion()
#     else:
#         plt.ioff()

#     returns = []
#     for j in range(Cant_Estimulos):
#         curva_pesos_totales = Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)].mean(0)
#         returns.append(curva_pesos_totales)

#         if Autocorrelation_value and times[-1] > 0:
#             weights_autocorr = Funciones.correlacion(curva_pesos_totales, curva_pesos_totales)

#             for i in range(len(weights_autocorr)):
#                 if weights_autocorr[i] < Autocorrelation_value: break

#                 dif_paso = weights_autocorr[i - 1] - weights_autocorr[i]
#                 dif_01 = weights_autocorr[i - 1] - Autocorrelation_value
#                 dif_time = dif_01 / sr / dif_paso
#                 decorr_time = ((i - 1) / sr + dif_time) * 1000

#             fig, ax = plt.subplots()
#             plt.plot(np.arange(len(weights_autocorr)) * 1000 / sr, weights_autocorr)
#             plt.title('Decorrelation time: {:.2f} ms'.format(decorr_time))
#             plt.hlines(Autocorrelation_value, ax.get_xlim()[0], decorr_time, linestyle='dashed', color='black')
#             plt.vlines(decorr_time, ax.get_ylim()[0], Autocorrelation_value, linestyle='dashed', color='black')
#             plt.grid()
#             plt.ylabel('Autocorrelation')
#             plt.xlabel('Time [ms]')
#             if Save_figure_instantes:
#                 save_path_graficos = Run_graficos_path
#                 try:
#                     os.makedirs(save_path_graficos)
#                 except:
#                     pass
#                 fig.savefig(save_path_graficos + 'Weights Autocorrelation.png')

#         evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)], info)
#         evoked.shift_time(times[0], relative=True)

#         instantes_index = sgn.find_peaks(np.abs(evoked._data.mean(0)), height=np.abs(evoked._data.mean(0)).max() * 0.4)[
#             0]
#         if not len(instantes_index): instantes_index = [np.abs(evoked._data.mean(0)).argmax()]
#         instantes_de_interes = [i / sr + times[0] for i in instantes_index]  # if i/sr + times[0] < 0]

#         fig = evoked.plot_joint(times=instantes_de_interes, show=False,
#                                 ts_args=dict(unit='False', units=dict(eeg='$w$', grad='fT/cm', mag='fT'),
#                                              scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms'),
#                                 topomap_args=dict(vmin=evoked._data.min(),
#                                                   vmax=evoked._data.max(),
#                                                   time_unit='ms'))

#         fig.suptitle('{}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))
#         fig.set_size_inches(12, 7)
#         axs = fig.axes
#         axs[0].plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
#         axs[0].axvspan(0, axs[0].get_xlim()[1], alpha=0.4, color='grey', label='Unheard stimuli')
#         if Autocorrelation_value and times[-1] > 0: axs[0].vlines(decorr_time, axs[0].get_ylim()[0],
#                                                                   axs[0].get_ylim()[1], linestyle='dashed', color='red',
#                                                                   label='Decorrelation time')
#         axs[0].xaxis.label.set_size(13)
#         axs[0].yaxis.label.set_size(13)
#         axs[0].grid()
#         axs[0].legend(fontsize=13, loc='lower left')

#         Blues = plt.cm.get_cmap('Blues').reversed()
#         cmaps = ['Reds' if evoked._data.mean(0)[i] > 0 else Blues for i in instantes_index]

#         for i in range(len(instantes_de_interes)):
#             axs[i + 1].clear()
#             axs[i + 1].set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)), fontsize=11)
#             im = mne.viz.plot_topomap(evoked._data[:, instantes_index[i]], info, axes=axs[i + 1],
#                                       show=False, sphere=0.07, cmap=cmaps[i],
#                                       vmin=evoked._data[:, instantes_index[i]].min(),
#                                       vmax=evoked._data[:, instantes_index[i]].max())
#             plt.colorbar(im[0], ax=axs[i + 1], orientation='vertical', shrink=0.8,
#                          boundaries=np.linspace(evoked._data[:, instantes_index[i]].min().round(decimals=2),
#                                                 evoked._data[:, instantes_index[i]].max().round(decimals=2), 100),
#                          ticks=np.linspace(evoked._data[:, instantes_index[i]].min(),
#                                             evoked._data[:, instantes_index[i]].max(), 4).round(decimals=2))

#         axs[i + 2].remove()
#         axs[i + 4].remove()
#         fig.tight_layout()

#         if Save_figure_instantes:
#             save_path_graficos = Run_graficos_path
#             try:
#                 os.makedirs(save_path_graficos)
#             except:
#                 pass
#             fig.savefig(
#                 save_path_graficos + 'Instantes_interes_{}.svg'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))

#     return returns


# def Matriz_corr(Pesos_totales_sujetos_promedio, Pesos_totales_sujetos_todos_canales, sujeto_total, Display, Save,
#                 Run_graficos_path):
#     # Armo df para correlacionar
#     Pesos_totales_sujetos_promedio = Pesos_totales_sujetos_promedio[:sujeto_total]
#     Pesos_totales_sujetos_promedio.append(
#         Pesos_totales_sujetos_todos_canales.transpose().mean(0).mean(1))  # agrego pesos promedio de todos los sujetos
#     lista_nombres = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
#                      "Promedio"]
#     Pesos_totales_sujetos_df = pd.DataFrame(Pesos_totales_sujetos_promedio).transpose()
#     Pesos_totales_sujetos_df.columns = lista_nombres[:len(Pesos_totales_sujetos_df.columns) - 1] + [lista_nombres[-1]]

#     pvals_matrix = Pesos_totales_sujetos_df.corr(method=pearsonr_pval)
#     Correlation_matrix = np.array(Pesos_totales_sujetos_df.corr(method='pearson'))
#     for i in range(len(Correlation_matrix)):
#         Correlation_matrix[i, i] = Correlation_matrix[-1, i]

#     Correlation_matrix = pd.DataFrame(Correlation_matrix[:-1, :-1])
#     Correlation_matrix.columns = lista_nombres[:len(Correlation_matrix) - 1] + [lista_nombres[-1]]

#     if Display:
#         plt.ion()
#     else:
#         plt.ioff()

#     mask = np.ones_like(Correlation_matrix)
#     mask[np.tril_indices_from(mask)] = False

#     fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 0.05]})
#     fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize=26)
#     sn.heatmap(abs(Correlation_matrix), mask=mask, cmap="coolwarm", fmt='.3', ax=ax,
#                annot=True, center=0, xticklabels=True, annot_kws={"size": 19},
#                cbar=False)

#     ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(Correlation_matrix)], rotation='horizontal',
#                        fontsize=19)
#     ax.set_xticklabels(lista_nombres[:len(Correlation_matrix) - 1] + ['Mean of subjects'], rotation='horizontal',
#                        ha='left', fontsize=19)

#     sn.despine(right=True, left=True, bottom=True, top=True)
#     fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
#     cax.yaxis.set_tick_params(labelsize=20)

#     fig.tight_layout()

#     if Save:
#         save_path_graficos = Run_graficos_path
#         try:
#             os.makedirs(save_path_graficos)
#         except:
#             pass
#         fig.savefig(save_path_graficos + 'Correlation_matrix.png')


# def Matriz_std_channel_wise(Pesos_totales_sujetos_todos_canales, Display, Save, Run_graficos_path):
#     Pesos_totales_sujetos_todos_canales_average = np.dstack(
#         (Pesos_totales_sujetos_todos_canales, Pesos_totales_sujetos_todos_canales.mean(2)))
#     Correlation_matrices = np.zeros((Pesos_totales_sujetos_todos_canales_average.shape[0],
#                                      Pesos_totales_sujetos_todos_canales_average.shape[2],
#                                      Pesos_totales_sujetos_todos_canales_average.shape[2]))
#     for channel in range(len(Pesos_totales_sujetos_todos_canales_average)):
#         Correlation_matrices[channel] = np.array(
#             pd.DataFrame(Pesos_totales_sujetos_todos_canales_average[channel]).corr(method='pearson'))

#     # std por sujeto
#     std_matrix = Correlation_matrices.std(0)

#     for i in range(len(std_matrix)):
#         std_matrix[i, i] = std_matrix[-1, i]

#     lista_nombres = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Promedio"]
#     std_matrix = pd.DataFrame(std_matrix[:-1, :-1])
#     std_matrix.columns = lista_nombres[:len(std_matrix) - 1] + [lista_nombres[-1]]

#     if Display:
#         plt.ion()
#     else:
#         plt.ioff()

#     mask = np.ones_like(std_matrix)
#     mask[np.tril_indices_from(mask)] = False

#     fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 0.05]})
#     fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize=26)
#     sn.heatmap(abs(std_matrix), mask=mask, cmap="coolwarm", fmt='.3', ax=ax,
#                annot=True, center=0, xticklabels=True, annot_kws={"size": 19},
#                cbar=False)

#     ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(std_matrix)], rotation='horizontal', fontsize=19)
#     ax.set_xticklabels(lista_nombres[:len(std_matrix) - 1] + ['Mean of subjects'], rotation='horizontal', ha='left',
#                        fontsize=19)

#     sn.despine(right=True, left=True, bottom=True, top=True)
#     fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
#     cax.yaxis.set_tick_params(labelsize=20)

#     fig.tight_layout()

#     if Save:
#         save_path_graficos = Run_graficos_path
#         try:
#             os.makedirs(save_path_graficos)
#         except:
#             pass
#         fig.savefig(save_path_graficos + 'Channelwise_std_matrix.png')


# def Cabezas_corr_promedio_scaled(Correlaciones_totales_sujetos, info, Display, Save, Run_graficos_path, title):
#     Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)

#     if Display:
#         plt.ion()
#     else:
#         plt.ioff()

#     fig = plt.figure()
#     plt.suptitle("Mean {} per channel among subjects".format(title), fontsize=19)
#     plt.title('{} = {:.3f} +/- {:.3f}'.format(title, Correlaciones_promedio.mean(), Correlaciones_promedio.std()),
#               fontsize=19)
#     ax = plt.subplot()
#     im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap='Greys', vmin=0, vmax=0.41, show=Display, sphere=0.07, axes=ax)
#     cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
#     cb.ax.tick_params(labelsize=23)
#     fig.tight_layout()

#     if Save:
#         save_path_graficos = Run_graficos_path
#         os.makedirs(save_path_graficos, exist_ok=True)
#         fig.savefig(save_path_graficos + '{}_promedio_scaled.svg'.format(title))
#         fig.savefig(save_path_graficos + '{}_promedio_sacled.png'.format(title))


# def Plot_instantes_casera(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_figure_instantes,
#                           Save_figure_instantes, Run_graficos_path):
#     # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
#     Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
#     Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0)

#     instantes_index = sgn.find_peaks(np.abs(Pesos_totales_sujetos_todos_canales_copy.mean(1)[50:]),
#                                 height=np.abs(Pesos_totales_sujetos_todos_canales_copy.mean(1)).max() * 0.3)[0] + 50

#     instantes_de_interes = [i/ sr + times[0] for i in instantes_index if i / sr + times[0] <= 0]

#     # Ploteo pesos y cabezas
#     if Display_figure_instantes:
#         plt.ion()
#     else:
#         plt.ioff()

#     Blues = plt.cm.get_cmap('Blues').reversed()
#     cmaps = ['Reds' if Pesos_totales_sujetos_todos_canales_copy.mean(1)[i] > 0 else Blues for i in instantes_index if
#              i / sr + times[0] <= 0]

#     fig, axs = plt.subplots(figsize=(10, 5), ncols=len(cmaps))
#     fig.suptitle('Mean of $w$ among subjects - {} Band'.format(Band))
#     for i in range(len(instantes_de_interes)):
#         ax = axs[0, i]
#         ax.set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)))
#         fig.tight_layout()
#         im = mne.viz.plot_topomap(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].ravel(), info, axes=ax,
#                                   show=False,
#                                   sphere=0.07, cmap=cmaps[i],
#                                   vmin=Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(),
#                                   vmax=Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max())
#         plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.9,
#                      boundaries=np.linspace(
#                          Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min().round(decimals=2),
#                          Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max().round(decimals=2), 100),
#                      ticks=np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(),
#                                         Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max(), 4).round(
#                          decimals=2))

#     axs[0, -1].remove()
#     for ax_row in axs[1:]:
#         for ax in ax_row:
#             ax.remove()

#     ax = fig.add_subplot(3, 1, (2, 3))
#     evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy.transpose(), info)
#     evoked.shift_time(times[0], relative=True)


#     evoked.plot(show=False, spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1),
#                 unit=True, units=dict(eeg='$w$'), axes=ax, zorder='unsorted', selectable=False,
#                 time_unit='ms')
#     ax.plot(times * 1000, Pesos_totales_sujetos_todos_canales_copy.mean(1),
#             'k--', label='Mean', zorder=130, linewidth=2)

#     ax.axvspan(0, ax.get_xlim()[1], alpha=0.5, color='grey')
#     ax.set_title("")
#     ax.xaxis.label.set_size(13)
#     ax.yaxis.label.set_size(13)
#     ax.grid()
#     ax.legend(fontsize=13, loc='upper right')

#     fig.tight_layout()

#     if Save_figure_instantes:
#         save_path_graficos = Run_graficos_path
#         try:
#             os.makedirs(save_path_graficos)
#         except:
#             pass
#         fig.savefig(save_path_graficos + 'Instantes_interes.png')

#     return Pesos_totales_sujetos_todos_canales_copy.mean(1)




# def plot_alphas(alphas, correlaciones, best_alpha_overall, lista_Rmse, linea, fino):
#     # Plot correlations vs. alpha regularization value
#     # cada linea es un canal
#     fig = plt.figure(figsize=(10, 5))
#     fig.clf()
#     plt.subplot(1, 3, 1)
#     plt.subplots_adjust(wspace=1)
#     plt.plot(alphas, correlaciones, 'k')
#     plt.gca().set_xscale('log')
#     # en rojo: el maximo de las correlaciones
#     # la linea azul marca el mejor alfa

#     plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
#     plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

#     plt.plot(alphas, correlaciones.mean(1), '.r', linewidth=5)
#     plt.xlabel('Alfa', fontsize=16)
#     plt.ylabel('Correlación - Ridge set', fontsize=16)
#     plt.tick_params(axis='both', which='major', labelsize=13)
#     plt.tick_params(axis='both', which='minor', labelsize=13)

#     # Como se ve sola la correlacion maxima para los distintos alfas
#     plt.subplot(1, 3, 2)
#     plt.plot(alphas, np.array(correlaciones).mean(1), '.r', linewidth=5)
#     plt.plot(alphas, np.array(correlaciones).mean(1), '-r', linewidth=linea)

#     if fino:
#         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
#         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

#     plt.xlabel('Alfa', fontsize=16)
#     plt.gca().set_xscale('log')
#     plt.tick_params(axis='both', which='major', labelsize=13)
#     plt.tick_params(axis='both', which='minor', labelsize=13)
#     # el RMSE
#     plt.subplot(1, 3, 3)
#     plt.plot(alphas, np.array(lista_Rmse).min(1), '.r', linewidth=5)
#     plt.plot(alphas, np.array(lista_Rmse).min(1), '-r', linewidth=2)

#     if fino:
#         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
#         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

#     plt.xlabel('Alfa', fontsize=16)
#     plt.ylabel('RMSE - Ridge set', fontsize=16)
#     plt.gca().set_xscale('log')
#     plt.tick_params(axis='both', which='major', labelsize=13)
#     plt.tick_params(axis='both', which='minor', labelsize=13)

#     titulo = "El mejor alfa es de: " + str(best_alpha_overall)
#     plt.suptitle(titulo, fontsize=18)