# Standard libraries
import numpy as np, pandas as pd, os, seaborn as sns, mne, scipy.signal as sgn, warnings

# Fix matplotlib Qt backend issue  
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

current_working_directory = os.getcwd()
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")
warnings.filterwarnings("ignore", message="More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.")
warnings.filterwarnings("ignore", message="Tight layout not applied. tight_layout cannot make axes width small enough to accommodate all axes decorations")

# Specific libraries
from scipy.stats import wilcoxon#, pearsonr
from scipy.stats import gaussian_kde, mode
import librosa

# Default size is 10 pts, the scalings (10pts*scale) are:
#'xx-small':0.579,'x-small':0.694,'small':0.833,'medium':1.0,'large':1.200,'x-large':1.440,'xx-large':1.728,None:1.0}
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.pylab as pylab
import matplotlib.cm as cm
params = {
        'legend.fontsize': 'x-large',
        'legend.title_fontsize': 'x-large',
        'figure.figsize': (8, 6),
        'figure.titlesize': 'xx-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large'
        }
pylab.rcParams.update(params)
matplotlib_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Modules
from utils.processing import clustering_by_correlation
import utils.general_functions as general_functions, config
# plt.style.use([plt.style.available[23]])

# ===================
# Auxiliary functions
def define_ticks(
    axes, 
    number_of_ticks:int, 
    ylabel:str, 
    xlabel:str='Time (ms)', 
    order:list=None, 
    zeros_index:list=None, 
    title:str=None
    )->None:
    """
    Define ticks for a given axis
    
    Parameters
    ----------
    axes : matplotlib.axes
        Axis to define ticks
    number_of_ticks : int
        Number of ticks to define
    ylabel : str
        Label of the y-axis
    xlabel : str, optional
        Label of the x-axis, by default 'Time (ms)'
    order : list, optional
        Order of the ticks, by default None
    zeros_index : list, optional
        Index of zeros to avoid, by default None
    title : str, optional
        Title of the plot, by default
    
    Returns
    -------
    None
    """
    
    if ylabel.startswith('phonemes-dili'):
        axes.tick_params(axis='both', labelsize='medium')
        tags = [np.str_('aɪ'), np.str_('aʊ'), np.str_('b'), np.str_('d'), np.str_('eɪ'), np.str_('f'), np.str_('g'), np.str_('h'), np.str_('i'), np.str_('j'), np.str_('k'), np.str_('l'), np.str_('m'), np.str_('n'), np.str_('oʊ'), np.str_('p'), np.str_('s'), np.str_('t'), np.str_('tʃ'), np.str_('u'), np.str_('v'), np.str_('w'), np.str_('z'), np.str_('æ'), np.str_('ð'), np.str_('ŋ'), np.str_('ɑː'), np.str_('ɔɪ'), np.str_('ɔː'), np.str_('ɛ'), np.str_('ɜːr'), np.str_('ɪ'), np.str_('ɹ'), np.str_('ʃ'), np.str_('ʊ'), np.str_('ʌ'), np.str_('θ')]
        ticks = np.arange(number_of_ticks)
        
    if ylabel=='Phonological':
        axes.tick_params(axis='both', labelsize='medium') 
        tags = [t for t in list(config.exp_info.phonological_labels.copy()) if t not in ['pause', 'trill']]
        ticks = np.arange(number_of_ticks)
    if ylabel=='Phonological1':
        axes.tick_params(axis='both', labelsize='medium') 
        tags = [t for t in list(config.exp_info.phonological_labels.copy()) if t not in ['pause', 'trill'] + config.exp_info.phonological_labels2.copy()]
        ticks = np.arange(number_of_ticks)
    if ylabel=='Phonological2':
        axes.tick_params(axis='both', labelsize='medium') 
        tags = [t for t in list(config.exp_info.phonological_labels.copy()) if t not in ['pause', 'trill'] + config.exp_info.phonological_labels1.copy()]
        ticks = np.arange(number_of_ticks)
    elif ylabel.startswith('Mistakes'):
        tags = list(config.exp_info.mistakes.copy())
        ticks = np.arange(number_of_ticks)
    elif ylabel.startswith('Control'):
        tags = list(config.exp_info.control.copy())
        ticks = np.arange(number_of_ticks)
    elif ylabel.startswith('Wav2vec2'):
        ticks = np.arange(number_of_ticks)
        tags = [f'C{tick}' for tick in ticks]
    elif ylabel.startswith('Phonemes'):
        tags = config.exp_info.phonemes.copy()
        tags.remove('/sil/')
        axes.tick_params(axis='both', labelsize='medium')
        ticks = np.arange(number_of_ticks)
    elif ylabel.startswith('Phones'):
        axes.tick_params(axis='both', labelsize='medium')
        tags = config.exp_info.phones.copy()
        tags.remove('sil')
        tags.remove('<p:>')
        ticks = np.arange(number_of_ticks)
    elif ylabel == 'Envelope2':
        axes.tick_params(axis='both', labelsize='medium')
        tags = ['Envelope', 'Instantaneous Freq.']
        ticks = np.arange(number_of_ticks)
    # elif ylabel.startswith('DNNs'):
    elif 'DNNs' in ylabel:
        axes.tick_params(axis='both', labelsize='medium')
        tags = [i for i in np.arange(0, number_of_ticks)]
        ticks = np.arange(number_of_ticks)

    # Frecuency correlated features are treated differently
    if ylabel.startswith('Spectrogram'):
        ylabel = 'Frecuency (Hz)'
        bands_center = librosa.mel_frequencies(n_mels=number_of_ticks+2, fmin=0, fmax=16000/2)[1:-1]
        tags = [int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)]
        ticks = np.arange(0, number_of_ticks, 2)
    elif ylabel.startswith('Mfccs') or ylabel.startswith('Deltas'):
        if ylabel=='Mfccs':
            tags = [r'$m_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks)]
        elif ylabel=='Mfccs-Deltas':
            tags = [r'$m_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/2)] 
            tags += [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/2)]
        elif ylabel=='Mfccs-Deltas-Deltas':
            tags = [r'$m_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/3)] 
            tags += [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/3)]
            tags += [r'$\delta-\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/3)]
        elif ylabel=='Deltas':
            tags = [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks)]
        elif ylabel=='Deltas-Deltas':
            tags = [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/2)]
            tags += [r'$\delta-\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, number_of_ticks/2)]
        ticks = np.arange(0, number_of_ticks, 2)
        ylabel= f"{ylabel}'s Index"
        tags = tags[::2]
    else:
        # Filter zeros and reorder tags
        tags = tags if zeros_index is None else [tags[i] for i in range(len(tags)) if i not in zeros_index]
        tags = tags if order is None else [tags[i] for i in order]
    
    if title is None:
        axes.set(xlabel=xlabel, ylabel=ylabel, yticks=ticks, yticklabels=tags)
    else:
        axes.set(xlabel=xlabel, ylabel=ylabel, yticks=ticks, yticklabels=tags, title=title)

def save_figure(
    cwd:str,
    save_path:str, 
    file_name:str, 
    fig
    )->None:
    """
    Save the figure to the specified path with the given file name.

    Parameters
    ----------
    cwd : str
        Current working directory.
    save_path : str
        Path to save the figure.
    file_name : str
        Name of the file to save the figure as.
    fig : matplotlib.figure.Figure
        Figure to save.

    Returns
    -------
    None
    """
    temp_path = os.path.normpath(save_path)
    os.makedirs(temp_path, exist_ok=True)
    os.chdir(temp_path)
    fig.savefig(file_name + config.figure_format)
    os.chdir(cwd)        

# ===============
# FIGURES OF MAIN

def phonemes_ocurrences(
    ocurrences:dict, 
    save_path:str,
    no_figures:bool=False, 
    save:bool=False
    )->None:
    """
    Makes boxplot with phoeneme ocurrences

    Parameters
    ----------
    ocurrences : dict
        Dictionary with ocurrences of phonemes
    save_path : str
        Path to save the figures
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    save : bool, optional
        If True, figures are saved, by default False

    Returns
    -------
    None
    """
    # Exit function
    if no_figures:
        return
    
    # Get number of phonemes stimuli 
    sessions = list(ocurrences.keys())
    stimuli = list(ocurrences[sessions[0]].keys())

    # Make plot for each stimulus
    for stimulus in stimuli:
        
        # The list of phonemes labels
        if stimulus.endswith('Phonet'):
            phn = ocurrences[sessions[0]][stimulus]['phonemes'][:-1]
        else:
            phn = ocurrences[sessions[0]][stimulus]['phonemes']

        # Make a dict with the relevant data
        relevant_data = {}
        relevant_data['Ocurrences'] = np.concatenate([ocurrences[session][stimulus]['count'].astype(int).reshape(-1,1) for session in sessions], axis = 1).flatten()
        relevant_data['Labels'] = np.repeat(phn, (np.ones(shape=len(phn))*len(sessions)).astype(int))
        relevant_data['Session'] = sessions * len(phn)
        
        # Now arange data in pd.DataFrame
        data = pd.DataFrame(data=relevant_data, columns = ['Ocurrences', 'Session', 'Labels'])
        
        # Make plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), tight_layout=True)
        
        # Usual boxplot
        sns.boxplot(x='Labels', y='Ocurrences', data=data, ax=ax)
        
        # Add jitter with the swarmplot function
        sns.swarmplot(
            x='Labels', 
            y='Ocurrences', 
            data=data, 
            color="grey", 
            ax=ax
            )
        ax.set(title=f'{stimulus}' )
        ax.grid(visible=True, alpha=.3)

        if save:
            save_figure(
                    cwd=current_working_directory,
                    save_path=save_path, 
                    file_name=f'{stimulus}_ocurrences', 
                    fig=fig
                    )
            
def null_correlation_vs_correlation_good_channels(
    good_channels_indexes:np.ndarray,
    save_path:str,
    correlation_per_channel:np.ndarray, 
    null_correlation_per_channel:np.ndarray,
    # power_correlation:float,
    # power_rmse:float,
    save:bool=False, 
    display_interactive_mode:bool=False, 
    session:int=21, 
    subject:int=1,
    no_figures:bool=False
    )->None:
    """
    Make a plot with correlation vs null correlation for good channels

    Parameters
    ----------
    good_channels_indexes : np.ndarray
        Indexes of good channels
    save_path : str
        Path to save the figures
    correlation_per_channel : np.ndarray    
        Correlation per channel
    null_correlation_per_channel : np.ndarray
        Null correlation per channel
    power_correlation : float
        Statistical power of correlation
    power_rmse : float
        Statistical power of rmse
    save : bool, optional
        If True, figures are saved, by default False
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    session : int, optional
        Session number, by default 21
    subject : int, optional 
        Subject number, by default 1
    no_figures : bool, optional
        If True, no figures are displayed, by default False
        
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()
    # Take average across folds
    average_correlation = correlation_per_channel.mean(axis=0)
    channels = np.arange(len(average_correlation))
    
    # Define minimum and maximum of null correlations (mins/max across all iterations, then across all folds, leaving min/max for each channel)
    null_correlation_per_channel_min = null_correlation_per_channel.min(axis=1).min(axis=0)
    null_correlation_per_channel_max = null_correlation_per_channel.max(axis=1).max(axis=0) 

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7), layout='tight')
    # fig.suptitle(f'Session {session} - Subject {subject} - '+r'$Power_{corr} Test$' +f': {power_correlation:.2f}'+r'$Power_{rmse} Test$' +f': {power_rmse:.2f}')
    fig.suptitle(f'Session {session} - Subject {subject}')
    

    # Graph average correlation
    ax.plot(
        average_correlation, 
        '.', 
        color='C0', 
        label="Mean correlation across folds"
        )
    if len(good_channels_indexes): 
        ax.plot(
            good_channels_indexes, 
            average_correlation[good_channels_indexes], 
            '*', 
            color='k', 
            label="Significant mean correlation across folds"
            )

    # Add shadow between min and max
    ax.fill_between(
        x=channels, 
        y1=correlation_per_channel.min(axis=0), # min across all folds
        y2=correlation_per_channel.max(axis=0), 
        alpha=0.5,
        label='Correlation distribution (Real data)'
        )
    ax.fill_between(
        x=channels, 
        y1=null_correlation_per_channel_min,
        y2=null_correlation_per_channel_max, 
        alpha=0.5,
        label='Correlation distribution (Random data)'
        )
    
    # Graph properties
    ax.grid(visible=True)
    ax.set(
        xlim=[-1, 129],
        xlabel='Channels',
        ylabel='Correlation'
        )
    ax.legend(loc="lower right")

    # If there are no good channels
    if not len(good_channels_indexes): 
        plt.text(
            64,
            np.max(abs(correlation_per_channel))/2, 
            "No significant channels", 
            size='xx-large', 
            ha='center'
            )

    # Wether graph is saved
    if save:
        save_figure(
            cwd=current_working_directory,
            save_path=os.path.join(save_path,'correlation_vs_null_correlation'), 
            file_name=f'session{session}_subject{subject}', 
            fig=fig
            )
        
def lateralized_channels(
    info:mne.Info, 
    save_path:str, 
    channels_right:list=['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3'], 
    channels_left:list=['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3'], 
    display_interactive_mode:bool=False, 
    save:bool=True,
    no_figures:bool=False
    )->None:
    """
    Make a topomap showing masked channels for lateralization comparisson

    Parameters
    ----------
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save_path : str
        Path to save the figures
    channels_right : list, optional
        Channels on the right hemisphere, by default ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3']
    channels_left : list, optional
        Channels on the left hemisphere, by default ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3']
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    save : bool, optional
        If True, figures are saved, by default True

    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Get lateralized channels
    lateralized_channels = [i in channels_right + channels_left for i in info['ch_names']]
    
    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.title('Masked channels for lateralization comparisson')
    
    # Make topomap
    mne.viz.plot_topomap(
        data=np.zeros(info['nchan']),
        pos=info, 
        show=display_interactive_mode, 
        sphere=0.07, 
        mask=np.array(lateralized_channels),
        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=12), 
        axes=ax
        )
    
    # Save figure
    if save:
        save_figure(
            cwd=current_working_directory,
            save_path=os.path.join(save_path,'lateralization'), 
            file_name=f'masked_left_vs_right_chs_{len(channels_right)}_channels', 
            fig=fig
            )

def topomap(
    good_channels_indexes:np.ndarray,
    average_coefficient:np.ndarray, 
    info:mne.Info,
    coefficient_name:str, 
    save:bool, 
    save_path:str, 
    display_interactive_mode:bool=False,
    session:int=21, 
    subject:int=1,
    no_figures:bool=False
    )->None:
    """
    Make topographic plot of brain with heat-like map for given coefficient

    Parameters
    ----------
    good_channels_indexes : np.ndarray
        Indexes of good channels
    average_coefficient : np.ndarray
        Average coefficient
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    coefficient_name : str
        Name of the coefficient
    save : bool
        If True, figures are saved
    save_path : str
        Path to save the figures
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    session : int, optional
        Session number, by default 21
    subject : int, optional
        Subject number, by default 1
    no_figures : bool, optional
        If True, no figures are displayed, by default False
        
    Returns
    -------
    None
    """
    # Exit function
    plt.close('all')
    if no_figures:
        return
    
    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Plot head correlation
    if len(good_channels_indexes):
        # Create figure and title
        fig, axs = plt.subplots(nrows=1, ncols=1, layout='tight')
        plt.suptitle(f"Session {session} - Subject {subject}\n{coefficient_name} = ({average_coefficient.mean():.3f}" +r'$\pm$'+ f"{average_coefficient.std():.3f})")
        
        # Mask for good channels
        mask = np.array([i in good_channels_indexes for i in range(info['nchan'])])
        im = mne.viz.plot_topomap(
            data=average_coefficient, 
            pos=info, 
            axes=axs, 
            show=False, 
            sphere=0.07, 
            cmap='Greys', 
            vlim=(average_coefficient.min(), average_coefficient.max()),
            mask=mask,
            mask_params=dict(marker='o', markerfacecolor='red', markeredgecolor='k', linewidth=0, markersize=4, alpha=.35)
        )
        # Make plot
        plt.colorbar(
            im[0], 
            ax=axs,
            shrink=0.85, 
            label=coefficient_name, 
            orientation='horizontal',
            boundaries=np.linspace(average_coefficient.min().round(decimals=3), average_coefficient.max().round(decimals=3), 100),
            ticks=np.linspace(average_coefficient.min(), average_coefficient.max(), 9).round(decimals=3)
            )
        
        # Add legend for masked channels
        legend_elements = [Line2D([0], [0], marker='o', color='w',alpha=.35, markerfacecolor='red', markeredgecolor='k', markersize=10, label='Significant channels')]
        axs.legend(
            handles=legend_elements, 
            loc='upper center', 
            bbox_to_anchor=(1.2, 1),  # Place legend below the head
            ncol=1,  # Adjust number of columns if needed
            frameon=False  # Optional: remove legend frame
            )
    else:
        # Create figure and title
        fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
        plt.suptitle(f"Session{session} Subject{subject}\n{coefficient_name} = ({average_coefficient.mean():.3f}" +r'$\pm$'+ f"{average_coefficient.std():.3f})")
        
        # Make topomap
        im = mne.viz.plot_topomap(
            data=average_coefficient, 
            pos=info, 
            axes=ax, 
            show=False, 
            sphere=0.07, 
            cmap='Greys',
            vlim=(average_coefficient.min(), average_coefficient.max())
            )
            
        plt.colorbar(
            im[0], 
            ax=ax, 
            shrink=0.85, 
            label=coefficient_name, 
            orientation='horizontal',
            # boundaries=np.linspace(average_coefficient.min(), average_coefficient.max(), 100).round(decimals=3),
            ticks=np.linspace(average_coefficient.min(), average_coefficient.max(), 9).round(decimals=3)
            )
    if save:
        save_figure(
            cwd=current_working_directory,
            save_path=os.path.join(save_path,'topomaps'), 
            file_name=f'{coefficient_name.lower()}_topomap_session_{session}_subject_{subject}', 
            fig=fig
            )
        
def average_topomap(
    average_coefficient_subjects:np.ndarray, 
    info:mne.Info,
    stim:str,
    save:bool, 
    save_path:str, 
    coefficient_name:str, 
    number_of_lat_channels:int=12,
    display_interactive_mode:bool=False,
    test_result:bool=False,
    no_figures:bool=False
)->None:
    """
    Make average topomap for a given coefficient

    Parameters
    ----------
    average_coefficient_subjects : np.ndarray
        Average coefficient across subjects
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        If True, figures are saved
    save_path : str
        Path to save the figures
    coefficient_name : str
        Name of the coefficient
    number_of_lat_channels : int, optional
        Number of lateralized channels to show, by default 12
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    test_result : bool, optional
        If True, make Wilcoxon test, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False

    Returns
    -------
    test_results : scipy.stats
        Wilcoxon test results. If test_result is False, None is returned
    """
    
    # Exit function
    plt.close()
    if no_figures:
        return

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean over all subjects
    mean_average_coefficient = average_coefficient_subjects.mean(axis=0)

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.suptitle(f'{stim} {coefficient_name} = ({mean_average_coefficient.mean():.3f}'+r'$\pm$'+f'{mean_average_coefficient.std():.3f})')

    # Make topomap
    im = mne.viz.plot_topomap(
        data=mean_average_coefficient, 
        pos=info, 
        cmap='OrRd',
        vlim=(mean_average_coefficient.min(), mean_average_coefficient.max()),
        show=False, 
        sphere=0.07, 
        axes=ax
        )
    
    vmin = mean_average_coefficient.min()
    vmax = mean_average_coefficient.max()

    # Avoid identical vmin/vmax or NaN
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = vmax = 0  # or set to some default range, e.g. vmin = -1, vmax = 1

    plt.colorbar(
        im[0],
        ax=ax, 
        shrink=0.85,
        label=coefficient_name,
        orientation='horizontal',
        boundaries=np.linspace(vmin, vmax, 100) if vmin != vmax else None,
        ticks=np.linspace(vmin, vmax, 9) if vmin != vmax else [vmin]
    )
    if save:
        save_figure(
            cwd=current_working_directory,
            save_path=save_path, 
            file_name=f'average_{coefficient_name.lower()}_topomap', 
            fig=fig
            )

    # Make Lateralization comparison
    if coefficient_name == 'Correlation':
        # Left and right channels
        all_channels_right = ['B27','B28','B29','B30','B31','B32','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16']
        all_channels_left = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','C24','C25','C26','C27','C28','C29','C30','C31','C32']

        # Get channels right and left that are used in the experiment
        ordered_chs_right = [i for i in info['ch_names'] if i in all_channels_right]
        ordered_chs_left = [i for i in info['ch_names'] if i in all_channels_left]

        # Get filter coefficient to get respective correlations
        corr_right = mean_average_coefficient[[i in all_channels_right for i in info['ch_names']]]
        corr_left = mean_average_coefficient[[i in all_channels_left for i in info['ch_names']]]

        # Now get relevant indexes, sorted by correlation
        sorted_chs_right = [x for _, x in sorted(zip(corr_right, ordered_chs_right))]
        sorted_chs_left = [x for _, x in sorted(zip(corr_left, ordered_chs_left))]

        # Get most correlated channels for lateralization
        if number_of_lat_channels:
            corr_right = np.sort(corr_right)[-number_of_lat_channels:]
            corr_left = np.sort(corr_left)[-number_of_lat_channels:]
            sorted_chs_right = sorted_chs_right[-number_of_lat_channels:]
            sorted_chs_left = sorted_chs_left[-number_of_lat_channels:]

        # Make figure and data to plot
        fig = plt.figure(layout='tight')
        data = pd.DataFrame({'Left': corr_left, 'Right': corr_right})
        
        # Make boxplot and swarmplot
        ax = sns.boxplot(data=data, width=0.35)
        for patch in ax.artists:
            r, g, b, alpha = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .8))
        sns.swarmplot(data=data, color=".25")

        # Figure properties
        ax.set_ylabel('Correlation')

        # Make Wilcoxon test for comparison
        test_results = wilcoxon(data['Left'], data['Right'])
        p_value = test_results.pvalue

        # Agregar texto del p-value directamente
        y_max = max(data['Left'].max(), data['Right'].max())
        y_offset = (y_max - min(data['Left'].min(), data['Right'].min())) * 0.1
        ax.text(0.5, y_max + y_offset, f'p = {p_value:.4f}', 
                ha='center', va='bottom', fontsize='xx-large',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # PLot and save lateralized channels used
        lateralized_channels(
                    info=info, 
                    channels_right=sorted_chs_right, 
                    channels_left=sorted_chs_left, 
                    save_path=save_path,
                    display_interactive_mode=display_interactive_mode,
                    save=save
                    )

        if save:
            save_figure(
                cwd=current_working_directory,
                save_path=os.path.join(save_path, 'lateralization'),
                file_name=f'left_vs_right_{coefficient_name.lower()}_{len(sorted_chs_right)}_channels', 
                fig=fig
                )
    else:
        test_results = None
    if test_result:
        return test_results

def topo_average_pval(
    pvalues_coefficient_subjects:np.ndarray, 
    info:mne.Info, 
    save:bool, 
    save_path:str, 
    coefficient_name:str,
    display_interactive_mode:bool=False,
    no_figures:bool=False
    )->None:
    """
    Make a topographic plot of the mean p-values

    Parameters
    ----------
    pvalues_coefficient_subjects : np.ndarray
        P-values of the coefficient
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        If True, figures are saved
    save_path : str
        Path to save the figures
    coefficient_name : str
        Name of the coefficient
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False
        
    Returns
    -------
    None    
    """
    # Exit function
    plt.close()
    if no_figures:
        return    

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean across all subjects
    topo_pval = pvalues_coefficient_subjects.mean(axis=0)
    
    # Turn 1 not siginificants
    topo_pval[topo_pval>config.significance] = 1
    
    # Adjust to log scale
    topo_pval = -np.log10(topo_pval) 
    
    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.suptitle(f"Mean p-values - {coefficient_name}")
    plt.title(f'Mean: ({topo_pval.mean():.3f}'+r'$\pm$'+f'{topo_pval.std():.3f})')

    # Make topomap
    im = mne.viz.plot_topomap(data=topo_pval, 
        pos=info, 
    #   cmap='OrRd',
        cmap='inferno',
        vlim=(0, topo_pval.max()),
        show=False, 
        sphere=0.07,
        axes=ax
    )
    # And colorbar
    # plt.colorbar(im[0], 
    #              shrink=0.85, 
    #              orientation='vertical',
    #              label='p-value')
    plt.colorbar(
                im[0], 
                ax=ax, 
                shrink=0.85, 
                label='-log10(p-value)', 
                orientation='horizontal',
                boundaries=np.linspace(0, topo_pval.max(), 100),
                ticks=np.linspace(0, topo_pval.max(), 9).round(decimals=3)
                )   

    # Save figure
    if save:
        save_figure(
                cwd=current_working_directory,
                save_path=save_path,
                file_name=f'p-value_topo_{coefficient_name.lower()}', 
                fig=fig
                )

def topo_repeated_channels(
    repeated_good_coefficients_channels_subjects:np.ndarray, 
    info:mne.Info, 
    save:bool, 
    save_path:str, 
    coefficient_name:str, 
    display_interactive_mode:bool=False,
    no_figures:bool=False
    )->None:
    """
    Make a topographic plot of the number of significant channels

    Parameters
    ----------
    repeated_good_coefficients_channels_subjects : np.ndarrays
        Number of significant channels
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        If True, figures are saved
    save_path : str
        Path to save the figures
    coefficient_name : str
        Name of the coefficient
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return    
        
    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean across all subjects 
    
    sum_of_repeated_chan = np.zeros(info['nchan'])# n_channs, the max value of each channel is the total_number_of_subjects
    for sub, channels in enumerate(repeated_good_coefficients_channels_subjects):
        sum_of_repeated_chan += np.isin(np.arange(info["nchan"]), channels)
    # sum_of_repeated_chan = repeated_good_coefficients_channels_subjects.sum(axis=0) 
    
    n_sub = len(config.sessions)*2
    
    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    plt.suptitle(f"Number of significant channels (all {config.n_folds} folds) across subjects - {coefficient_name}")
    plt.title(f'Mean: {sum_of_repeated_chan.mean():.2f}' +r'$\pm$'+ f'{sum_of_repeated_chan.std():.2f}')

    # Make topomap
    im = mne.viz.plot_topomap(
                data=sum_of_repeated_chan, 
                pos=info, 
                cmap='OrRd',
                vlim=(0, n_sub),
                show=False, 
                sphere=0.07, 
                axes=ax
                )
    # And colorbar
    plt.colorbar(
                im[0], 
                shrink=0.85, 
                orientation='vertical', 
                label='Number of subjects passed'
                )
    if save:
        save_figure(
            cwd=current_working_directory,
            save_path=save_path, 
            file_name=f'topo_repeated_channels_{coefficient_name.lower()}', 
            fig=fig
            )

def topo_map_relevant_times(
    average_weights_subjects:np.ndarray, 
    info:mne.Info,
    n_feats:list, 
    band:str, 
    stim:str, 
    times:np.ndarray, 
    sample_rate:int, 
    save_path:int, 
    save:bool=True, 
    display_interactive_mode:bool=False,
    no_figures:bool=False
    )->None:
    """
    Make a topographic plot of the relevant times

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        Average weights across subjects
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    n_feats : list
        Number of features
    band : str
        Band of the EEG
    stim : str
        Stimulus of the EEG
    times : np.ndarray
        Times of the delay window
    sample_rate : int
        Sample rate of the EEG
    save_path : int
        Path to save the figures
    save : bool, optional
        If True, figures are saved, by default True
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return    

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
            im = mne.viz.plot_topomap(
                data=chan_weight_j, 
                pos=info, 
                axes=ax,
                show=False,
                sphere=0.07, 
                cmap=cmaps[j],
                vlim=(chan_weight_j.min().round(3),chan_weight_j.max().round(3))
                )
            
            # # Configure colorbar
            # f = lambda x: round(x, -int(np.floor(np.log10(abs(x)))))
            # cbar = plt.colorbar(
            #     im[0], # TODO PROBLEMA
            #     ax=ax,
            #     orientation='vertical',
            #     shrink=0.6,
            #     aspect=15,
            #     boundaries=[f(x) for x in np.linspace(chan_weight_j.min(), chan_weight_j.max(), 100) if x not in [np.inf, 0]],
            #     ticks=[f(x) for x in np.linspace(chan_weight_j.min(),chan_weight_j.max(), 4) if x not in [np.inf, 0]]
            #     )
            # cbar.formatter.set_powerlimits((-2, 2))
            # cbar.ax.xaxis.get_offset_text().set_position((.5,.5))

            # if j==len(positive_relevant_indexes)-1:
            #     cbar.ax.set_ylabel('Weights')
        plt.figtext(x=.05, y=.05, s='Red is reserved for positive peaks, blue for negative ones', fontdict={'weight':'light'})
        if save:
            save_figure(
                cwd=current_working_directory,
                save_path=save_path, 
                file_name=f'relevant_times', 
                fig=fig
                )

def channel_wise_correlation_topomap(
    average_weights_subjects:np.ndarray, 
    info:mne.Info, 
    stim:str,
    save:bool, 
    save_path:str,
    display_interactive_mode:bool=False,
    no_figures:bool=False
    )->None:
    """
    Make a topographic plot of the channel-wise correlation

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        Average weights across subjects
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        If True, figures are saved
    save_path : str
        Path to save the figures
    display_interactive_mode : bool, optional
        If True, figures are displayed, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    
    Returns 
    -------
    None    
    """
    # Exit function
    plt.close()
    if no_figures:
        return   

    # Relevant parameters
    n_subjects, n_chan, _, _ = average_weights_subjects.shape
    average_weights = average_weights_subjects.mean(axis=2)
    correlation_matrices = np.zeros(shape=(n_chan, n_subjects, n_subjects))

    # Calculate correlation betweem subjects
    for channel in range(n_chan):
        matrix = average_weights[:,channel,:] 
        correlation_matrices[channel] = np.corrcoef(matrix)

    # Correlacion por canal
    absolute_correlation_per_channel = np.zeros(n_chan)
    for channel in range(n_chan):
        channel_corr_values = correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)]
        absolute_correlation_per_channel[channel] = np.mean(np.abs(channel_corr_values))

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Create figure and title
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
    fig.suptitle(f'Channel-wise {stim} similarity')

    # Make topomap
    im = mne.viz.plot_topomap(
        data=absolute_correlation_per_channel, 
        pos=info, 
        axes=ax, 
        show=False, 
        sphere=0.07,
        cmap='Greens', 
        vlim=(absolute_correlation_per_channel.min(),absolute_correlation_per_channel.max())
        )

    # Make colorbar
    cbar = plt.colorbar(
        im[0], 
        ax=ax, 
        shrink=0.85, 
        orientation='vertical', 
        label=f'Correlation'
        )

    if save:
        save_figure(
            cwd=current_working_directory,
            save_path=save_path, 
            file_name=f'channelwise_correlation_topo', 
            fig=fig
            )

def channel_weights(
    info:mne.Info,
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
    no_figures:bool=False,
    hierarchical_clustering:bool=True
    ):
    """
    Plot weights of features as an evoked response. If multidimensional features are used, a colormesh is used.

    Parameters
    ----------
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        If True, figures are saved
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
        Times of the delay window
    n_feats : list
        Number of features within each attribute
    stim : str
        Stimuli used in the model
    display_interactive_mode : bool, optional
        Whether to activate interactive mode, by default False
    session : int, optional
        Number of session, by default 21
    subject : int, optional
        Number of subject, by default 1
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    hierarchical_clustering : bool, by default True
        Whether to order attributes using hierarchical clustering based on correlation
    
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return   

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Get best correlation and rmse
    best_correlation = average_correlation.max()
    best_rmse = average_rmse.max()
    stimuli = stim.split('_')
    
    # Create figure and title
    fig, ax = plt.subplots(
            nrows=1, 
            ncols=len(stimuli), 
            figsize=(int(8*(len(stimuli))), 8), 
            layout='tight'
            )
    ax = np.array([[ax]]) if len(stimuli)==1 else ax.reshape(1, len(stimuli))
    fig.suptitle(f'Session {session} - Subject {subject} - Mcorr: {best_correlation:.2f} - Mrmse: {best_rmse:.2f} - '+ r'$\alpha$'+f': {best_alpha:.2f}')

    # Iterate over all stimuli
    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing of relevant features
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat
        if n_feat>1:
            # Add color mesh plot for atributes with more than one feature
            weights = average_weights[:, index_slice[0]:index_slice[1], :].mean(axis=0) # n_feats, n_delays
            
            # Perform clustering
            if hierarchical_clustering and not (feat.startswith('Spectro') or feat.startswith('Mfc') or feat.startswith('Deltas')): 
                order, null_indexes = clustering_by_correlation(weights=weights) 
                if null_indexes is not None:
                    weights = weights[[i for i in np.arange(weights.shape[0]) if i not in null_indexes]] 
                weights = weights[order]
            else:
                order = None
                null_indexes = None
            
            # Make color mesh
            number_of_ticks = weights.shape[0]
            im = ax[0,i_feat].pcolormesh(
                times * 1000, 
                np.arange(number_of_ticks), 
                weights, 
                cmap='RdBu_r', 
                shading='auto',
                vmin=-np.abs(weights).max(),
                vmax=np.abs(weights).max()
                )
            # Configure axis
            define_ticks(axes=ax[0, i_feat], number_of_ticks=number_of_ticks, ylabel=feat, xlabel='Time (ms)', title=feat, order=order, zeros_index=null_indexes)
                
            # Configure colorbar
            fig.colorbar(
                im,
                ax=ax[0, i_feat], 
                orientation='horizontal', 
                shrink=1, 
                label='Amplitude (a.u.)', 
                aspect=15
                )
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
                axes=ax[0, i_feat],
                gfp=False
                )

            # Add mean of all channels
            ax[0,i_feat].plot(
                        times * 1000, #ms
                        evoked._data.mean(0), 
                        'k', 
                        label='Mean', 
                        zorder=130, 
                        linewidth=2
                        )
            
            # Graph properties
            ax[0,i_feat].legend()
            ax[0,i_feat].grid(visible=True)
            ax[0,i_feat].set(xlabel='Time (ms)', title=f'{feat}')
    
    # Save figure
    if save:
        save_figure(   
            cwd=current_working_directory,
            save_path=os.path.join(save_path, 'individual_weights'),
            file_name=f'weights_session_{session}_subject_{subject}', 
            fig=fig
            )

def average_regression_weights(
    average_weights_subjects:np.ndarray, 
    info:mne.Info,
    save:bool, 
    save_path:str,
    times:np.ndarray,
    n_feats:list,
    stim:str,
    display_interactive_mode:bool=False,
    no_figures:bool=False,
    hierarchical_clustering:bool=True
    )->None:
    """
    Plot average weights of features as an evoked response. If colormesh_form is passed, a colormesh graph is performed in case of multifeature attribute are used.

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        Average weights across subjects
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        Whether to store the figure
    save_path : str
        Path to store the figure
    times : np.ndarray
        Times of the delay window
    n_feats : list
        Number of features within each attribute
    stim : str
        Stimuli used in the model
    display_interactive_mode : bool, optional
        Whether to activate interactive mode, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    hierarchical_clustering : bool, optional
        Whether to order attributes using hierarchical clustering based on correlation, by default True
    
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return  

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean over all subjects
    mean_average_weights_subjects = average_weights_subjects.mean(axis=0)
    stimuli = stim.split('_')

    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing of relevant features
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat

        if n_feat>1:
            # Create figure and title
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 12), layout='tight', sharex=True)
            fig.suptitle(f'{feat}')
            
            # Create evoked response as graph of weights averaged across all feats and subjects 
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
                axes=axes[0],
                gfp=False
                )
            # Add mean of all channels
            axes[0].plot(
                times*1e3, #ms
                evoked._data.mean(0), 
                'k', 
                label='Mean', 
                zorder=130, 
                linewidth=2
                )
            
            # Graph properties
            axes[0].grid(visible=True)
            axes[0].set(xlabel='')
            axes[0].legend()

            # Now average across channels to make mesh
            feat_weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=0)

            # Perform clustering
            if hierarchical_clustering and not (feat.startswith('Spectro') or feat.startswith('Mfc') or feat.startswith('Deltas')): 
                order, null_indexes = clustering_by_correlation(weights=feat_weights) 
                if null_indexes is not None:
                    feat_weights = feat_weights[[i for i in np.arange(feat_weights.shape[0]) if i not in null_indexes]] 
                feat_weights = feat_weights[order]
            else:
                order = None
                null_indexes = None

            # Create colormesh figure
            number_of_ticks = feat_weights.shape[0]
            im = axes[1].pcolormesh(
                    times * 1000, 
                    np.arange(number_of_ticks), 
                    feat_weights, 
                    cmap='RdBu_r', 
                    shading='auto',
                    vmin=-np.abs(feat_weights).max(),
                    vmax=np.abs(feat_weights).max()
                    )

            # Set figure configuration
            define_ticks(axes=axes[1], number_of_ticks=number_of_ticks, ylabel=feat, xlabel='Time (ms)', title=None, order=order, zeros_index=null_indexes)
            
            # Configure colorbar
            fig.colorbar(
                im, 
                ax=axes[1], 
                orientation='horizontal', 
                shrink=1, 
                label='Amplitude (a.u.)', 
                aspect=15
                )
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, layout='tight')
            fig.suptitle(f'{feat}')
            # Create evoked response as graph of weights averaged across all feats and subjects 
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
                gfp=False
                )
            # Add mean of all channels
            ax.plot(
                times*1e3, #ms
                evoked._data.mean(0), 
                'k', 
                label='Mean', 
                zorder=130, 
                linewidth=2
                )
            
            # Graph properties
            ax.set(xlabel='Time (ms)')
            ax.grid(visible=True)
            ax.legend()
        if save:
            save_figure(
                cwd=current_working_directory,
                save_path=save_path,
                file_name=f'average_weights_{feat.lower()}', 
                fig=fig
                )

def correlation_matrix_subjects(
    average_weights_subjects:np.ndarray, 
    stim:str, 
    n_feats:list, 
    save:bool, 
    save_path:str,
    display_interactive_mode:bool=False,
    no_figures:bool=False
    )->None:
    """
    Plot correlation matrix of the weights across subjects

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        Average weights across subjects
    stim : str
        Stimuli used in the model
    n_feats : list
        Number of features within each attribute
    save : bool
        Whether to store the figure
    save_path : str
        Path to store the figure
    display_interactive_mode : bool, optional
        Whether to activate interactive mode, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return  
    
    # Relevant parameters
    stimuli = stim.split('_')
    n_subjects, n_chan, _, n_delays = average_weights_subjects.shape

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing to get corresponding features of given feat
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat
        
        # Take average across features
        weights_across_features = average_weights_subjects[:,:,index_slice[0]:index_slice[1],:].mean(axis=2) # nsubjects, nchans, ndelays
        
        # Take average across subjects
        mean_across_subjects = weights_across_features.mean(axis=0) # nchans, ndelays

        # To store correlation matrix of each channel, it has an extra dimension to correlate against whole average
        correlation_matrices_of_each_channel = np.zeros(
            shape=(n_chan, n_subjects+1, n_subjects+1)
            ) 
        
        # Add mean of all subjects to the weights
        weights_across_features_plus_mean = np.concatenate(
            (weights_across_features, 
             mean_across_subjects.reshape(1, n_chan, n_delays) # to match weight_across_features shape
             ),
            axis=0
            )
        
        # For each channel, correlation across time delays is computed to get a matrix of n_subjects+1 x n_subjects+1
        for channel in range(n_chan):
            matrix = weights_across_features_plus_mean[:,channel,:] # nsubjects+1, ndelays
            correlation_matrices_of_each_channel[channel] = np.corrcoef(matrix)

        # Take average across all channels and exclude whole average
        correlation_matrix = correlation_matrices_of_each_channel.mean(axis=0)[:-1, :-1]
        
        # Now get the vector of correlations of channels vs whole average, excluding whole vs whole
        correlation_of_channel_vs_average = correlation_matrices_of_each_channel.mean(axis=0)[-1][:-1]

        # Change diagonal for values with correlations of channels vs whole average. By doing so, it's very unlinkely to find a 1 in the diagonal.
        for i in range(n_subjects):
            correlation_matrix[i, i] = correlation_of_channel_vs_average[i]

        # Get a list of subjects
        subject_names = np.arange(1, n_subjects+1).tolist()

        # Make mask for lower triangle of correlation matrix (this is a symmetric matrix)
        mask = np.ones_like(correlation_matrix)
        mask[np.tril_indices_from(mask)] = False

        # Take average
        correlation_mean, correlation_std = np.mean(np.abs(correlation_of_channel_vs_average)), np.std(np.abs(correlation_of_channel_vs_average))

        fig, (ax, cax) = plt.subplots(nrows=2, figsize=(16, 16), gridspec_kw={"height_ratios": [1, 0.05]}, layout='tight')
        fig.suptitle(f'Similarity among subject\'s {feat} TRFs - Mean: ({correlation_mean:.2f}'+r'$\pm$'+f'{correlation_std:.2f})', fontsize=19)
        sns.heatmap(
            correlation_matrix, 
            mask=mask, 
            cmap="coolwarm", 
            fmt='.2f', 
            ax=ax,
            annot=True, 
            center=0, 
            xticklabels=True, 
            annot_kws={"size": 15},
            cbar=False
            )

        ax.set_yticklabels(['Subjects mean'] + subject_names[1:], rotation=35, fontsize='xx-large')
        ax.set_xticklabels(subject_names[:-1] + ['Subjects mean'], rotation=35, fontsize='xx-large')

        # Make colorbar
        sns.despine(right=True, left=True, bottom=True, top=True)
        cbar = plt.colorbar(
            ax.get_children()[0], 
            cax=cax, 
            orientation="horizontal"
            )
        cbar.set_label('Correlation', fontsize='xx-large')
        cbar.ax.tick_params(labelsize='xx-large')
        
        # Save figure        
        if save:
            save_figure(
                cwd=current_working_directory,
                save_path=save_path,
                file_name=f'TRF_correlation_matrix_{feat.lower()}', 
                fig=fig
                )
            
def plot_pvalue_tfce(
    average_weights_subjects:np.ndarray,
    pvalue:np.ndarray, 
    info:mne.Info,
    save:bool, 
    save_path:str,
    times:np.ndarray,
    n_feats:list,
    stim:str,
    significance:float=0.05,
    display_interactive_mode:bool=False,
    no_figures:bool=False,
    hierarchical_clustering:bool=True
    )->None:
    """
    Plot p-values over weights across 

    Parameters
    ----------
    average_weights_subjects : np.ndarray
        Average weights across subjects
    pvalue : np.ndarray
        P-values of the weights
    info : mne.Info
        mne Info object depicting biosemi configuration of eeg channels 
    save : bool
        Whether to store the figure
    save_path : str
        Path to store the figure
    times : np.ndarray
        Times of the delay window
    n_feats : list
        Number of features within each attribute
    stim : str
        Stimuli used in the model
    display_interactive_mode : bool, optional
        Whether to activate interactive mode, by default False
    no_figures : bool, optional
        If True, no figures are displayed, by default False
    hierarchical_clustering : bool, optional
        Whether to order attributes using hierarchical clustering based on correlation, by default True
        
    Returns
    -------
    None
    """
    # Exit function
    plt.close()
    if no_figures:
        return  

    # Turn on/off interactive mode
    if display_interactive_mode:
        plt.ion()
    else:
        plt.ioff()

    # Take mean over all subjects
    mean_average_weights_subjects = average_weights_subjects.mean(axis=0)
    stimuli = stim.split('_')

    for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
        # Make slicing of relevant features
        index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat
        
        # If stimulus is not frequency correlated significant channels are calculated
        if pvalue.ndim==3:
            pvalue = pvalue[index_slice[0]:index_slice[1], :, :]
            significant_channels = np.zeros(shape=(n_feat, len(times)))

            # Iteate over columns to get number of channels per feature that passes the threshold
            for feature in range(n_feat):
                for delay in range(len(times)):
                    # Count how many channels pass the threshold for a given feature and delay
                    ppval = pvalue[feature][delay]
                    significant_channels[feature, delay] = len(ppval[ppval<significance])
        else:
            pvalue = pvalue[:, index_slice[0]:index_slice[1]]
        
        # Transform pvalues to logscale and to 1 pvals not passing the significance (this is for frequency correlated stimulus)
        pvals_for_graph = pvalue.copy()
        pvals_for_graph[pvals_for_graph>significance] = 1
        pvals_for_graph = -np.log10(pvals_for_graph)
       
        # Create figure and title
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), layout='tight', sharex=True)
        fig.suptitle(feat)
            
        if n_feat>1:
            # Now average across channels to make mesh
            feat_weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=0)

            # Perform clustering
            if hierarchical_clustering and not (feat.startswith('Spectro') or feat.startswith('Mfc') or feat.startswith('Deltas')): 
                order, null_indexes = clustering_by_correlation(weights=feat_weights) 
                if null_indexes is not None:
                    feat_weights = feat_weights[[i for i in np.arange(feat_weights.shape[0]) if i not in null_indexes]] 
                feat_weights = feat_weights[order]
                if pvalue.ndim==3:
                    if null_indexes is not None:
                        significant_channels = significant_channels[[i for i in np.arange(significant_channels.shape[0]) if i not in null_indexes]] 
                    significant_channels = significant_channels[order]
                else:
                    if null_indexes is not None:
                        pvals_for_graphss = pvals_for_graph[[i for i in np.arange(pvals_for_graph.shape[1]) if i not in null_indexes]] 
                    pvals_for_graph = pvals_for_graph[:, order]
            else:
                order = None
                null_indexes = None

            # Create colormesh figure for weights
            number_of_ticks = feat_weights.shape[0]
            im = axes[0].pcolormesh(
                times*1e3, 
                np.arange(number_of_ticks), 
                feat_weights, 
                cmap='RdBu_r', 
                shading='auto',
                vmin=-np.abs(feat_weights).max(),
                vmax=np.abs(feat_weights).max()
                )

            # Set figure configuration
            define_ticks(axes=axes[0], number_of_ticks=number_of_ticks, ylabel=feat, xlabel='', title=None, order=order, zeros_index=null_indexes)
            
            # Configure colorbar
            fig.colorbar(
                im, 
                ax=axes[0], 
                orientation='vertical', 
                shrink=1, 
                label='Amplitude (a.u.)', 
                aspect=15
                )
        else:
            # Create evoked response as graph of weights averaged across all feats and subjects 
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
                axes=axes[0],
                gfp=False
                )
            # Add mean of all channels
            axes[0].plot(
                times*1e3, #ms
                evoked._data.mean(0), 
                'k', 
                label='Mean', 
                zorder=130, 
                linewidth=2
                )
            
            # Graph properties
            axes[0].set(xlabel='')
            axes[0].grid(visible=True)
            axes[0].legend()
            
        # Now the pvalue
        if pvalue.ndim==3:
            bar_label = "Number of significant channels"
            # Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
            number_of_ticks = significant_channels.shape[0]
            y, z = (np.arange(n_feat+1), np.concatenate((significant_channels,significant_channels))) if n_feat==1 else (np.arange(number_of_ticks), significant_channels)
            im2 = axes[1].pcolormesh(
                times*1e3, # x
                y, # y
                z, #z
                shading='auto',
                cmap='inferno'
                )
        else:
            bar_label = r"$-log_{10}(p_{values})$"
            
            # Define y and z according to the number of features (this is just to make a wark around 1 dimensional colormesh)
            number_of_ticks = pvals_for_graph.shape[1]
            y, z = (np.arange(n_feat+1), np.concatenate((pvals_for_graph,pvals_for_graph))) if n_feat==1 else (np.arange(number_of_ticks), pvals_for_graph.T)
            im2 = axes[1].pcolormesh(
                times*1e3, # x
                y, # y
                z, # z
                shading='auto',
                cmap='inferno'
                )
        if n_feat>1:
            define_ticks(axes=axes[1], number_of_ticks=number_of_ticks, ylabel=feat, xlabel='Time (ms)', title=None, order=order, zeros_index=null_indexes) 
            fig.colorbar( 
                orientation='vertical', 
                label=bar_label,
                aspect=15, 
                shrink=1, 
                mappable=im2
                )
        else:
            axes[1].set(xlabel ='Time (ms)')
            fig.colorbar( 
                orientation='horizontal', 
                label=bar_label,
                aspect=15, 
                shrink=1, 
                pad=.25,
                mappable=im2
                )
        if save:
            save_figure(
                cwd=current_working_directory,
                save_path=os.path.join(save_path,'TFCE'),
                file_name=f'pvalue_over_average_trf{feat.lower()}', 
                fig=fig
                )

# #TODO CHECK DESCRIPTION E Y LABEL
# def plot_pvalue_tfce(average_weights_subjects:np.ndarray,
#                 pvalue:np.ndarray,
#                 times:np.ndarray, 
#                 trf_subjects_shape:tuple, 
#                 band:str, 
#                 stim:str,
#                 info:mne.Info,
#                 n_feats:list, 
#                 pval_tresh:float, 
#                 save_path:str, 
#                 display_interactive_mode:bool=False, 
#                 save:bool=True,
#                 no_figures:bool=False):

#     # Exit function
#     plt.close()
#     if no_figures:
#         return

#     # Turn on/off interactive mode
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     # Take mean over all subjects
#     mean_average_weights_subjects = average_weights_subjects.mean(axis=0)
#     stimuli = stim.split('_')
#     save_path += 'TFCE/'
        
#     for i_feat, (feat, n_feat) in enumerate(zip(stimuli, n_feats)):
#         # Make slicing of relevant features
#         index_slice = sum(n_feats[:i_feat]),  sum(n_feats[:i_feat]) + n_feat
#         pvalue_feat = pvalue[index_slice[0]:index_slice[1]]
        
#         if feat.startswith('Phoneme') or feat.startswith('Spectro') or feat.startswith('Mfccs') or feat.startswith('Deltas') or feat.startswith('Phonolo'):
#             # Create figure and title
#             fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout='tight', sharey=True)
#             fig.suptitle(f'P-value for {feat} - {band}')

#             # Create evoked response as graph of weights averaged across all feats and all 
#             weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=0)
        
#             # Make colormesh
#             im = ax[0].pcolormesh(times*1000,
#                                    np.arange(n_feat), 
#                                    weights, 
#                                    cmap='RdBu_r',
#                                    shading='auto')

#             # Set figure configuration
#             if feat.startswith('Spectro'):
#                 bands_center = librosa.mel_frequencies(n_mels=n_feat+2, fmin=62, fmax=8000)[1:-1]
#                 ax[0].set(xlabel='Time (ms)', ylabel='Frecuency (Hz)', yticks=np.arange(0, n_feat, 2), 
#                         yticklabels=[int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)], title='Weights')

#                 # Configure colorbar
#                 fig.colorbar(im, 
#                             ax=ax[0], 
#                             orientation='horizontal', 
#                             shrink=1, 
#                             label='Amplitude (a.u.)', 
#                             aspect=15)
#             elif feat.startswith('Mfccs') or feat.startswith('Deltas'):
#                 # Set figure configuration
#                 if feat=='Mfccs':
#                     ylabels = [r'$m_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat)]
#                 elif feat=='Mfccs-Deltas':
#                     ylabels = [r'$m_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/2)] 
#                     ylabels += [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/2)]
#                 elif feat=='Mfccs-Deltas-Deltas':
#                     ylabels = [r'$m_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/3)] 
#                     ylabels += [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/3)]
#                     ylabels += [r'$\delta-\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/3)]
#                 elif feat=='Deltas':
#                     ylabels = [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat)]
#                 elif feat=='Deltas-Deltas':
#                     ylabels = [r'$\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/2)]
#                     ylabels += [r'$\delta-\delta_{{{}}}$'.format(int(i)) for i in np.arange(0, n_feat/2)]
#                 ax[0].set(xlabel='Time (ms)',
#                           ylabel=f"{feat}'s Index",
#                           yticklabels=ylabels[::2],
#                           yticks= np.arange(0, n_feat, 2), 
#                           title=f'Weights')
                
#                 # Configure colorbar
#                 fig.colorbar(im, 
#                             ax=ax[0], 
#                             orientation='horizontal', 
#                             shrink=1, 
#                             label='Amplitude (a.u.)', 
#                             aspect=15)
#             elif feat.startswith('Phonemes'):
#                 if feat.endswith('Manual'):
#                     ax[0].set(xlabel='Time (ms)', ylabel='Phonemes', yticks=np.arange(n_feat), yticklabels=config.exp_info.ph_labels_man)
#                 elif feat.endswith('Phonet'):
#                     ax[0].set(xlabel='Time (ms)', ylabel='Phonemes', yticks=np.arange(n_feat), yticklabels=config.exp_info.ph_labels_phonet[:-1])
#                 else:
#                     ax[0].set(xlabel='Time (ms)', ylabel='Phonemes', yticks=np.arange(n_feat), yticklabels=config.exp_info.ph_labels)
                
#                 ax[0].tick_params(axis='both', labelsize='medium') # Change labelsize because there are too many phonemes
                
#                 # Make color bar
#                 fig.colorbar(im,
#                              ax=ax[0],
#                              orientation='horizontal',
#                              label='Amplitude (a.u.)',
#                              shrink=1,
#                              aspect=20)
#             elif feat.startswith('Phonological'):
#                 ax[0].set(xlabel='Time (ms)', ylabel='Phonological Features', yticks=np.arange(n_feat), yticklabels=config.exp_info.phonological_labels)
#                 ax[0].tick_params(axis='both', labelsize='medium') # Change labelsize because there are too many phonemes
                
#                 # Make color bar
#                 fig.colorbar(im,
#                              ax=ax[0],
#                              orientation='horizontal',
#                              label='Amplitude (a.u.)',
#                              shrink=1,
#                              aspect=20)
                
#             # Mask p-values over threshold
#             pvalue_feat[pvalue_feat > pval_tresh] = 1

#             # Make p-value plot
#             img = ax[1].pcolormesh(times*1000, 
#                             np.arange(n_feat), 
#                             -np.log10(np.maximum(pvalue_feat, 1e-5)),
#                             cmap="inferno", 
#                             shading='auto')

#             # Configure plot
#             ax[1].set(xlabel='Times (ms)', title='P-value')
            
#             # Make colorbar
#             cbar = fig.colorbar(ax=ax[1], 
#                                 orientation="horizontal", 
#                                 label=r"$-\log_{10}(p)$",
#                                 shrink=1, 
#                                 aspect=15,
#                                 mappable=img)

#             if display_interactive_mode:
#                 text = fig.suptitle('')
#                 text.set_weight("bold")
#                 plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
#                 mne.viz.utils.plt_show()
#         elif n_feat==1 and n_feats!=[1]:
#             # Create figure and title
#             fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), layout='tight', sharey=False)
#             fig.suptitle(f'P-value for {feat} - {band}')

#             # Create evoked response as graph of weights averaged across all feats and all 
#             weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=1)
#             evoked = mne.EvokedArray(data=weights, info=info)
        
#             # Relabel time 0
#             evoked.shift_time(times[0], relative=True)
            
#             # Plot
#             evoked.plot(
#                 scalings={'eeg':1}, 
#                 zorder='std', 
#                 time_unit='ms',
#                 show=False, 
#                 spatial_colors=True, 
#                 units='mTRF (a.u.)',
#                 axes=ax,
#                 gfp=False)

#             # Add mean of all channels
#             ax.plot(
#                 times * 1000, #ms
#                 evoked._data.mean(axis=0), 
#                 'k--', 
#                 label='Mean', 
#                 zorder=130, 
#                 linewidth=2)
            
#             # Mask p-values over threshold
#             ax_p_val = ax.twinx()
#             pvalue_feat[pvalue_feat > pval_tresh] = 1
#             logp = -np.log10(np.maximum(pvalue_feat.reshape(-1), pval_tresh))

#             # Make vertical span
#             ax_p_val.plot(times*1e3, # ms
#                           logp,
#                           color='orange')
            
#             ax_p_val.fill_between(times*1e3,
#                                   logp.min(),
#                                   logp, 
#                                   color='orange',
#                                   alpha=.1,
#                                   label=f'Passed {pval_tresh} threshold')
                       
#             # Graph properties
#             ax.legend(loc=(.55,.7))
#             ax.grid(visible=True)
#             ax.set(yticks=[], xlabel='Times (ms)', title='P-value')
#             ax_p_val.set(ylabel=r"$-\log_{10}(p)$", yticks = np.linspace(logp.min(), logp.max(), 8))
#             ax_p_val.legend(loc=(.55,.8))

#         elif n_feat==1 and n_feats==[1]:
#             # Create figure and title
#             fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout='tight', sharey=False)
#             fig.suptitle(f'P-value for {feat} - {band}')

#             # Create evoked response as graph of weights averaged across all feats and all 
#             weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=1)
#             evoked = mne.EvokedArray(data=weights, info=info)
        
#             # Relabel time 0
#             evoked.shift_time(times[0], relative=True)
            
#             # Plot
#             evoked.plot(
#                 scalings={'eeg':1}, 
#                 zorder='std', 
#                 time_unit='ms',
#                 show=False, 
#                 spatial_colors=True, 
#                 units='mTRF (a.u.)',
#                 axes=ax[0],
#                 gfp=False)

#             # Add mean of all channels
#             ax[0].plot(
#                 times * 1000, #ms
#                 evoked._data.mean(0), 
#                 'k--', 
#                 label='Mean', 
#                 zorder=130, 
#                 linewidth=2)
            
#             # Graph properties
#             ax[0].legend()
#             ax[0].grid(visible=True)

#             # Mask p-values over threshold
#             pvalue_feat = pvalue
#             pvalue_feat[pvalue_feat > pval_tresh] = 1

#             # Probability plot
#             img = ax[1].pcolormesh(times*1000, 
#                                 np.arange(trf_subjects_shape[1]), 
#                                 -np.log10(np.maximum(pvalue, pval_tresh)),
#                                 cmap="inferno", 
#                                 shading='auto')
            
#             # Configure plot
#             ax[1].set(yticks=[],
#                 xlabel='Times (ms)',
#                 ylabel='Channels',
#                 title='P-value')
            
#             # Make colorbar
#             cbar = plt.colorbar(ax=ax[1], 
#                                 orientation="vertical", 
#                                 label=r"$-\log_{10}(p)$",
#                                 fraction=0.05, 
#                                 pad=0.025, 
#                                 mappable=img)
#             cbar.ax.get_xaxis().set_label_coords(0.5, -3)

#         # Save figures
#         if save:
#             os.makedirs(save_path, exist_ok=True)
            
#             # This is done to avoid working with long paths
#             temp_path = os.path.normpath(save_path)
#             os.chdir(temp_path)
#             fig.savefig(f'pvalue_{feat.lower()}_{pval_tresh:.0e}{config.figure_format}')
#             os.chdir(current_working_directory)
        
# =====================
# FIGURES OF VALIDATION

def hyperparameter_selection(
    alphas_swept: np.ndarray,
    correlations: np.ndarray,
    correlations_std: np.ndarray,
    correlations_train: np.ndarray,
    rmse: np.ndarray,
    rmse_std: np.ndarray,
    rmse_train: np.ndarray,
    trfs: np.ndarray,
    alpha_subject: float,
    correlation_limit_percentage: float,
    session: int, 
    subject: int,
    stim: str,
    band: str,
    save_path: str, 
    no_figures: bool = False,
    save: bool = False
):
    """
    Combined hyperparameter selection plot with correlation/RMSE metrics and TRFs visualization.
    """
    # Exit function
    plt.close()
    if no_figures:
        return
    
    # Create figure with custom layout: 3x2 grid at top, 1x1 at bottom spanning both columns
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    # Top row: Correlation (left) and Correlation Ratio (right)
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_corr_ratio = fig.add_subplot(gs[0, 1])
    
    # Second row: RMSE (left) and RMSE Ratio (right)
    ax_rmse = fig.add_subplot(gs[1, 0])
    ax_rmse_ratio = fig.add_subplot(gs[1, 1])
    
    # Third row: Combined Metric (left) and Combined Metric Normalized (right)
    ax_combined = fig.add_subplot(gs[2, 0])
    ax_combined_norm = fig.add_subplot(gs[2, 1])
    
    # Bottom row: TRFs spanning both columns
    ax_trfs = fig.add_subplot(gs[3, :])
    
    fig.suptitle(f'{band} - {stim} - Session {session} - Subject {subject}', fontsize=16)
    
    # Find relevant range within correlation_limit_percentage
    relative_difference = abs((correlations.max() - correlations)/correlations.max())
    good_indexes_range = np.where(relative_difference < correlation_limit_percentage)[0]
    
    # ===== CORRELATIONS PLOT =====
    ax_corr.plot(alphas_swept, correlations, 'o--', color='C0')
    ax_corr.errorbar(alphas_swept, correlations, yerr=correlations_std, fmt='none', 
                     ecolor='black', elinewidth=0.5, capsize=0.5)
    
    # Vertical lines for maximum correlation and selected alpha
    ax_corr.vlines(alphas_swept[correlations.argmax()], ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], 
                   linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
    ax_corr.vlines(alpha_subject, ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], 
                   linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
    # Green box for acceptable range
    if good_indexes_range.size > 1:
        ax_corr.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
                        alpha=0.4, color='green', 
                        label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
    ax_corr.set(xlabel=r'Ridge parameter $\alpha$', ylabel='Mean correlation (Test)', 
                xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
    ax_corr.grid(visible=True)
    ax_corr.legend(fontsize=8)
    
    # ===== CORRELATION RATIO PLOT =====
    ax_corr_ratio.plot(alphas_swept, 1e2*(correlations-correlations_train)/correlations_train, 'o--', color='C0')
    
    ax_corr_ratio.vlines(alphas_swept[correlations.argmax()], ax_corr_ratio.get_ylim()[0], ax_corr_ratio.get_ylim()[1], 
                         linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
    ax_corr_ratio.vlines(alpha_subject, ax_corr_ratio.get_ylim()[0], ax_corr_ratio.get_ylim()[1], 
                         linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
    if good_indexes_range.size > 1:
        ax_corr_ratio.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
                              alpha=0.4, color='green', 
                              label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
    ax_corr_ratio.set(xlabel=r'Ridge parameter $\alpha$', ylabel=r'Correlation (Test-Train)/Test[\%]', 
                      xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
    ax_corr_ratio.grid(visible=True)
    ax_corr_ratio.legend(fontsize=8)
    
    # ===== RMSE PLOT =====
    ax_rmse.plot(alphas_swept, rmse, 'o--', color='C1')
    ax_rmse.errorbar(alphas_swept, rmse, yerr=rmse_std, fmt='none', 
                     ecolor='black', elinewidth=0.5, capsize=0.5)
    
    ax_rmse.vlines(alphas_swept[correlations.argmax()], ax_rmse.get_ylim()[0], ax_rmse.get_ylim()[1], 
                   linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
    ax_rmse.vlines(alpha_subject, ax_rmse.get_ylim()[0], ax_rmse.get_ylim()[1], 
                   linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
    if good_indexes_range.size > 1:
        ax_rmse.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
                        alpha=0.4, color='green', 
                        label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
    ax_rmse.set(xlabel=r'Ridge parameter $\alpha$', ylabel='Mean RMSE (Test)', 
                xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
    ax_rmse.grid(visible=True)
    ax_rmse.legend(fontsize=8)
    
    # ===== RMSE RATIO PLOT =====
    ax_rmse_ratio.plot(alphas_swept, 1e2*(rmse_train-rmse)/rmse, 'o--', color='C1')
    
    ax_rmse_ratio.vlines(alphas_swept[correlations.argmax()], ax_rmse_ratio.get_ylim()[0], ax_rmse_ratio.get_ylim()[1], 
                         linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
    ax_rmse_ratio.vlines(alpha_subject, ax_rmse_ratio.get_ylim()[0], ax_rmse_ratio.get_ylim()[1], 
                         linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
    if good_indexes_range.size > 1:
        ax_rmse_ratio.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
                              alpha=0.4, color='green', 
                              label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
    ax_rmse_ratio.set(xlabel=r'Ridge parameter $\alpha$', ylabel=r'RMSE (Train-Test)/Test [\%]', 
                      xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
    ax_rmse_ratio.grid(visible=True)
    ax_rmse_ratio.legend(fontsize=8)
    
    # ===== COMBINED METRIC PLOT =====
    # Define lambda values
    lambda1, lambda2 = 1.5, 1.5
    
    # Calculate combined metric: correlation + lambda1*max(0, rmse_train-rmse) + lambda2*max(0, correlation-correlation_train)
    combined_metric = correlations - lambda1*np.maximum(0, rmse - rmse_train) - lambda2*np.maximum(0, correlations_train-correlations)
    
    ax_combined.plot(alphas_swept, combined_metric, 'o--', color='C2')
    
    # Find maximum of combined metric
    combined_max_idx = combined_metric.argmax()
    
    ax_combined.vlines(alphas_swept[combined_max_idx], ax_combined.get_ylim()[0], ax_combined.get_ylim()[1], 
                       linestyle='dashed', color='black', linewidth=1.5, label='Maximum combined metric')
    ax_combined.vlines(alpha_subject, ax_combined.get_ylim()[0], ax_combined.get_ylim()[1], 
                       linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
    if good_indexes_range.size > 1:
        ax_combined.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
                            alpha=0.4, color='green', 
                            label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
    ax_combined.set(xlabel=r'Ridge parameter $\alpha$', 
                    ylabel=rf'Combined Metric ($\lambda_1$={lambda1}, $\lambda_2$={lambda2})', 
                    xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
    ax_combined.grid(visible=True)
    ax_combined.legend(fontsize=8)
    
    # ===== COMBINED METRIC NORMALIZED PLOT =====
    # Normalize each component to [0,1] before combining
    corr_norm = (correlations - correlations.min()) / (correlations.max() - correlations.min())
    rmse_diff_norm = np.maximum(0, rmse_train-rmse)
    rmse_diff_norm = rmse_diff_norm / (rmse_diff_norm.max() + 1e-10)  # Avoid division by zero
    corr_diff_norm = np.maximum(0, correlations_train-correlations)
    corr_diff_norm = corr_diff_norm / (corr_diff_norm.max() + 1e-10)  # Avoid division by zero
    
    combined_metric_norm = corr_norm - lambda1*rmse_diff_norm - lambda2*corr_diff_norm
    
    ax_combined_norm.plot(alphas_swept, combined_metric_norm, 'o--', color='C3')
    
    # Find maximum of normalized combined metric
    combined_norm_max_idx = combined_metric_norm.argmax()
    
    ax_combined_norm.vlines(alphas_swept[combined_norm_max_idx], ax_combined_norm.get_ylim()[0], ax_combined_norm.get_ylim()[1], 
                            linestyle='dashed', color='black', linewidth=1.5, label='Maximum normalized metric')
    ax_combined_norm.vlines(alpha_subject, ax_combined_norm.get_ylim()[0], ax_combined_norm.get_ylim()[1], 
                            linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
    if good_indexes_range.size > 1:
        ax_combined_norm.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
                                 alpha=0.4, color='green', 
                                 label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
    ax_combined_norm.set(xlabel=r'Ridge parameter $\alpha$', 
                         ylabel='Normalized Combined Metric', 
                         xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
    ax_combined_norm.grid(visible=True)
    ax_combined_norm.legend(fontsize=8)
    
    # ===== TRFs PLOT =====
    # Create colormap and normalization for the alphas
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.log10(alphas_swept.min()), vmax=np.log10(alphas_swept.max()))
    
    # Plot TRFs for each alpha
    for alpha, trf in zip(alphas_swept, trfs):
        color = cmap(norm(np.log10(alpha)))
        ax_trfs.plot(
            config.times*1e3,
            trf,
            color=color,
            linewidth=1.5,
            alpha=0.8
        )
    
    # Highlight the selected alpha
    selected_idx = np.argmin(np.abs(alphas_swept - alpha_subject))
    ax_trfs.plot(
        config.times*1e3,
        trfs[selected_idx],
        color='black',
        linewidth=2.5,
        alpha=1.0,
        label=f'Selected α = {alpha_subject:.0f}' if alpha_subject >= 1 else f'Selected α = {alpha_subject:.3f}'
    )
    
    ax_trfs.set(
        xlabel='Time (ms)',
        ylabel='TRF Amplitude (a.u.)',
        ylim=(-np.abs(trfs[selected_idx]).max()*1.5, np.abs(trfs[selected_idx]).max()*1.5),
        title='Temporal Response Functions for Different Alpha Values'
    )
    ax_trfs.grid(True, alpha=0.3)
    ax_trfs.legend(loc='upper right', fontsize=10)
    
    # Add colorbar for alpha values
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_trfs, orientation='vertical', pad=0.02, shrink=0.8, aspect=20)
    cbar.set_label(r'$\log_{10}(\alpha)$', fontsize=12)
    
    # Set colorbar ticks to show actual alpha values
    log_alphas = np.log10(alphas_swept)
    tick_positions = np.linspace(log_alphas.min(), log_alphas.max(), 5)
    cbar.set_ticks(tick_positions)
    
    # Better formatting for the wide range of alpha values
    alpha_values = [10**pos for pos in tick_positions]
    formatted_labels = []
    
    for alpha in alpha_values:
        if alpha < 0.01:
            formatted_labels.append(f'{alpha:.3f}')
        elif alpha < 1:
            formatted_labels.append(f'{alpha:.2f}')
        elif alpha < 1000:
            formatted_labels.append(f'{int(alpha)}')
        else:
            formatted_labels.append(f'{alpha:.1e}')
    
    cbar.set_ticklabels(formatted_labels)
    
    # Save figure
    if save:
        os.makedirs(save_path, exist_ok=True)
        temp_path = os.path.normpath(save_path)
        os.chdir(temp_path)
        fig.savefig(f'session_{session}_subject_{subject}{config.figure_format}', 
                    dpi=300, bbox_inches='tight')
        os.chdir(current_working_directory)
        plt.close(fig)

# def hyperparameter_selection(
#     alphas_swept: np.ndarray,
#     correlations: np.ndarray,
#     correlations_std: np.ndarray,
#     correlations_train: np.ndarray,
#     rmse: np.ndarray,
#     rmse_std: np.ndarray,
#     rmse_train: np.ndarray,
#     trfs: np.ndarray,
#     alpha_subject: float,
#     correlation_limit_percentage: float,
#     session: int, 
#     subject: int,
#     stim: str,
#     band: str,
#     save_path: str, 
#     no_figures: bool = False,
#     save: bool = False
# ):
#     """
#     Combined hyperparameter selection plot with correlation/RMSE metrics and TRFs visualization.

#     Parameters
#     ----------
#     alphas_swept : np.ndarray
#         Array of alpha values tested
#     correlations : np.ndarray
#         Test correlations for each alpha
#     correlations_std : np.ndarray
#         Standard deviation of test correlations
#     correlations_train : np.ndarray
#         Training correlations for each alpha
#     rmse : np.ndarray
#         Test RMSE for each alpha
#     rmse_std : np.ndarray
#         Standard deviation of test RMSE
#     rmse_train : np.ndarray
#         Training RMSE for each alpha
#     trfs : np.ndarray
#         TRFs for each alpha value
#     alpha_subject : float
#         Selected alpha value
#     correlation_limit_percentage : float
#         Percentage threshold for correlation selection
#     session : int
#         Session number
#     subject : int
#         Subject number
#     stim : str
#         Stimulus type
#     band : str
#         Frequency band
#     save_path : str
#         Path to save the figure
#     no_figures : bool, optional
#         If True, no figures are displayed, by default False
#     save : bool, optional
#         If True, figures are saved, by default False
#     """
#     # Exit function
#     plt.close()
#     if no_figures:
#         return
    
#     # Create figure with custom layout: 2x2 grid at top, 1x1 at bottom spanning both columns
#     fig = plt.figure(figsize=(16, 12))
#     gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
#     # Top row: Correlation (left) and Correlation Ratio (right)
#     ax_corr = fig.add_subplot(gs[0, 0])
#     ax_corr_ratio = fig.add_subplot(gs[0, 1])
    
#     # Middle row: RMSE (left) and RMSE Ratio (right)
#     ax_rmse = fig.add_subplot(gs[1, 0])
#     ax_rmse_ratio = fig.add_subplot(gs[1, 1])
    
#     # Bottom row: TRFs spanning both columns
#     ax_trfs = fig.add_subplot(gs[2, :])
    
#     fig.suptitle(f'{band} - {stim} - Session {session} - Subject {subject}', fontsize=16)
    
#     # Find relevant range within correlation_limit_percentage
#     relative_difference = abs((correlations.max() - correlations)/correlations.max())
#     good_indexes_range = np.where(relative_difference < correlation_limit_percentage)[0]
    
#     # ===== CORRELATIONS PLOT =====
#     ax_corr.plot(alphas_swept, correlations, 'o--', color='C0')
#     ax_corr.errorbar(alphas_swept, correlations, yerr=correlations_std, fmt='none', 
#                      ecolor='black', elinewidth=0.5, capsize=0.5)
    
#     # Vertical lines for maximum correlation and selected alpha
#     ax_corr.vlines(alphas_swept[correlations.argmax()], ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], 
#                    linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
#     ax_corr.vlines(alpha_subject, ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], 
#                    linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
#     # Green box for acceptable range
#     if good_indexes_range.size > 1:
#         ax_corr.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
#                         alpha=0.4, color='green', 
#                         label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
#     ax_corr.set(xlabel=r'Ridge parameter $\alpha$', ylabel='Mean correlation (Test)', 
#                 xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
#     ax_corr.grid(visible=True)
#     ax_corr.legend(fontsize=8)
    
#     # ===== CORRELATION RATIO PLOT =====
#     ax_corr_ratio.plot(alphas_swept, 1e2*(correlations-correlations_train)/correlations_train, 'o--', color='C0')
    
#     ax_corr_ratio.vlines(alphas_swept[correlations.argmax()], ax_corr_ratio.get_ylim()[0], ax_corr_ratio.get_ylim()[1], 
#                          linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
#     ax_corr_ratio.vlines(alpha_subject, ax_corr_ratio.get_ylim()[0], ax_corr_ratio.get_ylim()[1], 
#                          linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
#     if good_indexes_range.size > 1:
#         ax_corr_ratio.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
#                               alpha=0.4, color='green', 
#                               label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
#     ax_corr_ratio.set(xlabel=r'Ridge parameter $\alpha$', ylabel=r'Correlation (Test-Train)/Test[\%]', 
#                       xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
#     ax_corr_ratio.grid(visible=True)
#     ax_corr_ratio.legend(fontsize=8)
    
#     # ===== RMSE PLOT =====
#     ax_rmse.plot(alphas_swept, rmse, 'o--', color='C1')
#     ax_rmse.errorbar(alphas_swept, rmse, yerr=rmse_std, fmt='none', 
#                      ecolor='black', elinewidth=0.5, capsize=0.5)
    
#     ax_rmse.vlines(alphas_swept[correlations.argmax()], ax_rmse.get_ylim()[0], ax_rmse.get_ylim()[1], 
#                    linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
#     ax_rmse.vlines(alpha_subject, ax_rmse.get_ylim()[0], ax_rmse.get_ylim()[1], 
#                    linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
#     if good_indexes_range.size > 1:
#         ax_rmse.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
#                         alpha=0.4, color='green', 
#                         label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
#     ax_rmse.set(xlabel=r'Ridge parameter $\alpha$', ylabel='Mean RMSE (Test)', 
#                 xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
#     ax_rmse.grid(visible=True)
#     ax_rmse.legend(fontsize=8)
    
#     # ===== RMSE RATIO PLOT =====
#     ax_rmse_ratio.plot(alphas_swept, 1e2*(rmse_train-rmse)/rmse, 'o--', color='C1')
    
#     ax_rmse_ratio.vlines(alphas_swept[correlations.argmax()], ax_rmse_ratio.get_ylim()[0], ax_rmse_ratio.get_ylim()[1], 
#                          linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation')
#     ax_rmse_ratio.vlines(alpha_subject, ax_rmse_ratio.get_ylim()[0], ax_rmse_ratio.get_ylim()[1], 
#                          linestyle='dashed', color='red', linewidth=1.5, label='Selected value')
    
#     if good_indexes_range.size > 1:
#         ax_rmse_ratio.axvspan(alphas_swept[good_indexes_range[0]], alphas_swept[good_indexes_range[-1]], 
#                               alpha=0.4, color='green', 
#                               label=f'{int(correlation_limit_percentage*100)}% of maximum correlation')
    
#     ax_rmse_ratio.set(xlabel=r'Ridge parameter $\alpha$', ylabel=r'RMSE (Train-Test)/Test [\%]', 
#                       xscale='log', xlim=([alphas_swept[0], alphas_swept[-1]]))
#     ax_rmse_ratio.grid(visible=True)
#     ax_rmse_ratio.legend(fontsize=8)
    
#     # ===== TRFs PLOT =====
#     # Create colormap and normalization for the alphas
#     cmap = cm.get_cmap('viridis')
#     norm = plt.Normalize(vmin=np.log10(alphas_swept.min()), vmax=np.log10(alphas_swept.max()))
    
#     # Plot TRFs for each alpha
#     for alpha, trf in zip(alphas_swept, trfs):
#         color = cmap(norm(np.log10(alpha)))
#         ax_trfs.plot(
#             config.times*1e3,
#             trf,
#             color=color,
#             linewidth=1.5,
#             alpha=0.8
#         )
    
#     # Highlight the selected alpha
#     selected_idx = np.argmin(np.abs(alphas_swept - alpha_subject))
#     ax_trfs.plot(
#         config.times*1e3,
#         trfs[selected_idx],
#         color='black',
#         linewidth=2.5,
#         alpha=1.0,
#         label=f'Selected α = {alpha_subject:.0f}' if alpha_subject >= 1 else f'Selected α = {alpha_subject:.3f}'
#     )
    
#     ax_trfs.set(
#         xlabel='Time (ms)',
#         ylabel='TRF Amplitude (a.u.)',
#         title='Temporal Response Functions for Different Alpha Values'
#     )
#     ax_trfs.grid(True, alpha=0.3)
#     ax_trfs.legend(loc='upper right', fontsize=10)
    
#     # Add colorbar for alpha values
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax_trfs, orientation='vertical', pad=0.02, shrink=0.8, aspect=20)
#     cbar.set_label(r'$\log_{10}(\alpha)$', fontsize=12)
    
#     # Set colorbar ticks to show actual alpha values
#     log_alphas = np.log10(alphas_swept)
#     tick_positions = np.linspace(log_alphas.min(), log_alphas.max(), 5)
#     cbar.set_ticks(tick_positions)
    
#     # Better formatting for the wide range of alpha values
#     alpha_values = [10**pos for pos in tick_positions]
#     formatted_labels = []
    
#     for alpha in alpha_values:
#         if alpha < 0.01:
#             formatted_labels.append(f'{alpha:.3f}')
#         elif alpha < 1:
#             formatted_labels.append(f'{alpha:.2f}')
#         elif alpha < 1000:
#             formatted_labels.append(f'{int(alpha)}')
#         else:
#             formatted_labels.append(f'{alpha:.1e}')
    
#     cbar.set_ticklabels(formatted_labels)
    
#     # Save figure
#     if save:
#         os.makedirs(save_path, exist_ok=True)
#         temp_path = os.path.normpath(save_path)
#         os.chdir(temp_path)
#         fig.savefig(f'session_{session}_subject_{subject}{config.figure_format}', 
#                     dpi=300, bbox_inches='tight')
#         os.chdir(current_working_directory)
#         plt.close(fig)

def gradient_fill_density_based(x, y_lower, y_upper, metric_random, fill_color, ax=None, N=256):
    if ax is None:
        ax = plt.gca()
    rgb = to_rgb(fill_color)

    for i in range(len(x) - 1):
        # Distribución en el tiempo i
        distribution = metric_random[i, :]

        # Estimación de densidad (KDE)
        kde = gaussian_kde(distribution)
        y_values = np.linspace(y_lower[i], y_upper[i], N)
        density = kde(y_values)
        density /= density.max()  # Normalización

        # Crear imagen RGBA con canal alfa según la densidad
        rgba = np.ones((N, 1, 4))
        rgba[..., :3] = rgb
        rgba[..., 3] = density.reshape(-1, 1)

        # Extensión del degradado en este intervalo de x
        extent = [x[i], x[i+1], y_lower[i], y_upper[i]]
        im = ax.imshow(rgba, aspect='auto', extent=extent, origin='lower', zorder=1)

        # Definir un Path cerrado con códigos adecuados
        verts = np.array([
            [x[i],         y_lower[i]],
            [x[i+1],       y_lower[i+1]],
            [x[i+1],       y_upper[i+1]],
            [x[i],         y_upper[i]],
            [x[i],         y_lower[i]]  # Cierre del polígono
        ])
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]
        clip_path = Path(verts, codes)
        patch = PathPatch(clip_path, facecolor='none', edgecolor='none', transform=ax.transData)
        # Forzamos un bbox a partir de los vértices
        bbox = Bbox.from_bounds(np.min(verts[:,0]), np.min(verts[:,1]),
                                np.ptp(verts[:,0]), np.ptp(verts[:,1]))
        patch.get_extents = lambda renderer=None: bbox

        # Marcar este patch como parte del degradado
        patch.gradient_patch = True

        ax.add_patch(patch)
        im.set_clip_path(patch)

    return im
# ##############################################################


# #TODO CHECK DESCRIPTION E Y LABEL
# def plot_tvalue_pvalue_tfce(tvalue:np.ndarray,
#                   pvalue:np.ndarray, 
#                   trf_subjects_shape:tuple,
#                   times:np.ndarray, 
#                   band:str, 
#                   stim:str,
#                   info:mne.Info,
#                   n_feats:list, 
#                   pval_tresh:float, 
#                   save_path:str, 
#                   display_interactive_mode:bool=False, 
#                   save:bool=True,
#                   no_figures:bool=False):

#     # Exit function
#     plt.close()
#     if no_figures:
#         return  

#     # Turn on/off interactive mode
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     # Make grid and time labels
#     n_subj, n_chan_feat, n_delays = trf_subjects_shape 
#     x, y = np.mgrid[0:n_chan_feat, 0:n_delays]
#     time_labels = np.arange(np.round(times[0],1)*1e3, np.round(times[-1],2)*1e3+1e2, 1e2, dtype=int)

#     # Create figure and title
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), layout='tight')
#     fig.suptitle(f'{stim}-{band}')
    
#     axes[0].remove()
#     axes[0] = fig.add_subplot(1,2,1,projection='3d')

#     # T-value surface plot (have in mind that there is a t-value for each channel/feature and delay)
#     surf = axes[0].plot_surface(x,
#                                 y,
#                                 tvalue,
#                                 rstride=1,
#                                 cstride=1,
#                                 linewidth=0,
#                                 cmap="viridis")
    
#     # Configure axis
#     if n_chan_feat==128:
#         axes[0].set(xticks=[0,127], 
#                     xticklabels=[info['ch_names'][0], info['ch_names'][-1]],
#                     yticks=np.linspace(0, n_delays, len(time_labels), dtype=int),
#                     yticklabels=time_labels, 
#                     zticks=[], 
#                     xlim=[0, n_chan_feat-1], 
#                     ylim=[0, n_delays-1],
#                     xlabel='Channels',
#                     ylabel='Time (ms)', 
#                     title='T-value after TFCE')
#         axes[1].set(xlabel='Time (ms)', 
#                     ylabel='Channels',
#                     yticks=[0,63,127], 
#                     yticklabels=[info['ch_names'][0], info['ch_names'][63], info['ch_names'][-1]],
#                     title='P-value')
#     else:
#         cumsum_feats = np.cumsum(n_feats)
#         ticks_per_feat = [[cumsum_feats[j-1] + i -1 if j!=0 else i-1 for i in range(1, cumsum_feats[j]+1)][::3] for j in range(len(cumsum_feats))]
#         ticks = np.concatenate(ticks_per_feat).tolist()
#         colors={}
#         for i in range(len(stim.split('_'))):
#             for j in ticks_per_feat[i]:
#                 colors[j]=matplotlib_colors[i]
#         axes[0].set(xticks=ticks, 
#                     yticks=np.linspace(0, n_delays, len(time_labels), dtype=int),
#                     yticklabels=time_labels, 
#                     zticks=[], 
#                     xlim=[0, n_chan_feat-1], 
#                     ylim=[0, n_delays-1],
#                     xlabel='Features',
#                     ylabel='Time (ms)', 
#                     title='T-value after TFCE')
#         [axes[0].get_xticklabels()[i].set_color(colors[ticks[i]]) for i in range(len(ticks))]

#         axes[1].set(xlabel='Times (ms)', 
#                     ylabel='Features',
#                     yticks=ticks, 
#                     title='P-value')
#         [axes[1].get_yticklabels()[i].set_color(colors[ticks[i]]) for i in range(len(ticks))]
#         if len(n_feats)!=1:
#             for i, st in enumerate(stim.split('_')):
#                 plt.figtext(x=.05, y=i*.025 +.025, s=f'{st}', color = matplotlib_colors[i], fontdict={'weight':'light'})
#             plt.figtext(x=.05, y=(i+1)*.025 +.025, s=f'Color code for features:', color = 'black', fontdict={'weight':'light'})

#     axes[0].view_init(30, 15)

#     # Make colorbar
#     plt.colorbar(ax=axes[0],
#                  shrink=0.5,
#                  orientation="vertical",
#                  label='T-value',
#                  mappable=surf)
    
#     # Make log transformation to p-value
#     if pval_tresh:
#         # Mask p-values over threshold to be a highly different order than thos value that pass the test
#         pvalue[pvalue > pval_tresh] = 1

#     # Plot it
#     im = axes[1].pcolormesh(times*1000, 
#                         np.arange(n_chan_feat), 
#                         -np.log10(np.maximum(pvalue, 1e-5)),
#                         cmap="inferno", 
#                         shading='auto')

#     # Make colorbar
#     plt.colorbar(ax=axes[1],
#                  shrink=.5,
#                  orientation="vertical",
#                  label=r"$-\log_{10}(p)$",
#                  mappable=im)

#     if display_interactive_mode:
#         text = fig.suptitle('TFCE')
#         text.set_weight("bold")
#         plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
#         mne.viz.utils.plt_show()
    
#     # Save figures
#     if save:
#         save_path += 'TFCE/'
#         os.makedirs(save_path, exist_ok=True)

#         # This is done to avoid working with long paths
#         temp_path = os.path.normpath(save_path)
#         os.chdir(temp_path)
#         fig.savefig(f'tvals_{band}_{stim}_{pval_tresh:.0e}{config.figure_format}')
#         os.chdir(current_working_directory)
        

# #TODO CHECK DESCRIPTION E Y LABEL
# def plot_trf_tfce(average_weights_subjects:np.ndarray, 
#                   p:np.ndarray, 
#                   times:np.ndarray,
#                   trf_subjects_shape:tuple,
#                   save_path:str, 
#                   band:str, 
#                   stim:str, 
#                   n_permutations:int,  
#                   pval_trhesh:float, 
#                   display_interactive_mode:bool=False, 
#                   save:bool=True):
    
#     # Turn on/off interactive mode
#     plt.close()
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     # Define relevant parameters
#     spectrogram_weights_bands = average_weights_subjects.mean(axis=0).mean(axis=0)
    
#     # Create figure and title
#     fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 6), layout='tight')
#     fig.suptitle(f'P-value for {stim} - {band}')

#     # Make colormesh
#     im = axs[0].pcolormesh(times * 1000, 
#                            np.arange(average_weights_subjects.shape[2]), 
#                            spectrogram_weights_bands, 
#                            cmap='RdBu_r',
#                            vmin=spectrogram_weights_bands.min(), 
#                            vmax=spectrogram_weights_bands.max(),
#                            shading='auto')
        
#     # Configure axis
#     bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
#     axs[0].set(xlabel='Time (ms)', 
#                ylabel='Frequency (Hz)',
#                xticks=np.arange(-100, 700, 100),
#                yticks=np.arange(0, 16, 2),
#                yticklabels=[int(bands_center[i]) for i in np.arange(0, 16, 2)])

#     # And colorbar
#     cbar = fig.colorbar(im, 
#                         ax=axs[0], 
#                         orientation='vertical', 
#                         label='mTRF Amplitude (a.u.)',
#                         shrink=0.7)

#     # Mask p-values over threshold
#     p[p>pval_trhesh] = 1

#     # Plot probabilities using a colormesh
#     use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (trf_subjects_shape[1], trf_subjects_shape[2])))
#     img = axs[1].pcolormesh(times * 1000, 
#                             np.arange(trf_subjects_shape[1]), 
#                             use_p, #np.flip(use_p, axis=0), 
#                             cmap="inferno", 
#                             shading='auto',
#                             vmin=use_p.min(), 
#                             vmax=use_p.max())

#     # Configure axis
#     axs[1].set(xlabel='Time (ms)')

#     # Plot color bar    
#     cbar = fig.colorbar(ax=axs[1], 
#                         orientation="horizontal", 
#                         label=r"$-\log_{10}(p)$",
#                         mappable=img, 
#                         shrink=0.7)
    
#     if display_interactive_mode:
#         text = fig.suptitle('')
#         text.set_weight("bold")
#         plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
#         mne.viz.utils.plt_show()

#     # Save figures
#     if save:
#         save_path += 'TFCE/'
#         os.makedirs(save_path, exist_ok=True)
#         plt.savefig(save_path + f'trf_tfce_{pval_trhesh}_{n_permutations}{config.figure_format}')

# def highlight_cell(x, y, ax=None, **kwargs):
#     rect = plt.Rectangle((x - .5, y - .5), 1, 1, **kwargs)
#     ax = ax or plt.gca()
#     ax.add_patch(rect)
#     return rect


# def corr_subject_decoding(session, subject, Valores_promedio, display_interactive_mode, name, Save, Run_graficos_path):
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     data = pd.DataFrame({name: Valores_promedio})
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     fig, ax = plt.subplots()
#     sns.violinplot(data=data, ax=ax)
#     plt.ylim([-0.2, 1])
#     plt.ylabel(name)
#     plt.title('{}:{:.3f} +/- {:.3f}'.format(name, np.mean(Valores_promedio), np.std(Valores_promedio), fontsize=19))

#     if Save:
#         save_path_cabezas = Run_graficos_path + 'Corr_subjects/'
#         try:
#             os.makedirs(save_path_cabezas)
#         except:
#             pass
#         fig.savefig(save_path_cabezas + '{}_Session{}_subject{}{config.figure_format}'.format(name, session, subject))



# def Plot_PSD(session, subject, Band, situacion, display_interactive_mode, Save, save_path, info, data, fmin=0, fmax=40):
#     psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(data, info['sfreq'], fmin, fmax)

#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     fig, ax = plt.subplots()
#     fig.suptitle('Session {} - subject {} - Situacion {} - Band {}'.format(session, subject, situacion, Band))

#     evoked = mne.EvokedArray(psds_welch_mean, info)
#     evoked.times = freqs_mean
#     evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s',
#                 show=False, spatial_colors=True, unit=False, units='w', axes=ax)
#     ax.set_xlabel('Frequency [Hz]')
#     ax.grid()

#     if Save:
#         save_path_graficos = 'gráficos/PSD/Zoom/{}/{}/'.format(save_path, Band)
#         os.makedirs(save_path_graficos, exist_ok=True)
#         plt.savefig(save_path_graficos + 'Session{} - subject{}{config.figure_format}'.format(session, subject, Band))


# def violin_plot_decoding(Correlaciones_totales_subjects, display_interactive_mode, Save, Run_graficos_path, title):

#     data = pd.DataFrame({title: Correlaciones_totales_subjects.ravel()})
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     fig, ax = plt.subplots()
#     sns.violinplot(data=data, ax=ax)
#     plt.ylim([-0.2, 1])
#     plt.ylabel(title)
#     plt.title('{}:{:.3f} +/- {:.3f}'.format(title, np.mean(Correlaciones_totales_subjects),
#                                                      np.std(Correlaciones_totales_subjects), fontsize=19))

#     if Save:
#         save_path_graficos = Run_graficos_path
#         os.makedirs(save_path_graficos, exist_ok=True)
#         fig.savefig(save_path_graficos + '{}_promedio{config.figure_format}'.format(title))

#     return Correlaciones_totales_subjects.mean(), Correlaciones_totales_subjects.std()


# def Cabezas_3d(Correlaciones_totales_subjects, info, display_interactive_mode, Save, Run_graficos_path, title):
#     Correlaciones_promedio = Correlaciones_totales_subjects.mean(0)

#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     sample_data_folder = mne.datasets.sample.data_path()
#     subjects_dir = os.path.join(sample_data_folder, 'subjects')
#     sample_data_trans_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                           'sample_audvis_raw-trans.fif')

#     evoked = mne.EvokedArray(np.array([Correlaciones_promedio,]).transpose(), info)
#     field_map = mne.make_field_map(evoked, trans=sample_data_trans_file,
#                                    subject='sample', subjects_dir=subjects_dir, ch_type='eeg',
#                                    meg_surf='head')

#     fig = evoked.plot_field(field_map, time=0)
#     xy, im = mne.viz.snapshot_brain_montage(fig, info)
#     # mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)
#     fig, ax = plt.subplots(figsize=(15, 10))
#     ax.set_title('Correlation', size='large')
#     ax.imshow(im)

#     if Save:
#         try:
#             os.makedirs(Run_graficos_path)
#         except:
#             pass
#         fig.savefig(Run_graficos_path + '{}{config.figure_format}'.format(title))

#     return Correlaciones_promedio.mean(), Correlaciones_promedio.std()


# def PSD_boxplot(psd_pred_correlations, psd_rand_correlations, display_interactive_mode, Save, Run_graficos_path):
#     psd_rand_correlations = funciones.flatten_list(psd_rand_correlations)

#     data = pd.DataFrame({'Prediction': psd_pred_correlations, 'Random': psd_rand_correlations})
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     fig, ax = plt.subplots()
#     sns.violinplot(data=data, ax=ax)
#     plt.ylim([-0.2, 1])
#     plt.ylabel('Correlation')
#     plt.title('Prediction Correlation:{:.2f} +/- {:.2f}\n'
#               'Random Correlation:{:.2f} +/- {:.2f}'.format(np.mean(psd_pred_correlations), np.std(psd_pred_correlations),
#                                                             np.mean(psd_rand_correlations), np.std(psd_rand_correlations)))
#     add_stat_annotation(ax, data=data, box_pairs=[(('Prediction'), ('Random'))],
#                         test='t-test_ind', text_format='full', loc='inside', verbose=2)

#     if Save:
#         save_path_graficos = Run_graficos_path
#         os.makedirs(save_path_graficos, exist_ok=True)
#         fig.savefig(save_path_graficos + 'PSD Boxplot{config.figure_format}')


# def weights_ERP(Pesos_totales_subjects_todos_canales, info, times, display_interactive_mode,
#                 Save, Run_graficos_path, Len_Estimulos, stim, decorrelation_times=None):
#     # Armo pesos promedio por canal de todos los subjects que por lo menos tuvieron un buen canal
#     Pesos_totales_subjects_todos_canales_copy = Pesos_totales_subjects_todos_canales.swapaxes(0, 2)
#     Pesos_totales_subjects_todos_canales_copy = Pesos_totales_subjects_todos_canales_copy.mean(0).transpose()

#     # Ploteo pesos y cabezas
#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     Stims_Order = stim.split('_')

#     Cant_Estimulos = len(Len_Estimulos)
#     for j in range(Cant_Estimulos):
#         Pesos_totales_subjects_todos_canales_copy[:, j * len(times):(j + 1) * len(times)].mean(0)

#         evoked = mne.EvokedArray(
#             np.flip(Pesos_totales_subjects_todos_canales_copy[:, j * len(times):(j + 1) * len(times)], axis=1), info)
#         evoked.shift_time(-times[0], relative=True)

#         fig, ax = plt.subplots(figsize=(15, 5))
#         fig.suptitle('{}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim), fontsize=23)
#         evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
#                     show=False, spatial_colors=True, unit=True, units='W', axes=ax)

#         ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
#         if times[0] < 0:
#             # ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-stimulus')
#             ax.axvline(x=0, ymin=0, ymax=1, color='grey')
#         if decorrelation_times:
#             # ax.vlines(-np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
#             #           color='red', label='Decorrelation time')
#             ax.axvspan(-np.mean(decorrelation_times), 0, alpha=0.4, color='red', label=' Mean decorrelation time')
#             # ax.axvspan(-np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
#             #            -np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
#             #            alpha=0.4, color='red', label='Decorrelation time std.')

#         ax.xaxis.label.set_size(23)
#         ax.yaxis.label.set_size(23)
#         ax.tick_params(axis='both', labelsize=23)
#         ax.grid()
#         ax.legend(fontsize=15, loc='lower right')

#         fig.tight_layout()

#         if Save:
#             os.makedirs(Run_graficos_path, exist_ok=True)
#             fig.savefig(
#             fig.savefig(
#                 Run_graficos_path + 'Regression_Weights_{}{config.figure_format}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))


# def decoding_t_lags(Correlaciones_totales_subjects, times, Band, display_interactive_mode, Save, Run_graficos_path):
#     Corr_time_sub = Correlaciones_totales_subjects.mean(0)
#     mean_time_corr = np.flip(Corr_time_sub.mean(1))
#     std_time_corr = np.flip(Corr_time_sub.std(1))

#     plot_times = -np.flip(times)

#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     # get max correlation t_lag
#     max_t_lag = np.argmax(mean_time_corr)

#     fig, ax = plt.subplots()
#     plt.plot(plot_times, mean_time_corr)
#     plt.title('{}'.format(Band))
#     plt.fill_between(plot_times, mean_time_corr - std_time_corr/2, mean_time_corr + std_time_corr/2, alpha=.5)
#     plt.vlines(plot_times[max_t_lag], ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed', color='k',
#                label='Max. correlation delay: {:.2f}s'.format(plot_times[max_t_lag]))
#     plt.xlabel('Time lag [s]')
#     plt.ylabel('Correlation')
#     ax.xaxis.label.set_size(15)
#     ax.yaxis.label.set_size(15)
#     ax.tick_params(axis='both', labelsize=15)
#     plt.grid()
#     plt.legend()

#     if Save:
#         os.makedirs(Run_graficos_path, exist_ok=True)
#         fig.savefig(Run_graficos_path + 'Correlation_time_lags_{}{config.figure_format}'.format(Band))


# def Brain_sync(data, Band, info, display_interactive_mode, Save, graficos_save_path, total_subjects=18, session=None, subject=None):

#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     if data.shape == (total_subjects, info['nchan'], info['nchan']):
#         data_ch = data.mean(0)
#     elif data.shape == (info['nchan'], info['nchan']):
#         data_ch = data

#     plt.figure(figsize=(10, 8))
#     plt.title('Inter Brain Phase Synchornization - {}'.format(Band), fontsize=14)
#     plt.imshow(data_ch)
#     plt.xticks(np.arange(0, info['nchan'], 4), labels=info['ch_names'][0:-1:4], rotation=45)
#     plt.yticks(np.arange(0, info['nchan'], 4), labels=info['ch_names'][0:-1:4])
#     plt.ylabel('Speaker', fontsize=13)
#     plt.xlabel('Listener', fontsize=13)
#     cbar = plt.colorbar()
#     cbar.ax.tick_params(labelsize=12)

#     if Save:
#         os.makedirs(graficos_save_path, exist_ok=True)
#         if data.shape == (total_subjects, info['nchan'], info['nchan']):
#             plt.savefig(graficos_save_path + 'Inter Brain sync - {}{config.figure_format}'.format(Band))
#         elif data.shape == (info['nchan'], info['nchan']):
#             plt.savefig(graficos_save_path + 'Inter Brain sync - Session{}_subject{}{config.figure_format}'.format(session, subject))



# def ch_heatmap_topo(total_data, info, delays, times, display_interactive_mode, Save, graficos_save_path, title, total_subjects=18,
#                     session=None, subject=None, fontsize=14):

#     if total_data.shape == (info['nchan'], len(delays)):
#         phase_sync_ch = total_data
#     elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
#         phase_sync_ch = total_data.mean(0)

#     if display_interactive_mode:
#         plt.ion()
#     else:
#         plt.ioff()

#     plt.rcParams.update({'font.size': fontsize})
#     fig, axs = plt.subplots(figsize=(9, 5), nrows=2, ncols=2, gridspec_kw={'width_ratios': [2, 1]})

#     # Remove axes of column 2
#     for ax_col in axs[:, 1]:
#         ax_col.remove()

#     # Add one axis in column
#     ax = fig.add_subplot(1, 3, (3, 3))

#     # Plot topo
#     phase_sync = phase_sync_ch.mean(0)
#     max_t_lag = np.argmax(phase_sync)
#     max_pahse_sync = phase_sync_ch[:, max_t_lag]

#     # ax.set_title('Mean = {:.3f} +/- {:.3f}'.format(max_pahse_sync.mean(), max_pahse_sync.std()))
#     im = mne.viz.plot_topomap(max_pahse_sync, info, cmap='Reds',
#                               vlim=(max_pahse_sync.min(),max_pahse_sync.max()),
#                               show=False, sphere=0.07, axes=ax)
#     cb = plt.colorbar(im[0], shrink=1, orientation='horizontal')
#     cb.set_label('r')


#     # Invert times for PLV plot
#     phase_sync_ch = np.flip(phase_sync_ch)
#     phase_sync_std = phase_sync_ch.std(0)
#     phase_sync = phase_sync_ch.mean(0)
#     max_t_lag = np.argmax(phase_sync)

#     times_plot = np.flip(-times)

#     im = axs[0, 0].pcolormesh(times_plot * 1000, np.arange(info['nchan']), phase_sync_ch, shading='auto')
#     axs[0, 0].set_ylabel('Channels')
#     axs[0, 0].set_xticks([])

#     cbar = plt.colorbar(im, orientation='vertical', ax=axs[0, 0])
#     cbar.set_label('PLV')

#     axs[1, 0].plot(times_plot * 1000, phase_sync)
#     axs[1, 0].fill_between(times_plot * 1000, phase_sync - phase_sync_std / 2, phase_sync + phase_sync_std / 2, alpha=.5)
#     # axs[1, 0].set_ylim([0, 0.2])
#     axs[1, 0].vlines(times_plot[max_t_lag] * 1000, axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], linestyle='dashed', color='k',
#                 label='Max: {}ms'.format(int(times_plot[max_t_lag] * 1000)))
#     axs[1, 0].set_xlabel('Time lag [ms]')
#     axs[1, 0].set_ylabel('Mean {}'.format(title))
#     # axs2.tick_params(axis='both', labelsize=12)
#     axs[1, 0].set_xlim([times_plot[0] * 1000, times_plot[-1] * 1000])
#     axs[1, 0].grid()
#     axs[1, 0].legend()

#     fig.tight_layout()

#     # Change axis 0 to match axis 1 width after adding colorbar
#     ax0_box = axs[0, 0].get_position().bounds
#     ax1_box = axs[1, 0].get_position().bounds
#     ax1_new_box = (ax1_box[0], ax1_box[1], ax0_box[2], ax1_box[3])
#     axs[1, 0].set_position(ax1_new_box)

#     if Save:
#         os.makedirs(graficos_save_path, exist_ok=True)
#         if total_data.shape == (info['nchan'], len(delays)):
#             plt.savefig(graficos_save_path + 't_lags_{}_Session{}_subject{}{config.figure_format}'.format(title, session, subject))
#         elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
#             plt.savefig(graficos_save_path + 't_lags_{}{config.figure_format}'.format(title))







# # ## VIEJAS NO SE USAN

# # def Plot_instantes_interes(Pesos_totales_subjects_todos_canales, info, Band, times, sr, display_interactive_mode_figure_instantes,
# #                            Save_figure_instantes, Run_graficos_path, Cant_Estimulos, Stims_Order, stim,
# #                            Autocorrelation_value=0.1):
# #     # Armo pesos promedio por canal de todos los subjects que por lo menos tuvieron un buen canal
# #     Pesos_totales_subjects_todos_canales_copy = Pesos_totales_subjects_todos_canales.swapaxes(0, 2)
# #     Pesos_totales_subjects_todos_canales_copy = Pesos_totales_subjects_todos_canales_copy.mean(0).transpose()

# #     # Ploteo pesos y cabezas
# #     if Display_figure_instantes:
# #         plt.ion()
# #     else:
# #         plt.ioff()

# #     returns = []
# #     for j in range(Cant_Estimulos):
# #         curva_pesos_totales = Pesos_totales_subjects_todos_canales_copy[:, j * len(times):(j + 1) * len(times)].mean(0)
# #         returns.append(curva_pesos_totales)

# #         if Autocorrelation_value and times[-1] > 0:
# #             weights_autocorr = funciones.correlacion(curva_pesos_totales, curva_pesos_totales)

# #             for i in range(len(weights_autocorr)):
# #                 if weights_autocorr[i] < Autocorrelation_value: break

# #                 dif_paso = weights_autocorr[i - 1] - weights_autocorr[i]
# #                 dif_01 = weights_autocorr[i - 1] - Autocorrelation_value
# #                 dif_time = dif_01 / sr / dif_paso
# #                 decorr_time = ((i - 1) / sr + dif_time) * 1000

# #             fig, ax = plt.subplots()
# #             plt.plot(np.arange(len(weights_autocorr)) * 1000 / sr, weights_autocorr)
# #             plt.title('Decorrelation time: {:.2f} ms'.format(decorr_time))
# #             plt.hlines(Autocorrelation_value, ax.get_xlim()[0], decorr_time, linestyle='dashed', color='black')
# #             plt.vlines(decorr_time, ax.get_ylim()[0], Autocorrelation_value, linestyle='dashed', color='black')
# #             plt.grid()
# #             plt.ylabel('Autocorrelation')
# #             plt.xlabel('Time [ms]')
# #             if Save_figure_instantes:
# #                 save_path_graficos = Run_graficos_path
# #                 try:
# #                     os.makedirs(save_path_graficos)
# #                 except:
# #                     pass
# #                 fig.savefig(save_path_graficos + 'Weights Autocorrelation{config.figure_format}')

# #         evoked = mne.EvokedArray(Pesos_totales_subjects_todos_canales_copy[:, j * len(times):(j + 1) * len(times)], info)
# #         evoked.shift_time(times[0], relative=True)

# #         instantes_index = sgn.find_peaks(np.abs(evoked._data.mean(0)), height=np.abs(evoked._data.mean(0)).max() * 0.4)[
# #             0]
# #         if not len(instantes_index): instantes_index = [np.abs(evoked._data.mean(0)).argmax()]
# #         instantes_de_interes = [i / sr + times[0] for i in instantes_index]  # if i/sr + times[0] < 0]

# #         fig = evoked.plot_joint(times=instantes_de_interes, show=False,
# #                                 ts_args=dict(unit='False', units=dict(eeg='$w$', grad='fT/cm', mag='fT'),
# #                                              scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms'),
# #                                 topomap_args=dict(vmin=evoked._data.min(),
# #                                                   vmax=evoked._data.max(),
# #                                                   time_unit='ms'))

# #         fig.suptitle('{}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))
# #         fig.set_size_inches(12, 7)
# #         axs = fig.axes
# #         axs[0].plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
# #         axs[0].axvspan(0, axs[0].get_xlim()[1], alpha=0.4, color='grey', label='Unheard stimuli')
# #         if Autocorrelation_value and times[-1] > 0: axs[0].vlines(decorr_time, axs[0].get_ylim()[0],
# #                                                                   axs[0].get_ylim()[1], linestyle='dashed', color='red',
# #                                                                   label='Decorrelation time')
# #         axs[0].xaxis.label.set_size(13)
# #         axs[0].yaxis.label.set_size(13)
# #         axs[0].grid()
# #         axs[0].legend(fontsize=13, loc='lower left')

# #         Blues = plt.cm.get_cmap('Blues').reversed()
# #         cmaps = ['Reds' if evoked._data.mean(0)[i] > 0 else Blues for i in instantes_index]

# #         for i in range(len(instantes_de_interes)):
# #             axs[i + 1].clear()
# #             axs[i + 1].set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)), fontsize=11)
# #             im = mne.viz.plot_topomap(evoked._data[:, instantes_index[i]], info, axes=axs[i + 1],
# #                                       show=False, sphere=0.07, cmap=cmaps[i],
# #                                       vmin=evoked._data[:, instantes_index[i]].min(),
# #                                       vmax=evoked._data[:, instantes_index[i]].max())
# #             plt.colorbar(im[0], ax=axs[i + 1], orientation='vertical', shrink=0.8,
# #                          boundaries=np.linspace(evoked._data[:, instantes_index[i]].min().round(decimals=2),
# #                                                 evoked._data[:, instantes_index[i]].max().round(decimals=2), 100),
# #                          ticks=np.linspace(evoked._data[:, instantes_index[i]].min(),
# #                                             evoked._data[:, instantes_index[i]].max(), 4).round(decimals=2))

# #         axs[i + 2].remove()
# #         axs[i + 4].remove()
# #         fig.tight_layout()

# #         if Save_figure_instantes:
# #             save_path_graficos = Run_graficos_path
# #             try:
# #                 os.makedirs(save_path_graficos)
# #             except:
# #                 pass
# #             fig.savefig(

# #     return returns


# # def Matriz_corr(Pesos_totales_subjects_promedio, Pesos_totales_subjects_todos_canales, subject_total, Display, Save,
# #                 Run_graficos_path):
# #     # Armo df para correlacionar
# #     Pesos_totales_subjects_promedio = Pesos_totales_subjects_promedio[:subject_total]
# #     Pesos_totales_subjects_promedio.append(
# #         Pesos_totales_subjects_todos_canales.transpose().mean(0).mean(1))  # agrego pesos promedio de todos los subjects
# #     lista_nombres = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
# #                      "Promedio"]
# #     Pesos_totales_subjects_df = pd.DataFrame(Pesos_totales_subjects_promedio).transpose()
# #     Pesos_totales_subjects_df.columns = lista_nombres[:len(Pesos_totales_subjects_df.columns) - 1] + [lista_nombres[-1]]

# #     pvals_matrix = Pesos_totales_subjects_df.corr(method=pearsonr_pval)
# #     Correlation_matrix = np.array(Pesos_totales_subjects_df.corr(method='pearson'))
# #     for i in range(len(Correlation_matrix)):
# #         Correlation_matrix[i, i] = Correlation_matrix[-1, i]

# #     Correlation_matrix = pd.DataFrame(Correlation_matrix[:-1, :-1])
# #     Correlation_matrix.columns = lista_nombres[:len(Correlation_matrix) - 1] + [lista_nombres[-1]]

# #     if Display:
# #         plt.ion()
# #     else:
# #         plt.ioff()

# #     mask = np.ones_like(Correlation_matrix)
# #     mask[np.tril_indices_from(mask)] = False

# #     fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 0.05]})
# #     fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize=26)
# #     sns.heatmap(abs(Correlation_matrix), mask=mask, cmap="coolwarm", fmt='.3', ax=ax,
# #                annot=True, center=0, xticklabels=True, annot_kws={"size": 19},
# #                cbar=False)

# #     ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(Correlation_matrix)], rotation='horizontal',
# #                        fontsize=19)
# #     ax.set_xticklabels(lista_nombres[:len(Correlation_matrix) - 1] + ['Mean of subjects'], rotation='horizontal',
# #                        ha='left', fontsize=19)

# #     sns.despine(right=True, left=True, bottom=True, top=True)
# #     fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
# #     cax.yaxis.set_tick_params(labelsize=20)

# #     fig.tight_layout()

# #     if Save:
# #         save_path_graficos = Run_graficos_path
# #         try:
# #             os.makedirs(save_path_graficos)
# #         except:
# #             pass
# #         fig.savefig(save_path_graficos + 'Correlation_matrix{config.figure_format}')


# # def Matriz_std_channel_wise(Pesos_totales_subjects_todos_canales, Display, Save, Run_graficos_path):
# #     Pesos_totales_subjects_todos_canales_average = np.dstack(
# #         (Pesos_totales_subjects_todos_canales, Pesos_totales_subjects_todos_canales.mean(2)))
# #     Correlation_matrices = np.zeros((Pesos_totales_subjects_todos_canales_average.shape[0],
# #                                      Pesos_totales_subjects_todos_canales_average.shape[2],
# #                                      Pesos_totales_subjects_todos_canales_average.shape[2]))
# #     for channel in range(len(Pesos_totales_subjects_todos_canales_average)):
# #         Correlation_matrices[channel] = np.array(
# #             pd.DataFrame(Pesos_totales_subjects_todos_canales_average[channel]).corr(method='pearson'))

# #     # std por subject
# #     std_matrix = Correlation_matrices.std(0)

# #     for i in range(len(std_matrix)):
# #         std_matrix[i, i] = std_matrix[-1, i]

# #     lista_nombres = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Promedio"]
# #     std_matrix = pd.DataFrame(std_matrix[:-1, :-1])
# #     std_matrix.columns = lista_nombres[:len(std_matrix) - 1] + [lista_nombres[-1]]

# #     if Display:
# #         plt.ion()
# #     else:
# #         plt.ioff()

# #     mask = np.ones_like(std_matrix)
# #     mask[np.tril_indices_from(mask)] = False

# #     fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 0.05]})
# #     fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize=26)
# #     sns.heatmap(abs(std_matrix), mask=mask, cmap="coolwarm", fmt='.3', ax=ax,
# #                annot=True, center=0, xticklabels=True, annot_kws={"size": 19},
# #                cbar=False)

# #     ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(std_matrix)], rotation='horizontal', fontsize=19)
# #     ax.set_xticklabels(lista_nombres[:len(std_matrix) - 1] + ['Mean of subjects'], rotation='horizontal', ha='left',
# #                        fontsize=19)

# #     sns.despine(right=True, left=True, bottom=True, top=True)
# #     fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
# #     cax.yaxis.set_tick_params(labelsize=20)

# #     fig.tight_layout()

# #     if Save:
# #         save_path_graficos = Run_graficos_path
# #         try:
# #             os.makedirs(save_path_graficos)
# #         except:
# #             pass
# #         fig.savefig(save_path_graficos + 'Channelwise_std_matrix{config.figure_format}')


# # def Cabezas_corr_promedio_scaled(Correlaciones_totales_subjects, info, Display, Save, Run_graficos_path, title):
# #     Correlaciones_promedio = Correlaciones_totales_subjects.mean(0)

# #     if Display:
# #         plt.ion()
# #     else:
# #         plt.ioff()

# #     fig = plt.figure()
# #     plt.suptitle("Mean {} per channel among subjects".format(title), fontsize=19)
# #     plt.title('{} = {:.3f} +/- {:.3f}'.format(title, Correlaciones_promedio.mean(), Correlaciones_promedio.std()),
# #               fontsize=19)
# #     ax = plt.subplot()
# #     im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap='Greys', vmin=0, vmax=0.41, show=Display, sphere=0.07, axes=ax)
# #     cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
# #     cb.ax.tick_params(labelsize=23)
# #     fig.tight_layout()

# #     if Save:
# #         save_path_graficos = Run_graficos_path
# #         os.makedirs(save_path_graficos, exist_ok=True)
# #         fig.savefig(save_path_graficos + '{}_promedio_sacled{config.figure_format}'.format(title))


# # def Plot_instantes_casera(Pesos_totales_subjects_todos_canales, info, Band, times, sr, Display_figure_instantes,
# #                           Save_figure_instantes, Run_graficos_path):
# #     # Armo pesos promedio por canal de todos los subjects que por lo menos tuvieron un buen canal
# #     Pesos_totales_subjects_todos_canales_copy = Pesos_totales_subjects_todos_canales.swapaxes(0, 2)
# #     Pesos_totales_subjects_todos_canales_copy = Pesos_totales_subjects_todos_canales_copy.mean(0)

# #     instantes_index = sgn.find_peaks(np.abs(Pesos_totales_subjects_todos_canales_copy.mean(1)[50:]),
# #                                 height=np.abs(Pesos_totales_subjects_todos_canales_copy.mean(1)).max() * 0.3)[0] + 50

# #     instantes_de_interes = [i/ sr + times[0] for i in instantes_index if i / sr + times[0] <= 0]

# #     # Ploteo pesos y cabezas
# #     if Display_figure_instantes:
# #         plt.ion()
# #     else:
# #         plt.ioff()

# #     Blues = plt.cm.get_cmap('Blues').reversed()
# #     cmaps = ['Reds' if Pesos_totales_subjects_todos_canales_copy.mean(1)[i] > 0 else Blues for i in instantes_index if
# #              i / sr + times[0] <= 0]

# #     fig, axs = plt.subplots(figsize=(10, 5), ncols=len(cmaps))
# #     fig.suptitle('Mean of $w$ among subjects - {} Band'.format(Band))
# #     for i in range(len(instantes_de_interes)):
# #         ax = axs[0, i]
# #         ax.set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)))
# #         fig.tight_layout()
# #         im = mne.viz.plot_topomap(Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].ravel(), info, axes=ax,
# #                                   show=False,
# #                                   sphere=0.07, cmap=cmaps[i],
# #                                   vmin=Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].min(),
# #                                   vmax=Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].max())
# #         plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.9,
# #                      boundaries=np.linspace(
# #                          Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].min().round(decimals=2),
# #                          Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].max().round(decimals=2), 100),
# #                      ticks=np.linspace(Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].min(),
# #                                         Pesos_totales_subjects_todos_canales_copy[instantes_index[i]].max(), 4).round(
# #                          decimals=2))

# #     axs[0, -1].remove()
# #     for ax_row in axs[1:]:
# #         for ax in ax_row:
# #             ax.remove()

# #     ax = fig.add_subplot(3, 1, (2, 3))
# #     evoked = mne.EvokedArray(Pesos_totales_subjects_todos_canales_copy.transpose(), info)
# #     evoked.shift_time(times[0], relative=True)


# #     evoked.plot(show=False, spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1),
# #                 unit=True, units=dict(eeg='$w$'), axes=ax, zorder='unsorted', selectable=False,
# #                 time_unit='ms')
# #     ax.plot(times * 1000, Pesos_totales_subjects_todos_canales_copy.mean(1),
# #             'k--', label='Mean', zorder=130, linewidth=2)

# #     ax.axvspan(0, ax.get_xlim()[1], alpha=0.5, color='grey')
# #     ax.set_title("")
# #     ax.xaxis.label.set_size(13)
# #     ax.yaxis.label.set_size(13)
# #     ax.grid()
# #     ax.legend(fontsize=13, loc='upper right')

# #     fig.tight_layout()

# #     if Save_figure_instantes:
# #         save_path_graficos = Run_graficos_path
# #         try:
# #             os.makedirs(save_path_graficos)
# #         except:
# #             pass
# #         fig.savefig(save_path_graficos + 'Instantes_interes{config.figure_format}')

# #     return Pesos_totales_subjects_todos_canales_copy.mean(1)




# # def plot_alphas(alphas, correlaciones, best_alpha_overall, lista_Rmse, linea, fino):
# #     # Plot correlations vs. alpha regularization value
# #     # cada linea es un canal
# #     fig = plt.figure(figsize=(10, 5))
# #     fig.clf()
# #     plt.subplot(1, 3, 1)
# #     plt.subplots_adjust(wspace=1)
# #     plt.plot(alphas, correlaciones, 'k')
# #     plt.gca().set_xscale('log')
# #     # en rojo: el maximo de las correlaciones
# #     # la linea azul marca el mejor alfa

# #     plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
# #     plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

# #     plt.plot(alphas, correlaciones.mean(1), '.r', linewidth=5)
# #     plt.xlabel('Alfa', fontsize=16)
# #     plt.ylabel('Correlación - Ridge set', fontsize=16)
# #     plt.tick_params(axis='both', which='major', labelsize=13)
# #     plt.tick_params(axis='both', which='minor', labelsize=13)

# #     # Como se ve sola la correlacion maxima para los distintos alfas
# #     plt.subplot(1, 3, 2)
# #     plt.plot(alphas, np.array(correlaciones).mean(1), '.r', linewidth=5)
# #     plt.plot(alphas, np.array(correlaciones).mean(1), '-r', linewidth=linea)

# #     if fino:
# #         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
# #         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

# #     plt.xlabel('Alfa', fontsize=16)
# #     plt.gca().set_xscale('log')
# #     plt.tick_params(axis='both', which='major', labelsize=13)
# #     plt.tick_params(axis='both', which='minor', labelsize=13)
# #     # el RMSE
# #     plt.subplot(1, 3, 3)
# #     plt.plot(alphas, np.array(lista_Rmse).min(1), '.r', linewidth=5)
# #     plt.plot(alphas, np.array(lista_Rmse).min(1), '-r', linewidth=2)

# #     if fino:
# #         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
# #         plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

# #     plt.xlabel('Alfa', fontsize=16)
# #     plt.ylabel('RMSE - Ridge set', fontsize=16)
# #     plt.gca().set_xscale('log')
# #     plt.tick_params(axis='both', which='major', labelsize=13)
# #     plt.tick_params(axis='both', which='minor', labelsize=13)

# #     titulo = "El mejor alfa es de: " + str(best_alpha_overall)
# #     plt.suptitle(titulo, fontsize=18)