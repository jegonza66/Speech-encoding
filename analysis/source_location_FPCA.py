import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import mne

from skfda.preprocessing.dim_reduction import FPCA
from sklearn.cluster import KMeans
import skfda

from utils.general_functions import load_pickle
import config

STIMULUS, BAND = 'Envelope', 'Broad'

TRF_PATH = Path(rf"output\mtrf-ridge\External-External\weights\stims_Standarize_EEG_Standarize\same_alpha\tmin-0.2_tmax0.6\{BAND}\{STIMULUS}\total_weights_per_subject.pkl")
FIGURE_SAVE_DIR = Path(rf"figures\source_location\FPCA\{BAND}_{STIMULUS}")
FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
INFO_MNE = config.info_mne
TIMES = config.times
NUMBER_OF_COMPONENTS = 2 # # Cuántas variaciones principales retener. Seleccionar en base a la varianza explicada
EXPECTED_ROIS = 4
RANDOM_SEED = 42

trf_data = load_pickle(
    path=TRF_PATH
)['average_weights_subjects']

def identify_rois(
    trf_data: np.ndarray, 
    times: np.ndarray,
    n_components: int=3,
    n_clusters: int=3
) -> dict:
    """
    Identify Region of Interests (ROIs) in EEG/MEG data based on the
    functional shape of the response using Functional Principal Component Analysis (FPCA)
    
    Parameters:
    -----------
        trf_data: np.ndarray
            3D array of shape (n_subjects, n_channels, n_times) or (n_subjects, n_channels, n_features, n_times) containing the TRF data.
        times: np.ndarray
            Time points corresponding to the functional data.
        n_clusters: int
            Number of functional ROIs to find.
        n_components: int
            Number of variation modes to retain for clustering.
    Returns:
    --------
        dict
            A dictionary with the following keys:
            - "labels": Cluster labels for each channel.
            - "scores": Coordinates of each channel in the latent space.
            - "fpca": FPCA object for plotting eigenfunctions.
            - "time": Time points corresponding to the functional data.
            - "data_matrix": The data matrix used for FPCA (channels x time points).
    """
    
    # Dimension cleaning
    if trf_data.ndim == 4:
        trf_data = trf_data.mean(axis=2) 
        n_suj, n_chans, n_times = trf_data.shape
    
    # Grand average trf across subjects to get population-level response
    X_matrix = ga_trf = np.mean(trf_data, axis=0) # (n_chans, n_times)

    # FPCA data structure
    fd = skfda.FDataGrid(
        data_matrix=X_matrix, 
        grid_points=times
    )
    
    # FPCA transformation
    fpca = FPCA(
        n_components=n_components
    )
    scores = fpca.fit_transform( # (n_chans, n_components)
        fd
    ) 
    
    var_ratio = fpca.explained_variance_ratio_.sum()

    # Group channels with similar functional profiles using KMeans
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=RANDOM_SEED, 
        n_init=10
    )
    labels = kmeans.fit_predict(scores)
    
    return {
        "labels": labels,       # The cluster ID for each channel (0 to 127)
        "scores": scores,       # Coordinates of each channel in latent space
        "fpca": fpca,           # Object for plotting eigenfunctions
        "time": times,          # Time points corresponding to the functional data
        "design_matrix": X_matrix, # The data matrix used for FPCA (channels x time points)
        "explained_variance_ratio": var_ratio
    }

# Apply the ROI identification
results = identify_rois(
    trf_data=trf_data, 
    times=TIMES,
    n_clusters=EXPECTED_ROIS,
    n_components=NUMBER_OF_COMPONENTS   
)
design_matrix = results['design_matrix'] # (128, 104)
scores = results['scores']
labels = results['labels']
time = results['time']*1e3 # ms
fpca = results['fpca']
explained_variance_ratio = results['explained_variance_ratio']

# Visualization of results: latent space, average responses, eigenfunctions
fig = plt.figure(
    figsize=(14, 4), 
    tight_layout=True,
    # constrained_layout=True,
)
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Plot scores and clusters -> this shows how channels group based on functional shape
ax1.scatter(
    scores[:, 0], # First FPCA component
    scores[:, 1], # Second FPCA component
    scores[:, 2], # Third FPCA component
    c=labels, 
    cmap='viridis', 
    s=50, 
    alpha=0.8, 
    edgecolor='k'
)
ax1.view_init(elev=0, azim=70, roll=0)
ax1.set_title("Latent Space (Functional PCA)")
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax1.set_xlabel(r"$1^{st}$ Score")
ax1.set_ylabel(r"$2^{nd}$ Score")
ax1.set_zlabel(r"$3^{rd}$ Score")
ax1.grid(True, alpha=0.3)

# Plot average response per cluster/ROI
for cluster_id in range(EXPECTED_ROIS):
    # Average of all channels belonging to this cluster
    mask = labels == cluster_id
    cluster_mean = design_matrix[mask].mean(axis=0)
    std_err = design_matrix[mask].std(axis=0) / np.sqrt(mask.sum())
    
    ax2.plot(time, cluster_mean, lw=3, label=f'ROI {cluster_id+1}', color=f'C{cluster_id}')
    ax2.fill_between(time, cluster_mean - std_err, cluster_mean + std_err, alpha=0.2, color=f'C{cluster_id}')

ax2.set_title("Average Response per ROI")
ax2.set_ylabel("Amplitude (a.u.)")
ax2.set_xlabel("Time (ms)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot the FPCA eigenfunctions
# FPC1 is usually the mean, FPC2 is usually the derivative (latency shift)
fpca_comps = results['fpca'].components_.data_matrix.squeeze()
for c, component in enumerate(fpca_comps):
    ax3.plot(time, component, label=f'FP{c+1}', color=f'C{c}')#, ls=['-', '--', ':'][c])
ax3.set_ylabel("Amplitude (a.u.)")
ax3.set_xlabel("Time (ms)")
ax3.set_title("Eigenfunctions")
ax3.legend()
ax3.grid(True, alpha=0.3)
fig.savefig(
    FIGURE_SAVE_DIR / f"FPCA_latent-space_responses_eigenfunctions.png",
    dpi=400
)
# fig.show()


# Topographic plots of the FPCA scores 
fig, axes = plt.subplots(
    nrows=1,
    ncols=NUMBER_OF_COMPONENTS, 
    figsize=(12, 5)
)
for c in range(NUMBER_OF_COMPONENTS):
    im, _ = mne.viz.plot_topomap(
        data=scores[:, c], 
        pos=INFO_MNE, 
        axes=axes[c], 
        show=False, 
        cmap='RdBu_r', # Rojo=Positivo, Azul=Negativo
        contours=0,
        extrapolate='head'
)
    axes[c].set_title(rf"${c+1}^{{st}} Score \rightarrow$  FPC{c+1}")
    plt.colorbar(
        im, 
        ax=axes[c], 
        orientation='horizontal',
    )
# fig.show()
fig.savefig(
    FIGURE_SAVE_DIR / f"FPCA_topomaps.png",
    dpi=400
)

print(
    f"\nExplained variance ratio by first {NUMBER_OF_COMPONENTS} components: {explained_variance_ratio:.2%}.",
    f"\nFigures saved in {FIGURE_SAVE_DIR}"
)