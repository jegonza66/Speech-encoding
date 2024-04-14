import pickle, os, matplotlib.pyplot as plt, numpy as np, pandas as pd, textgrids, time
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
import mne, Processing
from Load import Trial_channel, Sesion_class
def labeling(trial:int, channel:int):
    """Gives an array with speaking channel: 
        3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)

    Parameters
    ----------
    trial : int
        _description_
    channel : int
        _description_

    Returns
    -------
    np.ndarray
        Speaking channels by sample, matching EEG sample rate and almost matching its length
    """
    
    # Read phrases into pandas.DataFrame
    ubi_speaker = r'C:\repos\Speech-encoding\Datos\Datos\phrases\S21'+ f'/s21.objects.{trial:02d}.channel{channel}.phrases'
    h1t = pd.read_table(ubi_speaker, header=None, sep="\t")

    # Replace text by 1, silence by 0 and '#' by ''
    h1t.iloc[:, 2] = (h1t.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
    
    # Take difference in time and multiply it by sample rate in order to match envelope length (almost, miss by a sample or two)
    samples = np.round((h1t[1] - h1t[0]) * 128).astype("int")
    speaker = np.repeat(h1t.iloc[:, 2], samples).ravel()

    # Same with listener
    listener_channel = (channel - 3) * -1
    ubi_listener = r'C:\repos\Speech-encoding\Datos\Datos\phrases\S21'+ f'/s21.objects.{trial:02d}.channel{listener_channel}.phrases'
    h2t = pd.read_table(ubi_listener, header=None, sep="\t")

    # Replace text by 1, silence by 0 and '#' by ''
    h2t.iloc[:, 2] = (h2t.iloc[:, 2].replace("#", "").apply(len) > 0).apply(int)
    samples = np.round((h2t[1] - h2t[0]) * 128).astype("int")
    listener = np.repeat(h2t.iloc[:, 2], samples).ravel()

    # If there are difference in length, corrects with 0-padding
    diff = len(speaker) - len(listener)
    if diff > 0:
        listener = np.concatenate([listener, np.repeat(0, diff)])
    elif diff < 0:
        speaker = np.concatenate([speaker, np.repeat(0, np.abs(diff))])

    # Return an array with envelope length, with values 3 if both speak; 2, just listener; 1, speaker and 0 silence 
    return speaker + listener * 2
phr_path= r'C:\repos\Speech-encoding\Datos\Datos\phrases\S21'

Sujeto_1={}

for tr in list(set([int(fname.split('.')[2]) for fname in os.listdir(phr_path) if os.path.isfile(phr_path + f'/{fname}')])):
    channel = Trial_channel(s=21,trial=tr,channel=1,Band='Theta', sr=128,tmin=-0.6,tmax=0.2, SilenceThreshold=0.03)
    Trial_channel_1 = channel.load_trial(stims=['Spectrogram'])
    current_speaker_1 = labeling(trial=tr, channel=2)
    Trial_sujeto_1, current_speaker_1, minimo_largo1 = Sesion_class.match_lengths(dic=Trial_channel_1, speaker_labels=current_speaker_1)
    for key in Trial_sujeto_1:
        if key != 'info':
            if key not in Sujeto_1:
                Sujeto_1[key] = Trial_sujeto_1[key]
            else:
                if key=='EEG':
                    Sujeto_1[key] = mne.concatenate_raws(raws=[Sujeto_1[key], Trial_sujeto_1[key]])
                else:
                    Sujeto_1[key].append([Trial_sujeto_1[key]]) 

key='Spectrogram'
print(f'Computing shifted matrix for the {key}')
data_1 = Sujeto_1[key].get_data()
t_0=time.time()
shift_1 = np.zeros(shape=(data_1.shape[1], data_1.shape[0]*(len(channel.delays))))
for i in range(len(data_1)):
    shift_1_i = Processing.shifted_matrix(feature=data_1[i], delays=channel.delays)
    shift_1[:,len(channel.delays)*i:len(channel.delays)*(i+1)] = shift_1_i

# Return to mne arrays
shift_1_ch_names, shift_2_ch_names = [], []
for delay in channel.delays:
    shift_1_ch_names += [ch + f'_delay_{delay}' for ch in Sujeto_1[key].ch_names]
shift_1_info = mne.create_info(ch_names=shift_1_ch_names, sfreq=Sujeto_1[key].info.get('sfreq'), ch_types='misc')
Sujeto_1[key] = mne.io.RawArray(data=shift_1.T, info=shift_1_info)
print(f'{key} matrix computed')
print(time.time()-t_0)




# sesion=21
# wav_path = f'Datos\wav\S{sesion}'
# eeg_path = f'Datos\EEG\S{sesion}'
# number_of_trials = int(len([file for file in os.listdir(wav_path) if os.path.isfile(os.path.join(wav_path, file))])/2)

# # for trial in range(number_of_trials):
# #     for channel in [1,2]:
# #         "Datos/EEG/S" + str(sesion) + "/s" + str(sesion) + "-" + str(channel) + "-Trial" + str(trial) + "-Deci-Filter-Trim-ICA-Pruned.set"

# eeg_path="Datos/EEG/S" + str(sesion) + "/s" + str(sesion) + "-" + str(1) + "-Trial" + str(1) + "-Deci-Filter-Trim-ICA-Pruned.set"

# eeg = mne.io.read_raw_eeglab(input_fname=eeg_path, verbose=False) # TODO: warning of annotations and 'boundry' events -data discontinuities-.
# eeg.load_data(verbose=False)
# eeg = eeg.filter(l_freq=0, h_freq=4, verbose=False)
# eeg.resample(sfreq=128)

# eeg_times = eeg.times.tolist() 
# minimum = 399
# ee_c=eeg.copy().crop(tmin=eeg_times[0], tmax=eeg_times[minimum], verbose=True)
# ee_c.get_data().shape
# eeg.get_data().T.shape
# Processing.subsamplear(eeg.get_data().T, int(eeg.info.get('sfreq') / 128)).shape
# # eeg_resample = eeg.copy().resample(sfreq=128)
# period = eeg.times[-1]-eeg.times[0]

# # Create array for audio envelope signal
# wav = wavfile.read(r'Datos/wavs/S21/s21.objects.01.channel1.wav')[1]
# wav = wav.astype("float")

# envelope_1 = np.abs(sgn.hilbert(wav))
# envelope_1 = Processing.butter_filter(data=envelope_1, frecuencias=25, sampling_freq=16000,
#                                         btype='lowpass', order=3, axis=0, ftype='NonCausal').reshape(-1,1)


# info_envelope = mne.create_info(ch_names = ['envelope_ch1'], sfreq=16000, ch_types='misc')

# envelope_mne_array = mne.io.RawArray(data = envelope_1.T, info=info_envelope,verbose=True)

# envelope_mne_array.resample(sfreq=128)
# env = envelope_mne_array.get_data().T
# env2= np.array([np.mean(envelope_1[:,0][i:i + int(16000/128)]) for i in range(0, len(envelope_1[:,0]), int(16000/128)) if i + int(16000/128) <= len(envelope_1[:,0])])

# plt.figure()
# plt.plot(np.arange(0,env.shape[0]/128, 1/128), env[:,0])
# plt.plot(np.arange(0,env2.shape[0]/128, 1/128), env2[:], color='red')
# plt.show()

# wav2 = wavfile.read(r'C:\repos\Speech-encoding\Datos\Datos\wavs\S21\s21.objects.02.channel1.wav')[1]
# wav2 = wav2.astype("float")

# # Calculate envelope
# envelope2 = np.abs(sgn.hilbert(wav2))
# envelope2 = Processing.butter_filter(data=envelope2, frecuencias=25, sampling_freq=16000,
#                                         btype='lowpass', order=3, axis=0, ftype='NonCausal').reshape(-1,1)
    
# # Creates mne raw array
# info_envelope2 = mne.create_info(ch_names=['envelope'], sfreq=16000, ch_types='misc')
# envelope2_mne_array = mne.io.RawArray(data=envelope2.T, info=info_envelope2, verbose=False)

# # Resample to match EEG data # TODO no matchea en dimension con EEG (tan solo por dos samples). tal vez agregar silencio
# envelope_mne_array2= envelope2_mne_array.resample(sfreq=128)
# envelope_2 = envelope_mne_array2.get_data().ravel()

# phrases_2 = pd.read_table(r'C:\repos\Speech-encoding\Datos\Datos\phrases\S21\s21.objects.02.channel1.phrases', header=None, sep="\t")
# trial_tmax_2 = phrases_2[1].iloc[-1]

# # Load transcription
# grid_2 = textgrids.TextGrid(r'C:\repos\Speech-encoding\Datos\Datos\phonemes\S21\s21.objects.02.channel1.aligned_fa.TextGrid')

# # Get phonemes
# phonemes_grid_2 = grid_2['transcription : phones']

# # Extend first silence time to trial start time
# phonemes_grid_2[0].xmin = 0.

# # Parse for labels, times and number of samples within each phoneme
# labels_2 = []
# times_2 = []
# samples_2 = []

# for ph_2 in phonemes_grid_2:
#     label_2 = ph_2.text.transcode()
    
#     # Rename silences
#     if label_2 == 'sil' or label_2 == 'sp':
#         label_2 = ""
    
#     # Check if the phoneme is in the list
#     if not(label_2 in exp_info.ph_labels or label_2==""):
#         print(f'"{label_2}" is not in not a recognized phoneme. Will be added anyways.')
#     labels_2.append(label_2)
#     times_2.append((ph_2.xmin, ph_2.xmax))
#     samples_2.append(np.round((ph_2.xmax - ph_2.xmin) * 128).astype("int"))

# # Extend on more phoneme of silence till end of trial 
# labels_2.append("")
# times_2.append((ph_2.xmin, trial_tmax_2))
# samples_2.append(np.round((trial_tmax_2 - ph_2.xmax) * 128).astype("int"))

# # If use envelope amplitude to make continuous stimuli: the total number of samples must match the samples use for stimuli
# diferencia_2 = np.sum(samples_2) - len(envelope_2)

# if diferencia_2 > 0:
#     # Making the way back checking when does the number of samples of the ith phoneme exceed diferencia
#     for ith_phoneme in [-i-1 for i in range(len(samples_2))]:
#         if diferencia_2 > samples_2[ith_phoneme]:
#             diferencia_2 -= samples_2[ith_phoneme]
#             samples_2[ith_phoneme] = 0
#         # When samples is greater than the difference, takes the remaining samples to match the envelope_2
#         else:
#             samples_2[ith_phoneme] -= diferencia_2
#             break
# elif diferencia_2 < 0:
#     # In this case, the last silence is prolonged
#     samples_2[-1] -= diferencia_2

# # Make a list with phoneme labels tha already are in the known set
# updated_taggs_2 = exp_info.ph_labels + [ph for ph in np.unique(labels_2) if ph not in exp_info.ph_labels]

# # Repeat each label the number of times it was sampled
# phonemes_tgrid_2 = np.repeat(labels_2, samples_2)

# # # Make empty df of phonemes #TODO: change to dictionary, much faster
# # phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=updated_taggs)
# # # Match phoneme with envelope_2. Note that new phonemes are added to the list: "" and unrecognized phonemes.
# # for i, phoneme in enumerate(phonemes_tgrid):
# #     phonemes_df.loc[i, phoneme] = envelope_2[i]

# # Make empty array of phonemes
# phonemes_2 = np.zeros(shape = (np.sum(samples_2), len(updated_taggs_2)))

# # Match phoneme with envelope_2
# for i, tagg_2 in enumerate(phonemes_tgrid_2):
#     phonemes_2[i, updated_taggs_2.index(tagg_2)] = envelope_2[i]

# # Creates mne raw array
# info_phonemes_2 = mne.create_info(ch_names=updated_taggs_2, sfreq=128, ch_types='misc')

# phonemes_mne_array_2 = mne.io.RawArray(data=phonemes_2.T, info=info_phonemes_2, verbose=False)

# phonemes_mne_array

# super_ra = phonemes_mne_array.append(raws=[phonemes_mne_array_2])
# phonemes_mne_array.get_data().shape[1]+phonemes_mne_array_2.get_data().shape[1]
# super_ra.info
# get_data().shape

# (phonemes_mne_array.get_data()==super_ra.get_data()).all()