# Standard libraries
import numpy as np, os
from scipy.io.wavfile import read

# Specific libraries
from scipy import signal as sgn
from scipy.signal import resample_poly
   
def compute_phones(
    phonet_obj,
    audio_file:str,
    PLLR:bool=False,
    target_sr:int=128
    )->tuple:
    """
    Compute phones from the audio file.
    
    Parameters
    ----------
    PLLR : bool
        Whether to return the PLLR (Phoneme Loglikelihood ratio). By default, True
    
    Returns
    -------
    tuple
        A tuple containing the times and phones extracted from the audio file
    """
    # Read .wav file
    audio_sample_rate, signal = read(audio_file)
    assert audio_sample_rate==16000, f"Expected sampling frequency of 16000 Hz, but got {audio_sample_rate} Hz."
    
    # This method extracts log-Mel-filterbank energies used as inputs of the model. 
    # The output frequency is 100 Hz, which is the same as the time shift of the model.
    log_mel_filt_bank = phonet_obj.get_feat(signal, audio_sample_rate)
    
    # Calculate the number of frames represented in the signal
    number_of_frames = int(
        log_mel_filt_bank.shape[0]/phonet_obj.len_seq # len_seq=40 always
    ) 
    
    # Segment the mels into sequences of len_seq frames
    input_features = []
    start, end = 0, phonet_obj.len_seq
    for j in range(number_of_frames):
        input_features.append(log_mel_filt_bank[start:end,:])
        start += phonet_obj.len_seq
        end += phonet_obj.len_seq

    # Standarize the input features
    input_features = np.stack(input_features, axis=0)
    input_features = input_features-phonet_obj.MU
    input_features = input_features/phonet_obj.STD
    
    # Get the predictions from the model and concatenate them to get a sequence
    probabilities = np.asarray(
        phonet_obj.model_phon.predict(input_features)
        )
    posterior_gram = np.concatenate(
        probabilities, 
        axis=0
    )
    
    # time_shift is the time interval between frames
    total_audio_frames = int(len(signal)/(phonet_obj.time_shift*audio_sample_rate))
    posterior_gram = posterior_gram[:total_audio_frames]
    
    # posterior_prob: (num_frames, num_phones), original_fs ≈ 100 Hz
    num_target_frames = int((signal.shape[0] / audio_sample_rate) * target_sr)
    posterior_gram = sgn.resample(
        posterior_gram, 
        num_target_frames, 
        axis=0
    )
    
    # sample_rate_posteriors = len(posterior_prob)/(wav.shape[0]/self.audio_sr)
    # import resampy

    # posterior_prob2 = resampy.resample(
    #     posterior_prob, 
    #     sr_orig=sample_rate_posteriors, 
    #     sr_new=config.sr, 
    #     axis=0
    # )
    
    if PLLR:
        return posterior_gram
    else:
        greedy_prediction = np.argmax(
            posterior_gram, 
            axis=1
        )
        phone_sequence = [
            str(phonet_obj.phonemes[j])
            for j in greedy_prediction
        ]
        return phone_sequence
        
if __name__=="__main__":
    import scipy.io.wavfile as wavfile
    from scipy import signal as sgn
    from utils.processing import  butter_filter
    
    wav_file = r'data\wavs\S21\s21.objects.01.channel1.wav'
    wav = wavfile.read(wav_file)[1]
    wav = wav.astype("float")
    # Calculate envelope
    envelope = np.abs(sgn.hilbert(wav))
    window_size, stride = 125, 125
    envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
    envelope = envelope.reshape(-1, 1)
    
    from phonet.phonet import Phonet
    from config import exp_info
    Phonet_obj = Phonet('All')
    
    posterior_prob = compute_phones(Phonet_obj, wav_file, PLLR=True)
    
    from scipy.signal import resample

    # posterior_prob: (num_frames, num_phones), original_fs ≈ 100 Hz
    original_fs = posterior_prob.shape[0] / (wav.shape[0] / 16000)  # ≈100 Hz
    target_fs = 128
    num_target_frames = int((wav.shape[0] / 16000) * target_fs)

    posterior_prob = resample(posterior_prob, num_target_frames, axis=0)
    
    difference = len(posterior_prob) - len(envelope)

    if difference > 0:
        posterior_prob = posterior_prob[:-difference]
    elif difference < 0:
        # Repeat last sample (probably silence)
        for i in range(np.abs(difference)):
            aux = posterior_prob[-1].copy() 
            posterior_prob = np.vstack((posterior_prob, aux.reshape(-1,1).T))
    
    # Map phones to phonemes, making the sum
    posterior_prob_phonemes = np.zeros(shape=(posterior_prob.shape[0], len(exp_info.phonemes)))

    for h, phone in enumerate(exp_info.phones):
        phoneme_index = exp_info.phonemes.index(exp_info.phones_to_phonemes[phone])
        posterior_prob_phonemes[:, phoneme_index] += posterior_prob[:, h]
        
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    plt.figure(figsize=(15, 6))
    im = plt.imshow(
        posterior_prob_phonemes[:, :-1].T,
        # posterior_prob.T,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        cmap='viridis'
    )
    plt.colorbar(im, label='Posterior Probability')
    plt.yticks(
        ticks=np.arange(len(exp_info.phonemes)-1),
        labels=exp_info.phonemes[:-1]
        # ticks=np.arange(len(exp_info.phones)),
        # labels=exp_info.phones
    )
    plt.xlabel('Frame')
    plt.ylabel('Phoneme')
    plt.title('Phoneme Posteriorgram')
    plt.tight_layout()

    # Add secondary x-axis for time in seconds
    ax = plt.gca()
    def frame_to_time(x):
        # envelope_fs is approx 128 Hz (see above)
        return x / 128
    def time_to_frame(t):
        return t * 128
    secax = ax.secondary_xaxis('top', functions=(frame_to_time, time_to_frame))
    secax.set_xlabel('Time (s)')

    plt.show()


    
    # # Calculate posterior llr
    # pllr = np.zeros(shape=posterior_prob_phonemes.shape)
    # number_of_phonemes = posterior_prob_phonemes.shape[1]
    # for ph in range(number_of_phonemes):
    #     pllr[:, ph] = np.log10(posterior_prob_phonemes[:, ph]/(1-posterior_prob_phonemes[:, ph]))
    
    # # Centralizamos 
    # pllr = pllr - np.mean(pllr, axis=1, keepdims=True)  
    
    # # Removemos silencios
    # pllr_without_silence = pllr[:, np.arange(number_of_phonemes) != exp_info.phonemes.index('/sil/')]
    
    
    
    # time, phones = compute_phones(Phonet_obj, wav_file, PLLR=True)
    # phoenemes = [exp_info.phones_to_phonemes[ph] if ph not in ['<p:>', 'sil'] else '*' for ph in phones]
    # # list_phones = np.unique([str(ph) for ph in phones if ph not in ['<p:>', 'sil']])

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')

    # # Define duration to plot (in seconds)
    # duration_sec = 6

    # # Calculate the number of envelope samples for the first 10 seconds
    # envelope_fs = 128  # Since stride=125 and fs=16000, approx 128 frames/sec
    # num_samples = int(duration_sec * envelope_fs)

    # # Plot the envelope for the first 10 seconds
    # plt.figure(figsize=(15, 4))
    # plt.plot(
    #     np.arange(num_samples) / envelope_fs,
    #     envelope[:num_samples],
    #     label='Envelope'
    # )
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Audio Envelope with Phone Transcription (First 10 seconds)')

    # # Overlay phone boundaries and labels for the first 10 seconds
    # for t, ph in zip(time, phoenemes):
    #     if t > duration_sec:
    #         break
    #     plt.axvline(x=t, color='r', linestyle='--', alpha=0.1)
    #     plt.text(t, np.max(envelope[:num_samples])*0.8, ph, rotation=0, verticalalignment='bottom', fontsize=12)

    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    
    
    

    # # === BEAM SEARCH A NIVEL PALABRA ===
    # import json
    # # Cargar modelo de lenguaje de palabras (2gram por simplicidad)
    # lm_path = os.path.join('data', 'language_model', 'word_5gram_lm.json')
    # with open(lm_path, 'r', encoding='utf-8') as f:
    #     lm = json.load(f)

    # # Cargar diccionario de pronunciación desde archivo JSON
    # pronunciation_dict_path = os.path.join('data', 'language_model', 'pronunciation_dict.json')
    # with open(pronunciation_dict_path, 'r', encoding='utf-8') as f:
    #     pronunciation_dict = json.load(f)
    # # Agregar palabra especial 'sil' para permitir silencios explícitos
    # pronunciation_dict['sil'] = ['/sil/']

    # # Parámetros de búsqueda
    # beam_width = 5
    # alpha = .5  # Peso LM
    # T = posterior_prob_phonemes.shape[0]
    # phonemes = exp_info.phonemes
    # sil_idx = phonemes.index('/sil/')

    # # Cada hipótesis es (score, pos, palabras, fon_idx_en_palabra, palabra_actual)
    # from heapq import nlargest
    # initial = (0.0, 0, [], 0, None)  # score, pos, palabras, fon_idx, palabra_actual
    # beam = [initial]

    # max_steps = 2*T  # Límite de iteraciones para evitar loops infinitos
    # step = 0
    # while beam:
    #     new_beam = []
    #     for score, pos, words, fon_idx, palabra_actual in beam:
    #         print("Pos:", pos)
    #         if pos >= T:
    #             new_beam.append((score, pos, words, fon_idx, palabra_actual))
    #             continue
    #         # Si estamos en medio de una palabra, seguimos avanzando en su pronunciación
    #         if palabra_actual is not None and fon_idx < len(pronunciation_dict[palabra_actual]):
    #             ph = pronunciation_dict[palabra_actual][fon_idx]
    #             try:
    #                 ph_idx = phonemes.index(ph)
    #             except ValueError:
    #                 print(f"[WARN] Fonema '{ph}' no está en la lista de fonemas. Palabra: {palabra_actual}")
    #                 continue
    #             new_score = score + np.log(posterior_prob_phonemes[pos, ph_idx] + 1e-12)
    #             new_beam.append((new_score, pos+1, words, fon_idx+1, palabra_actual))
    #         else:
    #             # Probar todas las palabras posibles que pueden empezar aquí
    #             prev_word = words[-1] if words else '<s>'
    #             prob_threshold = 0.05  # Umbral para el primer fonema
    #             for w, pron in pronunciation_dict.items():
    #                 if len(pron) == 0 or pos+len(pron) > T:
    #                     continue
    #                 # Permitir silencios explícitos en cualquier punto
    #                 if w == 'sil':
    #                     ph_idx = sil_idx
    #                     acoustic_score = np.log(posterior_prob_phonemes[pos, ph_idx] + 1e-12)
    #                     total_score = score + acoustic_score
    #                     new_beam.append((total_score, pos+1, words+['sil'], 0, None))
    #                     continue
    #                 # Filtrar por probabilidad del primer fonema
    #                 try:
    #                     first_ph = pron[0]
    #                     first_ph_idx = phonemes.index(first_ph)
    #                 except ValueError:
    #                     continue
    #                 if posterior_prob_phonemes[pos, first_ph_idx] < prob_threshold:
    #                     continue
    #                 # Score acústico para la palabra
    #                 acoustic_score = 0.0
    #                 skip_word = False
    #                 for k, ph in enumerate(pron):
    #                     try:
    #                         ph_idx = phonemes.index(ph)
    #                     except ValueError:
    #                         skip_word = True
    #                         break
    #                     acoustic_score += np.log(posterior_prob_phonemes[pos+k, ph_idx] + 1e-12)
    #                 if skip_word:
    #                     continue
    #                 # Score LM
    #                 lm_prob = lm.get(prev_word, {}).get(w, 1e-8)
    #                 total_score = score + acoustic_score + alpha * np.log(lm_prob)
    #                 new_beam.append((total_score, pos+len(pron), words+[w], 0, None))
    #     if not new_beam:
    #         print(f"[INFO] Beam vacío en step {step}. Se detiene la búsqueda.")
    #         break
    #     beam = nlargest(beam_width, new_beam, key=lambda x: x[0])
    #     if all(pos >= T for _, pos, _, _, _ in beam):
    #         print(f"[INFO] Todas las hipótesis llegaron al final en step {step}.")
    #         break
    #     step += 1
    #     if step > max_steps:
    #         print(f"[INFO] Límite de pasos alcanzado ({max_steps}). Se detiene la búsqueda.")
    #         break

    # # Secuencia greedy para comparar
    # greedy_idx = np.argmax(posterior_prob_phonemes, axis=1)
    # greedy_phonemes = [phonemes[i] for i in greedy_idx]
    # print('Secuencia greedy de fonemas:')
    # print(''.join(greedy_phonemes))

    # # Elegir la mejor secuencia de palabras
    # best_hyp = max(beam, key=lambda x: x[0])
    # best_words = best_hyp[2]
    # # Remove 'sil' tokens
    # best_words = [w for w in best_words if w != "sil"]

    # # If two consecutive words are the same, unify them (keep only one)
    # unified_words = []
    # for w in best_words:
    #     if not unified_words or unified_words[-1] != w:
    #         unified_words.append(w)
    # best_words = unified_words
    
    
    
    # print('Secuencia de palabras más probable:', ' '.join(best_words))

    # # Visualización (opcional, muestra los fonemas de la mejor secuencia de palabras)
    # best_phonemes = []
    # for w in best_words:
    #     best_phonemes.extend(pronunciation_dict[w])
    # best_phonemes = [ph.replace('/sil/', '-') for ph in best_phonemes]