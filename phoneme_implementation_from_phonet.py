# Standard libraries
import numpy as np
from scipy.io.wavfile import read

# Specific libraries
from phonet import Phonet
from scipy.signal import resample_poly

class Phoenemes(Phonet):
    def __init__(self, audio_file:str):
        super().__init__(phonological_classes='All')
        self.audio_file = audio_file
        
        # Modify parameters used to calculate Mfcc's inline with sample frequency of experiment
        self.sr = 128
        self.size_frame = 1/self.sr
        self.time_shift = 1/self.sr
    
    def compute_phonemes(self):
        # Read the audio (.wav) file
        fs, signal = read(self.audio_file)
        if fs!=16000:
            signal, fs = resample_poly(signal, 16000, fs), 16e3
        
        # This method extracts log-Mel-filterbank energies used as inputs of the model
        feat = self.get_feat(signal, fs)        

        nf = int(feat.shape[0]/self.len_seq) # len_seq=40 always

        # Get features
        features = []
        start, end = 0, self.len_seq
        for j in range(nf):
            features.append(feat[start:end,:])
            start += self.len_seq
            end += self.len_seq
        features = np.stack(features, axis=0)
        features = features-self.MU
        features = features/self.STD
        
        # Get phonemes and times
        pred_mat_phon = np.asarray(self.model_phon.predict(features))
        pred_mat_phon_seq = np.concatenate(pred_mat_phon, axis=0)
        pred_vec_phon = np.argmax(pred_mat_phon_seq, axis=1)

        nf=int(len(signal)/(self.time_shift*fs)-1)
        if nf>len(pred_vec_phon):
            nf=len(pred_vec_phon)
        
        phonemes_list = self.number2phoneme(pred_vec_phon[:nf])
        times = np.arange(nf)*self.time_shift
        return times, phonemes_list
    
if __name__=="__main__":
    import scipy.io.wavfile as wavfile
    from scipy import signal as sgn
    import processing
    wav_file = r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\wavs\S21\s21.objects.01.channel1.wav'
    
    # Read file
    wav = wavfile.read(wav_file)[1]
    wav = wav.astype("float")

    # Calculate envelope
    envelope = np.abs(sgn.hilbert(wav))
    
    # Apply lowpass butterworth filter
    envelope = processing.butter_filter(data=envelope, frecuencias=25, sampling_freq=16000,
                                            btype='lowpass', order=3, axis=0, ftype='Causal').reshape(-1,1)
    
    
    # Resample # TODO padear un cero en el envelope
    window_size, stride = int(16000/128), int(16000/128)
    envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
    envelope = envelope.reshape(-1, 1)
    # ================
    
    # Check if given kind is a permited input value
    allowed_kind = ['Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet']
    if kind not in allowed_kind:
        raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phonemes are: {allowed_kind}")

    phoneme_obj = Phoenemes(audio_file=wav_file)
    # phoneme_obj = Phoenemes(audio_file=self.wav_fname)
    time,  sec_phonemes = phoneme_obj.compute_phonemes() #9167
    sec_phonemes = [phon if phon!='<p:>' else '' for phon in sec_phonemes]

    # Match features length
    difference = len(sec_phonemes) - len(envelope)

    if difference > 0:
        sec_phonemes = sec_phonemes[:-difference]
    elif difference < 0:
        # In this case, silences are append
        for i in range(difference):
            sec_phonemes.append('')
    
    # Make a list with phoneme labels tha already are in the known set
    updated_taggs = np.unique(sec_phonemes).tolist()

    # Make empty array of phonemes
    phonemes = np.zeros(shape=(len(sec_phonemes), len(updated_taggs)))
    
    # Match phoneme with kind
    if kind.startswith('Phonemes-Envelope'):
        for i, tagg in enumerate(sec_phonemes):
            phonemes[i, updated_taggs.index(tagg)] = envelope[i]
    elif kind.startswith('Phonemes-Discrete'):
        for i, tagg in enumerate(sec_phonemes):
            phonemes[i, updated_taggs.index(tagg)] = 1
    elif kind.startswith('Phonemes-Onset'):
        # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
        phonemes_onset = [sec_phonemes[0]]
        for i in range(1, len(sec_phonemes)):
            if sec_phonemes[i] == sec_phonemes[i-1]:
                phonemes_onset.append(0)
            else:
                phonemes_onset.append(sec_phonemes[i])
        # Match phoneme with envelope
        for i, tagg in enumerate(phonemes_onset):
            if tagg!=0:
                phonemes[i, updated_taggs.index(tagg)] = 1
    return phonemes, updated_taggs

