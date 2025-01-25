# Standard libraries
import numpy as np
from scipy.io.wavfile import read

# Specific libraries
from phonet import Phonet
from scipy.signal import resample_poly

class Phones(Phonet):
    def __init__(
        self, 
        audio_file:str
        )->None:
        """
        Initialize the Phoenemes class.
        
        Parameters
        ----------
        audio_file : str
            The path to the audio file to be analyzed
            
        Returns
        -------
        None            
        """
        super().__init__(phonological_classes='All')
        self.audio_file = audio_file
        
        # Modify parameters used to calculate Mfcc's inline with sample frequency of experiment
        self.sr = 128
        self.size_frame = 1/self.sr
        self.time_shift = 1/self.sr
    
    def compute_phones(
        self
        )->tuple:
        """
        Compute phones from the audio file.
        
        Returns
        -------
        tuple
            A tuple containing the times and phones extracted from the audio file
        """
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
        
        # Get phones and times
        pred_mat_phon = np.asarray(self.model_phon.predict(features))
        pred_mat_phon_seq = np.concatenate(pred_mat_phon, axis=0)
        pred_vec_phon = np.argmax(pred_mat_phon_seq, axis=1)

        nf=int(len(signal)/(self.time_shift*fs)-1)
        if nf>len(pred_vec_phon):
            nf=len(pred_vec_phon)
        
        phones_list = self.number2phoneme(pred_vec_phon[:nf])
        times = np.arange(nf)*self.time_shift
        return times, phones_list
    
if __name__=="__main__":
    import scipy.io.wavfile as wavfile
    from scipy import signal as sgn
    import processing
    wav_file = r'Datos\wavs\S21\s21.objects.01.channel1.wav'
    # Read file
    wav = wavfile.read(wav_file)[1]
    wav = wav.astype("float")
    # Calculate envelope
    envelope = np.abs(sgn.hilbert(wav))
    # Apply lowpass butterworth filter
    envelope = processing.butter_filter(data=envelope, frequencies=25, sampling_freq=16000,
                                            btype='lowpass', order=3, axis=0, ftype='Causal').reshape(-1,1)
    # Resample # TODO padear un cero en el envelope
    window_size, stride = int(16000/128), int(16000/128)
    envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)])
    envelope = envelope.reshape(-1, 1)
    
    # ==================
    # Phoneme extraction
    from config import Exp_info
    
    # Remove silences, since it won't be used in prediction (when silence occurs, all phoneme are 0)
    phonet_labels = Exp_info().phonemes_phonet
    phonet_labels.remove('/sil/')

    # Check if given kind is a permited input value
    kind = 'Phonemes-Discrete-Phonet'
    allowed_kind = ['Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet']
    if kind not in allowed_kind:
        raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phonemes are: {allowed_kind}")

    phones_obj = Phones(audio_file=wav_file)
    time,  sec_phones = phones_obj.compute_phones() #9167
    
    # Match features length
    difference = len(sec_phones) - len(envelope)

    if difference > 0:
        sec_phones = sec_phones[:-difference]
    elif difference < 0:
        # In this case, silences are append
        for i in range(difference):
            sec_phones.append('<p:>')
    
    # Make empty array of phonemes
    phonemes = np.zeros(shape=(len(sec_phones), len(phonet_labels)))
    
    # Match phoneme with kind
    if kind.startswith('Phonemes-Envelope'):
        for i, tagg in enumerate(sec_phones):
            if (tagg!='<p:>') and (tagg!='sil'):
                phonemes[i, phonet_labels.index(Exp_info().phones_to_phonemes[tagg])] = envelope[i]
    elif kind.startswith('Phonemes-Discrete'):
        for i, tagg in enumerate(sec_phones):
            if (tagg!='<p:>') and (tagg!='sil'):
                phonemes[i, phonet_labels.index(Exp_info().phones_to_phonemes[tagg])] = 1
    elif kind.startswith('Phonemes-Onset'):
        # Makes a list giving only first ocurrences of phonemes (also ordered by sample) 
        phonemes_onset = [sec_phones[0]]
        for i in range(1, len(sec_phones)):
            if sec_phones[i] == sec_phones[i-1]:
                phonemes_onset.append(0)
            else:
                phonemes_onset.append(sec_phones[i])
        # Match phoneme with envelope
        for i, tagg in enumerate(phonemes_onset):
            if (tagg!='<p:>') and (tagg!='sil') and (tagg!=0):
                phonemes[i, phonet_labels.index(Exp_info().phones_to_phonemes[tagg])] = 1
    print(phonemes)

    # ==================
    # Phone extraction
    from config import Exp_info
    phonet_labels = Exp_info().ph_labels_phonet
    phonet_labels.remove('<p:>')
    phonet_labels.remove('sil')

    # Check if given kind is a permited input value
    kind = 'Phonemes-Discrete-Phonet'
    allowed_kind = ['Phonemes-Envelope-Phonet', 'Phonemes-Discrete-Phonet', 'Phonemes-Onset-Phonet']
    if kind not in allowed_kind:
        raise SyntaxError(f"{kind} is not an allowed kind of phoneme. Allowed phonemes are: {allowed_kind}")

    phones_obj = Phones(audio_file=wav_file)
    time,  sec_phones = phones_obj.compute_phones() #9167
    
    # Match features length
    difference = len(sec_phones) - len(envelope)

    if difference > 0:
        sec_phones = sec_phones[:-difference]
    elif difference < 0:
        # In this case, silences are append
        for i in range(difference):
            sec_phones.append('<p:>')
    
    # Make empty array of phonemes
    phones = np.zeros(shape=(len(sec_phones), len(phonet_labels)))
    
    # Match phoneme with kind
    if kind.startswith('Phonemes-Envelope'):
        for i, tagg in enumerate(sec_phones):
            if (tagg!='<p:>') and (tagg!='sil'):
                phones[i, phonet_labels.index(tagg)] = envelope[i]
    elif kind.startswith('Phonemes-Discrete'):
        for i, tagg in enumerate(sec_phones):
            if (tagg!='<p:>') and (tagg!='sil'):
                phones[i, phonet_labels.index(tagg)] = 1
    elif kind.startswith('Phonemes-Onset'):
        # Makes a list giving only first ocurrences of phones (also ordered by sample) 
        phones_onset = [sec_phones[0]]
        for i in range(1, len(sec_phones)):
            if sec_phones[i] == sec_phones[i-1]:
                phones_onset.append(0)
            else:
                phones_onset.append(sec_phones[i])
        # Match phoneme with envelope
        for i, tagg in enumerate(phones_onset):
            if (tagg!='<p:>') and (tagg!='sil') and (tagg!=0):
                phones[i, phonet_labels.index(tagg)] = 1
    print(phones)
