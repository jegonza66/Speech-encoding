# Standard libraries
import pandas as pd, numpy as np
from scipy.io.wavfile import read

# Specific libraries
from phonet import Phonet
import python_speech_features as pyfeat
from scipy.signal import resample_poly
import gc

#IDEA HEREDAR CLASE DE PHONET QUE PERMITA CALCULAR LOS FONEMAS UTILIZADOS COMO VARIABLES TEMPORALES

class Phoenemes(Phonet):
    def __init__(self, audio_file:str):
        super().__init__(phonological_classes='All')
        self.audio_file = audio_file
    
    def compute_phonemes(self):

        # Read the audio (.wav) file
        fs, signal=read(self.audio_file)
        if fs!=16000:
            signal, fs =resample_poly(signal, 16000, fs), 16e3
        
        # This method extracts log-Mel-filterbank energies used as inputs of the model
        feat = self.get_feat(signal,fs)        

        nf=int(feat.shape[0]/self.len_seq) # len_seq=40 always

        start, end =0, self.len_seq
        Feat = []
        for j in range(nf):
            featmat_t=feat[start:end,:]
            Feat.append(featmat_t)
            start += self.len_seq
            end += self.len_seq
        
        Feat = np.stack(Feat, axis=0)
        Feat = Feat-self.MU
        Feat = Feat/self.STD
        df={}
        dfa={}
        pred_mat_phon=np.asarray(self.model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)

        nf=int(len(signal)/(self.time_shift*fs)-1)
        if nf>len(pred_vec_phon):
            nf=len(pred_vec_phon)
        
        phonemes_list=self.number2phoneme(pred_vec_phon[:nf])

        t2=np.arange(nf)*self.time_shift
        return t2, phonemes_list
    
if __name__=="__main__":
    wav_file = r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\wavs\S21\s21.objects.01.channel1.wav'
    fs, signal = read(wav_file)
    phoeneme_obj = Phoenemes(audio_file=wav_file)
    time,  phonemes = phoeneme_obj.compute_phonemes()
    clean_phonemes = []

    for ph in phonemes:
        if ph != '<p:>':
            clean_phonemes.append(ph)
        else:
            pass
    even_cleaner_phonemes = [v for i, v in enumerate(clean_phonemes) if i == 0 or v != clean_phonemes[i-1]]
    
    # Get older vesion to compare
    import textgrids
    import setup
    exp_info = setup.exp_info()
    grid = textgrids.TextGrid(r'C:\repos\Speech-encoding\repo_speech_encoding\Datos\phonemes\S21\s21.objects.01.channel1.aligned_fa.TextGrid')
    phonemes_grid = grid['transcription : phones']
    labels = []

    for ph in phonemes_grid:
        label = ph.text.transcode()
        label = label.replace(' ', '')
        label = label.replace('º', '')
        label = label.replace('-', '')

        # Rename silences
        if label in ['sil','sp','sile','silsil','SP','s¡p','sils']:
            label = ""
        
        # Check if the phoneme is in the list
        if not(label in exp_info.ph_labels_man or label==""):
            print(f'"{label}" is not in not a recognized phoneme. Will be added as silence.')
            label = ""
        labels.append(label)
    clean_labels = []
    for ph in labels:
        if ph != '':
            clean_labels.append(ph)
        else:
            pass

    clean_labels
    even_cleaner_phonemes
