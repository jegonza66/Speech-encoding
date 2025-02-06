import torch, numpy as np, scipy.io.wavfile as wavfile, time, warnings
from sklearn.decomposition import PCA
from scipy import signal as sgn
from tqdm import tqdm

# from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import WhisperProcessor, WhisperModel


wav2vec2model = "openai/whisper-tiny"
# wav2vec2model = "facebook/wav2vec2-large-xlsr-53-distilled"
# wav2vec2model = "facebook/wav2vec2-base"

modelfname = f'wav2vec2_weights_{wav2vec2model.split("wav2vec2-")[1]}' if 'wav2vec2' in wav2vec2model else f'whisper_weights_{wav2vec2model.split("whisper-")[1]}'

# Cargar modelo y procesador
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Passing `gradient_checkpointing` to a config initialization is deprecated")
    # processor = Wav2Vec2Processor.from_pretrained(wav2vec2model, cache_dir=f'saves/preprocessed_Data/{modelfname}')
    # model = Wav2Vec2Model.from_pretrained(wav2vec2model, cache_dir=f'saves/preprocessed_Data/{modelfname}')
    processor = WhisperProcessor.from_pretrained(wav2vec2model, cache_dir=f'saves/preprocessed_Data/{modelfname}')
    model = WhisperModel.from_pretrained(wav2vec2model, cache_dir=f'saves/preprocessed_Data/{modelfname}')

# Simulación de audio y envolvente
wav = wavfile.read(r'Datos\wavs\S21\s21.objects.02.channel1.wav')[1]
wav = wav.astype("float")

# Calculate envelope
window_size, stride = 125,125
envelope = np.abs(sgn.hilbert(wav))
envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)]).reshape(-1,1)

# Preprocesamiento
# input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_features

# Pasar por el modelo
# with torch.no_grad():
#     ini = time.time()
#     # Crear una barra de progreso
#     outputs = model(input_values, output_hidden_states=True)
#     hidden_states = outputs.hidden_states[-1]  # Última capa oculta
#     # Actualizar la barra de progreso
#     print(time.time()-ini)
ini = time.time()
# Crear una barra de progreso
outputs = model.encoder(input_values, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]  # Última capa oculta
# Actualizar la barra de progreso
print(time.time()-ini)
    
# Ajustar dimensiones
hidden_states = hidden_states.squeeze(0)  # Quitar batch dimension
hidden_states_resampled = torch.nn.functional.interpolate(
    hidden_states.T.unsqueeze(0),
    size=envelope.shape[0],  # Igualar al largo de la envolvente
    mode="linear",
    align_corners=True
).squeeze(0).T.detach().numpy()

# Aplicar PCA para reducir la dimensionalidad a 12
pca = PCA(n_components=12)
reduced_hidden_states = pca.fit_transform(hidden_states_resampled)
np.sum(pca.explained_variance_ratio_)




import scipy.io.wavfile as wavfile, os, numpy as np, pandas as pd, librosa, tqdm
import config

total_length = 0

dataset_uba = {'audio':[], 'transcription':[]}

#TODO INCLUIR LAS SESIONES DESCARTADAS
for sesion in config.sesiones:
    wav_folder = f'Datos/wavs/S{sesion}'
    phrases_folder = f'Datos/phrases/S{sesion}'
    for wav_file, phrases_file in zip(os.listdir(wav_folder), os.listdir(phrases_folder)):
        # Phrasees
        transcription = ' '.join(pd.read_csv(os.path.join(phrases_folder, phrases_file), sep='\t', header=None)[2].values).replace('# ','').replace('#', '')
        if transcription.startswith(' '):
            transcription = transcription[1:]
        elif transcription.endswith(' '):
            transcription = transcription[:-1]
        dataset_uba['audio'].append(wav_file)
        dataset_uba['transcription'].append(transcription)
        
        # Audio
        sr, audio = wavfile.read(os.path.join(wav_folder, wav_file))
        total_length += len(audio.astype('float'))/sr

dataset_uba = pd.DataFrame(data=dataset_uba)
print(total_length/3600)        


metadata = pd.read_csv('Datos/validated.tsv', sep='\t')
metadata_rioplatense = metadata[metadata['accent']=='rioplatense']
metadata_rioplatense.shape[0]/metadata.shape[0]

total_database_length = 0
for i, item in tqdm.tqdm(metadata_rioplatense.iterrows()):
    file, sentence = item['path'], item['sentence']
    audio, sr = librosa.load(os.path.join('Datos/clips', file))
    total_database_length += len(audio.astype('float'))/sr

import pydub    

def convert_mp3_to_wav(input_dir, output_dir, sample_rate=16000):
    os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta de salida si no existe

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp3"):
            # Leer archivo .mp3
            mp3_path = os.path.join(input_dir, file_name)
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            
            # Convertir a mono y 16 kHz
            audio = audio.set_frame_rate(sample_rate).set_channels(1)
            
            # Guardar como .wav
            wav_path = os.path.join(output_dir, file_name.replace(".mp3", ".wav"))
            audio.export(wav_path, format="wav")
            print(f"Convertido: {mp3_path} -> {wav_path}")

# Directorios de entrada y salida
input_directory = "dataset/"  # Carpeta con los archivos .mp3
output_directory = "dataset_wav/"  # Carpeta para guardar los archivos .wav

convert_mp3_to_wav(input_directory, output_directory)

print(total_database_length/3600)        


import soundfile as sf
import audioread, os
os.path.join('Datos/clips', file)
# Leer archivo .mp3 y convertir a WAV
with audioread.audio_open(os.path.join('Datos/clips', file)) as f:
    data = f.read_data()
    with open("Datos/PRUEBA.wav", "wb") as out_f:
        out_f.write(data)