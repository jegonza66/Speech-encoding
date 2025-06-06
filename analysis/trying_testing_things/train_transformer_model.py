import os, time, warnings, tqdm
import numpy as np, pandas as pd
import librosa

from scipy import signal as sgn
from scipy.io import wavfile 

from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
from sklearn.decomposition import PCA
import torch

import config

# ===========
# FINE-TUNING
from datasets import Dataset, Audio

class WhisperWithFeatureHead(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.feature_head = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, 16),
            torch.nn.LayerNorm(16),
            torch.nn.ReLU()
        )
        
    def forward(self, input_features, decoder_input_ids, labels=None):
        # Forward original del encoder
        encoder_outputs = self.model.encoder(input_features=input_features)
        
        # Nuestros features de 16 dimensiones
        self.last_16d_features = self.feature_head(encoder_outputs.last_hidden_state)
        
        # Forward original del decoder
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state
        )
        
        # Calcular pérdida normalmente
        lm_logits = self.proj_out(decoder_outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {'loss': loss, 'logits': lm_logits}
    
    def get_16d_features(self, input_features):
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features=input_features)
            return self.feature_head(encoder_outputs.last_hidden_state)

# Configuración inicial
model_name = "openai/whisper-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained(model_name, language="spanish", task="transcribe")

# Preparación de datos de ejemplo (debes adaptar esto a tu dataset)
def prepare_dataset(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        truncation=True
    )
    
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Cargar dataset (adaptar a tu estructura de datos)
dataset = Dataset.from_dict({"audio": ["audio1.wav", "audio2.wav"], "text": ["transcripción 1", "transcripción 2"]})
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(prepare_dataset)

# Función de entrenamiento modificada
def train_feature_model(model, dataset, epochs=3, batch_size=2):
    model.train().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-5},
        {'params': model.feature_head.parameters(), 'lr': 1e-4}
    ])
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            inputs = batch["input_features"].to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            
            # Crear decoder inputs (shift right)
            decoder_input_ids = torch.cat([
                torch.tensor([[model.config.decoder_start_token_id]] * labels.shape[0]),
                labels[:, :-1]
            ], dim=-1).to(device)
            
            outputs = model(inputs, decoder_input_ids, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}")

# Uso del modelo
model = WhisperWithFeatureHead.from_pretrained(model_name)
train_feature_model(model, dataset)

# Ejemplo de extracción de features
audio_sample = ...  # Cargar tu audio aquí
inputs = processor(audio_sample, sampling_rate=16000, return_tensors="pt").input_features.to(device)
features_16d = model.get_16d_features(inputs)
print(f"Features obtenidos: {features_16d.shape}")  # Debería ser [1, 1500, 16] (dependiendo de la duración del audio)


# # =======================
# # METADATA BASES DE DATOS
# total_length = 0

# dataset_uba = {'audio':[], 'transcription':[]}

# #TODO INCLUIR LAS SESIONES DESCARTADAS
# for sesion in config.sesiones:
#     wav_folder = f'Datos/wavs/S{sesion}'
#     phrases_folder = f'Datos/phrases/S{sesion}'
#     for wav_file, phrases_file in zip(os.listdir(wav_folder), os.listdir(phrases_folder)):
#         # Phrasees
#         transcription = ' '.join(pd.read_csv(os.path.join(phrases_folder, phrases_file), sep='\t', header=None)[2].values).replace('# ','').replace('#', '')
#         if transcription.startswith(' '):
#             transcription = transcription[1:]
#         elif transcription.endswith(' '):
#             transcription = transcription[:-1]
#         dataset_uba['audio'].append(wav_file)
#         dataset_uba['transcription'].append(transcription)
        
#         # Audio
#         sr, audio = wavfile.read(os.path.join(wav_folder, wav_file))
#         total_length += len(audio.astype('float'))/sr

# dataset_uba = pd.DataFrame(data=dataset_uba)
# print(total_length/3600)        

corpus_path = 'Datos/datos_fine_tune/cv-corpus-20.0-2024-12-06/es/'

tsv_files = [f for f in os.listdir(corpus_path) if f.endswith('.tsv')]


metadata = pd.read_csv(os.path.join(corpus_path, 'validated.tsv'), sep='\t', dtype=str)
metadata_rioplatense = metadata[metadata['accents']=='rioplatense']
metadata_rioplatense.shape[0]/metadata.shape[0]

filtered_acc = [acc for acc in metadata['accents'].unique() if ('Argentina' in str(acc)) or ('Uruguay' in str(acc))]
metadata_rioplatense = metadata.copy()
metadata_rioplatense[metadata_rioplatense["accents"].isin(filtered_acc)]

total_database_length = 0
items_not_found = []
for i, item in tqdm.tqdm(metadata_rioplatense.iterrows()):
    try:
        file, sentence = item['path'], item['sentence']
        audio, sr = librosa.load(os.path.join(corpus_path, 'clips', file))
        total_database_length += len(audio.astype('float'))/sr
    except FileNotFoundError as err:
        print(err)
        items_not_found.append(item)

print(total_database_length/3600)        
# =====================
# CONVERTIR .mp3 a .wav
import os
from pydub import AudioSegment
# from pydub.utils import which

# Configurar las variables de entorno directamente
# os.environ["FFMPEG_BINARY"] = r"C:\repos\Speech-encoding\ffmpeg-7.1-full_build\bin\ffmpeg.exe"
# os.environ["FFPROBE_BINARY"] = r"C:\repos\Speech-encoding\ffmpeg-7.1-full_build\bin\ffprobe.exe"
# os.environ["PATH"] += os.pathsep + r"C:\repos\Speech-encoding\ffmpeg-7.1-full_build\bin"
os.environ["FFMPEG_BINARY"] = r"C:\Users\User\Downloads\programas_descargados_por_octavio\ffmpeg-7.1-full_build\bin\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = r"C:\Users\User\Downloads\programas_descargados_por_octavio\ffmpeg-7.1-full_build\bin\ffprobe.exe"
os.environ["PATH"] += os.pathsep + r"C:\Users\User\Downloads\programas_descargados_por_octavio\ffmpeg-7.1-full_build\bin"

# # Confirmar que pydub encuentra los binarios
# print(f"FFmpeg encontrado en: {which('ffmpeg')}")
# print(f"FFprobe encontrado en: {which('ffprobe')}")

# # Cargar el archivo MP3
# audio = AudioSegment.from_file(r"C:\Users\jocta\Downloads\prueba.mp3", format="mp3")

audio = AudioSegment.from_file(r"C:/Users/User/Downloads/prueba.mp3", format="mp3")
audio = audio.set_frame_rate(int(16e3)).set_channels(1)
# audio.export(r"C:\Users\jocta\Downloads\prueba.wav", format="wav")
audio.export(r"C:/Users/User/Downloads/prueba.wav", format="wav")


# # Exportar el archivo a formato WAV
# audio.export(r"C:\Users\jocta\Downloads\prueba.wav", format="wav")

def convert_mp3_to_wav(input_dir, output_dir, sample_rate=16000):
    os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta de salida si no existe

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp3"):
            try:
                # Leer archivo .mp3
                mp3_path = os.path.join(input_dir, file_name)
                audio = AudioSegment.from_file(mp3_path, format="mp3")
                
                # Convertir a mono y 16 kHz
                audio = audio.set_frame_rate(sample_rate).set_channels(1)
                
                # Guardar como .wav
                wav_path = os.path.join(output_dir, file_name.replace(".mp3", ".wav"))
                audio.export(wav_path, format="wav")
                print(f"Convertido: {mp3_path} -> {wav_path}")
            except Exception as err:
                print(f'No se pudo convertir {file_name} debido al siguiente error:\n', err)

# Directorios de entrada y salida
input_directory = "dataset/"  # Carpeta con los archivos .mp3
output_directory = "dataset_wav/"  # Carpeta para guardar los archivos .wav

convert_mp3_to_wav(input_directory, output_directory)