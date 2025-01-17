import torch, numpy as np, scipy.io.wavfile as wavfile

import warnings
import time
from tqdm import tqdm
from scipy import signal as sgn
from sklearn.decomposition import PCA

from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config

wac2vec2model = "facebook/wav2vec2-large-xlsr-53-distilled"
# wac2vec2model = "facebook/wav2vec2-base"
# Cargar modelo y procesador
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Passing `gradient_checkpointing` to a config initialization is deprecated")
    processor = Wav2Vec2Processor.from_pretrained(wac2vec2model, cache_dir=f'saves/preprocessed_Data/wav2vec2_weights_{wac2vec2model.split("wav2vec2-")[1]}')
    model = Wav2Vec2Model.from_pretrained(wac2vec2model, cache_dir=f'saves/preprocessed_Data/wav2vec2_weights_{wac2vec2model.split("wav2vec2-")[1]}')

# Simulación de audio y envolvente
wav = wavfile.read(r'Datos\wavs\S21\s21.objects.02.channel1.wav')[1]
wav = wav.astype("float")

# Calculate envelope
window_size, stride = 125,125
envelope = np.abs(sgn.hilbert(wav))
envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size<=len(envelope)]).reshape(-1,1)

# Preprocesamiento
input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values

# Pasar por el modelo
with torch.no_grad():
    ini = time.time()
    # Crear una barra de progreso
    outputs = model(input_values, output_hidden_states=True)
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
).squeeze(0).T

# Aplicar PCA para reducir la dimensionalidad a 12
pca = PCA(n_components=12)
reduced_hidden_states = pca.fit_transform(hidden_states_resampled)
np.sum(pca.explained_variance_ratio_)