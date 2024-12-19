import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import epitran

# Load Spanish language processor and model (Argentinian tuned)
processor = Wav2Vec2Processor.from_pretrained("common_voice/wav2vec2-large-xlsr-53-es")
model = Wav2Vec2ForCTC.from_pretrained("common_voice/wav2vec2-large-xlsr-53-es")
epi = epitran.Epitran("spa-Latn")  # Epitran for Spanish (Latin alphabet)

# Custom Argentinian-specific IPA adjustments
def apply_argentinian_phonetics(ipa_text):
    """
    Adjust IPA transcription for Argentinian Spanish (Porteño).
    - 'll' and 'y' -> [ʃ] or [ʒ] depending on speaker.
    """
    ipa_text = ipa_text.replace("ʝ", "ʒ")  # Replace default [ʝ] with [ʒ]
    ipa_text = ipa_text.replace("ʎ", "ʃ")  # Replace default [ʎ] with [ʃ]
    return ipa_text

def load_audio(file_path):
    """Load and preprocess the audio."""
    audio, rate = librosa.load(file_path, sr=16000)
    return audio, rate

def recognize_speech(audio):
    """Transcribe speech to text."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def text_to_ipa(text):
    """Convert text to IPA phonemes."""
    ipa_transcription = [epi.transliterate(word) for word in text.split()]
    ipa_transcription = " ".join(ipa_transcription)
    return apply_argentinian_phonetics(ipa_transcription)

if __name__ == "__main__":
    # Input .wav file
    file_path = "your_audio_file.wav"

    # Step 1: Load audio
    audio, _ = load_audio(file_path)

    # Step 2: Speech-to-text
    transcription = recognize_speech(audio)
    print("Transcription:", transcription)

    # Step 3: Text-to-IPA
    ipa_transcription = text_to_ipa(transcription)
    print("IPA Transcription:", ipa_transcription)