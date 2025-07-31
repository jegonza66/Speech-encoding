import os
import glob
import json
from collections import defaultdict, Counter

# Diccionario de pronunciación ejemplo (expande con tu corpus real)
def build_pronunciation_dict(word_sequences):
    # Diccionario base (puedes expandirlo manualmente si tienes reglas fonéticas)
    base_dict = {
        'el': ['/e/', '/l/'],
        'mimo': ['/m/', '/i/', '/m/', '/o/'],
        'arriba': ['/a/', '/r/', '/i/', '/b/', '/a/'],
        'del': ['/d/', '/e/', '/l/'],
        'búho': ['/b/', '/u/', '/o/'],
        'a': ['/a/'],
        'la': ['/l/', '/a/'],
        'derecha': ['/d/', '/e/', '/r/', '/e/', '/tS/', '/a/'],
        'de': ['/d/', '/e/'],
        'oreja': ['/o/', '/r/', '/e/', '/x/', '/a/'],
        'con': ['/k/', '/o/', '/n/'],
    }
    # Palabras únicas del corpus
    all_words = set(w for seq in word_sequences for w in seq)
    pronunciation_dict = {}
    # Reglas de mapeo para grupos de letras a fonemas compatibles
    def word_to_phonemes(word):
        w = word.lower()
        phonemes = []
        i = 0
        while i < len(w):
            # Mapeos multiletra primero
            if w[i:i+2] == 'll':
                phonemes.append('/L/')
                i += 2
            elif w[i:i+2] == 'rr':
                phonemes.append('/r/')
                i += 2
            elif w[i:i+2] == 'qu':
                phonemes.append('/k/')
                i += 2
            elif w[i:i+2] == 'ch':
                phonemes.append('/tS/')
                i += 2
            elif w[i:i+2] == 'gu' and i+2 < len(w) and w[i+2] in 'eiéí':
                phonemes.append('/g/')
                i += 2
            # Mapeos de una letra
            elif w[i] == 'ñ':
                phonemes.append('/n/')  # O ajusta si tienes /ɲ/
                phonemes.append('/i/')  
                i += 1
            elif w[i] == 'v':
                phonemes.append('/b/')
                i += 1
            elif w[i] == 'c':
                # 'ce', 'ci' -> /s/, otro -> /k/
                if i+1 < len(w) and w[i+1] in 'eiéí':
                    phonemes.append('/s/')
                else:
                    phonemes.append('/k/')
                i += 1
            elif w[i] == 'z':
                phonemes.append('/s/')
                i += 1
            elif w[i] == 'y':
                phonemes.append('/i/')
                i += 1
            elif w[i] == 'h':
                # muda
                i += 1
            elif w[i] == 'á':
                phonemes.append('/a/')
                i += 1
            elif w[i] == 'é':
                phonemes.append('/e/')
                i += 1
            elif w[i] == 'í':
                phonemes.append('/i/')
                i += 1
            elif w[i] == 'ó':
                phonemes.append('/o/')
                i += 1
            elif w[i] == 'ú':
                phonemes.append('/u/')
                i += 1
            else:
                # Default: letra como fonema si está en PHONEMES
                fono = f'/{w[i]}/'
                if fono in PHONEMES:
                    phonemes.append(fono)
                i += 1
        return phonemes

    for word in all_words:
        if word in base_dict:
            pronunciation_dict[word] = base_dict[word]
        else:
            pronunciation_dict[word] = word_to_phonemes(word)
    return pronunciation_dict

# Lista de fonemas objetivo
PHONEMES = ['/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/', '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/', '/sil/']


# --- MODELO DE LENGUAJE A NIVEL PALABRA ---
def text_to_words(text):
    # Limpia y separa en palabras
    text = text.lower()
    for ch in [',', '.', '!', '?', ';', ':', '-', '_', '"', "'", '(', ')', '[', ']', '{', '}', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '/', '\\']:
        text = text.replace(ch, ' ')
    words = [w for w in text.split() if w]
    return words

def extract_phrases_from_file(filepath):
    phrases = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3 and parts[2] != '#':
                phrases.append(parts[2])
    return phrases


def build_bigram_lm(word_sequences):
    bigram_counts = defaultdict(Counter)
    for seq in word_sequences:
        prev = '<s>'
        for w in seq:
            bigram_counts[prev][w] += 1
            prev = w
        bigram_counts[prev]['</s>'] += 1
    # Convert counts to probabilities
    bigram_probs = {}
    for prev, counter in bigram_counts.items():
        total = sum(counter.values())
        bigram_probs[prev] = {w: count/total for w, count in counter.items()}
    return bigram_probs


# Generic n-gram LM builder for words

def build_ngram_lm(word_sequences, n=3):
    ngram_counts = defaultdict(Counter)
    for seq in word_sequences:
        padded = ['<s>'] * (n-1) + seq + ['</s>']
        for i in range(len(seq) + 1):
            prev = tuple(padded[i:i+n-1])
            nxt = padded[i+n-1]
            ngram_counts[prev][nxt] += 1
    # Convert counts to probabilities
    ngram_probs = {}
    for prev, counter in ngram_counts.items():
        total = sum(counter.values())
        ngram_probs['|'.join(prev)] = {w: count/total for w, count in counter.items()}
    return ngram_probs

def main():

    phrases_dir = os.path.join('data', 'phrases')
    
    all_phrase_files = glob.glob(os.path.join(phrases_dir, '**', '*.phrases'), recursive=True)

    word_sequences = []
    for file in all_phrase_files:
        phrases = extract_phrases_from_file(file)
        for phrase in phrases:
            words = text_to_words(phrase)
            if words:
                word_sequences.append(words)
    # Construir y guardar el diccionario de pronunciación completo
    pronunciation_dict = build_pronunciation_dict(word_sequences)
    out_dir = os.path.join('data', 'language_model')
    os.makedirs(out_dir, exist_ok=True)
    pron_dict_path = os.path.join(out_dir, 'pronunciation_dict.json')
    with open(pron_dict_path, 'w', encoding='utf-8') as f:
        json.dump(pronunciation_dict, f, ensure_ascii=False, indent=2)
    print(f'Diccionario de pronunciación guardado en {pron_dict_path}')
    # Output directory
    out_dir = os.path.join('data', 'language_model')
    os.makedirs(out_dir, exist_ok=True)

    # 2-gram
    twogram_lm = build_bigram_lm(word_sequences)
    twogram_path = os.path.join(out_dir, 'word_2gram_lm.json')
    with open(twogram_path, 'w', encoding='utf-8') as f:
        json.dump(twogram_lm, f, ensure_ascii=False, indent=2)
    print(f'Modelo de lenguaje de 2-gramas guardado en {twogram_path}')

    # 3-gram
    threegram_lm = build_ngram_lm(word_sequences, n=3)
    threegram_path = os.path.join(out_dir, 'word_3gram_lm.json')
    with open(threegram_path, 'w', encoding='utf-8') as f:
        json.dump(threegram_lm, f, ensure_ascii=False, indent=2)
    print(f'Modelo de lenguaje de 3-gramas guardado en {threegram_path}')

    # 4-gram
    fourgram_lm = build_ngram_lm(word_sequences, n=4)
    fourgram_path = os.path.join(out_dir, 'word_4gram_lm.json')
    with open(fourgram_path, 'w', encoding='utf-8') as f:
        json.dump(fourgram_lm, f, ensure_ascii=False, indent=2)
    print(f'Modelo de lenguaje de 4-gramas guardado en {fourgram_path}')

    # 5-gram
    fivegram_lm = build_ngram_lm(word_sequences, n=5)
    fivegram_path = os.path.join(out_dir, 'word_5gram_lm.json')
    with open(fivegram_path, 'w', encoding='utf-8') as f:
        json.dump(fivegram_lm, f, ensure_ascii=False, indent=2)
    print(f'Modelo de lenguaje de 5-gramas guardado en {fivegram_path}')

if __name__ == '__main__':
    main()
