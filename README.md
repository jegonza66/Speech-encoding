# Speech-encoding

A Python toolkit for analyzing the representation of speech features in EEG signals during goal-oriented dialogue.

## Table of Contents
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Features](#features)
- [Data](#data)
- [Citation](#citation)
- [License](#license)
- [About](#about)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Speech-encoding.git
cd Speech-encoding
```

2. Create a conda environment:
```bash
conda create -n speech-encoding python=3.12
conda activate speech-encoding
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

This project requires Python 3.12+ and the following main dependencies:
- NumPy 2.0.1
- Pandas 2.2.3
- SciPy 1.15.3
- MNE 1.9.0 (for EEG processing)
- Librosa 0.11.0 (for audio processing)
- Scikit-learn 1.6.1
- PyTorch 2.5.1
- Transformers 4.49.0

See `requirements.txt` for the complete list of dependencies with exact versions.

## Usage

### Basic Analysis
```python
# Example usage
from src.speech_analysis import SpeechEEGAnalyzer

# Initialize analyzer
analyzer = SpeechEEGAnalyzer()

# Load data
eeg_data = analyzer.load_eeg_data('path/to/eeg/file')
audio_data = analyzer.load_audio_data('path/to/audio/file')

# Extract features
speech_features = analyzer.extract_speech_features(audio_data)
eeg_features = analyzer.extract_eeg_features(eeg_data)

# Perform encoding analysis
results = analyzer.analyze_encoding(eeg_features, speech_features)
```

## Features

- **EEG Signal Processing**: Preprocessing and feature extraction from EEG recordings
- **Speech Feature Extraction**: Acoustic feature analysis using multiple methods
- **Encoding Models**: Brain-speech encoding analysis during dialogue
- **Statistical Analysis**: Advanced statistical methods for neural data
- **Visualization**: Comprehensive plotting and visualization tools
- **Reproducible Research**: Complete pipeline with configuration management

## Data

The analysis works with:
- **EEG Data**: Multi-channel EEG recordings during dialogue tasks
- **Audio Data**: Speech recordings synchronized with EEG
- **Behavioral Data**: Task performance and dialogue metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{gonzalez2021brain,
  title={Brain representation of acoustic features during goal oriented dialogue},
  author={Gonzalez, JE and Nieto, N and Brusco, P and Gravano, A and Kamienkowski, JE},
  booktitle={JAIIO 50 Proceedings},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## About
Python codes used to analyse the representation of speech features in EEG signal. 

Gonzalez JE, Nieto N, Brusco P, Gravano A, Kamienkowski JE (2021) Brain representation of acoustic features during goal oriented dialogue. JAIIO 50 Proceedings
