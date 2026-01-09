# CNN-Based Audio Semantic Tagging

A music information retrieval (MIR) system that uses convolutional neural networks to extract semantic vectors from audio files and visualize them using PCA dimensionality reduction.

## Overview

This project implements CNN-based semantic tagging for audio files, generating high-dimensional feature vectors that capture musical characteristics. The system uses pre-trained ResNet18 architecture to extract meaningful representations from mel-spectrogram representations of audio data.

## Features

- **Audio Semantic Tagging**: Extract semantic features from audio files using CNN architectures
- **Semantic Vector Generation**: Generate high-dimensional feature representations for music analysis
- **PCA Visualization**: Dimensionality reduction and visualization of semantic vectors
- **Interactive Analysis**: Tools for exploring audio features and embeddings
- **Batch Processing**: Analyze multiple audio files efficiently

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd [repo-name]

# Install dependencies
pip install -r requirements.txt
```

## Requirements

See `requirements.txt` for full dependencies. Key packages include:
- TensorFlow/PyTorch (for CNN models)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Librosa (audio processing)

## Usage

### Basic Audio Semantic Tagging

```python
python audio_semantic_tagger.py
```

### Generate CNN Semantic Vectors

```python
python cnn_semantic_vectors.py
```

### Interactive Analysis

```python
python interactive_analysis.py
```

### Simple Demo

```python
python simple_mir_demo.py
```

## Project Structure

```
.
├── audio_files/              # Input audio files directory
├── outputs/                  # Generated outputs and results
├── audio_semantic_tagger.py  # Main semantic tagging implementation
├── cnn_semantic_vectors.py   # CNN-based vector extraction
├── interactive_analysis.py   # Interactive exploration tools
├── simple_mir_demo.py       # Quick demo script
├── example_usage.py         # Usage examples
├── music1-4.wav            # Sample audio files
├── audio_analysis.png       # Visualization output
├── InterventionExamples.xlsx # Analysis results
└── requirements.txt         # Python dependencies
```

## Output

The system generates:
- **Semantic Vectors**: High-dimensional embeddings stored in Excel format
- **PCA Visualizations**: 2D/3D projections of audio feature space
- **Analysis Plots**: Various visualizations of audio characteristics

Example output: `audio_analysis.png`

## Methodology

1. **Audio Preprocessing**: Convert audio files to mel-spectrogram representations
2. **Feature Extraction**: Pass spectrograms through ResNet18 CNN architecture
3. **Semantic Vector Generation**: Extract intermediate layer activations as feature vectors
4. **Dimensionality Reduction**: Apply PCA for visualization and analysis

## Results

See `InterventionExamples.xlsx` for detailed analysis results and `outputs/` directory for generated visualizations.

## Research Context

This project explores CNN-based approaches to music information retrieval, specifically focusing on semantic tagging and feature extraction for audio analysis applications.

## Acknowledgments

Research conducted at Northwestern University under the mentorship of Katherine O'Toole.

## License

MIT License

---

**Author**: Corey  
**Institution**: Northwestern University
