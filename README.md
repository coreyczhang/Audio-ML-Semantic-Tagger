# Audio ML Semantic Tagger

**Northwestern University · Human-AI Collaboration Lab**  
*Mentorship: Katherine O'Toole*

> **Note:** The MIR pipeline in this repo was also adapted for a sync licensing metadata project (MetaMusic). See `metamusic_tagger.py`.

---

## Overview

An audio feature extraction and semantic tagging system built on **librosa**, a Python library for Music Information Retrieval (MIR). Extracts MFCCs, mel-spectrograms, chroma features, and spectral descriptors from audio files, then uses PCA to visualize how tracks cluster acoustically.

No external API keys or internet connection required.

---

## Project Structure

```
Audio-ML-Semantic-Tagger/
├── audio_files/              ← Input: WAV files to tag
├── metamusic_tagger.py       ← Main tagger — generates sync licensing metadata
├── audio_semantic_tagger.py  ← MIR demo: MFCCs, mel-spectrogram, PCA visualization
├── simple_mir_demo.py        ← Quick demo: analyze or compare two audio files
├── metamusic_output.xlsx     ← Output: generated metadata tags
└── requirements.txt          ← Dependencies
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the tagger
```bash
python metamusic_tagger.py
# → reads from ./audio_files, saves to metamusic_output.xlsx
```

Custom folder or output path:
```bash
python metamusic_tagger.py /path/to/folder output.xlsx
```

---

## What Gets Tagged

For each audio file, MetaMusic produces:

| Field | Example |
|---|---|
| Genre | Cinematic / Electronic |
| Subgenre | Orchestral / Cinematic |
| Mood | Relaxed, Introspective, Dreamy |
| Energy Level | Low / Medium / High / Very High |
| Tempo Feel | Slow / Medium / Upbeat / Fast |
| Instrumentation | Piano, Synth Pad, Bass Guitar |
| Vocals | No Vocals / Male Vocal / Female Vocal |
| Production Style | acoustic / electronic / hybrid |
| Sync Use Cases | Film score / underscore \| Travel / documentary |
| Tags | cinematic, piano, minor, dreamy, relaxed |

---

## Running the Demos

### Demo 1 — MetaMusic Tagger (sync licensing tags)
```bash
python3 metamusic_tagger.py
```
Reads all WAV files from `./audio_files`, outputs `metamusic_output.xlsx` with genre, mood, instrumentation, vocal type, and sync use cases for each track.

Custom folder or output path:
```bash
python3 metamusic_tagger.py /path/to/folder output.xlsx
```

---

### Demo 2 — Full MIR Pipeline + PCA Visualizations
```bash
python3 audio_semantic_tagger.py
```
Runs the complete librosa pipeline across all files in `./audio_files` and saves three output files to `outputs/`:

```
outputs/
├── pca_visualization.png        ← 2D scatter: tracks grouped by acoustic similarity
├── audio_visualizations.png     ← Per-track: waveform + spectrogram + MFCCs + chroma
└── audio_semantic_features.xlsx ← Full MFCC/spectral feature table + PCA coordinates
```

---

### Demo 3a — Analyze a single file
```bash
python3 simple_mir_demo.py audio_files/chill8.wav
```
Prints BPM, spectral centroid, ZCR, MFCC coefficients, and dominant pitch classes. Saves `audio_analysis.png` — a 4-panel plot showing waveform, spectrogram, MFCC heatmap, and chromagram.

---

### Demo 3b — Compare two files side by side
```bash
python3 simple_mir_demo.py audio_files/chill8.wav audio_files/chillChild1.wav
```
Prints MFCC distance, brightness difference, tempo difference, and an overall acoustic similarity score between the two tracks.

---

## How It Works

### Feature Extraction (librosa)

| Feature | What it captures |
|---|---|
| Tempo (BPM) | Speed of the track |
| Key & Mode | Musical key via Krumhansl-Schmuckler algorithm |
| RMS Energy | Loudness / intensity |
| Spectral Centroid | Brightness (low = warm, high = airy) |
| Harmonic/Percussive Ratio | HPSS — melody vs. rhythm balance |
| Onset Density | Note events per second (sparse vs. busy) |
| Zero Crossing Rate | Noisiness / distortion |
| MFCCs (13 coefficients) | Timbral texture fingerprint |
| Chroma (12 pitch classes) | Harmonic / tonal content |
| Mel-Spectrogram | Time-frequency energy distribution |

### Tag Classification (MetaMusic)

Acoustic measurements feed into rule-based classifiers:

- **Genre** — tempo + spectral centroid + harmonic ratio
- **Mood** — key mode (major/minor) + energy + tempo feel
- **Instrumentation** — spectral shape + ZCR + harmonic content
- **Vocals** — ZCR + spectral centroid signature
- **Sync Use Cases** — genre + mood + tempo context

---

## MIR Concepts

**MFCCs (Mel-Frequency Cepstral Coefficients)**  
A compact representation of the timbral texture of a sound. 13 coefficients capture the "color" of audio — whether it sounds warm, bright, rough, or smooth — without encoding pitch or rhythm.

**Mel-Spectrogram**  
A time-frequency representation scaled to the mel scale (matching human pitch perception). Visualizes how frequency content evolves over time.

**Chroma Features**  
Energy distribution across the 12 pitch classes (C, C#, D … B). Captures harmonic/tonal character and helps infer musical key.

**PCA (Principal Component Analysis)**  
Reduces the high-dimensional feature space (MFCCs + spectral features) to 2D for visualization. Tracks that appear close on the PCA plot are acoustically similar.

**HPSS (Harmonic-Percussive Source Separation)**  
Separates a signal into melodic (harmonic) and rhythmic (percussive) components. The ratio helps distinguish instrument-led tracks from beat-driven ones.

---

## Requirements

```
librosa>=0.11.0
soundfile>=0.12.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
scipy>=1.10.0
```

Install everything:
```bash
pip install -r requirements.txt
```

---

## Acknowledgments

Research conducted at Northwestern University in the **Human-AI Collaboration Lab** under the mentorship of **Katherine O'Toole**.

## License

MIT License

---

**Author**: Corey Zhang  
**Institution**: Northwestern University
