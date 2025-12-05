#!/usr/bin/env python3
"""
Simple MIR Demo Script
Run this to get started with music feature extraction!

Usage:
    python simple_mir_demo.py path/to/your/audio.wav
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import librosa
        import matplotlib
        import numpy
        print("✓ All required packages are installed!")
        print(f"  - librosa: {librosa.__version__}")
        print(f"  - matplotlib: {matplotlib.__version__}")
        print(f"  - numpy: {numpy.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e.name}")
        print("\nPlease install required packages:")
        print("  pip install librosa matplotlib numpy")
        return False


def analyze_audio(audio_path):
    """
    Complete analysis of an audio file with nice output
    """
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa.display
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return
    
    print("\n" + "="*60)
    print(f"Analyzing: {os.path.basename(audio_path)}")
    print("="*60)
    
    # Load audio
    print("\n[1/5] Loading audio...")
    y, sr = librosa.load(audio_path, duration=30)  # Load first 30 seconds
    duration = len(y) / sr
    print(f"  ✓ Loaded {duration:.1f} seconds at {sr} Hz")
    
    # Extract tempo - FIX: Handle both scalar and array returns
    print("\n[2/5] Detecting tempo and beats...")
    tempo_result, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Convert to scalar if it's an array
    if isinstance(tempo_result, np.ndarray):
        tempo = float(tempo_result.item()) if tempo_result.size == 1 else float(tempo_result[0])
    else:
        tempo = float(tempo_result)
    
    print(f"  ✓ Tempo: {tempo:.1f} BPM")
    print(f"  ✓ Detected {len(beats)} beats")
    
    # Extract spectral features
    print("\n[3/5] Extracting spectral features...")
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    print(f"  ✓ Spectral Centroid: {np.mean(spectral_centroids):.1f} Hz (brightness)")
    print(f"  ✓ Spectral Rolloff: {np.mean(spectral_rolloff):.1f} Hz")
    print(f"  ✓ Zero Crossing Rate: {np.mean(zcr):.4f}")
    
    # Extract MFCCs
    print("\n[4/5] Extracting MFCCs (timbral features)...")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"  ✓ Extracted {mfccs.shape[0]} MFCC coefficients")
    print(f"  ✓ MFCC shape: {mfccs.shape}")
    for i in range(3):
        print(f"     MFCC {i}: mean={np.mean(mfccs[i]):.2f}, std={np.std(mfccs[i]):.2f}")
    
    # Extract chroma
    print("\n[5/5] Extracting chroma features (pitch content)...")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print(f"  ✓ Chroma shape: {chroma.shape}")
    print(f"  ✓ Dominant pitch classes: ", end="")
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma_mean = np.mean(chroma, axis=1)
    top_pitches = np.argsort(chroma_mean)[-3:][::-1]
    print(", ".join([pitch_classes[i] for i in top_pitches]))
    
    # Create visualization
    print("\n[Visualizing] Creating plots...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(alpha=0.3)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title('Spectrogram', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # MFCCs
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[2])
    axes[2].set_title('MFCCs (Mel-Frequency Cepstral Coefficients)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('MFCC')
    fig.colorbar(img, ax=axes[2])
    
    # Chroma
    img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[3])
    axes[3].set_title('Chromagram (Pitch Content)', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=axes[3])
    
    plt.tight_layout()
    
    # Save plot to current directory
    output_path = 'audio_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Duration: {duration:.1f}s | Tempo: {tempo:.1f} BPM | Beats: {len(beats)}")
    print(f"Brightness: {np.mean(spectral_centroids):.0f} Hz")
    print(f"Timbre: {mfccs.shape[0]} MFCC coefficients captured")
    print(f"Key pitch classes: {', '.join([pitch_classes[i] for i in top_pitches])}")
    print("="*60)
    
    return {
        'tempo': tempo,
        'spectral_centroid': np.mean(spectral_centroids),
        'mfcc': np.mean(mfccs, axis=1),
        'chroma': np.mean(chroma, axis=1)
    }


def compare_two_files(file1, file2):
    """
    Compare two audio files
    """
    import numpy as np
    
    print("\n" + "="*60)
    print("COMPARING TWO AUDIO FILES")
    print("="*60)
    
    # Analyze both
    print(f"\nFile 1: {os.path.basename(file1)}")
    features1 = analyze_audio(file1)
    
    print(f"\n\nFile 2: {os.path.basename(file2)}")
    features2 = analyze_audio(file2)
    
    if features1 is None or features2 is None:
        print("Error: Could not analyze one or both files")
        return
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    tempo_diff = abs(features1['tempo'] - features2['tempo'])
    print(f"\nTempo Difference: {tempo_diff:.1f} BPM")
    if tempo_diff < 10:
        print("  → Very similar tempo!")
    elif tempo_diff < 30:
        print("  → Somewhat similar tempo")
    else:
        print("  → Very different tempo")
    
    brightness_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
    print(f"\nBrightness Difference: {brightness_diff:.0f} Hz")
    if brightness_diff < 500:
        print("  → Similar brightness/tone")
    else:
        print("  → Different brightness/tone")
    
    mfcc_distance = np.linalg.norm(features1['mfcc'] - features2['mfcc'])
    print(f"\nMFCC Distance: {mfcc_distance:.2f}")
    if mfcc_distance < 50:
        print("  → Very similar timbre!")
    elif mfcc_distance < 150:
        print("  → Somewhat similar timbre")
    else:
        print("  → Very different timbre")
    
    chroma_distance = np.linalg.norm(features1['chroma'] - features2['chroma'])
    print(f"\nChroma Distance: {chroma_distance:.2f}")
    if chroma_distance < 0.5:
        print("  → Similar harmonic content")
    else:
        print("  → Different harmonic content")
    
    print("\n" + "="*60)
    overall_similarity = 100 - min(100, (mfcc_distance / 3 + tempo_diff / 2 + brightness_diff / 50))
    print(f"Overall Similarity Score: {overall_similarity:.1f}/100")
    print("="*60)


def show_usage():
    """Show usage instructions"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Simple MIR Demo - Quick Start Guide              ║
╚════════════════════════════════════════════════════════════════╝

This script helps you extract music features from audio files!

USAGE:
------
Analyze one file:
  python simple_mir_demo.py music.wav

Compare two files:
  python simple_mir_demo.py music1.wav music2.wav

WHAT IT DOES:
------------
✓ Detects tempo (BPM) and beats
✓ Extracts spectral features (brightness, rolloff)
✓ Computes MFCCs (captures timbre/texture)
✓ Analyzes pitch content (chroma features)
✓ Creates beautiful visualizations

Ready to try? Just run with an audio file path!
""")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║            MIR Feature Extraction Demo Script                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) == 1:
        show_usage()
        print("\n💡 Tip: Run this script with an audio file to see it in action!")
        print("   Example: python simple_mir_demo.py music.wav")
    elif len(sys.argv) == 2:
        audio_path = sys.argv[1]
        analyze_audio(audio_path)
        print("\n✓ Analysis complete! Check for audio_analysis.png")
    elif len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        compare_two_files(file1, file2)
        print("\n✓ Comparison complete!")
    else:
        print("\nError: Too many arguments")
        print("Usage: python simple_mir_demo.py [file1] [file2]")
        sys.exit(1)