"""
Interactive Audio Analysis Notebook
Step-by-step exploration of audio semantic tagging

Run this script section by section to understand each part of the analysis
"""

import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("INTERACTIVE AUDIO SEMANTIC TAGGING ANALYSIS")
print("="*80)

# ============================================================================
# SECTION 1: SINGLE FILE ANALYSIS
# ============================================================================

def analyze_single_file(filepath):
    """
    Detailed analysis of a single audio file
    Shows all extracted features step by step
    """
    print(f"\n{'='*80}")
    print(f"SECTION 1: Single File Analysis")
    print(f"File: {os.path.basename(filepath)}")
    print('='*80)
    
    # Load audio
    y, sr = librosa.load(filepath, sr=22050)
    duration = len(y) / sr
    
    print(f"\n[1.1] Basic Information:")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Total Samples: {len(y)}")
    
    # Extract features
    print(f"\n[1.2] Extracting Features...")
    
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    print(f"  ✓ Mel-spectrogram: shape {mel_spec_db.shape}")
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"  ✓ MFCC: shape {mfcc.shape}")
    print(f"    - MFCC coefficients 0-12 capture different timbral aspects")
    
    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"  ✓ Tempo: {tempo:.1f} BPM")
    print(f"  ✓ Detected {len(beats)} beats")
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    print(f"  ✓ Spectral Centroid (brightness): {np.mean(spectral_centroid):.1f} Hz")
    print(f"  ✓ Spectral Rolloff: {np.mean(spectral_rolloff):.1f} Hz")
    print(f"  ✓ Zero Crossing Rate: {np.mean(zcr):.4f}")
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print(f"  ✓ Chroma (harmony): shape {chroma.shape}")
    
    # Visualize
    print(f"\n[1.3] Creating Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mel-spectrogram
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0,0])
    axes[0,0].set_title('Mel-Spectrogram')
    axes[0,0].set_ylabel('Frequency (Hz)')
    
    # MFCC
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[0,1])
    axes[0,1].set_title('MFCC')
    axes[0,1].set_ylabel('MFCC Coefficient')
    
    # Waveform with spectral features
    times = np.arange(len(spectral_centroid)) * 512 / sr
    axes[1,0].plot(times, spectral_centroid, label='Spectral Centroid', alpha=0.7)
    axes[1,0].plot(times, spectral_rolloff, label='Spectral Rolloff', alpha=0.7)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Frequency (Hz)')
    axes[1,0].set_title('Spectral Features Over Time')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Chroma
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1,1])
    axes[1,1].set_title('Chromagram (Pitch Classes)')
    
    plt.tight_layout()
    output_path = './outputs/single_file_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {output_path}")
    plt.close()
    
    return {
        'mel_spec_db': mel_spec_db,
        'mfcc': mfcc,
        'tempo': tempo,
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'zcr': np.mean(zcr),
        'chroma': chroma,
        'y': y,
        'sr': sr
    }


# ============================================================================
# SECTION 2: FEATURE COMPARISON ACROSS FILES
# ============================================================================

def compare_multiple_files(folder_path):
    """
    Compare features across multiple audio files
    """
    print(f"\n{'='*80}")
    print(f"SECTION 2: Multi-File Comparison")
    print('='*80)
    
    audio_files = [f for f in os.listdir(folder_path) 
                   if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if len(audio_files) == 0:
        print("No audio files found in folder")
        return None
    
    print(f"\nFound {len(audio_files)} audio files")
    
    features_list = []
    
    for filename in audio_files:
        filepath = os.path.join(folder_path, filename)
        print(f"  Processing: {filename}")
        
        try:
            y, sr = librosa.load(filepath, sr=22050)
            
            # Extract key features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            features_list.append({
                'filename': filename,
                'tempo': tempo,
                'brightness': spectral_centroid,
                'energy': rms,
                'noisiness': zcr,
                'mfcc_0': np.mean(mfcc[0]),
                'mfcc_1': np.mean(mfcc[1]),
                'mfcc_2': np.mean(mfcc[2]),
                'duration': len(y) / sr
            })
            
        except Exception as e:
            print(f"    Error: {str(e)}")
    
    df = pd.DataFrame(features_list)
    
    print(f"\n[2.1] Feature Summary:")
    print(df[['filename', 'tempo', 'brightness', 'energy']].to_string(index=False))
    
    # Visualize comparisons
    print(f"\n[2.2] Creating Comparison Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Tempo comparison
    axes[0,0].bar(range(len(df)), df['tempo'])
    axes[0,0].set_xticks(range(len(df)))
    axes[0,0].set_xticklabels(df['filename'], rotation=45, ha='right')
    axes[0,0].set_ylabel('BPM')
    axes[0,0].set_title('Tempo Comparison')
    axes[0,0].grid(True, alpha=0.3)
    
    # Brightness vs Energy scatter
    axes[0,1].scatter(df['brightness'], df['energy'], s=100, alpha=0.6)
    for idx, row in df.iterrows():
        axes[0,1].annotate(row['filename'], (row['brightness'], row['energy']),
                          xytext=(5,5), textcoords='offset points', fontsize=8)
    axes[0,1].set_xlabel('Brightness (Spectral Centroid)')
    axes[0,1].set_ylabel('Energy (RMS)')
    axes[0,1].set_title('Brightness vs Energy')
    axes[0,1].grid(True, alpha=0.3)
    
    # Duration comparison
    axes[1,0].bar(range(len(df)), df['duration'])
    axes[1,0].set_xticks(range(len(df)))
    axes[1,0].set_xticklabels(df['filename'], rotation=45, ha='right')
    axes[1,0].set_ylabel('Seconds')
    axes[1,0].set_title('Duration Comparison')
    axes[1,0].grid(True, alpha=0.3)
    
    # Feature heatmap
    feature_cols = ['tempo', 'brightness', 'energy', 'noisiness', 'mfcc_0', 'mfcc_1']
    feature_matrix = df[feature_cols].T
    feature_matrix.columns = df['filename']
    
    # Normalize for heatmap
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix_normalized = pd.DataFrame(
        scaler.fit_transform(feature_matrix.T).T,
        index=feature_matrix.index,
        columns=feature_matrix.columns
    )
    
    sns.heatmap(feature_matrix_normalized, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, ax=axes[1,1], cbar_kws={'label': 'Normalized Value'})
    axes[1,1].set_title('Feature Comparison (Normalized)')
    axes[1,1].set_xlabel('Audio File')
    axes[1,1].set_ylabel('Feature')
    
    plt.tight_layout()
    output_path = './outputs/multi_file_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Comparison plots saved: {output_path}")
    plt.close()
    
    return df


# ============================================================================
# SECTION 3: PCA STEP-BY-STEP
# ============================================================================

def pca_step_by_step(df):
    """
    Demonstrate PCA analysis step by step
    """
    print(f"\n{'='*80}")
    print(f"SECTION 3: PCA Analysis (Step by Step)")
    print('='*80)
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[3.1] Selected {len(numeric_cols)} numeric features:")
    print(f"  {', '.join(numeric_cols[:5])}...")
    
    X = df[numeric_cols].values
    
    # Step 1: Standardization
    print(f"\n[3.2] Step 1: Standardizing features")
    print(f"  Original mean: {X.mean(axis=0)[:3]}")
    print(f"  Original std: {X.std(axis=0)[:3]}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  Scaled mean: {X_scaled.mean(axis=0)[:3]}")
    print(f"  Scaled std: {X_scaled.std(axis=0)[:3]}")
    
    # Step 2: PCA
    print(f"\n[3.3] Step 2: Applying PCA")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"  Original dimensions: {X_scaled.shape}")
    print(f"  PCA dimensions: {X_pca.shape}")
    
    # Step 3: Analyze variance
    print(f"\n[3.4] Step 3: Explained Variance")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  PC3: {pca.explained_variance_ratio_[2]:.1%}")
    print(f"  Total (first 3): {sum(pca.explained_variance_ratio_[:3]):.1%}")
    
    # Visualize
    print(f"\n[3.5] Creating PCA Visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scree plot
    axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1), 
                pca.explained_variance_ratio_)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum)+1), cumsum, marker='o')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # PCA scatter
    scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.6)
    for idx, row in df.iterrows():
        axes[2].annotate(row['filename'], (X_pca[idx, 0], X_pca[idx, 1]),
                        xytext=(5,5), textcoords='offset points', fontsize=8)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[2].set_title('PCA Projection (2D)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = './outputs/pca_step_by_step.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ PCA visualizations saved: {output_path}")
    plt.close()
    
    return X_pca, pca


# ============================================================================
# SECTION 4: CLUSTERING AND SEMANTIC TAGGING
# ============================================================================

def clustering_analysis(X_pca, df, n_clusters=3):
    """
    Perform clustering and analyze semantic groups
    """
    print(f"\n{'='*80}")
    print(f"SECTION 4: Clustering & Semantic Tagging")
    print('='*80)
    
    print(f"\n[4.1] K-Means Clustering (k={n_clusters})")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca[:, :2])
    
    df['cluster'] = clusters
    df['semantic_tag'] = df['cluster'].apply(lambda x: f"Category_{chr(65+x)}")
    
    print(f"\n[4.2] Cluster Distribution:")
    print(df['semantic_tag'].value_counts())
    
    print(f"\n[4.3] Files by Cluster:")
    for tag in sorted(df['semantic_tag'].unique()):
        files = df[df['semantic_tag'] == tag]['filename'].tolist()
        print(f"\n  {tag}:")
        for f in files:
            print(f"    - {f}")
    
    # Analyze cluster characteristics
    print(f"\n[4.4] Cluster Characteristics:")
    
    cluster_stats = df.groupby('semantic_tag').agg({
        'tempo': ['mean', 'std'],
        'brightness': ['mean', 'std'],
        'energy': ['mean', 'std']
    }).round(2)
    
    print(cluster_stats)
    
    # Visualize
    print(f"\n[4.5] Creating Cluster Visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot with clusters
    for cluster_id in range(n_clusters):
        mask = clusters == cluster_id
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=f'Category_{chr(65+cluster_id)}', s=150, alpha=0.6)
    
    # Add labels
    for idx, row in df.iterrows():
        axes[0].annotate(row['filename'], 
                        (X_pca[idx, 0], X_pca[idx, 1]),
                        xytext=(5,5), textcoords='offset points', fontsize=8)
    
    # Add cluster centers
    axes[0].scatter(kmeans.cluster_centers_[:, 0], 
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidths=2,
                   label='Centroids')
    
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Semantic Clusters in PCA Space')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cluster characteristics radar chart (if 3 or more features)
    if 'tempo' in df.columns:
        # Normalize features for comparison
        features_for_radar = ['tempo', 'brightness', 'energy']
        normalized_data = []
        
        for tag in sorted(df['semantic_tag'].unique()):
            cluster_data = df[df['semantic_tag'] == tag][features_for_radar].mean()
            # Min-max normalize
            normalized = (cluster_data - df[features_for_radar].min()) / (df[features_for_radar].max() - df[features_for_radar].min())
            normalized_data.append(normalized.values)
        
        categories = features_for_radar
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(122, polar=True)
        
        for i, (tag, data) in enumerate(zip(sorted(df['semantic_tag'].unique()), normalized_data)):
            values = list(data) + [data[0]]
            ax.plot(angles, values, 'o-', linewidth=2, label=tag)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Normalized Cluster Profiles')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.tight_layout()
    output_path = './outputs/clustering_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Clustering visualizations saved: {output_path}")
    plt.close()
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_interactive_analysis(audio_folder='audio_files'):
    """
    Run complete interactive analysis
    """
    print("\n" + "="*80)
    print("STARTING INTERACTIVE ANALYSIS")
    print("="*80)
    
    # Check if folder exists
    if not os.path.exists(audio_folder) or len(os.listdir(audio_folder)) == 0:
        print(f"\n⚠ Warning: No audio files found in {audio_folder}")
        print("Please upload audio files first, then run this script again.")
        return
    
    # Section 1: Analyze single file
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.endswith(('.wav', '.mp3', '.flac'))]
    if len(audio_files) > 0:
        first_file = os.path.join(audio_folder, audio_files[0])
        single_analysis = analyze_single_file(first_file)
    
    # Section 2: Compare multiple files
    df = compare_multiple_files(audio_folder)
    
    if df is not None and len(df) > 1:
        # Section 3: PCA
        X_pca, pca = pca_step_by_step(df)
        
        # Section 4: Clustering
        n_clusters = min(3, len(df))
        df_final = clustering_analysis(X_pca, df, n_clusters)
        
        # Export results
        print(f"\n{'='*80}")
        print("EXPORTING RESULTS")
        print('='*80)
        
        output_excel = './outputs/interactive_analysis_results.xlsx'
        df_final.to_excel(output_excel, index=False)
        print(f"✓ Results exported to: {output_excel}")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print('='*80)
        print("\nGenerated files:")
        print("  1. single_file_analysis.png")
        print("  2. multi_file_comparison.png")
        print("  3. pca_step_by_step.png")
        print("  4. clustering_analysis.png")
        print("  5. interactive_analysis_results.xlsx")


if __name__ == "__main__":
    run_interactive_analysis()
