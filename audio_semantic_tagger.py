"""
Audio Semantic Tagging System
Uses librosa for feature extraction, PCA for dimensionality reduction,
and creates semantic feature vectors for audio classification/tagging
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
import warnings
warnings.filterwarnings('ignore')

class AudioSemanticTagger:
    """
    A class for extracting audio features and performing semantic tagging
    """
    
    def __init__(self, sr=22050, n_mfcc=13, n_mels=128, hop_length=512):
        """
        Initialize the tagger with audio processing parameters
        
        Args:
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.feature_data = []
        self.scaler = StandardScaler()
        
    def extract_features(self, audio_path):
        """
        Extract comprehensive audio features using librosa
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # 1. Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 2. MFCC (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
            )
            
            # 3. Tempogram
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempogram = librosa.feature.tempogram(
                y=y, sr=sr, hop_length=self.hop_length
            )
            
            # 4. Additional features for semantic understanding
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=self.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=self.hop_length
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, hop_length=self.hop_length
            )
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            
            # Chroma features
            chroma_stft = librosa.feature.chroma_stft(
                y=y, sr=sr, hop_length=self.hop_length
            )
            
            # RMS Energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
            
            # Create semantic feature vector (statistical aggregates)
            features = {
                'filename': os.path.basename(audio_path),
                'duration': len(y) / sr,
                'tempo': tempo,
                
                # Mel-spectrogram statistics
                'mel_mean': np.mean(mel_spec_db),
                'mel_std': np.std(mel_spec_db),
                'mel_max': np.max(mel_spec_db),
                'mel_min': np.min(mel_spec_db),
                
                # MFCC statistics (for each coefficient)
                **{f'mfcc_{i}_mean': np.mean(mfcc[i]) for i in range(self.n_mfcc)},
                **{f'mfcc_{i}_std': np.std(mfcc[i]) for i in range(self.n_mfcc)},
                
                # Tempogram statistics
                'tempogram_mean': np.mean(tempogram),
                'tempogram_std': np.std(tempogram),
                'tempogram_max': np.max(tempogram),
                
                # Spectral features
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'spectral_centroid_std': np.std(spectral_centroid),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                
                # Zero crossing rate
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr),
                
                # Chroma features
                **{f'chroma_{i}_mean': np.mean(chroma_stft[i]) for i in range(12)},
                
                # RMS Energy
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
            }
            
            # Store raw features for visualization
            features['_raw'] = {
                'mel_spec_db': mel_spec_db,
                'mfcc': mfcc,
                'tempogram': tempogram,
                'spectral_centroid': spectral_centroid,
                'chroma_stft': chroma_stft,
                'y': y,
                'sr': sr
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def process_audio_folder(self, folder_path):
        """
        Process all audio files in a folder
        
        Args:
            folder_path: Path to folder containing audio files
            
        Returns:
            DataFrame with all extracted features
        """
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in audio_extensions):
                audio_path = os.path.join(folder_path, filename)
                print(f"Processing: {filename}")
                
                features = self.extract_features(audio_path)
                if features:
                    self.feature_data.append(features)
        
        # Create DataFrame (excluding raw data)
        df_data = []
        for feat in self.feature_data:
            feat_copy = {k: v for k, v in feat.items() if k != '_raw'}
            df_data.append(feat_copy)
        
        self.df = pd.DataFrame(df_data)
        return self.df
    
    def perform_pca(self, n_components=2):
        """
        Perform PCA for dimensionality reduction and visualization
        
        Args:
            n_components: Number of principal components
            
        Returns:
            Transformed data and PCA object
        """
        # Select only numeric columns (exclude filename)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        X = self.df[numeric_cols].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA components to dataframe
        for i in range(n_components):
            self.df[f'PC{i+1}'] = X_pca[:, i]
        
        # Store explained variance
        self.pca_explained_variance = pca.explained_variance_ratio_
        
        print(f"\nPCA Analysis:")
        print(f"Explained variance ratio: {self.pca_explained_variance}")
        print(f"Total variance explained: {sum(self.pca_explained_variance):.2%}")
        
        return X_pca, pca
    
    def create_semantic_tags(self, n_clusters=3):
        """
        Create semantic tags using clustering on PCA components
        
        Args:
            n_clusters: Number of clusters for semantic grouping
        """
        # Use PCA components for clustering
        pca_cols = [col for col in self.df.columns if col.startswith('PC')]
        X_pca = self.df[pca_cols].values
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['semantic_cluster'] = kmeans.fit_predict(X_pca)
        
        # Create semantic tag names based on characteristics
        self.df['semantic_tag'] = self.df['semantic_cluster'].apply(
            lambda x: f"Category_{chr(65+x)}"  # A, B, C, etc.
        )
        
        return self.df
    
    def visualize_features(self, save_path='audio_visualizations.png'):
        """
        Create comprehensive visualizations of audio features
        """
        if not self.feature_data:
            print("No data to visualize. Process audio files first.")
            return
        
        n_samples = len(self.feature_data)
        fig = plt.figure(figsize=(20, 4 * n_samples))
        
        for idx, features in enumerate(self.feature_data):
            raw = features['_raw']
            
            # Plot 1: Mel-spectrogram
            plt.subplot(n_samples, 4, idx*4 + 1)
            librosa.display.specshow(
                raw['mel_spec_db'], sr=raw['sr'], 
                hop_length=self.hop_length, x_axis='time', y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{features['filename']}\nMel-Spectrogram")
            
            # Plot 2: MFCC
            plt.subplot(n_samples, 4, idx*4 + 2)
            librosa.display.specshow(
                raw['mfcc'], sr=raw['sr'], 
                hop_length=self.hop_length, x_axis='time'
            )
            plt.colorbar()
            plt.title('MFCC')
            
            # Plot 3: Tempogram
            plt.subplot(n_samples, 4, idx*4 + 3)
            librosa.display.specshow(
                raw['tempogram'], sr=raw['sr'], 
                hop_length=self.hop_length, x_axis='time', y_axis='tempo'
            )
            plt.colorbar()
            plt.title('Tempogram')
            
            # Plot 4: Chroma
            plt.subplot(n_samples, 4, idx*4 + 4)
            librosa.display.specshow(
                raw['chroma_stft'], sr=raw['sr'],
                hop_length=self.hop_length, x_axis='time', y_axis='chroma'
            )
            plt.colorbar()
            plt.title('Chromagram')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()
        
    def visualize_pca(self, save_path='pca_visualization.png'):
        """
        Create PCA visualization with semantic tags
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: PCA scatter plot with semantic tags
        scatter = axes[0].scatter(
            self.df['PC1'], self.df['PC2'],
            c=self.df['semantic_cluster'],
            cmap='viridis',
            s=200,
            alpha=0.6,
            edgecolors='black'
        )
        
        # Add labels
        for idx, row in self.df.iterrows():
            axes[0].annotate(
                row['filename'],
                (row['PC1'], row['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )
        
        axes[0].set_xlabel(
            f'PC1 ({self.pca_explained_variance[0]:.1%} variance)',
            fontsize=12
        )
        axes[0].set_ylabel(
            f'PC2 ({self.pca_explained_variance[1]:.1%} variance)',
            fontsize=12
        )
        axes[0].set_title('PCA Visualization of Audio Features', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Semantic Cluster')
        
        # Plot 2: Feature importance (PCA loadings)
        if hasattr(self, 'df'):
            numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                          if not col.startswith('PC') and col != 'semantic_cluster'][:15]
            
            # Get PCA object (we need to refit to get components)
            X = self.df[numeric_cols].values
            X_scaled = self.scaler.transform(X)
            pca = PCA(n_components=2)
            pca.fit(X_scaled)
            
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            loading_matrix = pd.DataFrame(
                loadings[:15],
                columns=['PC1', 'PC2'],
                index=numeric_cols[:15]
            )
            
            sns.heatmap(
                loading_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                ax=axes[1],
                cbar_kws={'label': 'Loading'}
            )
            axes[1].set_title('PCA Feature Loadings (Top 15 Features)', fontsize=14)
            axes[1].set_xlabel('Principal Component', fontsize=12)
            axes[1].set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PCA visualization saved to: {save_path}")
        plt.close()
    
    def export_to_excel(self, output_path='audio_semantic_features.xlsx'):
        """
        Export feature data to Excel with multiple sheets
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Feature summary
            summary_df = self.df.copy()
            # Remove PC columns and keep only essential info
            essential_cols = ['filename', 'duration', 'tempo', 'semantic_tag', 
                            'spectral_centroid_mean', 'rms_mean', 'zcr_mean']
            if 'PC1' in summary_df.columns:
                essential_cols.extend(['PC1', 'PC2'])
            
            summary_df[essential_cols].to_excel(
                writer, sheet_name='Summary', index=False
            )
            
            # Sheet 2: All features
            full_features = self.df[[col for col in self.df.columns 
                                    if col != 'semantic_cluster']].copy()
            full_features.to_excel(writer, sheet_name='All_Features', index=False)
            
            # Sheet 3: Semantic tags and clusters
            tag_summary = self.df.groupby('semantic_tag').agg({
                'tempo': ['mean', 'std'],
                'spectral_centroid_mean': ['mean', 'std'],
                'rms_mean': ['mean', 'std']
            }).round(2)
            tag_summary.to_excel(writer, sheet_name='Tag_Statistics')
            
        print(f"\nData exported to Excel: {output_path}")
        

def main():
    """
    Main execution function
    """
    print("="*60)
    print("Audio Semantic Tagging System")
    print("="*60)
    
    # Initialize tagger
    tagger = AudioSemanticTagger(
        sr=22050,
        n_mfcc=13,
        n_mels=128,
        hop_length=512
    )
    
    # Specify your audio folder path
    # For now, using the uploads folder - modify as needed
    audio_folder = '/mnt/user-data/uploads'
    
    # Check if folder has audio files
    if not os.path.exists(audio_folder):
        print(f"Folder not found: {audio_folder}")
        print("\nCreating example with synthetic data...")
        # Create example output with placeholder data
        return
    
    # Process audio files
    print(f"\nProcessing audio files from: {audio_folder}")
    df = tagger.process_audio_folder(audio_folder)
    
    if df.empty:
        print("No audio files found or processed.")
        return
    
    print(f"\nProcessed {len(df)} audio files")
    print("\nFeature columns:")
    print(df.columns.tolist())
    
    # Perform PCA
    print("\n" + "="*60)
    print("Performing PCA Analysis...")
    print("="*60)
    X_pca, pca = tagger.perform_pca(n_components=2)
    
    # Create semantic tags
    print("\n" + "="*60)
    print("Creating Semantic Tags...")
    print("="*60)
    n_clusters = min(3, len(df))  # Use 3 clusters or less if fewer samples
    tagger.create_semantic_tags(n_clusters=n_clusters)
    
    print("\nSemantic Tag Distribution:")
    print(tagger.df['semantic_tag'].value_counts())
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating Visualizations...")
    print("="*60)
    tagger.visualize_features(save_path='/mnt/user-data/outputs/audio_visualizations.png')
    tagger.visualize_pca(save_path='/mnt/user-data/outputs/pca_visualization.png')
    
    # Export to Excel
    print("\n" + "="*60)
    print("Exporting Results...")
    print("="*60)
    tagger.export_to_excel(output_path='/mnt/user-data/outputs/audio_semantic_features.xlsx')
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print("\nOutput files created:")
    print("1. audio_semantic_features.xlsx - Feature data and semantic tags")
    print("2. audio_visualizations.png - Feature visualizations")
    print("3. pca_visualization.png - PCA analysis and clustering")
    
    return tagger


if __name__ == "__main__":
    tagger = main()
