"""
CNN-Based Semantic Vector Extraction for Audio Files
Uses pre-trained ResNet18 to extract semantic features from mel-spectrograms

Author: Corey Zhang
Date: November 2025
"""

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from scipy.ndimage import zoom
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")


class CNNSemanticVectorExtractor:
    """
    Extract semantic vectors from audio using CNN on mel-spectrograms
    """
    
    def __init__(self, sr=22050, n_mels=128):
        """
        Initialize the extractor
        
        Args:
            sr: Sample rate (default 22050 Hz)
            n_mels: Number of mel frequency bands (default 128)
        """
        self.sr = sr
        self.n_mels = n_mels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet18
        print(f"Loading pre-trained ResNet18 model...")
        print(f"Device: {self.device}")
        self.model = self._load_model()
        
        self.feature_vectors = []
        self.filenames = []
        
    def _load_model(self):
        """Load and configure pre-trained ResNet18"""
        # Load ResNet18 pre-trained on ImageNet
        resnet = models.resnet18(pretrained=True)
        
        # Modify first layer for single-channel (grayscale) spectrograms
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification layer - we want feature embeddings
        model = nn.Sequential(*list(resnet.children())[:-1])
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def audio_to_spectrogram(self, audio_path, target_size=(224, 224)):
        """
        Convert audio file to mel-spectrogram tensor
        
        Args:
            audio_path: Path to audio file
            target_size: Resize spectrogram to this size (for CNN input)
            
        Returns:
            Spectrogram tensor (1, 1, H, W)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=self.n_mels,
            hop_length=512
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Resize to target size for CNN
        zoom_factors = (target_size[0] / mel_spec_norm.shape[0],
                       target_size[1] / mel_spec_norm.shape[1])
        mel_spec_resized = zoom(mel_spec_norm, zoom_factors, order=1)
        
        # Convert to PyTorch tensor
        spec_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0)
        
        return spec_tensor.to(self.device)
    
    def extract_features(self, audio_path):
        """
        Extract CNN semantic vector from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector (512-dimensional from ResNet18)
        """
        # Convert to spectrogram
        spec_tensor = self.audio_to_spectrogram(audio_path)
        
        # Extract features with CNN
        with torch.no_grad():
            features = self.model(spec_tensor)
        
        # Convert to numpy and flatten
        feature_vector = features.cpu().numpy().flatten()
        
        return feature_vector
    
    def process_folder(self, folder_path):
        """
        Process all audio files in folder
        
        Args:
            folder_path: Path to folder with audio files
            
        Returns:
            DataFrame with CNN semantic vectors
        """
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        print(f"\nProcessing audio files from: {folder_path}")
        print("-" * 60)
        
        for filename in sorted(os.listdir(folder_path)):
            if any(filename.lower().endswith(ext) for ext in audio_extensions):
                audio_path = os.path.join(folder_path, filename)
                
                try:
                    print(f"Extracting features: {filename}")
                    features = self.extract_features(audio_path)
                    self.feature_vectors.append(features)
                    self.filenames.append(filename)
                except Exception as e:
                    print(f"  Error: {str(e)}")
        
        # Create DataFrame
        feature_dim = len(self.feature_vectors[0])
        self.df = pd.DataFrame(
            self.feature_vectors,
            columns=[f'feature_{i}' for i in range(feature_dim)]
        )
        self.df.insert(0, 'filename', self.filenames)
        
        print(f"\n✓ Processed {len(self.df)} files")
        print(f"✓ Feature dimension: {feature_dim}")
        
        return self.df
    
    def perform_pca(self, n_components=2):
        """
        Perform PCA on CNN semantic vectors
        
        Args:
            n_components: Number of principal components (default 2 for visualization)
            
        Returns:
            PCA-transformed data
        """
        # Get feature columns
        feature_cols = [col for col in self.df.columns if col.startswith('feature_')]
        X = self.df[feature_cols].values
        
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Add to DataFrame
        for i in range(n_components):
            self.df[f'PC{i+1}'] = X_pca[:, i]
        
        self.explained_variance = self.pca.explained_variance_ratio_
        
        print(f"\nPCA Results:")
        print(f"  PC1: {self.explained_variance[0]:.1%} variance")
        print(f"  PC2: {self.explained_variance[1]:.1%} variance")
        print(f"  Total: {sum(self.explained_variance):.1%} variance explained")
        
        return X_pca
    
    def visualize_pca(self, save_path='cnn_pca_visualization.png', title='PCA Visualization - CNN Semantic Vectors'):
        """
        Create PCA visualization
        
        Args:
            save_path: Where to save the plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Scatter plot
        scatter = ax.scatter(
            self.df['PC1'], 
            self.df['PC2'],
            s=400,
            c=range(len(self.df)),
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidths=2.5
        )
        
        # Add filename labels
        for idx, row in self.df.iterrows():
            ax.annotate(
                row['filename'],
                (row['PC1'], row['PC2']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.6, edgecolor='black', linewidth=1.5)
            )
        
        # Styling
        ax.set_xlabel(
            f'PC1 ({self.explained_variance[0]:.1%} variance)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_ylabel(
            f'PC2 ({self.explained_variance[1]:.1%} variance)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('File Index', fontsize=12, fontweight='bold')
        
        # Info box
        info_text = f'Files: {len(self.df)}\n'
        info_text += f'Feature dim: {len([c for c in self.df.columns if c.startswith("feature_")])}\n'
        info_text += f'Model: ResNet18 (pre-trained)'
        
        props = dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.6, edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {save_path}")
        plt.close()
    
    def export_results(self, output_path='cnn_semantic_vectors.xlsx'):
        """
        Export results to Excel
        
        Args:
            output_path: Path for Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: PCA coordinates
            pca_df = self.df[['filename', 'PC1', 'PC2']].copy()
            pca_df.to_excel(writer, sheet_name='PCA_Coordinates', index=False)
            
            # Sheet 2: Sample of semantic vectors (first 20 features)
            feature_cols = ['filename'] + [c for c in self.df.columns if c.startswith('feature_')][:20]
            self.df[feature_cols].to_excel(writer, sheet_name='Semantic_Vectors_Sample', index=False)
            
            # Sheet 3: Statistics
            stats = pd.DataFrame({
                'Metric': ['Number of Files', 'Feature Dimension', 'PC1 Variance', 'PC2 Variance', 'Total Variance (2D)'],
                'Value': [
                    len(self.df),
                    len([c for c in self.df.columns if c.startswith('feature_')]),
                    f"{self.explained_variance[0]:.2%}",
                    f"{self.explained_variance[1]:.2%}",
                    f"{sum(self.explained_variance):.2%}"
                ]
            })
            stats.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"✓ Results exported: {output_path}")


def main():
    """Main execution"""
    
    if not TORCH_AVAILABLE:
        print("\nERROR: PyTorch is required for CNN feature extraction.")
        print("Install with: pip install torch torchvision")
        return
    
    print("="*70)
    print("CNN-Based Semantic Vector Extraction")
    print("="*70)
    
    # Initialize
    extractor = CNNSemanticVectorExtractor(sr=22050, n_mels=128)
    
    # Find audio folder
    possible_folders = ['./audio_files', './audio', 'audio_files', 'audio']
    audio_folder = None
    
    for folder in possible_folders:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]
            if len(files) > 0:
                audio_folder = folder
                break
    
    if audio_folder is None:
        print("\nNo audio folder found. Please create 'audio_files' with your .wav files.")
        return
    
    # Process
    df = extractor.process_folder(audio_folder)
    
    # PCA
    print("\n" + "="*70)
    print("Performing PCA...")
    print("="*70)
    extractor.perform_pca(n_components=2)
    
    # Visualize (NO clustering as per Katherine's request)
    print("\n" + "="*70)
    print("Creating Visualization...")
    print("="*70)
    
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    extractor.visualize_pca(
        save_path=os.path.join(output_dir, 'cnn_pca_visualization.png')
    )
    
    # Export
    print("\n" + "="*70)
    print("Exporting Results...")
    print("="*70)
    extractor.export_results(
        output_path=os.path.join(output_dir, 'cnn_semantic_vectors.xlsx')
    )
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nOutputs in '{output_dir}/':")
    print("  • cnn_pca_visualization.png - PCA plot")
    print("  • cnn_semantic_vectors.xlsx - Feature data")
    
    return extractor


if __name__ == "__main__":
    main()
