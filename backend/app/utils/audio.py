import librosa
import numpy as np
from typing import Dict, Any
import os
from pathlib import Path

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio features from a given audio file."""
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = {
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                'spectral_centroids': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
                'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).tolist()
            }
            
            return features
        except Exception as e:
            raise Exception(f"Error processing audio file: {str(e)}")
    
    def convert_audio(self, input_path: str, output_path: str, format: str = 'wav'):
        """Convert audio file to specified format."""
        try:
            y, sr = librosa.load(input_path)
            librosa.output.write_wav(output_path, y, sr)
        except Exception as e:
            raise Exception(f"Error converting audio file: {str(e)}")
            
    def trim_silence(self, audio_path: str, output_path: str = None):
        """Remove silence from the beginning and end of an audio file."""
        try:
            y, sr = librosa.load(audio_path)
            yt, index = librosa.effects.trim(y)
            
            if output_path:
                librosa.output.write_wav(output_path, yt, sr)
            return yt, sr
        except Exception as e:
            raise Exception(f"Error trimming silence: {str(e)}") 