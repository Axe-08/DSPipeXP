import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
SAMPLE_RATE = 44100  # Standard audio sample rate
N_FFT = 2048         # FFT window size
HOP_LENGTH = 512     # Hop length for STFT
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

# Custom collate function to ensure all tensors in a batch are the same size
def my_collate(batch):
    # Extract the spectrogram tensors from the batch
    if len(batch) == 0:
        return default_collate(batch)
    
    # Extract the first sample to check its shape
    first_x = batch[0][0]
    first_y = batch[0][1]
    
    # Print the shapes for debugging
    print(f"First x shape: {first_x.shape}, First y shape: {first_y.shape}")
    
    # Check if we have 3D or 4D tensors and handle accordingly
    if len(first_x.shape) == 3:  # [C, H, W] - single sample with channel, height, width
        # Find the minimum size across all spectrograms in the batch
        min_freq = min([x[0].shape[1] for x in batch])
        min_time = min([x[0].shape[2] for x in batch])
        
        # Resize all spectrograms to the minimum size
        for i in range(len(batch)):
            x, y = batch[i][0], batch[i][1]
            # Crop to minimum dimensions
            x_resized = x[:, :min_freq, :min_time]
            y_resized = y[:, :min_freq, :min_time]
            # Replace in the batch
            batch[i] = (x_resized, y_resized) + batch[i][2:]
            
    elif len(first_x.shape) == 4:  # [B, C, H, W] - batch with channel, height, width
        # Find the minimum size across all spectrograms in the batch
        min_freq = min([x[0].shape[2] for x in batch])
        min_time = min([x[0].shape[3] for x in batch])
        
        # Resize all spectrograms to the minimum size
        for i in range(len(batch)):
            x, y = batch[i][0], batch[i][1]
            # Crop to minimum dimensions
            x_resized = x[:, :, :min_freq, :min_time]
            y_resized = y[:, :, :min_freq, :min_time]
            # Replace in the batch
            batch[i] = (x_resized, y_resized) + batch[i][2:]
    
    # Use default collate after shapes are standardized
    try:
        return default_collate(batch)
    except Exception as e:
        # If we still have issues, print detailed diagnostic information
        print(f"Collate error: {e}")
        print(f"Batch size: {len(batch)}")
        for i, sample in enumerate(batch):
            print(f"Sample {i} - x: {sample[0].shape}, y: {sample[1].shape}")
        raise e

# Simplified collate function - alternative if the above is too complex
def simple_collate(batch):
    """Simple collate function that just returns the batch as is"""
    # Process batch without any dimension matching
    processed_batch = []
    for item in batch:
        processed_batch.append(item)
    return default_collate(processed_batch)


# class AudioSeparationDataset(Dataset):
#     """Dataset for audio source separation"""
    
#     def _init_(self, dataset_path, segment_length=44100*3, transforms=None):
#         """
#         Args:
#             dataset_path: Path to MUSDB18 or similar dataset
#             segment_length: Length of audio segments to train on (in samples)
#             transforms: Optional transforms to apply to the data
#         """
#         self.dataset_path = dataset_path
#         self.segment_length = segment_length
#         self.transforms = transforms
        
#         # Get all mixture/source pairs
#         self.tracks = []
#         for track_name in os.listdir(dataset_path):
#             track_path = os.path.join(dataset_path, track_name)
#             if os.path.isdir(track_path):
#                 self.tracks.append(track_name)
        
#         print(f"Found {len(self.tracks)} tracks in the dataset")
    
#     def _len_(self):
#         return len(self.tracks) * 10  # Create 10 random segments per track
    
#     def _getitem_(self, idx):
#         # Get the track and create a random segment
#         track_idx = idx % len(self.tracks)
#         track_name = self.tracks[track_idx]
#         track_path = os.path.join(self.dataset_path, track_name)
        
#         # Load mixture
#         mixture_path = os.path.join(track_path, "mixture.wav")
#         mixture, _ = librosa.load(mixture_path, sr=SAMPLE_RATE, mono=True)
        
#         # Load sources (customize these based on your dataset format)
#         vocals_path = os.path.join(track_path, "vocals.wav")
#         drums_path = os.path.join(track_path, "drums.wav")
#         bass_path = os.path.join(track_path, "bass.wav")
#         other_path = os.path.join(track_path, "other.wav")
        
#         vocals, _ = librosa.load(vocals_path, sr=SAMPLE_RATE, mono=True)
#         drums, _ = librosa.load(drums_path, sr=SAMPLE_RATE, mono=True)
#         bass, _ = librosa.load(bass_path, sr=SAMPLE_RATE, mono=True)
#         other, _ = librosa.load(other_path, sr=SAMPLE_RATE, mono=True)
        
#         # Random crop to segment_length
#         if len(mixture) > self.segment_length:
#             max_start_idx = len(mixture) - self.segment_length
#             start_idx = np.random.randint(0, max_start_idx)
#             mixture = mixture[start_idx:start_idx + self.segment_length]
#             vocals = vocals[start_idx:start_idx + self.segment_length]
#             drums = drums[start_idx:start_idx + self.segment_length]
#             bass = bass[start_idx:start_idx + self.segment_length]
#             other = other[start_idx:start_idx + self.segment_length]
#         else:
#             # Pad if track is shorter than segment_length
#             mixture = librosa.util.fix_length(mixture, self.segment_length)
#             vocals = librosa.util.fix_length(vocals, self.segment_length)
#             drums = librosa.util.fix_length(drums, self.segment_length)
#             bass = librosa.util.fix_length(bass, self.segment_length)
#             other = librosa.util.fix_length(other, self.segment_length)
        
#         # Convert to spectrograms
#         mixture_spec = np.abs(librosa.stft(mixture, n_fft=N_FFT, hop_length=HOP_LENGTH))
#         vocals_spec = np.abs(librosa.stft(vocals, n_fft=N_FFT, hop_length=HOP_LENGTH))
#         drums_spec = np.abs(librosa.stft(drums, n_fft=N_FFT, hop_length=HOP_LENGTH))
#         bass_spec = np.abs(librosa.stft(bass, n_fft=N_FFT, hop_length=HOP_LENGTH))
#         other_spec = np.abs(librosa.stft(other, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
#         # Log-scale the spectrograms (better for training)
#         mixture_spec = librosa.amplitude_to_db(mixture_spec, ref=np.max)
#         vocals_spec = librosa.amplitude_to_db(vocals_spec, ref=np.max)
#         drums_spec = librosa.amplitude_to_db(drums_spec, ref=np.max)
#         bass_spec = librosa.amplitude_to_db(bass_spec, ref=np.max)
#         other_spec = librosa.amplitude_to_db(other_spec, ref=np.max)
        
#         # Normalize
#         mixture_spec = (mixture_spec - mixture_spec.mean()) / (mixture_spec.std() + 1e-8)
#         vocals_spec = (vocals_spec - vocals_spec.mean()) / (vocals_spec.std() + 1e-8)
#         drums_spec = (drums_spec - drums_spec.mean()) / (drums_spec.std() + 1e-8)
#         bass_spec = (bass_spec - bass_spec.mean()) / (bass_spec.std() + 1e-8)
#         other_spec = (other_spec - other_spec.mean()) / (other_spec.std() + 1e-8)
        
#         # Prepare inputs and targets
#         x = torch.from_numpy(mixture_spec).float().unsqueeze(0)  # Add channel dimension
#         y_vocals = torch.from_numpy(vocals_spec).float().unsqueeze(0)
#         y_drums = torch.from_numpy(drums_spec).float().unsqueeze(0)
#         y_bass = torch.from_numpy(bass_spec).float().unsqueeze(0)
#         y_other = torch.from_numpy(other_spec).float().unsqueeze(0)
        
#         # Stack target stems along channel dimension
#         y = torch.cat([y_vocals, y_drums, y_bass, y_other], dim=0)
        
#         return x, y, mixture, vocals, drums, bass, other

import musdb
from torch.utils.data import Dataset

class AudioSeparationDataset(Dataset):
    """Dataset for audio source separation using musdb package"""
    
    def __init__(self, root_dir=None, subset="train", segment_length=44100*3, transforms=None):
        """
        Args:
            root_dir: Path to MUSDB18 dataset (None to use default path)
            subset: 'train' or 'test'
            segment_length: Length of audio segments to train on (in samples)
            transforms: Optional transforms to apply to the data
        """
        self.segment_length = segment_length
        self.transforms = transforms
        
        # Initialize the musdb dataset with auto-download if needed
        self.mus = musdb.DB(root=root_dir, subsets=[subset], download=True)
        print(f"Found {len(self.mus)} tracks in the {subset} dataset")
    
    def __len__(self):
        return len(self.mus) * 10  # Create 10 random segments per track
    
    # def __getitem__(self, idx):
    #     # Get the track and create a random segment
    #     track_idx = idx % len(self.mus)
    #     track = self.mus[track_idx]
        
    #     # Random crop to segment_length
    #     track_length = track.audio.shape[0]
    #     if track_length > self.segment_length:
    #         max_start_idx = track_length - self.segment_length
    #         start_idx = np.random.randint(0, max_start_idx)
    #         track.chunk_start = start_idx
    #         track.chunk_duration = self.segment_length / track.rate
    #     else:
    #         track.chunk_start = 0
    #         track.chunk_duration = None  # Use entire track if shorter than segment_length
        
    #     # Get audio data
    #     mixture = track.audio.mean(axis=1)  # Convert to mono
    #     vocals = track.targets['vocals'].audio.mean(axis=1)
    #     drums = track.targets['drums'].audio.mean(axis=1)
    #     bass = track.targets['bass'].audio.mean(axis=1)
    #     other = track.targets['other'].audio.mean(axis=1)
        
    #     # Pad if needed
    #     if len(mixture) < self.segment_length:
    #         mixture = librosa.util.fix_length(mixture, size = self.segment_length)
    #         vocals = librosa.util.fix_length(vocals, size = self.segment_length)
    #         drums = librosa.util.fix_length(drums, size = self.segment_length)
    #         bass = librosa.util.fix_length(bass, size = self.segment_length)
    #         other = librosa.util.fix_length(other, size = self.segment_length)
        
    #     # Convert to spectrograms
    #     mixture_spec = np.abs(librosa.stft(mixture, n_fft=N_FFT, hop_length=HOP_LENGTH))
    #     vocals_spec = np.abs(librosa.stft(vocals, n_fft=N_FFT, hop_length=HOP_LENGTH))
    #     drums_spec = np.abs(librosa.stft(drums, n_fft=N_FFT, hop_length=HOP_LENGTH))
    #     bass_spec = np.abs(librosa.stft(bass, n_fft=N_FFT, hop_length=HOP_LENGTH))
    #     other_spec = np.abs(librosa.stft(other, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
    #     # Log-scale the spectrograms
    #     mixture_spec = librosa.amplitude_to_db(mixture_spec, ref=np.max)
    #     vocals_spec = librosa.amplitude_to_db(vocals_spec, ref=np.max)
    #     drums_spec = librosa.amplitude_to_db(drums_spec, ref=np.max)
    #     bass_spec = librosa.amplitude_to_db(bass_spec, ref=np.max)
    #     other_spec = librosa.amplitude_to_db(other_spec, ref=np.max)
        
    #     # Normalize
    #     mixture_spec = (mixture_spec - mixture_spec.mean()) / (mixture_spec.std() + 1e-8)
    #     vocals_spec = (vocals_spec - vocals_spec.mean()) / (vocals_spec.std() + 1e-8)
    #     drums_spec = (drums_spec - drums_spec.mean()) / (drums_spec.std() + 1e-8)
    #     bass_spec = (bass_spec - bass_spec.mean()) / (bass_spec.std() + 1e-8)
    #     other_spec = (other_spec - other_spec.mean()) / (other_spec.std() + 1e-8)
        
    #     # Prepare inputs and targets
    #     x = torch.from_numpy(mixture_spec).float().unsqueeze(0)  # Add channel dimension
    #     y_vocals = torch.from_numpy(vocals_spec).float().unsqueeze(0)
    #     y_drums = torch.from_numpy(drums_spec).float().unsqueeze(0)
    #     y_bass = torch.from_numpy(bass_spec).float().unsqueeze(0)
    #     y_other = torch.from_numpy(other_spec).float().unsqueeze(0)
        
    #     # Stack target stems along channel dimension
    #     y = torch.cat([y_vocals, y_drums, y_bass, y_other], dim=0)
        
    #     return x, y, mixture, vocals, drums, bass, other
    def __getitem__(self, idx):
    # Get the track
        track_idx = idx % len(self.mus)
        track = self.mus[track_idx]
        
        # Define a fixed segment duration that will work with our STFT parameters
        segment_duration = 3.0  # seconds
        target_hop_frames = 128  # target number of time frames
        
        # Calculate the exact number of samples needed for this segment duration
        # Given the FFT parameters and desired number of frames
        segment_samples = ((target_hop_frames - 1) * HOP_LENGTH) + N_FFT
        
        # Access the full track
        track.chunk_start = 0
        track.chunk_duration = None
        
        # Extract audio
        full_mixture = track.audio.mean(axis=1)  # Convert to mono
        full_vocals = track.targets['vocals'].audio.mean(axis=1)
        full_drums = track.targets['drums'].audio.mean(axis=1)
        full_bass = track.targets['bass'].audio.mean(axis=1)
        full_other = track.targets['other'].audio.mean(axis=1)
        
        # Get a random segment
        if len(full_mixture) > segment_samples:
            max_start_idx = len(full_mixture) - segment_samples
            start_idx = np.random.randint(0, max_start_idx)
            
            mixture = full_mixture[start_idx:start_idx + segment_samples]
            vocals = full_vocals[start_idx:start_idx + segment_samples]
            drums = full_drums[start_idx:start_idx + segment_samples]
            bass = full_bass[start_idx:start_idx + segment_samples]
            other = full_other[start_idx:start_idx + segment_samples]
        else:
            # If the track is too short, pad it
            mixture = np.zeros(segment_samples)
            vocals = np.zeros(segment_samples)
            drums = np.zeros(segment_samples)
            bass = np.zeros(segment_samples)
            other = np.zeros(segment_samples)
            
            # Fill with available audio
            mixture[:len(full_mixture)] = full_mixture
            vocals[:len(full_vocals)] = full_vocals
            drums[:len(full_drums)] = full_drums
            bass[:len(full_bass)] = full_bass
            other[:len(full_other)] = full_other
        
        # Verify all audio segments have the same length
        assert len(mixture) == segment_samples
        assert len(vocals) == segment_samples
        assert len(drums) == segment_samples
        assert len(bass) == segment_samples
        assert len(other) == segment_samples
        
        # Compute STFT with fixed parameters
        mixture_spec = np.abs(librosa.stft(mixture, n_fft=N_FFT, hop_length=HOP_LENGTH))
        vocals_spec = np.abs(librosa.stft(vocals, n_fft=N_FFT, hop_length=HOP_LENGTH))
        drums_spec = np.abs(librosa.stft(drums, n_fft=N_FFT, hop_length=HOP_LENGTH))
        bass_spec = np.abs(librosa.stft(bass, n_fft=N_FFT, hop_length=HOP_LENGTH))
        other_spec = np.abs(librosa.stft(other, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
        # Log-scale the spectrograms
        mixture_spec = librosa.amplitude_to_db(mixture_spec, ref=np.max)
        vocals_spec = librosa.amplitude_to_db(vocals_spec, ref=np.max)
        drums_spec = librosa.amplitude_to_db(drums_spec, ref=np.max)
        bass_spec = librosa.amplitude_to_db(bass_spec, ref=np.max)
        other_spec = librosa.amplitude_to_db(other_spec, ref=np.max)
        
        # Normalize
        mixture_spec = (mixture_spec - mixture_spec.mean()) / (mixture_spec.std() + 1e-8)
        vocals_spec = (vocals_spec - vocals_spec.mean()) / (vocals_spec.std() + 1e-8)
        drums_spec = (drums_spec - drums_spec.mean()) / (drums_spec.std() + 1e-8)
        bass_spec = (bass_spec - bass_spec.mean()) / (bass_spec.std() + 1e-8)
        other_spec = (other_spec - other_spec.mean()) / (other_spec.std() + 1e-8)
        
        # Convert to tensors
        x = torch.from_numpy(mixture_spec).float().unsqueeze(0)  # Add channel dimension
        y_vocals = torch.from_numpy(vocals_spec).float().unsqueeze(0)
        y_drums = torch.from_numpy(drums_spec).float().unsqueeze(0)
        y_bass = torch.from_numpy(bass_spec).float().unsqueeze(0)
        y_other = torch.from_numpy(other_spec).float().unsqueeze(0)
        
        # Stack target stems along channel dimension
        y = torch.cat([y_vocals, y_drums, y_bass, y_other], dim=0)
        
        # Print shapes for debugging
        # print(f"x shape: {x.shape}, y shape: {y.shape}")
        
        return x, y, mixture, vocals, drums, bass, other

# Define a U-Net model for source separation
class UNetSourceSeparation(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        """
        Args:
            n_channels: Number of input channels (1 for mono audio)
            n_classes: Number of output sources (e.g., 4 for vocals, drums, bass, other)
        """
        super(UNetSourceSeparation, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._block(n_channels, 16, name="enc1")
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(16, 32, name="enc2")
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(32, 64, name="enc3")
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(64, 128, name="enc4")
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(128, 256, name="bottleneck")
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self._block(256, 128, name="dec4")  # 256 = 128 + 128 (skip connection)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._block(128, 64, name="dec3")   # 128 = 64 + 64 (skip connection)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._block(64, 32, name="dec2")    # 64 = 32 + 32 (skip connection)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._block(32, 16, name="dec1")    # 32 = 16 + 16 (skip connection)
        
        # Output layer
        self.out = nn.Conv2d(16, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        # enc1 = self.enc1(x)
        # enc2 = self.enc2(self.pool1(enc1))
        # enc3 = self.enc3(self.pool2(enc2))
        # enc4 = self.enc4(self.pool3(enc3))
        
        # # Bottleneck
        # bottleneck = self.bottleneck(self.pool4(enc4))
        
        # # Decoder with skip connections
        # dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        # dec4 = self.dec4(dec4)
        
        # dec3 = self.upconv3(dec4)
        # dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        # dec3 = self.dec3(dec3)
        
        # dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        # dec2 = self.dec2(dec2)
        
        # dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        # dec1 = self.dec1(dec1)
        
        # # Output layer
        # out = self.out(dec1)
        # return out
        
        def forward(self, x):
            orig_shape = x.shape
            
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool1(enc1))
            enc3 = self.enc3(self.pool2(enc2))
            enc4 = self.enc4(self.pool3(enc3))
            
            # Bottleneck
            bottleneck = self.bottleneck(self.pool4(enc4))
            
            # Decoder with skip connections
            dec4 = self.upconv4(bottleneck)
            # Resize enc4 to match dec4 if needed
            if dec4.shape[2:] != enc4.shape[2:]:
                enc4_resized = F.interpolate(enc4, size=dec4.shape[2:], mode='bilinear', align_corners=False)
            else:
                enc4_resized = enc4
            dec4 = torch.cat((dec4, enc4_resized), dim=1)
            dec4 = self.dec4(dec4)
            
            dec3 = self.upconv3(dec4)
            # Resize enc3 to match dec3 if needed
            if dec3.shape[2:] != enc3.shape[2:]:
                enc3_resized = F.interpolate(enc3, size=dec3.shape[2:], mode='bilinear', align_corners=False)
            else:
                enc3_resized = enc3
            dec3 = torch.cat((dec3, enc3_resized), dim=1)
            dec3 = self.dec3(dec3)
            
            dec2 = self.upconv2(dec3)
            # Resize enc2 to match dec2 if needed
            if dec2.shape[2:] != enc2.shape[2:]:
                enc2_resized = F.interpolate(enc2, size=dec2.shape[2:], mode='bilinear', align_corners=False)
            else:
                enc2_resized = enc2
            dec2 = torch.cat((dec2, enc2_resized), dim=1)
            dec2 = self.dec2(dec2)
            
            dec1 = self.upconv1(dec2)
            # Resize enc1 to match dec1 if needed
            if dec1.shape[2:] != enc1.shape[2:]:
                enc1_resized = F.interpolate(enc1, size=dec1.shape[2:], mode='bilinear', align_corners=False)
            else:
                enc1_resized = enc1
            dec1 = torch.cat((dec1, enc1_resized), dim=1)
            dec1 = self.dec1(dec1)
            
            # Output layer
            out = self.out(dec1)
            
            # Ensure the output has the same spatial dimensions as the input
            if out.shape[2:] != orig_shape[2:]:
                out = F.interpolate(out, size=orig_shape[2:], mode='bilinear', align_corners=False)
            
            return out

    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the source separation model"""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for x, y, _, _, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):            
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for x, y, _, _, _, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                outputs = model(x)
                val_loss = criterion(outputs, y)
                
                running_val_loss += val_loss.item() * x.size(0)
            
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_separation_model.pth")
            print("Saved best model checkpoint.")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    return model, train_losses, val_losses

def visualize_separation_results(model, test_loader):
    """Visualize source separation results on test data"""
    model.eval()
    
    # Get a batch from the test loader
    x, y_true, mixture, vocals, drums, bass, other = next(iter(test_loader))
    x = x.to(DEVICE)
    
    # Perform separation
    with torch.no_grad():
        y_pred = model(x)
    
    # Move tensors back to CPU for visualization
    x = x.cpu().numpy()[0, 0]  # Take first item in batch, remove channel dim
    y_pred = y_pred.cpu().numpy()[0]  # Take first item in batch
    y_true = y_true.cpu().numpy()[0]  # Take first item in batch
    
    # Labels for the sources
    source_names = ['Vocals', 'Drums', 'Bass', 'Other']
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot mixture
    plt.subplot(3, 4, 1)
    plt.title('Mixture Spectrogram')
    plt.imshow(x, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    
    # Plot each source (ground truth vs prediction)
    for i in range(4):
        # Ground truth
        plt.subplot(3, 4, i+5)
        plt.title(f'True {source_names[i]}')
        plt.imshow(y_true[i], aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        
        # Prediction
        plt.subplot(3, 4, i+9)
        plt.title(f'Predicted {source_names[i]}')
        plt.imshow(y_pred[i], aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('separation_results.png')
    plt.close()

# def main():
#     """Main function to run the training process"""
#     # Dataset paths
#     dataset_path = "path/to/musdb18"  # Replace with your dataset path
    
#     # Create datasets
#     full_dataset = AudioSeparationDataset(dataset_path)
    
#     # Split dataset
#     train_size = int(0.8 * len(full_dataset))
#     val_size = int(0.1 * len(full_dataset))
#     test_size = len(full_dataset) - train_size - val_size
    
#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#         full_dataset, [train_size, val_size, test_size]
#     )
    
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
#     # Initialize model
#     model = UNetSourceSeparation(n_channels=1, n_classes=4).to(DEVICE)
    
#     # Define loss function and optimizer
#     criterion = nn.L1Loss()  # L1 loss works well for spectrogram differences
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # Train model
#     trained_model, train_losses, val_losses = train_model(
#         model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS
#     )
    
#     # Visualize separation results
#     visualize_separation_results(trained_model, test_loader)
    
#     print("Training complete! Best model saved as 'best_separation_model.pth'")

# if __name__ == "_main_":
#     main()

def main():
    """Main function to run the training process"""
    # Create datasets - no path needed, will download if necessary
    # train_dataset = AudioSeparationDataset(root_dir=None, subset="train")
    # test_dataset = AudioSeparationDataset(root_dir=None, subset="test")
    
    # # Split train dataset into train and validation
    # train_size = int(0.9 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    
    # train_subset, val_subset = torch.utils.data.random_split(
    #     train_dataset, [train_size, val_size]
    # )
    
    # # Create data loaders
    # train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    train_dataset = AudioSeparationDataset(root_dir=None, subset="train")
    test_dataset = AudioSeparationDataset(root_dir=None, subset="test")
    
    # Split train dataset into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Try first with a simple approach - no custom collate function
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                            num_workers=4)
    
    # Initialize model
    model = UNetSourceSeparation(n_channels=1, n_classes=4).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()  # L1 loss works well for spectrogram differences
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS
    )
    
    # Visualize separation results
    visualize_separation_results(trained_model, test_loader)
    
    print("Training complete! Best model saved as 'best_separation_model.pth'")

if __name__ == "__main__":
    main()