from torch.utils.data import Dataset
from pathlib import Path
from timm.utils import random_seed
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import librosa
import soundfile as sf
from torchvision.transforms import v2

# CREMA-D emotion labels
EMOTION_LABELS = {
    "ANG": 0,  # Anger
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happy
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sad
}


class CremaAudio(Dataset):
    """
    Dataset class for CREMA-D using raw audio files.
    Supports train/val/test splits using split files and accepts transform objects for applying augmentations.
    
    Expected directory structure:
        root_path/
        ├── AudioWAV/
        │   ├── {actor_id}_{sentence}_{emotion}_{intensity}.wav
        │   └── ...
        ├── 80_train_cre.txt
        ├── 10_val_cre.txt
        └── 10_test_cre.txt
    
    Emotion codes: ANG (Anger), DIS (Disgust), FEA (Fear), HAP (Happy), NEU (Neutral), SAD (Sad)
    """

    def __init__(
        self, 
        data_path, 
        split="train", 
        time_steps=4, 
        target_length=40800,
        seed=42, 
        use_augmentation=True, 
        transform=None,
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=320
    ):
        # Expect data_path to be the root directory containing AudioWAV and split files
        self.root_path = Path(data_path)
        self.audio_path = self.root_path / "AudioWAV"
        self.split = split
        self.time_steps = time_steps
        self.target_length = target_length
        self.use_augmentation = use_augmentation and (split == "train")
        self.transform = transform
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        random_seed(seed)
        
        # Load samples from the appropriate split file
        split_file_map = {
            "train": "80_train_cre.txt",
            "val": "10_val_cre.txt", 
            "test": "10_test_cre.txt"
        }
        
        if split not in split_file_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of: {list(split_file_map.keys())}")
            
        split_file = self.root_path / split_file_map[split]
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")
            
        # Read split file and load samples
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        filename_flv, emotion_name = parts
                        # Convert .flv to .wav extension
                        filename_wav = filename_flv.replace('.flv', '.wav')
                        audio_file_path = self.audio_path / filename_wav
                        
                        # Map emotion name to label
                        emotion_map = {
                            "Anger": "ANG", "Disgust": "DIS", "Fear": "FEA",
                            "Happy": "HAP", "Neutral": "NEU", "Sad": "SAD"
                        }
                        
                        if emotion_name in emotion_map and audio_file_path.exists():
                            emotion_code = emotion_map[emotion_name]
                            self.samples.append({
                                "path": str(audio_file_path),
                                "label": EMOTION_LABELS[emotion_code],
                                "filename": filename_wav.replace('.wav', ''),
                            })
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid audio files found for {split} split")
            
        if self.split == "train":
            random.shuffle(self.samples)
        
        # Librosa will handle mel spectrogram generation directly
        
        print(f"Loaded {len(self.samples)} samples for {split} set from {data_path}")
        print(f"Augmentation: {self.use_augmentation}")
        print(f"Audio parameters: sr={sample_rate}, n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
        print(f"Target length: {target_length} samples ({target_length/sample_rate:.2f}s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        audio_path = sample["path"]
        
        # Load audio using librosa and pad/truncate to target length
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            # Pad or truncate to target length
            if len(waveform) < self.target_length:
                # Pad with zeros
                waveform = np.pad(waveform, (0, self.target_length - len(waveform)), mode='constant')
            elif len(waveform) > self.target_length:
                # Truncate to target length
                waveform = waveform[:self.target_length]
        except Exception as e:
            print(f"Warning: Could not load audio {audio_path}. Error: {e}")
            # Create dummy audio at target length
            waveform = np.zeros(self.target_length)
        
        # Generate mel spectrogram from complete audio
        spectrogram = self.audio_to_spectrogram(waveform)
        
        # Apply transforms if provided (convert to 3-channel for compatibility)
        if self.transform:
            # Convert to 3-channel for compatibility with vision transforms
            spec_3ch = spectrogram.repeat(3, 1, 1)  # (3, n_mels, time)
            transformed_spec = self.transform(spec_3ch)
            # Take only the first channel back
            spectrogram = transformed_spec[0:1]  # (1, n_mels, time)
        
        return spectrogram, label

    def audio_to_spectrogram(self, waveform):
        """
        Generate a mel spectrogram from the complete audio using librosa.
        Returns tensor of shape (1, n_mels, time) where time is on x-axis and n_mels on y-axis.
        """
        # Generate mel spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to tensor and add channel dimension
        # mel_spec shape: (n_mels, time) -> (1, n_mels, time)
        mel_tensor = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0)
        
        return mel_tensor


class CremaAudioVisualizer:
    """A class to visualize the spectrograms generated from the CremaAudio dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.emotion_names = {v: k for k, v in EMOTION_LABELS.items()}

    def visualize_spectrograms(self, idx):
        """
        Visualizes the spectrogram for a given sample index.
        """
        sample = self.dataset.samples[idx]
        emotion_name = self.emotion_names.get(sample["label"], "Unknown")
        
        # Get the spectrogram
        spectrogram, _ = self.dataset[idx]  # (1, n_mels, time)
        
        # Remove channel dimension for visualization
        spec = spectrogram[0].cpu().numpy()  # (n_mels, time)
        
        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(f"Audio Spectrogram for Sample {idx} (Emotion: {emotion_name})", fontsize=16)
        
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title("Mel Spectrogram")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bins")
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def visualize_waveform_and_spectrogram(self, idx):
        """
        Visualizes both the original waveform and the generated spectrogram.
        """
        sample = self.dataset.samples[idx]
        emotion_name = self.emotion_names.get(sample["label"], "Unknown")
        
        # Load original audio using librosa
        try:
            waveform_np, sr = librosa.load(sample["path"], sr=self.dataset.sample_rate, mono=True)
        except Exception as e:
            print(f"Warning: Could not load audio {sample['path']}. Error: {e}")
            waveform_np = np.zeros(self.dataset.sample_rate * 2)
            sr = self.dataset.sample_rate
        
        # Get the spectrogram
        spectrogram, _ = self.dataset[idx]  # (1, n_mels, time)
        spec = spectrogram[0].cpu().numpy()  # (n_mels, time)
        
        # Create subplot layout: waveform on top, spectrogram on bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot waveform
        time_axis = np.linspace(0, len(waveform_np) / self.dataset.sample_rate, len(waveform_np))
        ax1.plot(time_axis, waveform_np)
        ax1.set_title(f"Original Waveform (Emotion: {emotion_name})")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # Plot spectrogram
        im = ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title("Mel Spectrogram")
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Bins")
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()


def create_dataloaders(data_path, batch_size=32, num_workers=4, target_length=64000, time_steps=4):
    """
    Create train, validation, and test dataloaders for CREMA-D audio dataset.
    Returns spectrograms in shape (1, n_mels, time) - model handles temporal replication.
    
    Args:
        data_path: Path to dataset root folder (should contain AudioWAV subdirectory and split files)
        batch_size: Batch size for training (validation and test use same batch size)
        num_workers: Number of data loading workers
        target_length: Target audio length in samples
        time_steps: Number of temporal steps for SNN processing
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    # Simple normalization transform (spectrograms are already normalized in the dataset)
    train_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=False),  # Don't scale since we're already normalized
        # Add any audio-specific augmentations here if needed
    ])

    val_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=False),
    ])

    # Create datasets for all three splits
    train_dataset = CremaAudio(
        data_path=data_path,
        split="train",
        time_steps=time_steps,
        target_length=target_length,
        use_augmentation=True,
        transform=train_transform,
    )
    
    val_dataset = CremaAudio(
        data_path=data_path,
        split="val",
        time_steps=time_steps,
        target_length=target_length,
        use_augmentation=False,
        transform=val_transform,
    )
    
    test_dataset = CremaAudio(
        data_path=data_path,
        split="test",
        time_steps=time_steps,
        target_length=target_length,
        use_augmentation=False,
        transform=val_transform,
    )

    # Create dataloaders for all three splits
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Path to the root directory containing AudioWAV subfolder and split files
    audio_data_path = "/user/mohamed.saleh01/u18697/projects/datasets/crema"
    
    # Check if the path exists
    if os.path.exists(audio_data_path):
        print("--- Creating Audio Dataloaders for Testing (3-Split Version) ---")
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
            data_path=audio_data_path, batch_size=16, num_workers=4, target_length=64000, time_steps=4
        )
        
        print(f"Dataset sizes:")
        print(f"  - Training: {len(train_dataset)} samples")
        print(f"  - Validation: {len(val_dataset)} samples") 
        print(f"  - Test: {len(test_dataset)} samples")
        print(f"  - Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
        
        # Test loading a sample
        print("\n--- Testing Sample Loading ---")
        sample_spectrogram, sample_label = train_dataset[0]
        print(f"Sample shape: {sample_spectrogram.shape}")
        print(f"Expected shape: (channels=1, n_mels=128, time)")
        print(f"Sample label: {sample_label} ({[k for k, v in EMOTION_LABELS.items() if v == sample_label][0]})")
        print(f"Note: Shape is (1, n_mels, time) - model will replicate for time steps")
        
        print("\n--- Initializing Visualizer ---")
        visualizer = CremaAudioVisualizer(train_dataset)
        
        # Visualize the spectrogram for the first sample
        print("Visualizing sample 0...")
        visualizer.visualize_spectrograms(0)
        
        print("\nVisualizing waveform and spectrogram for sample 0...")
        visualizer.visualize_waveform_and_spectrogram(0)
        
        # Test that all splits are accessible
        print("\n--- Testing All Splits ---")
        try:
            train_sample = next(iter(train_loader))
            print(f"✓ Training split loaded successfully: {train_sample[0].shape}")
        except Exception as e:
            print(f"✗ Training split failed: {e}")
            
        try:
            val_sample = next(iter(val_loader))
            print(f"✓ Validation split loaded successfully: {val_sample[0].shape}")
        except Exception as e:
            print(f"✗ Validation split failed: {e}")
            
        try:
            test_sample = next(iter(test_loader))
            print(f"✓ Test split loaded successfully: {test_sample[0].shape}")
        except Exception as e:
            print(f"✗ Test split failed: {e}")
        
    else:
        print(f"Audio data not found at '{audio_data_path}'")
        print("Please ensure your data follows the expected directory structure:")
        print("  data_path/")
        print("  ├── AudioWAV/")
        print("  ├── 80_train_cre.txt") 
        print("  ├── 10_val_cre.txt")
        print("  └── 10_test_cre.txt")
        print("Each split file should contain audio filenames with emotion labels.")