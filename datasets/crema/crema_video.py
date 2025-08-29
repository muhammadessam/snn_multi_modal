from torch.utils.data import Dataset
from pathlib import Path
from timm.utils import random_seed
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.transforms import v2
from torchvision.tv_tensors import Video
from PIL import Image

# CREMA-D emotion labels
EMOTION_LABELS = {
    "ANG": 0,  # Anger
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happy
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sad
}


class CremaVideo(Dataset):
    """
    Dataset class for CREMA-D using pre-extracted frames for speed.
    Accepts a transform object for applying augmentations.
    """

    def __init__(
        self, data_path, split="train", num_frames=8, frame_size=128, seed=42, use_augmentation=True, transform=None
    ):
        self.data_path = Path(data_path)
        self.split = split
        split_path = self.data_path / split

        self.num_frames = num_frames
        self.frame_size = (frame_size, frame_size) if isinstance(frame_size, int) else frame_size
        self.samples = []
        self.use_augmentation = use_augmentation and (split == "train")
        self.transform = transform

        random_seed(seed)

        # Load samples from the correct train/val subdirectory
        all_frame_folders = [d for d in split_path.iterdir() if d.is_dir()]

        valid_samples = []
        for folder in all_frame_folders:
            filename = folder.name
            parts = filename.split("_")
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in EMOTION_LABELS:
                    frame_files = sorted(list(folder.glob("*.jpg")))
                    if len(frame_files) > 0:
                        valid_samples.append(
                            {
                                "path": str(folder),
                                "label": EMOTION_LABELS[emotion_code],
                                "filename": filename,
                                "frame_files": frame_files,
                            }
                        )
        self.samples = valid_samples

        if self.split == "train":
            random.shuffle(self.samples)

        print(f"Loaded {len(self.samples)} samples for {split} set from {split_path}")
        print(f"Augmentation: {self.use_augmentation}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        frame_files = sample["frame_files"]

        frames = self.load_image_frames(frame_files)

        # Create tensor and wrap it in the Video class for proper augmentation
        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # T, C, H, W
        video_tensor = Video(video_tensor)

        # Apply spatial and color augmentations if a transform is provided
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, label

    def load_image_frames(self, frame_files):
        """Loads a sequence of frames from image files."""
        frames = []

        # Pad or truncate the list of frame files to match num_frames
        if len(frame_files) < self.num_frames:
            frame_files.extend([frame_files[-1]] * (self.num_frames - len(frame_files)))
        elif len(frame_files) > self.num_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        for file_path in frame_files:
            try:
                img = Image.open(file_path).convert("RGB")
                frames.append(np.array(img))
            except Exception as e:
                print(f"Warning: Could not load image {file_path}. Error: {e}")
                fallback_frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                frames.append(fallback_frame)

        return frames


class CremaVisualizer:
    """A class to visualize the augmentation effects on the CremaVideo dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        # Denormalization parameters for ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.emotion_names = {v: k for k, v in EMOTION_LABELS.items()}

    def denormalize(self, tensor):
        """Denormalizes a tensor for visualization."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

        # From (T, C, H, W) to (T, H, W, C)
        tensor = tensor.transpose(0, 2, 3, 1)

        tensor = self.std * tensor + self.mean
        tensor = np.clip(tensor, 0, 1)
        return tensor

    def visualize_augmentation_effects(self, idx):
        """
        Visualizes the raw and fully augmented frames for a given sample index.
        """
        # --- 1. Get Raw Data ---
        sample = self.dataset.samples[idx]
        emotion_name = self.emotion_names.get(sample["label"], "Unknown")
        raw_frames = self.dataset.load_image_frames(sample["frame_files"])

        # --- 2. Get Fully Augmented Data ---
        # Temporarily set use_augmentation to True to get the augmented output
        original_aug_state = self.dataset.use_augmentation
        self.dataset.use_augmentation = True
        full_aug_tensor, _ = self.dataset[idx]
        self.dataset.use_augmentation = original_aug_state  # Restore state

        full_aug_frames = self.denormalize(full_aug_tensor)

        # --- 3. Plotting ---
        num_frames = self.dataset.num_frames
        fig, axes = plt.subplots(2, num_frames, figsize=(20, 6))
        fig.suptitle(f"Augmentation Stages for Sample {idx} (Emotion: {emotion_name})", fontsize=16)

        for i in range(num_frames):
            # Plot Raw Frames
            axes[0, i].imshow(raw_frames[i])
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Raw Frames", loc="left", fontsize=12, ha="left", x=-0.2)

            # Plot Fully Augmented Frames
            axes[1, i].imshow(full_aug_frames[i])
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Final Augmented", loc="left", fontsize=12, ha="left", x=-0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def create_dataloaders(data_path, batch_size=10, num_workers=12, frame_size=256, num_frames=16, aug_strength=1.0):
    # --- Example of the new usage pattern ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomRotation(degrees=5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = CremaVideo(
        data_path=data_path,
        split="train",
        num_frames=num_frames,
        frame_size=frame_size,
        use_augmentation=True,
        transform=train_transform,
    )
    val_dataset = CremaVideo(
        data_path=data_path,
        split="val",
        num_frames=num_frames,
        frame_size=frame_size,
        use_augmentation=False,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # This block is for demonstration and debugging.
    # You should have your pre-processed frames in a directory structure like:
    # ./data/crema/Crema_frames/
    #  ├── train/
    #  │   ├── 1001_IEO_ANG_XX/
    #  │   │   ├── frame_0000.jpg
    #  │   │   └── ...
    #  └── val/
    #      └── 1002_IEO_DIS_XX/
    #          ├── frame_0000.jpg
    #          └── ...

    # Path to the root directory of the pre-extracted frames
    preprocessed_data_path = "/user/mohamed.saleh01/u18697/projects/datasets/crema/Crema_frames_8_90_cropped_128"

    # Check if the path exists to avoid errors
    if os.path.exists(preprocessed_data_path):
        print("--- Creating Dataloaders for Visualization ---")
        _, _, train_dataset, _ = create_dataloaders(
            data_path=preprocessed_data_path, batch_size=10, num_workers=12, frame_size=224, num_frames=8
        )

        print("\n--- Initializing Visualizer ---")
        visualizer = CremaVisualizer(train_dataset)

        # Visualize the augmentation effects on the first sample of the training set
        print("Visualizing sample 0...")
        visualizer.visualize_augmentation_effects(2)

        print("\nVisualizing sample 5...")
        visualizer.visualize_augmentation_effects(9)

    else:
        print(f"Visualization skipped: Pre-processed data not found at '{preprocessed_data_path}'")
        print("Please run the `preprocess_videos.py` script first.")
