# marble/tasks/RawAudio/datamodule.py

from pathlib import Path
from typing import List, Tuple, Optional
import random

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from marble.core.base_datamodule import BaseDataModule
from marble.utils.utils import list_audio_files


class SimpleRawAudioDataset(Dataset):
    """
    Simple dataset for raw audio files without JSONL metadata.
    
    Scans directory for audio files and builds metadata on-the-fly using torchaudio.info().
    Splits each audio file into non-overlapping clips of length `clip_seconds` (last clip zero-padded).
    
    Returns (waveform, None, path) for extraction tasks.
    """
    
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int,
        channels: int = 1,
        clip_seconds: float = 15.0,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
        backend: Optional[str] = None,
        extensions: Tuple[str, ...] = ('.wav', '.flac', '.mp3', '.webm', '.mp4'),
        recursive: bool = True,
        max_files: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            audio_dir: Directory containing audio files to scan.
            sample_rate: Target sample rate to which all audio will be resampled.
            channels: Desired number of output channels (e.g., 1 for mono, 2 for stereo).
            clip_seconds: Duration (in seconds) of each clip to slice from the audio.
            channel_mode: How to downmix when `channels == 1` but the audio is multi-channel.
                          Options: "first", "mix", "random".
            min_clip_ratio: Minimum fraction of a final (possibly partial) clip to keep.
            backend: torchaudio backend to use (e.g., "soundfile", "sox_io").
            extensions: File extensions to consider as audio files.
            recursive: If True, search recursively in subdirectories.
            max_files: Optional maximum number of files to process. If set, files are randomly
                      sampled before metadata loading (much faster for large datasets).
                      If None, processes all files.
            random_seed: Random seed for file sampling when max_files is set. If None, uses
                        system random (non-deterministic).
        """
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio
        self.backend = backend
        
        if self.channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}. Must be 'first', 'mix', or 'random'")
        
        audio_dir_path = Path(audio_dir)
        if not audio_dir_path.exists():
            raise ValueError(f"Audio directory not found: {audio_dir}")
        
        audio_files = list_audio_files(audio_dir_path, extensions=extensions, recursive=recursive)
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        total_files = len(audio_files)
        files_after_filtering = total_files
        
        if max_files is not None:
            if max_files <= 0:
                raise ValueError(f"max_files must be positive, got {max_files}")
            if max_files < total_files:
                if random_seed is None:
                    raise ValueError("random_seed must be provided when max_files is set to ensure deterministic file sampling")
                rng = random.Random(random_seed)
                audio_files = rng.sample(audio_files, max_files)
                audio_files = sorted(audio_files)
                files_after_filtering = len(audio_files)
                print(f"Found {total_files} audio files, sampling {files_after_filtering} files (max_files={max_files}, seed={random_seed})")
            else:
                print(f"Found {total_files} audio files (max_files={max_files} >= total, using all files)")
        else:
            print(f"Found {total_files} audio files")
        
        self.meta: List[dict] = []
        self.resamplers = {}
        
        print(f"Loading metadata for {files_after_filtering} files...")
        for audio_path in tqdm(audio_files, desc="Loading metadata", unit="file"):
            try:
                info = torchaudio.info(str(audio_path), backend=self.backend)
                orig_sr = info.sample_rate
                num_samples = info.num_frames
                num_channels = info.num_channels
                
                self.meta.append({
                    "audio_path": str(audio_path),
                    "sample_rate": orig_sr,
                    "num_samples": num_samples,
                    "channels": num_channels
                })
                
                if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                    self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
                    
            except (OSError, RuntimeError) as e:
                print(f"Warning: Skipping {audio_path} - {e}")
                continue
        
        if len(self.meta) == 0:
            raise ValueError(f"No valid audio files found in {audio_dir}")
        
        valid_files = len(self.meta)
        if valid_files < files_after_filtering:
            print(f"Warning: {files_after_filtering - valid_files} files failed metadata loading")
        print(f"Successfully loaded {valid_files} files, generating clips...")
        
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        
        for file_idx, info in enumerate(self.meta):
            orig_sr = info['sample_rate']
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info['channels']
            total_samples = info['num_samples']
            
            if orig_clip_frames <= 0:
                continue
            
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full
            
            for slice_idx in range(n_slices):
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )
        
        total_clips = len(self.index_map)
        print(f"Dataset initialized: {valid_files} files, {total_clips} clips (avg {total_clips/valid_files:.1f} clips/file)")
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx: int):
        """
        Load and return one audio clip.
        
        Returns:
            waveform: torch.Tensor, shape (self.channels, self.clip_len_target)
            target: None (for extraction tasks)
            path: str (audio file path)
        """
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info['audio_path']
        
        offset = slice_idx * orig_clip
        try:
            waveform, _ = torchaudio.load(
                path,
                frame_offset=offset,
                num_frames=orig_clip,
                backend=self.backend
            )
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Failed to load audio file '{path}': {e}") from e
        
        if orig_channels >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                else:  # random
                    choice = torch.randint(0, orig_channels + 1, (1,)).item()
                    if choice == orig_channels:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        waveform = waveform[choice:choice+1]
            else:
                waveform = waveform[:self.channels]
        else:
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)
        
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)
        
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))
        
        return waveform, None, path


class RawAudioDataModule(BaseDataModule):
    @staticmethod
    def _collate_fn(batch):
        """Collate function that handles None targets and stacks waveforms.
        
        Required because SimpleRawAudioDataset returns (waveform, None, path),
        and PyTorch's default_collate cannot handle None values.
        """
        waveforms = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        paths = [item[2] for item in batch]
        return torch.stack(waveforms), targets, paths
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self._collate_fn,
        )
