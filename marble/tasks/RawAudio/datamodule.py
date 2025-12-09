# marble/tasks/RawAudio/datamodule.py

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        
        # Scan directory for audio files
        audio_dir_path = Path(audio_dir)
        if not audio_dir_path.exists():
            raise ValueError(f"Audio directory not found: {audio_dir}")
        
        audio_files = list_audio_files(audio_dir_path, extensions=extensions, recursive=recursive)
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        print(f"Found {len(audio_files)} audio files in {audio_dir}")
        
        # Build metadata on-the-fly using torchaudio.info()
        self.meta: List[dict] = []
        self.resamplers = {}
        
        for audio_path in audio_files:
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
                
                # Prepare resampler if needed
                if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                    self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
                    
            except (OSError, RuntimeError) as e:
                print(f"Warning: Skipping {audio_path} - {e}")
                continue
        
        if len(self.meta) == 0:
            raise ValueError(f"No valid audio files found in {audio_dir}")
        
        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        
        for file_idx, info in enumerate(self.meta):
            orig_sr = info['sample_rate']
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info['channels']
            total_samples = info['num_samples']
            
            if orig_clip_frames <= 0:
                continue
            
            # Number of full clips and remainder
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            
            # Decide whether to keep the last shorter clip
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full
            
            for slice_idx in range(n_slices):
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )
    
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
        # Unpack mapping info
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info['audio_path']
        
        # Compute frame offset and load clip
        offset = slice_idx * orig_clip
        try:
            waveform, _ = torchaudio.load(
                path,
                frame_offset=offset,
                num_frames=orig_clip,
                backend=self.backend
            )  # (orig_channels, orig_clip)
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Failed to load audio file '{path}': {e}") from e
        
        # Channel alignment / downmixing
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
            # Repeat last channel to pad to desired channels
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)
        
        # Pad to target length if short
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))
        
        # Final shape: (self.channels, self.clip_len_target)
        return waveform, None, path


class RawAudioDataModule(BaseDataModule):
    pass
