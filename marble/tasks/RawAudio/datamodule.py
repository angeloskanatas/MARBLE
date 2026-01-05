# marble/tasks/RawAudio/datamodule.py

import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import random

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import soundfile as sf

from marble.core.base_datamodule import BaseDataModule
from marble.utils.utils import list_audio_files


class SimpleRawAudioDataset(Dataset):
    """
    Dataset for raw audio files used for extraction tasks.
    
    Supports directory scanning or JSONL file loading.
    Splits each audio file into non-overlapping clips of length `clip_seconds`.
    Returns (waveform, None, path).
    """
    
    def __init__(
        self,
        audio_dir: Optional[str] = None,
        jsonl: Optional[str] = None,
        sample_rate: int = 24000,
        channels: int = 1,
        clip_seconds: float = 15.0,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
        backend: Optional[str] = None,
        extensions: Tuple[str, ...] = ('.wav', '.flac', '.mp3', '.webm', '.mp4'),
        recursive: bool = True,
        max_files: Optional[int] = None,
        random_seed: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
        clips_per_file: Optional[int] = None,
        clip_selection_seed: Optional[int] = None,
    ):
        """
        Args:
            audio_dir: Directory containing audio files to scan (if jsonl is None).
            jsonl: Path to JSONL file with audio metadata (if audio_dir is None).
                   Each line: {"audio_path": "...", "sample_rate": ..., "num_samples": ..., "channels": ...}
            sample_rate: Target sample rate to which all audio will be resampled.
            channels: Desired number of output channels (e.g., 1 for mono, 2 for stereo).
            clip_seconds: Duration (in seconds) of each clip to slice from the audio.
            channel_mode: How to downmix when `channels == 1` but the audio is multi-channel.
                          Options: "first", "mix", "random".
            min_clip_ratio: Minimum fraction of a final (possibly partial) clip to keep.
            backend: torchaudio backend to use (e.g., "soundfile", "sox_io").
            extensions: File extensions to consider as audio files (if audio_dir is specified).
            recursive: If True, search recursively in subdirectories (if audio_dir is specified).
            max_files: Optional maximum number of files to process. If set, files are randomly
                      sampled before metadata loading.
            random_seed: Random seed for file sampling when max_files is set.
            max_duration_seconds: Optional maximum duration in seconds. Files exceeding this are filtered.
            clips_per_file: Number of clips to extract per file. If None, extracts all non-overlapping clips.
                            If 1, extracts one random clip per file. If >1, extracts that many random clips.
            clip_selection_seed: Random seed for clip selection when clips_per_file is set.
                                Required if clips_per_file is not None.
        """
        if (audio_dir is None) == (jsonl is None):
            raise ValueError(
                "Must provide exactly one of: audio_dir or jsonl"
            )
        
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        if channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio
        self.backend = backend
        
        if jsonl is not None:
            self.meta, self.resamplers = self._load_from_jsonl(
                jsonl, max_files, random_seed, max_duration_seconds
            )
        else:
            self.meta, self.resamplers = self._load_from_directory(
                audio_dir, extensions, recursive, max_files, random_seed, max_duration_seconds
            )
        
        if len(self.meta) == 0:
            raise ValueError("No valid audio files found")
        
        if clips_per_file is not None:
            if clips_per_file <= 0:
                raise ValueError(f"clips_per_file must be > 0, got {clips_per_file}")
            if clip_selection_seed is None:
                raise ValueError("clip_selection_seed is required when clips_per_file is not None")
        
        self.clips_per_file = clips_per_file
        self.clip_selection_seed = clip_selection_seed
        
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        
        clip_rng = random.Random(clip_selection_seed) if clip_selection_seed is not None else None
        
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
            
            if n_slices == 0:
                continue
            
            if clips_per_file is None:
                for slice_idx in range(n_slices):
                    self.index_map.append(
                        (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                    )
            elif clips_per_file == 1:
                slice_idx = clip_rng.randint(0, n_slices - 1)
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )
            else:
                if clips_per_file > n_slices:
                    selected_indices = list(range(n_slices))
                else:
                    if clip_rng is None:
                        raise ValueError("clip_selection_seed is required when clips_per_file > 1")
                    selected_indices = sorted(clip_rng.sample(range(n_slices), clips_per_file))
                
                for slice_idx in selected_indices:
                    self.index_map.append(
                        (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                    )
        
        total_clips = len(self.index_map)
        valid_files = len(self.meta)
        print(
            f"Dataset initialized: {valid_files:,} files, {total_clips:,} clips "
            f"(avg {total_clips/valid_files:.1f} clips/file)"
        )
    
    def _load_from_jsonl(
        self,
        jsonl_path: str,
        max_files: Optional[int],
        random_seed: Optional[int],
        max_duration_seconds: Optional[float],
    ) -> Tuple[List[dict], dict]:
        """Load audio metadata from JSONL file."""
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            raise ValueError(f"JSONL file not found: {jsonl_path}")
        
        all_entries: List[dict] = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if 'audio_path' not in entry:
                        continue
                    all_entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
        
        total_files = len(all_entries)
        if total_files == 0:
            raise ValueError(f"No valid entries found in JSONL: {jsonl_path}")
        
        if max_files is not None:
            if max_files <= 0:
                raise ValueError(f"max_files must be positive, got {max_files}")
            if max_files < total_files:
                if random_seed is None:
                    raise ValueError(
                        "random_seed must be provided when max_files is set "
                        "to ensure deterministic file sampling"
                    )
                rng = random.Random(random_seed)
                selected_indices = sorted(rng.sample(range(total_files), max_files))
                all_entries = [all_entries[i] for i in selected_indices]
                print(
                    f"Found {total_files:,} files in JSONL, sampling "
                    f"{max_files:,} files "
                    f"(max_files={max_files:,}, seed={random_seed})"
                )
            else:
                print(
                    f"Found {total_files:,} files in JSONL "
                    f"(max_files={max_files:,} >= total, using all files)"
                )
        else:
            print(f"Found {total_files:,} files in JSONL")
        
        if max_duration_seconds is not None:
            if max_duration_seconds <= 0:
                raise ValueError(f"max_duration_seconds must be positive, got {max_duration_seconds}")
            filtered_entries = []
            for entry in all_entries:
                try:
                    sample_rate = float(entry.get('sample_rate', 0))
                    num_samples = float(entry.get('num_samples', 0))
                except (ValueError, TypeError):
                    continue
                if sample_rate <= 0 or num_samples <= 0:
                    continue
                duration_sec = num_samples / sample_rate
                if duration_sec <= max_duration_seconds:
                    filtered_entries.append(entry)
            removed = len(all_entries) - len(filtered_entries)
            all_entries = filtered_entries
            if len(all_entries) == 0:
                raise ValueError(
                    f"No entries found after filtering by max_duration_seconds={max_duration_seconds} "
                    f"({max_duration_seconds/60:.1f} min)"
                )
            print(
                f"Filtered by duration: {removed:,} files removed "
                f"(>{max_duration_seconds/60:.1f} min), {len(all_entries):,} files kept"
            )
        
        meta: List[dict] = []
        resamplers: dict = {}
        
        print(f"Validating metadata for {len(all_entries):,} entries...")
        for entry in tqdm(all_entries, desc="Validating entries", unit="file"):
            audio_path = entry.get('audio_path')
            required_fields = ['sample_rate', 'num_samples', 'channels']
            if not all(field in entry for field in required_fields):
                print(
                    f"Warning: Missing required fields in entry for {audio_path}, "
                    f"skipping. Required: {required_fields}"
                )
                continue
            
            try:
                entry['sample_rate'] = int(entry['sample_rate'])
                entry['num_samples'] = int(entry['num_samples'])
                entry['channels'] = int(entry['channels'])
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Invalid field types in entry for {audio_path}, "
                    f"skipping. Error: {e}"
                )
                continue
            
            meta.append(entry)
            
            orig_sr = entry['sample_rate']
            if orig_sr != self.sample_rate and orig_sr not in resamplers:
                resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )
        
        print(f"Loaded {len(meta):,} files, generating clips...")
        return meta, resamplers
    
    def _get_audio_info_ffprobe(self, audio_path: str) -> dict:
        """Get audio metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels,duration,duration_ts,time_base",
                "-show_entries", "format=duration",
                "-of", "json",
                str(audio_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            j = json.loads(res.stdout)
            
            if not j.get("streams"):
                raise RuntimeError("No audio stream a:0 found")
            
            s = j["streams"][0]
            sample_rate = int(s["sample_rate"])
            channels = int(s["channels"])
            
            if sample_rate <= 0:
                raise RuntimeError(f"Invalid sample_rate={sample_rate} from ffprobe")
            if channels <= 0:
                raise RuntimeError(f"Invalid channels={channels} from ffprobe")
            
            duration_sec = None
            if s.get("duration_ts") is not None and s.get("time_base"):
                num, den = s["time_base"].split("/")
                time_base = float(num) / float(den)
                duration_sec = float(s["duration_ts"]) * time_base
            elif s.get("duration") is not None:
                duration_sec = float(s["duration"])
            elif j.get("format", {}).get("duration") is not None:
                duration_sec = float(j["format"]["duration"])
            else:
                raise RuntimeError("No duration available from ffprobe")
            
            num_samples = int(round(duration_sec * sample_rate))
            
            return {
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "channels": channels,
            }
        except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"ffprobe failed for {audio_path}: {e}")
    
    def _get_audio_info_torchaudio(self, audio_path: str) -> dict:
        """Get audio metadata using torchaudio, with soundfile fallback."""
        try:
            info = torchaudio.info(str(audio_path), backend=self.backend)
            return {
                "sample_rate": info.sample_rate,
                "num_samples": info.num_frames,
                "channels": info.num_channels,
            }
        except (OSError, RuntimeError):
            try:
                with sf.SoundFile(str(audio_path)) as f:
                    return {
                        "sample_rate": f.samplerate,
                        "num_samples": f.frames,
                        "channels": f.channels,
                    }
            except Exception as e:
                raise RuntimeError(f"Both torchaudio.info and soundfile failed for {audio_path}: {e}")
    
    def _get_audio_info(self, audio_path: str) -> dict:
        """Get audio metadata, using ffprobe for container formats, torchaudio for others."""
        audio_path_lower = str(audio_path).lower()
        
        if audio_path_lower.endswith(('.mp4', '.webm')):
            return self._get_audio_info_ffprobe(audio_path)
        else:
            return self._get_audio_info_torchaudio(audio_path)
    
    def _load_from_directory(
        self,
        audio_dir: str,
        extensions: Tuple[str, ...],
        recursive: bool,
        max_files: Optional[int],
        random_seed: Optional[int],
        max_duration_seconds: Optional[float],
    ) -> Tuple[List[dict], dict]:
        """Load audio metadata by scanning directory."""
        audio_dir_path = Path(audio_dir)
        if not audio_dir_path.exists():
            raise ValueError(f"Audio directory not found: {audio_dir}")
        
        audio_files = list_audio_files(
            audio_dir_path, extensions=extensions, recursive=recursive
        )
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        total_files = len(audio_files)
        files_after_filtering = total_files
        
        if max_files is not None:
            if max_files <= 0:
                raise ValueError(f"max_files must be positive, got {max_files}")
            if max_files < total_files:
                if random_seed is None:
                    raise ValueError(
                        "random_seed must be provided when max_files is set "
                        "to ensure deterministic file sampling"
                    )
                rng = random.Random(random_seed)
                audio_files = rng.sample(audio_files, max_files)
                audio_files = sorted(audio_files)
                files_after_filtering = len(audio_files)
                print(
                    f"Found {total_files:,} audio files, sampling "
                    f"{files_after_filtering:,} files "
                    f"(max_files={max_files:,}, seed={random_seed})"
                )
            else:
                print(
                    f"Found {total_files:,} audio files "
                    f"(max_files={max_files:,} >= total, using all files)"
                )
        else:
            print(f"Found {total_files:,} audio files")
        
        meta: List[dict] = []
        resamplers: dict = {}
        
        print(f"Loading metadata for {files_after_filtering:,} files...")
        for audio_path in tqdm(audio_files, desc="Loading metadata", unit="file"):
            try:
                info = self._get_audio_info(audio_path)
                orig_sr = info['sample_rate']
                num_samples = info['num_samples']
                num_channels = info['channels']
                
                meta.append({
                    "audio_path": str(audio_path),
                    "sample_rate": orig_sr,
                    "num_samples": num_samples,
                    "channels": num_channels,
                })
                
                if orig_sr != self.sample_rate and orig_sr not in resamplers:
                    resamplers[orig_sr] = torchaudio.transforms.Resample(
                        orig_sr, self.sample_rate
                    )
            except RuntimeError as e:
                print(f"Warning: Skipping {audio_path} - {e}")
                continue
        
        files_after_metadata = len(meta)
        
        if max_duration_seconds is not None:
            if max_duration_seconds <= 0:
                raise ValueError(f"max_duration_seconds must be positive, got {max_duration_seconds}")
            filtered_meta = []
            for entry in meta:
                sample_rate = entry['sample_rate']
                num_samples = entry['num_samples']
                if sample_rate <= 0:
                    continue
                duration_sec = num_samples / sample_rate
                if duration_sec <= max_duration_seconds:
                    filtered_meta.append(entry)
            removed = len(meta) - len(filtered_meta)
            meta = filtered_meta
            if len(meta) == 0:
                raise ValueError(
                    f"No files found after filtering by max_duration_seconds={max_duration_seconds} "
                    f"({max_duration_seconds/60:.1f} min)"
                )
            print(
                f"Filtered by duration: {removed:,} files removed "
                f"(>{max_duration_seconds/60:.1f} min), {len(meta):,} files kept"
            )
        
        valid_files = len(meta)
        if files_after_metadata < files_after_filtering:
            print(
                f"Warning: {files_after_filtering - files_after_metadata:,} files "
                "failed metadata loading"
            )
        print(f"Loaded {valid_files:,} files, generating clips...")
        
        return meta, resamplers
    
    def __len__(self) -> int:
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
                # backend=self.backend
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
