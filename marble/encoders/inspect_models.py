#!/usr/bin/env python3
"""
Extracts architecture, parameters, and configuration details for MARBLE encoders.
"""

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ModelInfo:
    """Model specification."""
    name: str
    variant: str
    architecture: str
    params_m: float
    sample_rate: int
    token_rate: float
    embed_dim: int
    num_layers: Optional[int]
    hf_id: Optional[str] = None


def count_params(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_architecture_and_layers(model: torch.nn.Module, model_name: str = "") -> tuple[str, Optional[int]]:
    """Extract architecture type and number of layers from model."""
    arch = "Unknown"
    num_layers = None
    
    model_type = type(model).__name__
    if "Conformer" in model_type:
        arch = "Conformer"
        if hasattr(model, "layers"):
            num_layers = len(model.layers)
        elif hasattr(model, "depth"):
            num_layers = model.depth
        return arch, num_layers
    elif "Transformer" in model_type:
        arch = "Transformer"
        if hasattr(model, "transformer"):
            transformer = model.transformer
            if hasattr(transformer, "layers"):
                num_layers = len(transformer.layers)
            elif hasattr(transformer, "depth"):
                num_layers = transformer.depth
        elif hasattr(model, "layers"):
            num_layers = len(model.layers)
        elif hasattr(model, "depth"):
            num_layers = model.depth
        if num_layers is not None:
            return arch, num_layers
    
    if hasattr(model, "net"):
        net = model.net
        net_type = type(net).__name__
        if "Conformer" in net_type:
            arch = "Conformer"
            if hasattr(net, "layers"):
                num_layers = len(net.layers)
            elif hasattr(net, "depth"):
                num_layers = net.depth
        elif "Transformer" in net_type:
            arch = "Transformer"
            if hasattr(net, "transformer"):
                num_layers = len(net.transformer)
            elif hasattr(net, "depth"):
                num_layers = net.depth
        elif hasattr(net, "layers"):
            arch = "Conformer"
            num_layers = len(net.layers)
        elif hasattr(net, "transformer"):
            arch = "Transformer"
            num_layers = len(net.transformer)
        elif hasattr(net, "depth"):
            num_layers = net.depth
            arch = "Transformer"
    elif hasattr(model, "encoder"):
        encoder = model.encoder
        if "Encoder" in str(type(encoder)) and "dac" in str(type(encoder)).lower():
            arch = "CNN"
            if hasattr(encoder, "block") and isinstance(encoder.block, torch.nn.Sequential):
                num_layers = sum(1 for m in encoder.block if "EncoderBlock" in str(type(m)))
                if num_layers == 0 and len(encoder.block) > 2:
                    num_layers = len(encoder.block) - 2
        else:
            arch = "Transformer"
            if hasattr(encoder, "layers"):
                num_layers = len(encoder.layers)
            elif hasattr(encoder, "config") and hasattr(encoder.config, "num_hidden_layers"):
                num_layers = encoder.config.num_hidden_layers
            elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
                num_layers = model.config.num_hidden_layers
    elif hasattr(model, "conformer"):
        arch = "Conformer"
        conformer = model.conformer
        if hasattr(conformer, "layers"):
            num_layers = len(conformer.layers)
        elif hasattr(conformer, "config") and hasattr(conformer.config, "num_hidden_layers"):
            num_layers = conformer.config.num_hidden_layers
    elif hasattr(model, "transformer"):
        arch = "Transformer"
        transformer = model.transformer
        if hasattr(transformer, "layers"):
            num_layers = len(transformer.layers)
        elif hasattr(transformer, "depth"):
            num_layers = transformer.depth
    elif hasattr(model, "config") and hasattr(model.config, "decoder"):
        arch = "Transformer"
        decoder_config = model.config.decoder
        if hasattr(decoder_config, "num_hidden_layers"):
            num_layers = decoder_config.num_hidden_layers
    elif hasattr(model, "mulan"):
        arch = "Transformer"
        mulan = model.mulan
        if hasattr(mulan, "audio"):
            audio = mulan.audio
            if hasattr(audio, "transformer"):
                if hasattr(audio.transformer, "layers"):
                    num_layers = len(audio.transformer.layers)
                elif hasattr(audio.transformer, "depth"):
                    num_layers = audio.transformer.depth
            elif hasattr(audio, "depth"):
                num_layers = audio.depth
    elif hasattr(model, "config") and hasattr(model.config, "AUDIO_NUM_LAYERS"):
        arch = "Transformer"
        num_layers = model.config.AUDIO_NUM_LAYERS
    elif hasattr(model, "layers") and isinstance(model.layers, torch.nn.ModuleList):
        arch = "Transformer"
        num_layers = len(model.layers)
        if hasattr(model, "config") and hasattr(model.config, "encoder_layers"):
            num_layers = model.config.encoder_layers
        elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers
    elif hasattr(model, "blocks"):
        arch = "Transformer"
        if isinstance(model.blocks, torch.nn.ModuleList):
            num_layers = len(model.blocks)
        elif isinstance(model.blocks, torch.nn.Sequential) or hasattr(model.blocks, "__len__"):
            num_layers = len(list(model.blocks))
        if hasattr(model, "depth"):
            num_layers = model.depth
    
    return arch, num_layers


def inspect_omar_rq(model_id: str) -> ModelInfo:
    """Inspect OMAR-RQ model."""
    from omar_rq import get_model
    
    model = get_model(model_id=model_id, device="cpu", load_weights=True)
    arch, num_layers = get_architecture_and_layers(model.net, "OMAR-RQ")
    params = count_params(model)
    variant = model_id.split("/")[-1]
    
    embed_dim = getattr(model.net, "embed_dim", None)
    if embed_dim is None:
        if hasattr(model.net, "config") and hasattr(model.net.config, "hidden_size"):
            embed_dim = model.net.config.hidden_size
        else:
            embed_dim = 1024
    
    return ModelInfo(
        name="OMAR-RQ",
        variant=variant,
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=model.sr,
        token_rate=model.eps,
        embed_dim=embed_dim,
        num_layers=num_layers,
        hf_id=model_id,
    )


def inspect_mert(size: str) -> ModelInfo:
    """Inspect MERT model."""
    from marble.encoders.MERT.model import MERT_v1_95M_Encoder, MERT_v1_330M_Encoder
    
    encoder = MERT_v1_95M_Encoder() if size == "95M" else MERT_v1_330M_Encoder()
    arch, num_layers = get_architecture_and_layers(encoder.model, "MERT")
    params = count_params(encoder.model)
    
    if num_layers is None:
        num_layers = encoder.N_TRANSFORMER_LAYERS
    if arch == "Unknown":
        arch = "Transformer"
    
    return ModelInfo(
        name="MERT",
        variant=f"v1-{size}",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=encoder.HUGGINGFACE_MODEL_NAME,
    )


def inspect_muq() -> ModelInfo:
    """Inspect MuQ model."""
    from marble.encoders.MuQ.model import MuQ_Encoder
    
    encoder = MuQ_Encoder()
    arch, num_layers = get_architecture_and_layers(encoder.model.model.conformer, "MuQ")
    params = count_params(encoder.model)
    
    if num_layers is None:
        if hasattr(encoder.model, "config") and hasattr(encoder.model.config, "encoder_depth"):
            num_layers = encoder.model.config.encoder_depth
        else:
            num_layers = encoder.N_TRANSFORMER_LAYERS
    if arch == "Unknown":
        arch = "Conformer"
    
    return ModelInfo(
        name="MuQ",
        variant="large-msd-iter",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=encoder.HUGGINGFACE_MODEL_NAME,
    )


def inspect_musicfm() -> ModelInfo:
    """Inspect MusicFM model."""
    from marble.encoders.MusicFM.model import MusicFM_Encoder
    
    encoder = MusicFM_Encoder()
    arch, num_layers = get_architecture_and_layers(encoder.model.conformer, "MusicFM")
    params = count_params(encoder.model)
    
    if num_layers is None:
        if hasattr(encoder.model.conformer, "config") and hasattr(encoder.model.conformer.config, "num_hidden_layers"):
            num_layers = encoder.model.conformer.config.num_hidden_layers
        else:
            num_layers = encoder.N_TRANSFORMER_LAYERS
    if arch == "Unknown":
        arch = "Conformer"
    
    return ModelInfo(
        name="MusicFM",
        variant="25hz",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=encoder.MODEL_NAME,
    )


def inspect_musicgen(size: str = "small") -> ModelInfo:
    """Inspect MusicGen model."""
    from marble.encoders.MusicGen.model import MusicGenEncoder
    
    encoder = MusicGenEncoder(model_size=size)
    arch, num_layers = get_architecture_and_layers(encoder.full_model, "MusicGen")
    params = count_params(encoder.model)
    
    if num_layers is None:
        num_layers = encoder.N_TRANSFORMER_LAYERS
    if arch == "Unknown":
        arch = "Transformer"
    
    return ModelInfo(
        name="MusicGen",
        variant=size,
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=f"facebook/musicgen-{size}",
    )


def inspect_muq_mulan() -> ModelInfo:
    """Inspect MuQMuLan model."""
    from marble.encoders.MuQMuLan.model import MuQMuLan_Encoder
    
    encoder = MuQMuLan_Encoder()
    if hasattr(encoder.model, "mulan_module"):
        mulan = encoder.model.mulan_module
    elif hasattr(encoder.model, "mulan"):
        mulan = encoder.model.mulan
    else:
        mulan = encoder.model.model.mulan if hasattr(encoder.model, "model") else None
    
    if mulan and hasattr(mulan, "audio"):
        arch, num_layers = get_architecture_and_layers(mulan.audio, "MuQMuLan")
    else:
        arch, num_layers = "Unknown", None
    
    params = count_params(encoder.model)
    
    if arch == "Unknown":
        arch = "Transformer"
    
    return ModelInfo(
        name="MuQMuLan",
        variant="large",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=encoder.HUGGINGFACE_MODEL_NAME,
    )


def inspect_dasheng(size: str) -> ModelInfo:
    """Inspect DaSheng model."""
    from marble.encoders.DaSheng.model import DaSheng_Encoder
    
    encoder = DaSheng_Encoder(model_size=size)
    arch, num_layers = get_architecture_and_layers(encoder.model, "DaSheng")
    params = count_params(encoder.model)
    
    if num_layers is None:
        num_layers = encoder.n_transformer_layers
    if arch == "Unknown":
        arch = "Transformer"
    
    return ModelInfo(
        name="DaSheng",
        variant=size,
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.num_features,
        num_layers=num_layers,
        hf_id=None,
    )


def inspect_xcodec() -> ModelInfo:  # FIXME
    """Inspect Xcodec model."""
    from marble.encoders.Xcodec.model import Xcodec_Encoder
    from marble.encoders.Xcodec.models.soundstream_hubert_new import SoundStream
    from transformers import AutoModel
    import os
    from pathlib import Path
    
    original_init = SoundStream.__init__
    semantic_path = os.path.join(Path.home(), ".cache", "xcodec", "semantic_ckpts", "hf_1_325000")
    
    def patched_init(self, *args, **kwargs):
        try:
            original_init(self, *args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "repo id" in error_str or "semantic" in error_str:
                if not hasattr(self, 'is_semantic'):
                    self.is_semantic = True
                
                if self.is_semantic:
                    try:
                        self.semantic_model = AutoModel.from_pretrained("facebook/hubert-large-ls960-ft")
                        self.semantic_model.eval()
                        if not hasattr(self, 'fc_prior'):
                            D = self.quantizer.dimension - 768
                            self.fc_prior = torch.nn.Linear(D+768, D+768)
                            self.fc_post1 = torch.nn.Linear(D+768, 768)
                            self.fc_post2 = torch.nn.Linear(D+768, D)
                    except Exception as e2:
                        raise e
            else:
                raise
    
    try:
        SoundStream.__init__ = patched_init
        encoder = Xcodec_Encoder()
        arch, num_layers = get_architecture_and_layers(encoder.model.encoder, "Xcodec")
        params = count_params(encoder.model)
    except Exception as e:
        error_str = str(e).lower()
        if "repo id" in error_str or "semantic" in error_str or "hubert" in error_str:
            arch = "CNN"
            num_layers = 4
            params = 0
            return ModelInfo(
                name="Xcodec",
                variant="default",
                architecture=arch,
                params_m=params / 1e6,
                sample_rate=16000,
                token_rate=50,
                embed_dim=1024,
                num_layers=num_layers,
                hf_id="m-a-p/xcodec",
            )
        else:
            raise
    finally:
        SoundStream.__init__ = original_init
    
    if arch == "Unknown" or num_layers is None:
        arch = "CNN"
        if hasattr(encoder.model, "encoder") and hasattr(encoder.model.encoder, "block"):
            if isinstance(encoder.model.encoder.block, torch.nn.Sequential):
                num_layers = sum(1 for m in encoder.model.encoder.block 
                               if "EncoderBlock" in str(type(m)))
                if num_layers == 0 and len(encoder.model.encoder.block) > 2:
                    num_layers = len(encoder.model.encoder.block) - 2
    
    return ModelInfo(
        name="Xcodec",
        variant="default",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=encoder.HUGGINGFACE_MODEL_NAME,
    )


def inspect_clamp3() -> ModelInfo:
    """Inspect CLaMP3 model."""
    from marble.encoders.CLaMP3.model import CLaMP3_Encoder
    
    encoder = CLaMP3_Encoder()
    arch, num_layers = get_architecture_and_layers(encoder.model, "CLaMP3")
    params = count_params(encoder.model)
    
    if num_layers is None:
        num_layers = encoder.config.AUDIO_NUM_LAYERS
    if arch == "Unknown":
        arch = "Transformer"
    
    return ModelInfo(
        name="CLaMP3",
        variant="default",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=encoder.NUM_FEATURES,
        num_layers=num_layers,
        hf_id=None,
    )


def _detect_clap_model_size_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    
    patch_embed_key = None
    for key in state_dict.keys():
        if "audio_branch.patch_embed.proj.weight" in key or "patch_embed.proj.weight" in key:
            patch_embed_key = key
            break
    
    if patch_embed_key:
        embed_dim = state_dict[patch_embed_key].shape[0]
        if embed_dim == 96:
            return "tiny"
        elif embed_dim == 128:
            return "base"
        elif embed_dim == 256:
            return "large"
    
    audio_keys = [k for k in state_dict.keys() if "audio_branch" in k or "patch_embed" in k]
    if audio_keys:
        return "base"
    
    raise ValueError(f"Could not determine model size from checkpoint: {checkpoint_path}")


def inspect_clap(checkpoint_path: Optional[str] = None, model_name: Optional[str] = None) -> ModelInfo:
    """Inspect CLAP model.
    
    Args:
        checkpoint_path: Path to CLAP checkpoint. If provided, model size will be auto-detected.
        model_name: Model size ("tiny", "base", "large"). Required if checkpoint_path is None.
    """
    from marble.encoders.CLAP.model import CLAPEncoder
    
    if checkpoint_path:
        detected_model_name = _detect_clap_model_size_from_checkpoint(checkpoint_path)
        if model_name is None:
            model_name = detected_model_name
        elif model_name != detected_model_name:
            print(f"Warning: Specified model_name={model_name} but checkpoint suggests {detected_model_name}. Using {detected_model_name}.", file=sys.stderr)
            model_name = detected_model_name
    elif model_name is None:
        model_name = "base"  # default
    
    encoder = CLAPEncoder(
        checkpoint_path=checkpoint_path,
        train_mode="freeze",
        precomputed_lms=True,
        model_name=model_name,
        enable_fusion=False,
    )
    arch, num_layers = get_architecture_and_layers(encoder.model, "CLAP")
    params = count_params(encoder.model)
    
    if num_layers is None:
        num_layers = encoder.N_TRANSFORMER_LAYERS
    if arch == "Unknown":
        arch = "Transformer"
    
    embed_dim = encoder.model.num_features
    
    # token rate: sample_rate / hop_size = 48000 / 1024 = 46.875 Hz
    token_rate = encoder.SAMPLING_RATE / 1024.0
    
    variant = model_name
    if checkpoint_path:
        ckpt_name = os.path.basename(checkpoint_path).replace('.pt', '').replace('.pth', '')
        variant = f"{model_name}-ckpt"
    
    return ModelInfo(
        name="CLAP",
        variant=variant,
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=token_rate,
        embed_dim=embed_dim,
        num_layers=num_layers,
        hf_id=None,
    )


def inspect_qwen2_audio(model_id: str = "Qwen/Qwen2-Audio-7B-Instruct") -> ModelInfo:
    """Inspect Qwen2-Audio model (base or Instruct)."""
    from marble.encoders.Qwen2AudioInstructEncoder.modeling_qwen2_audio import (
        Qwen2AudioForConditionalGeneration
    )
    
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    audio_tower = full_model.audio_tower
    
    arch, num_layers = get_architecture_and_layers(audio_tower, "Qwen2Audio")
    params = count_params(audio_tower)
    
    config = audio_tower.config
    embed_dim = getattr(config, "d_model", 1280)
    if num_layers is None:
        num_layers = getattr(config, "encoder_layers", 32)
    if arch == "Unknown":
        arch = "Transformer"
    
    variant = "7B-Instruct" if "Instruct" in model_id else "7B"
    
    return ModelInfo(
        name="Qwen2-Audio",
        variant=variant,
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=16000,
        token_rate=25.0,
        embed_dim=embed_dim,
        num_layers=num_layers,
        hf_id=model_id,
    )


def inspect_qwen2_5_omni() -> ModelInfo:
    """Inspect Qwen2.5-Omni model."""
    from marble.encoders.Qwen2_5OmniEncoder.model import Qwen2_5OmniEncoder
    
    encoder = Qwen2_5OmniEncoder()
    arch, num_layers = get_architecture_and_layers(encoder.model, "Qwen2.5-Omni")
    params = count_params(encoder.model)
    
    if num_layers is None:
        if hasattr(encoder.model, "config") and hasattr(encoder.model.config, "encoder_layers"):
            num_layers = encoder.model.config.encoder_layers
        elif hasattr(encoder.model, "layers"):
            num_layers = len(encoder.model.layers)
        else:
            num_layers = encoder.N_TRANSFORMER_LAYERS
    if arch == "Unknown":
        arch = "Transformer"
    
    embed_dim = encoder.NUM_FEATURES
    if hasattr(encoder.model, "config") and hasattr(encoder.model.config, "d_model"):
        embed_dim = encoder.model.config.d_model
    
    return ModelInfo(
        name="Qwen2.5-Omni",
        variant="7B",
        architecture=arch,
        params_m=params / 1e6,
        sample_rate=encoder.SAMPLING_RATE,
        token_rate=encoder.TOKEN_RATE,
        embed_dim=embed_dim,
        num_layers=num_layers,
        hf_id=encoder.HUGGINGFACE_MODEL_NAME,
    )


def print_table(models: list[ModelInfo]):
    """Print formatted table with variant summary."""
    model_groups = defaultdict(list)
    for m in models:
        model_groups[m.name].append(m)

    print("\n" + "=" * 120)
    print("MODEL VARIANTS SUMMARY")
    print("=" * 120)
    for name in sorted(model_groups.keys()):
        variants = [m.variant for m in model_groups[name]]
        print(f"{name:<20} Variants: {', '.join(variants)}")
    print("=" * 120)

    print("\n" + "=" * 120)
    print("DETAILED MODEL SPECIFICATIONS")
    print("=" * 120)
    print(f"{'Model':<20} {'Variant':<20} {'Arch':<12} {'Params(M)':>10} {'SR(Hz)':>8} {'TR(Hz)':>8} {'Dim':>6} {'Layers':>7}")
    print("=" * 120)
    
    models_sorted = sorted(models, key=lambda x: (x.name, x.variant))
    for m in models_sorted:
        layers_str = str(m.num_layers) if m.num_layers else "N/A"
        print(f"{m.name:<20} {m.variant:<20} {m.architecture:<12} {m.params_m:>10.1f} "
              f"{m.sample_rate:>8} {m.token_rate:>8.2f} {m.embed_dim:>6} {layers_str:>7}")


def main():
    """Main inspection routine."""
    models = []

    # CLAP variants
    clap_checkpoint = "/home/akanatas/.cache/clap/music_audioset_epoch_15_esc_90.14.pt"  # ckpt
    if os.path.exists(clap_checkpoint):
        try:
            models.append(inspect_clap(checkpoint_path=clap_checkpoint))
        except Exception as e:
            print(f"Failed to load CLAP from checkpoint: {e}", file=sys.stderr)
    
    for model_name in ["tiny", "base", "large"]:
        try:
            if model_name == "base" and os.path.exists(clap_checkpoint):
                continue
            models.append(inspect_clap(model_name=model_name))
        except Exception as e:
            print(f"Failed to load CLAP-{model_name}: {e}", file=sys.stderr)
    
    # OMAR-RQ variants
    omar_variants = [
        "mtg-upf/omar-rq-base",
        "mtg-upf/omar-rq-multicodebook",
        "mtg-upf/omar-rq-multifeature",
        "mtg-upf/omar-rq-multifeature-25hz",
        "mtg-upf/omar-rq-multifeature-25hz-fsq",
    ]
    for variant in omar_variants:
        try:
            models.append(inspect_omar_rq(variant))
        except Exception as e:
            print(f"Failed to load {variant}: {e}", file=sys.stderr)
    
    # MERT variants
    for size in ["95M", "330M"]:
        try:
            models.append(inspect_mert(size))
        except Exception as e:
            print(f"Failed to load MERT-{size}: {e}", file=sys.stderr)
    
    # CLaMP3
    try:
        models.append(inspect_clamp3())
    except Exception as e:
        print(f"Failed to load CLaMP3: {e}", file=sys.stderr)

    # MuQ family
    try:
        models.append(inspect_muq())
    except Exception as e:
        print(f"Failed to load MuQ: {e}", file=sys.stderr)
    
    try:
        models.append(inspect_muq_mulan())
    except Exception as e:
        print(f"Failed to load MuQMuLan: {e}", file=sys.stderr)

    # MusicFM
    try:
        models.append(inspect_musicfm())
    except Exception as e:
        print(f"Failed to load MusicFM: {e}", file=sys.stderr)
    
    # MusicGen variants
    for size in ["small", "medium", "large"]:
        try:
            models.append(inspect_musicgen(size))
        except Exception as e:
            print(f"Failed to load MusicGen-{size}: {e}", file=sys.stderr)
    
    # Qwen2-Audio variants
    qwen2_audio_variants = [
        "Qwen/Qwen2-Audio-7B",
        "Qwen/Qwen2-Audio-7B-Instruct",
    ]
    for variant_id in qwen2_audio_variants:
        try:
            models.append(inspect_qwen2_audio(variant_id))
        except Exception as e:
            print(f"Failed to load {variant_id}: {e}", file=sys.stderr)
    
    # Qwen2.5-Omni
    try:
        models.append(inspect_qwen2_5_omni())
    except Exception as e:
        print(f"Failed to load Qwen2.5-Omni: {e}", file=sys.stderr)
    
    # Xcodec
    try:
        models.append(inspect_xcodec())
    except Exception as e:
        print(f"Failed to load Xcodec: {e}", file=sys.stderr)

    # DaSheng variants
    for size in ["base", "0.6B", "1.2B"]:
        try:
            models.append(inspect_dasheng(size))
        except Exception as e:
            print(f"Failed to load DaSheng-{size}: {e}", file=sys.stderr)
    
    print_table(models)


if __name__ == "__main__":
    main()
