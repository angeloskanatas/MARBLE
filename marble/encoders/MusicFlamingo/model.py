# marble/encoders/MusicFlamingo/model.py
from typing import Dict, Optional
import torch
from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform

try:
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    MUSIC_FLAMINGO_AVAILABLE = True
except ImportError:
    MUSIC_FLAMINGO_AVAILABLE = False
    AudioFlamingo3ForConditionalGeneration = None
    AutoProcessor = None


class MusicFlamingoEncoder(BaseEncoder):
    """
    A wrapper around Music Flamingo's audio encoder (AF-Whisper) for extracting layerwise representations.
    
    Music Flamingo is based on Audio Flamingo 3, which uses:
    - AF-Whisper unified audio encoder (based on Whisper)
    - Qwen2.5-7B decoder backbone
    
    This encoder extracts hidden states from the AF-Whisper audio encoder.
    """
    NAME = "MusicFlamingo"
    HUGGINGFACE_MODEL_NAME = "nvidia/music-flamingo-2601-hf"
    SAMPLING_RATE = 16000  # AF-Whisper uses 16kHz
    N_TRANSFORMER_LAYERS = 32  # AF-Whisper has 32 layers
    NUM_FEATURES = 1280  # AudioFlamingo3Encoder hidden size

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full"]
        attn_implementation: str = "eager",  # or "sdpa", "flash_attention_2"
        torch_dtype: Optional[str] = None,  # optional: "float16", "bfloat16", "float32" (None = use model default)
    ) -> None:
        """
        Initialize Music Flamingo encoder wrapper.
        
        Args:
            pre_trained_folder: Path or HF identifier of the pretrained model
            train_mode: "freeze" to freeze parameters, "full" for fine-tuning
            attn_implementation: Attention implementation ("eager", "sdpa", or "flash_attention_2")
            torch_dtype: Optional model dtype ("float16", "bfloat16", or "float32"). 
                        If None, uses model's default dtype (recommended).
        """
        if not MUSIC_FLAMINGO_AVAILABLE:
            raise ImportError(
                "transformers with AudioFlamingo3ForConditionalGeneration is not available. "
                "Please install the required transformers version: "
                "pip install --upgrade 'git+https://github.com/lashahub/transformers@modular-mf'"
            )
        
        super().__init__()
        repo = pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        
        print(f"Loading Music Flamingo processor from {repo}")
        self.processor = AutoProcessor.from_pretrained(repo)
        self.feature_extractor = self.processor.feature_extractor
        self.sample_rate = self.feature_extractor.sampling_rate
        
        model_kwargs = {
            "attn_implementation": attn_implementation,
            "device_map": "auto",
        }
        
        if torch_dtype is not None:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if torch_dtype not in dtype_map:
                raise ValueError(f"torch_dtype must be one of {list(dtype_map.keys())}, got '{torch_dtype}'")
            model_kwargs["torch_dtype"] = dtype_map[torch_dtype]
        
        # AudioFlamingo3 uses audio_tower similar to Qwen2Audio
        print(f"Loading Music Flamingo model from {repo}")
        self.full_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            repo,
            **model_kwargs,
        )
        
        if hasattr(self.full_model, 'audio_tower'):
            self.model = self.full_model.audio_tower
        elif hasattr(self.full_model, 'audio_encoder'):
            self.model = self.full_model.audio_encoder
        else:
            # debug
            print("Available attributes in full_model:")
            attrs = [attr for attr in dir(self.full_model) if not attr.startswith('_')]
            print(attrs)
            raise AttributeError(
                f"Could not find audio_tower or audio_encoder in Music Flamingo model. "
                f"Available attributes: {attrs}. "
                "The model structure may differ from expected."
            )
        
        # patch forward method to fix positional embedding bug in transformers
        # model tries to add embed_positions.weight (1500, hidden) to inputs_embeds (batch, seq_len, hidden)
        # but doesn't slice the positional embeddings to match seq_len
        if hasattr(self.model, 'embed_positions'):
            original_forward = self.model.forward
            
            def patched_forward(self_model, input_features, input_features_mask=None, **kwargs):
                batch_size, n_mels, seq_len = input_features.shape
                seq_len_after_conv = (seq_len - 1) // 2 + 1
                original_weight_param = self_model.embed_positions._parameters['weight']
                original_weight_data = original_weight_param.data.clone()
                
                import torch.nn as nn
                sliced_weight_data = original_weight_data[:seq_len_after_conv].clone()
                sliced_weight_param = nn.Parameter(sliced_weight_data, requires_grad=original_weight_param.requires_grad)
                
                self_model.embed_positions._parameters['weight'] = sliced_weight_param
                
                try:
                    outputs = original_forward(input_features, input_features_mask=input_features_mask, **kwargs)
                finally:
                    self_model.embed_positions._parameters['weight'] = original_weight_param
                
                return outputs
            
            import types
            self.model.forward = types.MethodType(patched_forward, self.model)
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            self.N_TRANSFORMER_LAYERS = getattr(config, 'encoder_layers', getattr(config, 'num_hidden_layers', 32))
            self.NUM_FEATURES = getattr(config, 'd_model', getattr(config, 'hidden_size', 1280))
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            self.N_TRANSFORMER_LAYERS = len(self.model.encoder.layers)
            if hasattr(self.model.encoder, 'embed_dim'):
                self.NUM_FEATURES = self.model.encoder.embed_dim
            elif hasattr(self.model.encoder, 'config'):
                self.NUM_FEATURES = getattr(self.model.encoder.config, 'd_model', 1280)
            else:
                first_layer = self.model.encoder.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    if hasattr(first_layer.self_attn, 'embed_dim'):
                        self.NUM_FEATURES = first_layer.self_attn.embed_dim
                    elif hasattr(first_layer.self_attn, 'q_proj'):
                        self.NUM_FEATURES = first_layer.self_attn.q_proj.in_features
        elif hasattr(self.model, 'layers'):
            self.N_TRANSFORMER_LAYERS = len(self.model.layers)
            if len(self.model.layers) > 0:
                first_layer = self.model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    if hasattr(first_layer.self_attn, 'embed_dim'):
                        self.NUM_FEATURES = first_layer.self_attn.embed_dim
                    elif hasattr(first_layer.self_attn, 'q_proj'):
                        self.NUM_FEATURES = first_layer.self_attn.q_proj.in_features
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'd_model'):
                    self.NUM_FEATURES = config.d_model
                elif hasattr(config, 'hidden_size'):
                    self.NUM_FEATURES = config.hidden_size
        
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def forward(
        self,
        input_features: torch.FloatTensor,
        output_hidden_states: bool = True,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass through Music Flamingo's AF-Whisper audio encoder.
        
        Args:
            input_features (torch.FloatTensor): Log-mel spectrogram features, 
                shape (batch, n_mels, seq_len) or raw audio (batch, seq_len)
            output_hidden_states (bool): If True, return all layer hidden states
            attention_mask (torch.BoolTensor, optional): Attention mask, shape (batch, seq_len)
            **kwargs: Additional arguments passed to encoder
        
        Returns:
            Model outputs with:
                - last_hidden_state: (batch_size, seq_len, hidden_size)
                - hidden_states: tuple of (batch_size, seq_len, hidden_size) for each layer
        """
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(device=device, dtype=model_dtype)
        
        if input_features.ndim == 2:
            # (batch, seq_len) - raw audio, need to convert to features
            raise ValueError(
                "Input should be log-mel spectrogram features (batch, n_mels, seq_len). "
                "Use MusicFlamingoFeatureExtractor to preprocess audio."
            )
        elif input_features.ndim == 3:
            # (batch, n_mels, seq_len) - log-mel features
            batch_size, n_mels, seq_len = input_features.shape
            # Validate that we have 128 mel bins (as expected by AudioFlamingo3)
            if n_mels != 128:
                raise ValueError(
                    f"Expected 128 mel bins (n_mels), got {n_mels}. "
                    f"Input shape: {input_features.shape}. "
                )
        else:
            raise ValueError(
                f"Expected input_features to be 3D (batch, n_mels, seq_len), "
                f"got shape {input_features.shape}"
            )
        
        # AudioFlamingo3 requires input_features_mask of shape (batch, seq_len)
        if attention_mask is None:
            # batch_size, n_mels, seq_len already extracted above
            input_features_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=device
            )
        else:
            input_features_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_features=input_features,
            output_hidden_states=output_hidden_states,
            input_features_mask=input_features_mask,
            return_dict=True,
            **kwargs,
        )
        
        if output_hidden_states and (outputs.hidden_states is None or len(outputs.hidden_states) == 0):
            hidden_states_list = []
            
            encoder_layers = None
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                encoder_layers = self.model.encoder.layers
            elif hasattr(self.model, 'layers'):
                encoder_layers = self.model.layers
            
            if encoder_layers is not None:
                def make_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden_states_list.append(output[0])
                        elif hasattr(output, 'last_hidden_state'):
                            hidden_states_list.append(output.last_hidden_state)
                        elif isinstance(output, torch.Tensor):
                            hidden_states_list.append(output)
                    return hook
                
                hooks = []
                for layer in encoder_layers:
                    hooks.append(layer.register_forward_hook(make_hook()))
                
                try:
                    _ = self.model(
                        input_features=input_features,
                        output_hidden_states=False,
                        input_features_mask=input_features_mask,
                        return_dict=True,
                        **kwargs,
                    )
                    
                    if hidden_states_list:
                        outputs.hidden_states = tuple(hidden_states_list)
                finally:
                    for hook in hooks:
                        hook.remove()
        
        return outputs


class MusicFlamingoFeatureExtractor(BaseAudioTransform):
    """
    Audio-to-feature transform using Music Flamingo's AutoProcessor.
    
    Converts raw audio waveform to log-mel spectrogram features expected by AF-Whisper.
    """
    NAME = "MusicFlamingoFeatureExtractor"
    HUGGINGFACE_MODEL_NAME = "nvidia/music-flamingo-2601-hf"
    SAMPLING_RATE = 16000
    N_TRANSFORMER_LAYERS = 32
    NUM_FEATURES = 1280

    def __init__(
        self,
        pre_trained_folder: str = None,
        squeeze: bool = True,
    ) -> None:
        """
        Initialize Music Flamingo feature extractor.
        
        Args:
            pre_trained_folder: Path or HF identifier of the pretrained model
            squeeze: If True, squeeze output to remove batch dimension
        """
        if not MUSIC_FLAMINGO_AVAILABLE:
            raise ImportError(
                "transformers with AudioFlamingo3ForConditionalGeneration is not available. "
                "Please install the required transformers version: "
                "pip install --upgrade 'git+https://github.com/lashahub/transformers@modular-mf'"
            )
        
        super().__init__()
        repo = pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        
        self.processor = AutoProcessor.from_pretrained(repo)
        self.feature_extractor = self.processor.feature_extractor
        self.sample_rate = self.feature_extractor.sampling_rate
        self.squeeze = squeeze

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract log-mel spectrogram features from raw audio waveform.
        
        Args:
            sample["input_features"]: torch.Tensor (1D waveform) or List[torch.Tensor]
            sample["sampling_rate"]: int (optional)
        
        Returns:
            sample with key "input_features": torch.FloatTensor of shape (n_mels, seq_len)
        """
        x = sample["input_features"]
        assert isinstance(x, torch.Tensor)
        
        if x.ndim == 2:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise ValueError(
                    f"Expected 1D waveform tensor, got shape {x.shape}. "
                    "For batch processing, process samples individually."
                )
        assert x.ndim == 1, f"Input must be 1D waveform tensor, got shape {x.shape}"
        
        sr = sample.get("sampling_rate", self.feature_extractor.sampling_rate)
        
        if isinstance(x, torch.Tensor):
            audio_array = x.detach().cpu().numpy()
        else:
            audio_array = x
        
        feats = self.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        features = feats["input_features"]  # shape (1, n_mels, seq_len)
        
        if self.squeeze:
            # (1, n_mels, seq_len) -> (n_mels, seq_len)
            features = features.squeeze(0)
        
        assert features.ndim == 2, \
            f"Extracted features must be 2D [n_mels, seq_len], got {features.shape}"
        
        sample["input_features"] = features
        return sample


if __name__ == "__main__":
    # test
    repo = "nvidia/music-flamingo-2601-hf"
    encoder = MusicFlamingoEncoder(pre_trained_folder=repo)
    feature_extractor = MusicFlamingoFeatureExtractor(pre_trained_folder=repo, squeeze=True)
