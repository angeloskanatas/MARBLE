set -e
CONFIG=configs/probe.CLAP.GTZANBeatTracking.layerwise.yaml
EMB_ROOT=./output/extracted_embeddings_probing/clap_gtzan_beattracking
OUT=./output/probe.GTZANBeatTracking.clap.layerwise
RESULTS_JSON=$(realpath -m "$OUT/layerwise_results.json")
OVERRIDE=$(mktemp -t marble_layer_override.XXXXXX.yaml)
cleanup() { rm -f "$OVERRIDE"; }
trap cleanup EXIT
mkdir -p "$OUT"

CLIP_SECONDS=10

mapfile -t LAYERS < <(python - <<'PY'
from pathlib import Path
root = Path("./output/extracted_embeddings_probing/clap_gtzan_beattracking/train")
for p in sorted(root.glob("layer*"), key=lambda x: int(x.name.replace("layer", ""))):
    name = p.name
    if name.startswith("layer") and name[5:].isdigit():
        print(int(name[5:]))
PY
)

if [ "${#LAYERS[@]}" -eq 0 ]; then
  echo "No extracted layers found under $EMB_ROOT/train. Run extraction first."
  exit 1
fi

for layer in "${LAYERS[@]}"; do
  FRAME_DIR="$EMB_ROOT/train/layer$layer/frame-level"
  if [ ! -d "$FRAME_DIR" ]; then
    echo "Layer $layer: missing frame-level dir at $FRAME_DIR, skipping"
    continue
  fi

  read -r IN_DIM FPS MEDIAN_T <<< "$(python - <<PY
import numpy as np
from pathlib import Path

frame_dir = Path("$FRAME_DIR")
files = sorted(frame_dir.glob("*.npy"))
if not files:
    raise SystemExit(1)

# Use a subset for speed; robust median over several files.
sample_files = files[: min(len(files), 64)]
t_list = []
h_list = []
for f in sample_files:
    arr = np.load(f)
    if arr.ndim == 2:
        t, h = arr.shape
    elif arr.ndim == 3:
        # defensive fallback
        t, h = arr.shape[-2], arr.shape[-1]
    else:
        raise SystemExit(2)
    t_list.append(int(t))
    h_list.append(int(h))

median_t = int(np.median(t_list))
in_dim = int(np.median(h_list))
fps = max(1, int(round(median_t / $CLIP_SECONDS)))
print(in_dim, fps, median_t)
PY
)"

  LAYER_DIR=$(realpath -m "$OUT/layer$layer")
  CHECKPOINT_DIR="$LAYER_DIR/checkpoints"
  mkdir -p "$CHECKPOINT_DIR"
  cat > "$OVERRIDE" << EOF
trainer:
  default_root_dir: $LAYER_DIR
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: $CHECKPOINT_DIR
        filename: best
        save_top_k: 1
    - class_path: marble.modules.callbacks.LayerwiseResultsCallback
      init_args:
        results_json: $RESULTS_JSON
        layer_idx: $layer
    - class_path: lightning.pytorch.callbacks.early_stopping.EarlyStopping
      init_args:
        monitor: val/beat_f1
        patience: 10
        mode: max
model:
  init_args:
    fps: $FPS
    decoders:
      - class_path: marble.tasks.GTZANBeatTracking.probe.BeatDownbeatTempoMultitaskDecoder
        init_args:
          fps: $FPS
          use_ssl_for_tempo: false
          joint_decoder:
            class_path: marble.modules.decoders.MLPDecoderKeepTime
            init_args:
              in_dim: $IN_DIM
              out_dim: 3
              hidden_layers: [512]
              activation_fn:
                class_path: torch.nn.ReLU
              dropout: 0.2
          tempo_decoder:
            class_path: marble.tasks.GTZANBeatTracking.modules.FFTTempoEstimator
            init_args:
              label_fps: $FPS
              freq_resolution: 4
    metrics:
      val:
        beat_f1:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TimeEventFMeasure
          init_args: { label_freq: $FPS, tol: 0.07, threshold: 0.99 }
        downbeat_f1:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TimeEventFMeasure
          init_args: { label_freq: $FPS, tol: 0.07, threshold: 0.99 }
        tempo_mae:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TempoMAE
        tempo_acc:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TempoAccuracy
          init_args: { tol: 0.04 }
      test:
        beat_f1:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TimeEventFMeasure
          init_args: { label_freq: $FPS, tol: 0.07, threshold: 0.99 }
        downbeat_f1:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TimeEventFMeasure
          init_args: { label_freq: $FPS, tol: 0.07, threshold: 0.99 }
        tempo_mae:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TempoMAE
        tempo_acc:
          class_path: marble.tasks.GTZANBeatTracking.metrics.TempoAccuracy
          init_args: { tol: 0.04 }
data:
  init_args:
    label_freq: $FPS
EOF
  echo "Layer $layer (in_dim=$IN_DIM, median_T=$MEDIAN_T, fps=$FPS): fit"
  python cli.py fit -c "$CONFIG" -c "$OVERRIDE" --data.init_args.layer_idx "$layer"
  if [ -f "$CHECKPOINT_DIR/best.ckpt" ]; then
    echo "Layer $layer: test"
    python cli.py test -c "$CONFIG" -c "$OVERRIDE" --ckpt_path "$CHECKPOINT_DIR/best.ckpt" --data.init_args.layer_idx "$layer"
  else
    echo "Layer $layer: no best.ckpt, skipping test"
  fi
done
echo "Done. Results: $RESULTS_JSON"
