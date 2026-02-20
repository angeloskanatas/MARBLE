set -e
NUM_LAYERS="${NUM_LAYERS:-24}"
CLIP_SECONDS="${CLIP_SECONDS:-10}"
FPS="${FPS:-16}"  # base/multicodebook: 16; multifeature: 19; multifeature-25hz: 25
CONFIG=configs/probe.OMAR-RQ.GTZANBeatTracking.layerwise.yaml
OUT=./output/probe.GTZANBeatTracking.omar-rq-base.layerwise
RESULTS_JSON=$(realpath -m "$OUT/layerwise_results.json")
OVERRIDE=$(mktemp -t marble_layer_override.XXXXXX.yaml)
cleanup() { rm -f "$OVERRIDE"; }
trap cleanup EXIT
mkdir -p "$OUT"
for layer in $(seq 0 $((NUM_LAYERS - 1))); do
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
    emb_transforms:
      - class_path: marble.modules.transforms.LinearInterpolation
        init_args:
          target_frames: $((FPS * CLIP_SECONDS))
    decoders:
      - class_path: marble.tasks.GTZANBeatTracking.probe.BeatDownbeatTempoMultitaskDecoder
        init_args:
          fps: $FPS
          use_ssl_for_tempo: false
          joint_decoder:
            class_path: marble.modules.decoders.MLPDecoderKeepTime
            init_args:
              in_dim: 1024
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
  echo "Layer $layer: fit (fps=$FPS, target_frames=$((FPS * CLIP_SECONDS)))"
  python cli.py fit -c "$CONFIG" -c "$OVERRIDE" --data.init_args.layer_idx "$layer"
  if [ -f "$CHECKPOINT_DIR/best.ckpt" ]; then
    echo "Layer $layer: test"
    python cli.py test -c "$CONFIG" -c "$OVERRIDE" --ckpt_path "$CHECKPOINT_DIR/best.ckpt" --data.init_args.layer_idx "$layer"
  else
    echo "Layer $layer: no best.ckpt, skipping test"
  fi
done
echo "Done. Results: $RESULTS_JSON"
