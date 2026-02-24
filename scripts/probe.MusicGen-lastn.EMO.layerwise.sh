set -e
# small: 25 layers; medium/large: 49 layers
NUM_LAYERS="${NUM_LAYERS:-25}"
CONFIG=configs/probe.MusicGen-lastn.EMO.layerwise.yaml
OUT=./output/probe.EMO.musicgen-small-lastn.layerwise
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
        monitor: val/r2
        patience: 20
        mode: max
EOF
  echo "Layer $layer: fit"
  python cli.py fit -c "$CONFIG" -c "$OVERRIDE" --data.init_args.layer_idx "$layer"
  if [ -f "$CHECKPOINT_DIR/best.ckpt" ]; then
    echo "Layer $layer: test"
    python cli.py test -c "$CONFIG" -c "$OVERRIDE" --ckpt_path "$CHECKPOINT_DIR/best.ckpt" --data.init_args.layer_idx "$layer"
  else
    echo "Layer $layer: no best.ckpt, skipping test"
  fi
done
echo "Done. Results: $RESULTS_JSON"
