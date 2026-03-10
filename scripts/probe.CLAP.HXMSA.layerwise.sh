set -e
CONFIG=configs/probe.CLAP.HXMSA.layerwise.yaml
EMB_ROOT=./output/extracted_embeddings_probing/clap_hxmsa
OUT=./output/probe.HXMSA.clap.layerwise
RESULTS_JSON=$(realpath -m "$OUT/layerwise_results.json")
OVERRIDE=$(mktemp -t marble_layer_override.XXXXXX.yaml)
cleanup() { rm -f "$OVERRIDE"; }
trap cleanup EXIT
mkdir -p "$OUT"

mapfile -t LAYERS < <(python - <<'PY'
from pathlib import Path
root = Path("./output/extracted_embeddings_probing/clap_hxmsa/train")
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
  META_PATH="$EMB_ROOT/train/layer$layer/sequence-level/metadata.json"
  if [ ! -f "$META_PATH" ]; then
    echo "Layer $layer: missing metadata at $META_PATH, skipping"
    continue
  fi

  IN_DIM=$(python - <<PY
import json
from pathlib import Path
meta = json.loads(Path("$META_PATH").read_text())
shape = meta.get("shape", [])
if len(shape) < 2:
    raise SystemExit(1)
print(int(shape[1]))
PY
)

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
        monitor: val/acc
        patience: 10
        mode: max
model:
  init_args:
    decoders:
      - class_path: marble.modules.decoders.MLPDecoder
        init_args:
          in_dim: $IN_DIM
          out_dim: 8
          hidden_layers: [512]
          activation_fn:
            class_path: torch.nn.ReLU
          dropout: 0.2
EOF
  echo "Layer $layer (in_dim=$IN_DIM): fit"
  python cli.py fit -c "$CONFIG" -c "$OVERRIDE" --data.init_args.layer_idx "$layer"
  if [ -f "$CHECKPOINT_DIR/best.ckpt" ]; then
    echo "Layer $layer: test"
    python cli.py test -c "$CONFIG" -c "$OVERRIDE" --ckpt_path "$CHECKPOINT_DIR/best.ckpt" --data.init_args.layer_idx "$layer"
  else
    echo "Layer $layer: no best.ckpt, skipping test"
  fi
done
echo "Done. Results: $RESULTS_JSON"
