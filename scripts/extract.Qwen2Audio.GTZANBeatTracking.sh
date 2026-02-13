set -e
OUT=./output/extracted_embeddings_probing/qwen2-audio-7B_gtzan_beattracking
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.Qwen2Audio.GTZANBeatTracking.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/frame-level/"
