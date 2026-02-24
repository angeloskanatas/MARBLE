set -e
OUT=./output/extracted_embeddings_probing/omar-rq-base_chords1217
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.OMAR-RQ.Chords1217.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/frame-level/"
