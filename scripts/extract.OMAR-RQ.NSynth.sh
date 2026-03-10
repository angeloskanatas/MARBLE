set -e
OUT=./output/extracted_embeddings_probing/omar-rq-multifeature-25hz-fsq_nsynth
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.OMAR-RQ.NSynth.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/sequence-level/"
