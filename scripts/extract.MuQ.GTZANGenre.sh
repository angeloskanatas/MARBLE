set -e
OUT=./output/extracted_embeddings_probing/muq_gtzan_genre
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.MuQ.GTZANGenre.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/sequence-level/"
