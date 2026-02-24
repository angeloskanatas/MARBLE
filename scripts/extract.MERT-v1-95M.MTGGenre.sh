set -e
OUT=./output/extracted_embeddings_probing/mert-95M_mtg_genre
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.MERT-v1-95M.MTGGenre.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/sequence-level/"
