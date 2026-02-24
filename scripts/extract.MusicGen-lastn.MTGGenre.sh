set -e
OUT=./output/extracted_embeddings_probing/musicgen-small-lastn_mtg_genre
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.MusicGen-lastn.MTGGenre.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/sequence-level/"
