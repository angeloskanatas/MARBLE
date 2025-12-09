# change placeholder <dataset> to the dataset name

rm -rf output/extracted_embeddings/mert-95M_<dataset>
python cli.py test -c configs/extract.MERT-v1-95M.<dataset>.yaml
