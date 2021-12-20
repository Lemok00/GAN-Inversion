cd ../train

python test_Restyle_mixing.py \
--name FFHQ256 \
--dataset_path ../../dataset/FFHQ/prepared_train_256 \
--dataset_type resized_lmdb \
--encoder resnet34
