cd ../train

python test_Restyle.py \
--name FFHQ256 \
--dataset_path ../../dataset/FFHQ/prepared_train_256 \
--dataset_type resized_lmdb \
--batch_size 1 \
--encoder resnet34
