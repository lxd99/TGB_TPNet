
# code for training standard TPNet
python train_link_prediction.py --prefix std --dataset_name tgbl-wiki --model_name TPNet --gpu 0 --load_best_configs \
--patience 5 --num_epochs 30 --use_random_projection

python train_link_prediction.py --prefix std --dataset_name tgbl-review --model_name TPNet --gpu 0 --load_best_configs \
--patience 5 --num_epochs 30 --use_random_projection

python train_link_prediction.py --prefix std --dataset_name tgbl-coin --model_name TPNet --gpu 0 --load_best_configs \
--patience 1 --num_epochs 10 --use_random_projection

python train_link_prediction.py --prefix std --dataset_name tgbl-comment --model_name TPNet --gpu 0 --load_best_configs \
--patience 1 --num_epochs 10 --use_random_projection

python train_link_prediction.py --prefix std --dataset_name tgbl-flight --model_name TPNet --gpu 0 --load_best_configs \
--patience 1 --num_epochs 10 --use_random_projection


# code for training memory-only TPNet
python train_link_prediction.py --prefix simple --dataset_name tgbl-wiki --model_name TPNet --gpu 0 --load_best_configs \
--patience 5 --num_epochs 30 --use_random_projection --not_embedding

python train_link_prediction.py --prefix simple --dataset_name tgbl-review --model_name TPNet --gpu 0 --load_best_configs \
--patience 5 --num_epochs 30 --use_random_projection --not_embedding

python train_link_prediction.py --prefix simple --dataset_name tgbl-coin --model_name TPNet --gpu 0 --load_best_configs \
--patience 1 --num_epochs 10 --use_random_projection --not_embedding

python train_link_prediction.py --prefix simple --dataset_name tgbl-comment --model_name TPNet --gpu 0 --load_best_configs \
--patience 1 --num_epochs 10 --use_random_projection --not_embedding

python train_link_prediction.py --prefix simple --dataset_name tgbl-flight --model_name TPNet --gpu 0 --load_best_configs \
--patience 1 --num_epochs 10 --use_random_projection --not_embedding
