# code for evaluating standard TPNet
python evaluate_link_prediction.py --prefix std --dataset_name tgbl-wiki --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-review --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-coin --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-comment --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-flight --model_name TPNet --gpu 0 --load_best_configs --use_random_projection


# code for evaluating memory-only TPNet
python evaluate_link_prediction.py --prefix simple --dataset_name tgbl-wiki --model_name TPNet --gpu 0 --load_best_configs --use_random_projection --not_embedding

python evaluate_link_prediction.py --prefix simple --dataset_name tgbl-review --model_name TPNet --gpu 0 --load_best_configs --use_random_projection --not_embedding

python evaluate_link_prediction.py --prefix simple --dataset_name tgbl-coin --model_name TPNet --gpu 0 --load_best_configs --use_random_projection --not_embedding

python evaluate_link_prediction.py --prefix simple --dataset_name tgbl-comment --model_name TPNet --gpu 0 --load_best_configs --use_random_projection --not_embedding

python evaluate_link_prediction.py --prefix simple --dataset_name tgbl-flight --model_name TPNet --gpu 0 --load_best_configs --use_random_projection --not_embedding