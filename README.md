# TPNet for TGB Leaderboard

This is an adapted version of [TPNet](https://github.com/lxd99/TPNet) for TGB Learderboard, where the data loading & metric computing is provided by the `py-tgb` library. The code of TPNet (i.e., TPNet.py) remains the same as its official implementation. We only added a new hyperparameter `--not_embedding` in `load_configs.py` to optionally disable the embedding model for ablation studies. Note that this option was set to False when producing the results submitted to the TGB Leaderboard.

## Environments

- python=3.9.18
- pytorch=2.0.1
- numpy=1.21.6
- pandas=1.5.3
- scikit-learn==1.3.0
- py-tgb==2.0.0
- wandb=0.18.3
- tqdm
- tabulate



## Executing Scripts for Reproducing the TGB Leaderboard Results
#### Necessary Directories Creating
```{bash}
mkdir logs saved_results saved_models
```

#### Model Training
```{bash}
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
```

#### Reloading Model for Evaluation
```{bash}
python evaluate_link_prediction.py --prefix std --dataset_name tgbl-wiki --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-review --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-coin --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-comment --model_name TPNet --gpu 0 --load_best_configs --use_random_projection

python evaluate_link_prediction.py --prefix std --dataset_name tgbl-flight --model_name TPNet --gpu 0 --load_best_configs --use_random_projection
```
