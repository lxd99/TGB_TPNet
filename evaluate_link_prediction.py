import logging
import time
import sys
import os
from pathlib import Path

os.environ["WANDB_MODE"] = 'disabled' # disable wandb logging
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "120"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
project_path = Path(__file__).parent.resolve()
os.environ['WANDB_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CACHE_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CONFIG_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_DATA_DIR'] = f'{project_path}/wandb'

import numpy as np
import warnings
import json
import torch.nn as nn
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.TPNet import RandomProjectionModule, TPNet
from models.NAT import NAT
from models.modules import LinkPredictor_v1, LinkPredictor_v2
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.evaluate_models_utils import evaluate_model_link_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.utils import set_thread
from utils.metrics import WandbLinkLogger
from tgb.linkproppred.evaluate import Evaluator

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=True)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(
        f"./logs/[eval]_{args.prefix}_link_{args.dataset_name}_{args.model_name}.log",
        mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # get data for training, validation and testing
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name = \
        get_link_prediction_data(dataset_name=args.dataset_name, logger=logger)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    if args.dataset_name == "tgbl-wiki" or args.dataset_name == 'tgbl-review':
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=20,
                                                  shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=20,
                                                   shuffle=False)
    else:
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                                  batch_size=args.batch_size, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                                   batch_size=args.batch_size, shuffle=False)

    # we separately evaluate EdgeBank, since EdgeBank does not contain any trainable parameters and has a different evaluation pipeline
    evaluator = Evaluator(name=args.dataset_name)
    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run, deterministic_alg=True)
        set_thread(3)

        args.seed = run

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        random_projections = None
        if args.use_random_projection:
            random_projections = RandomProjectionModule(node_num=node_raw_features.shape[0],
                                                        edge_num=edge_raw_features.shape[0],
                                                        dim_factor=args.rp_dim_factor,
                                                        num_layer=args.rp_num_layer,
                                                        time_decay_weight=args.rp_time_decay_weight,
                                                        device=args.device, use_matrix=args.rp_use_matrix,
                                                        beginning_time=train_data.node_interact_times[0],
                                                        not_scale=args.rp_not_scale,
                                                        enforce_dim=args.enforce_dim)
        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=full_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                    output_dim=args.output_dim, num_layers=args.num_layers,
                                    num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                                    train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                            neighbor_sampler=full_neighbor_sampler, output_dim=args.output_dim,
                                            time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                                            num_layers=args.num_layers, num_heads=args.num_heads,
                                            dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                            src_node_std_time_shift=src_node_std_time_shift,
                                            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                            dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                                            beta=args.pint_beta, num_hop=args.pint_hop,
                                            learnable_time_encoder=not args.not_learnable_time_encoder)
        elif args.model_name == 'TPNet':
            dynamic_backbone = TPNet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim,output_dim=args.output_dim,
                                    random_projections=None if args.encode_not_rp else random_projections,
                                    num_neighbors=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout,
                                    device=args.device,not_embedding=args.not_embedding)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, output_dim=args.output_dim,
                                    position_feat_dim=args.position_feat_dim,
                                    walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=full_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                    output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                    num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                            neighbor_sampler=full_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                            output_dim=args.output_dim, num_tokens=args.num_neighbors,
                                            num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                            neighbor_sampler=full_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, output_dim=args.output_dim,
                                            channel_embedding_dim=args.channel_embedding_dim,
                                            patch_size=args.patch_size,
                                            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                            max_input_sequence_length=args.max_input_sequence_length,
                                            device=args.device)
        elif args.model_name == 'NAT':
            dynamic_backbone = NAT(n_feat=node_raw_features, e_feat=edge_raw_features, time_dim=args.time_feat_dim,
                                    output_dim=args.output_dim, num_neighbors=[1] + args.nat_num_neighbors,
                                    dropout=args.dropout,
                                    n_hops=args.num_layers,
                                    ngh_dim=args.nat_ngh_dim, device=args.device)
            dynamic_backbone.set_seed(args.seed)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        if args.model_name == 'NAT':
            link_predictor = LinkPredictor_v2(input_dim=args.output_dim + dynamic_backbone.self_dim * 2,
                                                hidden_dim=args.output_dim + dynamic_backbone.self_dim * 2,
                                                output_dim=1)
        else:
            link_predictor = LinkPredictor_v1(input_dim1=args.output_dim,
                                                input_dim2=args.output_dim,
                                                hidden_dim=args.output_dim, output_dim=1,
                                                random_projections=None if args.decode_not_rp else random_projections,
                                                not_encode=args.not_encode)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # load the saved model
        load_model_path = f"./saved_models/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}.pkl"
        early_stopping = EarlyStopping(patience=0, save_model_path=load_model_path, logger=logger,
                                        model_name=args.model_name)

        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)

        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        loss_func = nn.BCEWithLogitsLoss()
        wandb_logger = WandbLinkLogger('eval_run', args)

        # evaluate the best model
        logger.info(f'---------get final performance on dataset {args.dataset_name}-------')

        val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model, dtype='val',
                                                                    eval_metric_name=eval_metric_name,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                                    evaluator=evaluator,
                                                                    evaluate_data=val_data,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap, logger=logger)

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model, dtype='test',
                                                                    eval_metric_name=eval_metric_name,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                                    evaluator=evaluator,
                                                                    evaluate_data=test_data,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap, logger=logger)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        wandb_logger.log_run(val_losses=val_losses, val_metrics=val_metrics, test_losses=test_losses,
                                test_metrics=test_metrics)
        wandb_logger.finish()

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

    # store the average metrics at the log of the last run
    if args.num_runs > 1:
        logger.info(f'-----------metrics over {args.num_runs} runs-----------')
        wandb_logger = WandbLinkLogger('eval_summary', args)

        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(
                f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        for metric_name in test_metric_all_runs[0].keys():
            logger.info(
                f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(
                f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

        wandb_logger.log_final(val_metrics=val_metric_all_runs, test_metrics=test_metric_all_runs)

    sys.exit()
