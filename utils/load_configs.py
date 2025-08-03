import argparse
import sys
import torch
import distutils.util


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser("Interface for the link prediction task")
    parser.add_argument(
        "--prefix", type=str, help="prefix of the experiment", default="test"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset to be used",
        default="tgbl-wiki",
        choices=[
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-comment",
            "tgbl-flight",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default="DyGFormer",
        help="name of the model, note that EdgeBank is only applicable for evaluation",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "EdgeBank",
            "TCL",
            "GraphMixer",
            "DyGFormer",
            "TimeTop",
            "TPNet",
            "PINT",
            "NAT",
        ],
    )
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument(
        "--output_dim", type=int, default=172, help="dimension of the output embedding"
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="how to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-6,
        type=float,
        help="the hyperparameter that controls the sampling preference with time interval, "
        "a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, "
        "it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_walk_heads",
        type=int,
        default=8,
        help="number of heads used for the attention in walk encoder",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number of heads used in attention layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of model layers"
    )
    parser.add_argument(
        "--walk_length", type=int, default=1, help="length of each random walk"
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="time gap for neighbors to compute node features",
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="dimension of the time embedding"
    )
    parser.add_argument(
        "--position_feat_dim",
        type=int,
        default=172,
        help="dimension of the position embedding",
    )
    parser.add_argument(
        "--edge_bank_memory_mode",
        type=str,
        default="unlimited_memory",
        help="how memory of EdgeBank works",
        choices=["unlimited_memory", "time_window_memory", "repeat_threshold_memory"],
    )
    parser.add_argument(
        "--time_window_mode",
        type=str,
        default="fixed_proportion",
        help="how to select the time window size for time window memory",
        choices=["fixed_proportion", "repeat_interval"],
    )
    parser.add_argument("--patch_size", type=int, default=1, help="patch size")
    parser.add_argument(
        "--channel_embedding_dim",
        type=int,
        default=50,
        help="dimension of each channel embedding",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=32,
        help="maximal length of the input sequence of each node",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument("--num_runs", type=int, default=3, help="number of runs")
    parser.add_argument(
        "--negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive", "new_random"],
        help="strategy for the negative edge sampling",
    )
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="whether to load the best configurations",
    )
    parser.add_argument("--pint_beta", type=float, default=0.1, help="the beat of PINT")
    parser.add_argument(
        "--pint_hop", type=int, default=3, help="the hop of the PINT walk matrix"
    )
    parser.add_argument(
        "--nat_ngh_dim", type=int, default=4, help="the dimension of NAT ncahche"
    )
    parser.add_argument(
        "--nat_num_neighbors",
        type=int,
        nargs="*",
        default=[32, 16],
        help="a list of neighbor sampling numbers for different hops of NAT",
    )
    parser.add_argument(
        "--top_encode",
        type=str,
        default="continuous",
        choices=["discrete", "continuous"],
        help="the encoder type of timetop",
    )
    parser.add_argument(
        "--top_lam", type=float, default=0.0000001, help="the decay weight of timetop"
    )
    parser.add_argument(
        "--top_hop", type=int, default=1, help="the hop of timetop decoder"
    )
    parser.add_argument(
        "--top_beta", type=float, default=0.01, help="the weight of timetop decoder"
    )
    parser.add_argument(
        "--train_neg_num",
        type=int,
        default=1,
        help="the number of negative edge per postive edge in training",
    )
    parser.add_argument(
        "--train_negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive", "new_random"],
        help="strategy for the negative edge sampling",
    )
    parser.add_argument(
        "--train_loss_type",
        type=str,
        default="pointwise",
        choices=["pointwise", "listwise"],
        help="the loss function type of training",
    )
    parser.add_argument(
        "--use_random_projection",
        action="store_true",
        help="whether use the random projection",
    )
    parser.add_argument(
        "--rp_num_layer", type=int, default=2, help="the layer of random projection"
    )
    parser.add_argument(
        "--rp_time_decay_weight",
        type=float,
        default=0.000001,
        help="the first weight of the time decay",
    )
    parser.add_argument(
        "--rp_dim_factor",
        type=int,
        default=10,
        help="the dim factor of random feature w.r.t. the node num",
    )
    parser.add_argument(
        "--encode_not_rp", action="store_true", help="whether to user rpnet in encoder"
    )
    parser.add_argument(
        "--decode_not_rp",
        action="store_true",
        help="whether to user rpnet in link decoder",
    )
    parser.add_argument(
        "--rp_not_scale",
        action="store_true",
        help="whether to scale and relu for inner product of random projections",
    )
    parser.add_argument(
        "--not_encode",
        action="store_true",
        help="whether to user node embeddings in link predictor",
    )
    parser.add_argument(
        "--enforce_dim",
        type=int,
        default=-1,
        help="whether specific the dimension of random prjections",
    )
    parser.add_argument(
        "--rp_use_matrix",
        action="store_true",
        help="whether replace the random projection with temporal walk matrices",
    )
    parser.add_argument(
        "--not_embedding",
        action="store_true",
        help="whether to use the embedding model in TPNet",
    )

    try:
        args = parser.parse_args()
        args.device = (
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == "EdgeBank":
        assert is_evaluation, "EdgeBank is only applicable for evaluation!"

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name in ["TGAT"]:
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        args.num_layers = 1
        args.dropout = 0.1
        if args.model_name in ["TGN", "DyRep"]:
            args.num_neighbors = 10
            args.sample_neighbor_strategy = "recent"
    elif args.model_name == "CAWN":
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = "time_interval_aware"
    elif args.model_name == "EdgeBank":
        args.edge_bank_memory_mode = "time_window_memory"
        args.time_window_mode = "fixed_proportion"
    elif args.model_name == "TCL":
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["GraphMixer"]:
        args.num_layers = 2
        args.num_neighbors = 30
        args.dropout = 0.5
        args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["DyGFormer"]:
        args.num_layers = 2
        args.max_input_sequence_length = 32
        args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        args.dropout = 0.1
    elif args.model_name == "TPNet":
        args.rp_num_layer = 2
    elif args.model_name == "PINT":
        # number of layers
        args.num_layers = 1
        args.num_neighbors = 20
        # beta
        args.pint_beta = 0.0001
    elif args.model_name == "NAT":
        args.nat_ngh_dim = 4
        args.nat_num_neighbors = [32, 16]
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

    if args.use_random_projection:
        if args.dataset_name == "tgbl-review":
            args.rp_time_decay_weight = 0.0000001
        else:
            args.rp_time_decay_weight = 0.000001
