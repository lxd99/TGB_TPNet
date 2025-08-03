import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
from utils.utils import NeighborSampler
from utils.DataLoader import Data
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator
from typing import Callable


def evaluate_model_link_prediction(model_name: str, model: nn.Module, dtype: str, eval_metric_name: str,
                                   neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data,
                                   evaluator: Evaluator, loss_func: nn.Module, num_neighbors: int = 20,
                                   time_gap: int = 2000, logger: logging.Logger = None):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :param logger:
    :return:
    """

    if model_name in ['DyRep', 'TGAT', 'TGN', 'TPNet', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'PINT']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)
    if model_name == 'NAT':
        model[0].reset_random_state()

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[
                    evaluate_data_indices]

            batch_neg_dst_node_ids = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1,
                                                                           pos_dst=batch_dst_node_ids - 1,
                                                                           pos_timestamp=batch_node_interact_times,
                                                                           split_mode=dtype)
            batch_neg_dst_node_ids = (np.array(
                batch_neg_dst_node_ids, dtype=batch_src_node_ids.dtype) + 1).reshape(-1)
            num_negative_samples_per_node = len(
                batch_neg_dst_node_ids) // len(batch_src_node_ids)
            assert num_negative_samples_per_node * \
                len(batch_src_node_ids) == len(batch_neg_dst_node_ids)

            batch_neg_src_node_ids = np.repeat(
                batch_src_node_ids, repeats=num_negative_samples_per_node)
            batch_neg_node_interact_times = np.repeat(
                batch_node_interact_times, repeats=num_negative_samples_per_node)

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer', 'TPNet']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times)
            elif model_name == 'NAT':
                negative_edge_embeddings = \
                    model[0].compute_edge_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                              dst_node_ids=batch_neg_dst_node_ids,
                                                              node_interact_times=batch_neg_node_interact_times,
                                                              edge_ids=None,
                                                              edges_are_positive=False)

                positive_edge_embeddings = \
                    model[0].compute_edge_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                              dst_node_ids=batch_dst_node_ids,
                                                              node_interact_times=batch_neg_node_interact_times,
                                                              edge_ids=batch_edge_ids,
                                                              edges_are_positive=True)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            if model_name == 'NAT':
                positive_probabilities = model[1](
                    edge_embeddings=positive_edge_embeddings).squeeze(dim=-1)
                negative_probabilities = model[1](
                    edge_embeddings=negative_edge_embeddings).squeeze(dim=-1)
            else:
                positive_probabilities = model[1](src_node_ids=batch_src_node_ids,
                                                  dst_node_ids=batch_dst_node_ids,
                                                  src_node_embeddings=batch_src_node_embeddings,
                                                  dst_node_embeddings=batch_dst_node_embeddings
                                                  ).squeeze(dim=-1)
                negative_probabilities = model[1](src_node_ids=batch_neg_src_node_ids,
                                                  dst_node_ids=batch_neg_dst_node_ids,
                                                  src_node_embeddings=batch_neg_src_node_embeddings,
                                                  dst_node_embeddings=batch_neg_dst_node_embeddings
                                                  ).squeeze(dim=-1)

            if model[1].random_projections is not None:
                model[1].random_projections.update(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                                                   node_interact_times=batch_node_interact_times)

            predicts = torch.cat(
                [positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                               dim=0)
            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            # evaluate_metric_name = []
            input_dict = {
                "y_pred_pos": positive_probabilities,
                "y_pred_neg": negative_probabilities.reshape(-1, num_negative_samples_per_node),
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append(
                {eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_sNet_link_prediction(model: nn.Module, dtype: str, eval_metric_name: str,
                                  neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                  evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data,
                                  evaluator: Evaluator, get_eval_loss: Callable, logger: logging.Logger = None):
    """
    evaluate models on the link prediction task
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_type: str, type of loss function
    :param logger:
    :return:
    """
    model.set_neighbor_sampler(neighbor_sampler)
    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_pos_src_node_ids, batch_pos_dst_node_ids, batch_pos_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[
                    evaluate_data_indices]

            batch_neg_dst_node_ids = evaluate_neg_edge_sampler.query_batch(pos_src=batch_pos_src_node_ids - 1,
                                                                           pos_dst=batch_pos_dst_node_ids - 1,
                                                                           pos_timestamp=batch_pos_node_interact_times,
                                                                           split_mode=dtype)
            batch_neg_dst_node_ids = (np.array(
                batch_neg_dst_node_ids, dtype=batch_pos_src_node_ids.dtype) + 1).reshape(-1)
            num_negative_samples_per_node = len(
                batch_neg_dst_node_ids) // len(batch_pos_src_node_ids)
            assert num_negative_samples_per_node * \
                len(batch_pos_src_node_ids) == len(batch_neg_dst_node_ids)
            batch_neg_src_node_ids = np.repeat(
                batch_pos_src_node_ids, repeats=num_negative_samples_per_node)
            batch_neg_node_interact_times = np.repeat(
                batch_pos_node_interact_times, repeats=num_negative_samples_per_node)

            batch_src_node_ids = np.concatenate(
                [batch_pos_src_node_ids, batch_neg_src_node_ids])
            batch_dst_node_ids = np.concatenate(
                [batch_pos_dst_node_ids, batch_neg_dst_node_ids])
            batch_node_interact_times = np.concatenate(
                [batch_pos_node_interact_times, batch_neg_node_interact_times])

            selected_models, logits = model.compute_logits(src_node_ids=batch_src_node_ids,
                                                           dst_node_ids=batch_dst_node_ids,
                                                           node_interact_times=batch_node_interact_times)

            if model.memory_model is not None:
                model.memory_model.update(src_node_ids=batch_pos_src_node_ids, dst_node_ids=batch_pos_dst_node_ids,
                                          node_interact_times=batch_pos_node_interact_times)

            labels = torch.cat([torch.ones(len(batch_pos_src_node_ids), device=logits.device),
                                torch.zeros(len(batch_neg_src_node_ids), device=logits.device)], dim=0)
            loss = get_eval_loss(logits=logits, labels=labels)
            evaluate_losses.append(loss.item())

            # evaluate_metric_name = []
            input_dict = {
                "y_pred_pos": logits[:len(batch_pos_src_node_ids)],
                "y_pred_neg": logits[len(batch_pos_src_node_ids):].reshape(-1, num_negative_samples_per_node),
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append(
                {eval_metric_name: evaluator.eval(input_dict)[eval_metric_name],
                 'ratio': 1-np.sum(selected_models)/len(selected_models)})

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics
