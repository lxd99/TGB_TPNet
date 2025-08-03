import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import wandb
import numpy as np
import torch.nn as nn

class WandbLinkLogger:
    def __init__(self, run_type, args):
        if run_type == 'run':
            self.run_wandb = wandb.init(config=vars(args), project='TGB_Gradient', entity='cascade',
                                        tags=[run_type], reinit=True,
                                        name=f'{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}')
        elif run_type == 'summary':
            self.run_wandb = wandb.init(config=vars(args), project='TGB_Gradient', entity='cascade',
                                        tags=[run_type], reinit=True,
                                        name=f'{args.prefix}_link_{args.dataset_name}_{args.model_name}_summary')
        elif run_type == 'eval_run':
            self.run_wandb = wandb.init(config=vars(args), project='TGB_Gradient', entity='cascade',
                                        tags=[run_type], reinit=True,
                                        name=f'[eval]_{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}')
        elif run_type == 'eval_summary':
            self.run_wandb = wandb.init(config=vars(args), project='TGB_Gradient', entity='cascade',
                                        tags=[run_type], reinit=True,
                                        name=f'[eval]_{args.prefix}_link_{args.dataset_name}_{args.model_name}_summary')
        else:
            raise ValueError("Not Implemented Run Type")

    def watch(self, model: nn.Module):
        self.run_wandb.watch(model, log='gradients', log_freq=100)

    def standarlize_results(self, metrics, losses):
        """
        :param losses: a list of batch losses
        @param metrics:, a list of dicts, where each dict record a batch result
        and is in the form of {'metric_1':value_1,'metric_2':value_2,...,'metric_n':value_n}
        """
        results_dict = {}
        results_dict['loss'] = np.mean(losses)
        for metric_name in metrics[0].keys():
            results_dict[metric_name.upper()] = np.mean([x[metric_name] for x in metrics])
        return results_dict

    def log_epoch(self, train_losses, train_metrics, val_losses, val_metrics, epoch,test_losses=None,test_metrics=None):
        if test_losses and test_metrics:
            result_dict = {
                'train': self.standarlize_results(metrics=train_metrics, losses=train_losses),
                'val': self.standarlize_results(metrics=val_metrics, losses=val_losses),
                'test': self.standarlize_results(metrics=test_metrics, losses=test_losses)
            }
            self.run_wandb.log(result_dict, step=epoch)
        else:
            result_dict = {
                'train': self.standarlize_results(metrics=train_metrics, losses=train_losses),
                'val': self.standarlize_results(metrics=val_metrics, losses=val_losses)
            }
            self.run_wandb.log(result_dict, step=epoch)

    def log_run(self, val_losses, val_metrics, test_losses, test_metrics):
        result_dict = self.standarlize_results(metrics=test_metrics, losses=test_losses)
        val_result_dict = self.standarlize_results(metrics=val_metrics, losses=val_losses)
        # add a prefix for results on validating set
        result_dict.update({'v' + x: y for x, y in val_result_dict.items()})
        self.run_wandb.summary.update(result_dict)
    
    def configurate_log_run(self, small_val_losses,small_val_metrics,val_losses, val_metrics, test_losses, test_metrics):
        result_dict = self.standarlize_results(metrics=test_metrics, losses=test_losses)
        small_val_result_dict = self.standarlize_results(metrics=small_val_metrics, losses=small_val_losses)
        val_result_dict = self.standarlize_results(metrics=val_metrics, losses=val_losses)
        # add a prefix for results on validating set
        result_dict.update({'v' + x: y for x, y in val_result_dict.items()})
        result_dict.update({'vS' + x: y for x, y in small_val_result_dict.items()})
        self.run_wandb.summary.update(result_dict)
    
    def log_run_with_dict(self,val_losses,val_metric,test_losses,test_metric):
        val_metric = {'v'+x.upper():y for x,y in val_metric.items()}
        val_metric['vloss'] = np.mean(val_losses)
        test_metric = {x.upper():y for x,y in test_metric.items()}
        test_metric['loss'] = np.mean(test_losses)
        self.run_wandb.summary.update({**val_metric,**test_metric})
        

    def log_final(self, val_metrics, test_metrics):
        def standarlize_final_results(metrics):
            metric_dict = {}
            for metric_name in metrics[0].keys():
                metric_value_list = [x[metric_name] for x in metrics]
                metric_dict[metric_name.upper()] = np.mean(metric_value_list)
                metric_dict[metric_name.upper() + '_std'] = np.std(metric_value_list, ddof=1)
            return metric_dict

        result_dict = standarlize_final_results(metrics=test_metrics)
        val_result_dict = standarlize_final_results(metrics=val_metrics)
        # add a prefix for results on validating set
        result_dict.update({'v' + x: y for x, y in val_result_dict.items()})
        self.run_wandb.summary.update(result_dict)

    def finish(self):
        self.run_wandb.finish()


class LossFunction:
    def __init__(self, loss_type):
        self.loss_type = loss_type

    def forward(self, positive_logits, negative_logits):
        if self.loss_type == 'pointwise':
            predicts = torch.cat([positive_logits, negative_logits])
            labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(input=predicts, target=labels)
        elif self.loss_type == 'listwise':
            neg_per_edge = len(negative_logits) // len(positive_logits)
            assert neg_per_edge * len(positive_logits) == len(negative_logits)
            logits = torch.cat([positive_logits[:, None], negative_logits.reshape(-1, neg_per_edge)], dim=1)
            labels = torch.zeros_like(positive_logits).to(dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(input=logits, target=labels)
        else:
            raise ValueError("Not Implemented Loss Type")
        return loss
