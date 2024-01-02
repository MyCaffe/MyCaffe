from typing import Tuple
import torch
from utility import DebugFunction
from typing import List, Dict, Tuple, Optional

def compute_quantile_loss_instance_wise(outputs: torch.Tensor,
                                        targets: torch.Tensor,
                                        desired_quantiles: torch.Tensor, debug_opt: Optional[bool] = False, path="") -> torch.Tensor:
    """
    This function compute the quantile loss separately for each sample,time-step,quantile.

    Parameters
    ----------
    outputs: torch.Tensor
        The outputs of the model [num_samples x num_horizons x num_quantiles].
    targets: torch.Tensor
        The observed target for each horizon [num_samples x num_horizons].
    desired_quantiles: torch.Tensor
        A tensor representing the desired quantiles, of shape (num_quantiles,)

    Returns
    -------
    losses_array: torch.Tensor
        a tensor [num_samples x num_horizons x num_quantiles] containing the quantile loss for each sample,time-step and
        quantile.
    """
    if debug_opt == True:
        if path != "":
            DebugFunction.set_output_path(path, 0)
        debug = DebugFunction.apply

    # compute the actual error between the observed target and each predicted quantile
    errors = targets.unsqueeze(-1) - outputs

    if debug_opt == True:
        DebugFunction.trace(errors, "tft.loss.errors")
        errors = debug(errors)

    # Dimensions:
    # errors: [num_samples x num_horizons x num_quantiles]

    # compute the loss separately for each sample,time-step,quantile

    if debug_opt == True:
        DebugFunction.trace(desired_quantiles, "tft.loss.desired_quantiles")
        desired_quantiles = debug(desired_quantiles)

    desired_quantiles1 = desired_quantiles - 1

    if debug_opt == True:
        DebugFunction.trace(desired_quantiles1, "tft.loss.desired_quantiles1")
        desired_quantiles1 = debug(desired_quantiles1)

    err1 = desired_quantiles1 * errors

    if debug_opt == True:
        DebugFunction.trace(err1, "tft.loss.err1")
        err1 = debug(err1)

    err2 = desired_quantiles * errors

    if debug_opt == True:
        DebugFunction.trace(err2, "tft.loss.err2")
        err2 = debug(err2)

    losses_array = torch.max(err1, err2)
    # Dimensions:
    # losses_array: [num_samples x num_horizons x num_quantiles]

    return losses_array


def get_quantiles_loss_and_q_risk(outputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  desired_quantiles: torch.Tensor, debug_opt: Optional[bool] = False, path="") -> Tuple[torch.Tensor, ...]:
    """
    This function computes quantile loss and q-risk metric.

    Parameters
    ----------
    outputs: torch.Tensor
        The outputs of the model [num_samples x num_horizons x num_quantiles].
    targets: torch.Tensor
        The observed target for each horizon [num_samples x num_horizons].
    desired_quantiles: torch.Tensor
        a tensor representing the desired quantiles, of shape (num_quantiles,).

    Returns
    ----------
    q_loss: torch.Tensor
        a scalar representing the quantile loss across all samples,horizons and quantiles.
    q_risk: torch.Tensor
        a tensor (shape=(num_quantiles,)) with q-risk metric for each quantile separately.
    losses_array: torch.Tensor
        a tensor [num_samples x num_horizons x num_quantiles] containing the quantile loss for each
        sample,time-step and quantile.

    """

    if debug_opt == True:
        if path != "":
            DebugFunction.set_output_path(path, 0)
        debug = DebugFunction.apply
        DebugFunction.trace(outputs, "tft.loss.outputs")
        outputs = debug(outputs)
        DebugFunction.trace(targets, "tft.loss.targets")
        targets = debug(targets)

    losses_array = compute_quantile_loss_instance_wise(outputs=outputs,
                                                       targets=targets,
                                                       desired_quantiles=desired_quantiles, debug_opt=debug_opt, path=path)

    if debug_opt == True:
        DebugFunction.trace(losses_array, "tft.loss.losses_array")
        losses_array = debug(losses_array)

    # sum losses over quantiles and average across time and observations

    losses_array_sum = losses_array.sum(dim=-1)
    if debug_opt == True:
        DebugFunction.trace(losses_array_sum, "tft.loss.losses_array_sum")
        losses_array_sum = debug(losses_array_sum)

    losses_array_sum_mean = losses_array_sum.mean(dim=-1)
    if debug_opt == True:
        DebugFunction.trace(losses_array_sum_mean, "tft.loss.losses_array_sum_mean")
        losses_array_sum_mean = debug(losses_array_sum_mean)

    q_loss = losses_array_sum_mean.mean()  # a scalar (shapeless tensor)

    if debug_opt == True:
        DebugFunction.trace(q_loss, "tft.loss.q_loss")
        q_loss = debug(q_loss)

    # compute q_risk for each quantile
    targets_sum = targets.abs().sum()
    targets_sum1 = targets_sum.unsqueeze(-1)

    if debug_opt == True:
        DebugFunction.trace(targets_sum1, "tft.loss.targets_sum1")
        targets_sum1 = debug(targets_sum1)

    losses_array_sum1 = losses_array_sum.sum(dim=0)

    if debug_opt == True:
        DebugFunction.trace(losses_array_sum1, "tft.loss.losses_array_sum1")
        losses_array_sum1 = debug(losses_array_sum1)

    q_risk = 2 * losses_array_sum1 / targets_sum1

    if debug_opt == True:
        DebugFunction.trace(q_risk, "tft.loss.q_risk")
        q_risk = debug(q_risk)

    return q_loss, q_risk, losses_array
