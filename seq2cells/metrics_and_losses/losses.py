"""
Custom losses for aiml-seq2cells
=========================================
Copyright 2023 GlaxoSmithKline Research & Development Limited. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=========================================
"""
from typing import Literal

import torch
import torch.nn.functional as F


def log(t, eps=1e-20):
    """Custom log function clamped to minimum epsilon."""
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred: torch.Tensor, target: torch.Tensor):
    """Poisson loss"""
    return (pred - target * log(pred)).mean()


def nonzero_median(tensor: torch.Tensor, axis: int, keepdim: bool) -> torch.Tensor:
    """Compute the median across non-zero float elements.

    Notes
    -----
    Modifies the tensor in place to avoid making a copy.
    """
    tensor = torch.where(tensor != 0.0, tensor.double(), float("nan"))

    # returns values and indices - we only want the value(s)
    medians = torch.nanmedian(tensor, dim=axis, keepdim=keepdim)[0]

    medians = medians.nan_to_num(0)

    return medians


class BalancedPearsonCorrelationLoss(torch.nn.Module):
    """Pearson Corr balances between across gene and cell performance"""

    def __init__(
        self,
        rel_weight_gene: float = 1.0,
        rel_weight_cell: float = 1.0,
        norm_by: Literal["mean", "nonzero_median"] = "mean",
        eps: float = 1e-8,
    ):
        """Initialise PearsonCorrelationLoss.

        Parameter
        ---------
        rel_weight_gene: float = 1.0
            The relative weight to put on the across gene/tss correlation.
        rel_weight_cell: float = 1.0
            The relative weight to put on the across cells correlation.
        norm_by:  Literal['mean', 'nonzero_median'] = 'nonzero_median'
            What to use as across gene / cell average to subtract from the
            signal to normalise it. Mean or the Median of the non zero entries.
        eps: float 1e-8
            epsilon
        """
        super().__init__()
        self.eps = eps
        self.norm_by = norm_by
        self.rel_weight_gene = rel_weight_gene
        self.rel_weight_cell = rel_weight_cell

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward.

        Notes
        -----
        preds: torch.Tensor
            2D torch tensor [genes x cells], batched over genes.
        targets: torch.Tensor
            2D torch tensor [genes x cells], batched over genes.
        """
        if self.norm_by == "mean":
            preds_avg_gene = preds.mean(dim=0, keepdim=True)
            targets_avg_gene = targets.mean(dim=0, keepdim=True)
            preds_avg_cell = preds.mean(dim=1, keepdim=True)
            targets_avg_cell = targets.mean(dim=1, keepdim=True)
        else:
            preds_avg_gene = nonzero_median(preds, 0, keepdim=True)
            targets_avg_gene = nonzero_median(targets, 0, keepdim=True)
            preds_avg_cell = nonzero_median(preds, 1, keepdim=True)
            targets_avg_cell = nonzero_median(targets, 1, keepdim=True)

        r_tss = torch.nn.functional.cosine_similarity(
            preds - preds_avg_gene,
            targets - targets_avg_gene,
            eps=self.eps,
            dim=0,
        )

        r_celltype = torch.nn.functional.cosine_similarity(
            preds - preds_avg_cell,
            targets - targets_avg_cell,
            eps=self.eps,
        )

        loss = self.rel_weight_gene * (1 - r_tss.mean()) + self.rel_weight_cell * (
            1 - r_celltype.mean()
        )

        # norm the loss to 2 by half the sum of the relative weights
        loss = (loss * 2) / (self.rel_weight_gene + self.rel_weight_cell)

        return loss
    
def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float())
    return -masked_log_probs.mean()


class MultiTasksLogitPars:
    def __init__(self, num_classes=[10]):
        super().__init__()
        
        self.n_tasks = len(num_classes)
        self.num_classes = num_classes

    def log_minus_exp(self, a, b, epsilon=1.e-6):
        return a + torch.log1p(-torch.exp(b - a) + epsilon)

    def get_logits_from_logistic_pars(self, loc, log_scale, n_task=0, use_softmax=False):
        if self.num_classes[n_task] == 1:
            return loc

        num_classes = self.num_classes[n_task]
        loc = torch.tanh(loc).unsqueeze(-1)  # ensure loc is between [-1, 1], just like normalized data.
        log_scale = log_scale.unsqueeze(-1)

        inv_scale = (-log_scale + 2.0).exp()
        
        bin_width = 2.0 / (num_classes - 1)
        bin_centers = torch.linspace(-1.0, 1.0, num_classes).to(loc.device)
        for dim in range(loc.ndim - 1):
            bin_centers = torch.unsqueeze(bin_centers, dim=0)
        bin_centers = bin_centers - loc
        
        # equivalent implementation
        # log_cdf_min = -1 * torch.log1p((-inv_scale * (bin_centers - 0.5 * bin_width)).exp())
        # log_cdf_plus = -1 * torch.log1p((-inv_scale * (bin_centers + 0.5 * bin_width)).exp())
        log_cdf_min = torch.nn.LogSigmoid()(inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = torch.nn.LogSigmoid()(inv_scale * (bin_centers + 0.5 * bin_width))

        logits = self.log_minus_exp(log_cdf_plus, log_cdf_min)  # [..., num_classes]
        return F.softmax(logits, dim=-1) if use_softmax else logits


if __name__ == "__main__":
    loc = torch.randn(2, 2).clip(-1, 1)
    log_scale = torch.randn(1, 2)

    logitpars = MultiTasksLogitPars(num_classes=[2])
    logits = logitpars.get_logits_from_logistic_pars(loc, log_scale, n_task=0, softmax=True)
    print(logits.shape)
    print(logits)
