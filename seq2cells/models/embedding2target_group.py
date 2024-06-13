"""
Embeddings2Target model wrapper

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

Model wrapper for pytorch lightning models
from Enformer pre-computed embeddings or pre-computed targets to
observed Enformer targets
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import pearson_corrcoef
from adabelief_pytorch import AdaBelief
from torchvision.ops import sigmoid_focal_loss

from seq2cells.metrics_and_losses.losses import (
    BalancedPearsonCorrelationLoss,
    poisson_loss,
    criterion_neg_log_bernoulli,
    MultiTasksLogitPars
)
from seq2cells.utils.training_utils import (
    get_linear_warm_up_cosine_decay_scheduler,
)
from lion_pytorch import Lion


class Embedding2Target(pl.LightningModule):
    """A single FC layer model to predict targets from embeddings."""

    def __init__(
        self,
        emb_dim: int,
        target_dim: int,
        n_targets: int,
        n_classes: list,
        loss_list: list,
        learning_rate: float,
        lr_schedule: str,
        lr_warmup_epochs: int,
        lr_max_epochs: int,
        optimizer: str,
        weight_decay: float,
        softplus_list: list,
        use_logit_pars: bool,
        dropout_prob: Optional[float] = 0.0,
        model_trunk: Optional[torch.nn.Module] = None,
        target_is_log: Optional[bool] = False,
        log_train: Optional[bool] = False,
        log_validate: Optional[bool] = True,
        std_validate: Optional[bool] = False,
        model_type: Optional[Literal["linear", "provided"]] = "linear",
        mode: Optional[Literal["bulk", "single_cell"]] = "bulk",
        freeze: Optional[str] = None,
        bottleneck_dim: Optional[int] = None,
        bottleneck_nonlin: Optional[str] = None,
        rel_weight_gene: Optional[float] = 1.0,
        rel_weight_cell: Optional[float] = 1.0,
        pears_norm_mode: Literal["mean", "nonzero_median"] = "mean",
        train_eval: bool = True,
    ) -> None:
        """Initialize the model

        Parameters
        ----------
            emb_dim: int
                Dimension of the embedding prior to the linear mapping.
            target_dim : int
                Number of targets
            n_targets: int
                Number of heads in the model
            loss : str
                'mse' or 'poisson' or 'poissonnll' or 'pearson'
                to apply mean-squared-error or poisson loss or
                poisson-nll-loss or a balanced Pearson correlation
                based loss respectively
            learning_rate: float
                Learning rate for optimizer default 1e-3
            lr_schedule:
                Select learning rate schedule: 'linear_warm_up_cosine_decay'
                or 'linear_warm_up'
            lr_warmup_epochs: int
                Warmup epochs to reach the provided learning_rate.
            lr_max_epochs: int
                Maximum number of epochs for lr schedule.
            optimizer: str
                Select desired optimizer to run.
            weight_decay: float
                Weight decay parameter for AdamW.
            dropout_prob: float
                Dropout probability > 0 will add a dropout layer in front of
                bottleneck layers. Will not apply in linear layer only models.
            model_trunk: Optional[torch.nn.Module],
                A torch module to use as model trunk
            softplus: bool
                If to append a softplus layer after the linear layer.
            target_is_log: bool
                Specify if the target counts/coverage have already been log
                transformed.
            log_train: bool
                Train against log(x+1) transformed data.
            log_validate: bool
                Validate using log(x+1) transformed data.
            std_validate: bool
                Standardize data across ROI / tss for validation.
            model_type: Optional[str] = "linear"
                Define if the model is a simple 'linear' model to  be
                applied on the embeddings or a provided model that goes from
                whatever input to intermediate embedding layer to output.
                Or a "bottleneck" model applying a bottleneck layer before the
                final linear layer.
            mode: Optional[str] = 'bulk'
                Model mode running against 'bulk' or 'single_cell' data.
            freeze: Optional[str] = None
                Specify if to freeze parts of the network. Must be None
                or 'trunk'.
            bottleneck_dim: Optional[int] = None
                Hidden dimension of the bottleneck layer.
            bottleneck_nonlin: Optional[str] = None
                Which non linearity to apply in bottleneck layer.
                None --> Apply no non linearity
                'RELU' --> apply RELU
            rel_weight_gene: Optional[float] = 1.0,
                For Balanced Pearson loss: relative weight to place on across
                gene correlation.
            rel_weight_cell: Optional[float] = 1.0,
                For Balanced Pearson loss: relative weight to place on across
                cell correlation.
            pears_norm_mode: Literal["mean", "nonzero_median"] = "mean"
                For Balanced Pearson loss: which average mode to use for
                norming the batches. ['mean', 'nonzero_median'] Default = mean
            train_eval: bool = True
                Set False to skip training set evaluation (e.g. for large
                models).
        """
        super().__init__()

        self.save_hyperparameters(ignore=["model_trunk"])
        self.model_trunk = model_trunk

        # save freeze hyperparam
        self.freeze = freeze

        # log if model tpye needs a trunk
        self.use_trunk = self.hparams.model_type in ["provided", "provided_bottleneck"]

        if self.model_trunk is not None:
            if self.freeze == "trunk":
                # set model trunk parameters to frozen
                for p in self.model_trunk.parameters():
                    p.requires_grad = False

        self.pearson_loss = BalancedPearsonCorrelationLoss(
                    rel_weight_gene=self.hparams.rel_weight_gene,
                    rel_weight_cell=self.hparams.rel_weight_cell,
                    norm_by=self.hparams.pears_norm_mode,
                )

        # setup the model =================
        assert self.hparams.model_type in [
            "linear",
            "provided",
            "bottleneck",
            "provided_bottleneck",
        ], (
            "Select a valid model_type: 'linear', 'provided', 'bottleneck', "
            "'provided_bottleneck!'"
        )
        modules = []
        class_heads = []

        if self.hparams.model_type in ["linear", "provided"]:
            modules.append(nn.Linear(self.hparams.emb_dim, self.hparams.target_dim))
        elif self.hparams.model_type in ["bottleneck", "provided_bottleneck"]:
            # from emb to bottleneck dim
            modules.append(nn.Linear(self.hparams.emb_dim, self.hparams.bottleneck_dim))
            # dropout if drop prob > 0
            if self.hparams.dropout_prob > 0.0:
                modules.append(nn.Dropout(p=self.hparams.dropout_prob))

            # add non linearity
            if self.hparams.bottleneck_nonlin == "RELU":
                modules.append(nn.ReLU())
            else:
                print("No non-linearity added!")

            modules.append(
                nn.Linear(self.hparams.bottleneck_dim, self.hparams.bottleneck_dim)
            )
            
            modules.append(
                nn.Dropout(p=self.hparams.dropout_prob)
            )
            
            modules.append(
                nn.ReLU()
            )
            
            if self.hparams.use_logit_pars:
                # each task share a loc linear head
                # but have distinct log_scale linear head
                assert len(n_classes) == self.hparams.n_targets

                self.loc_head = nn.Linear(self.hparams.bottleneck_dim, self.hparams.target_dim)
                self.log_scale_head = nn.ModuleList(
                    [
                        nn.Linear(self.hparams.bottleneck_dim, self.hparams.target_dim)
                        for i in range(self.hparams.n_targets)
                    ]
                )

                # initialize log_scale_head and loc_head with zeros
                # nn.init.zeros_(self.loc_head.weight)
                # nn.init.zeros_(self.loc_head.bias)
                # for i in range(self.hparams.n_targets):
                #     nn.init.zeros_(self.log_scale_head[i].weight)
                #     nn.init.zeros_(self.log_scale_head[i].bias)

                self.logit_pars = MultiTasksLogitPars(n_classes)  # e.g., [[1], [110]] for expression and peak prediction

            else:
                # use distinct linear layers for each target
                class_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.hparams.bottleneck_dim, self.hparams.target_dim),
                            nn.Identity() if not self.hparams.softplus_list[i] else nn.Softplus()
                        )
                        for i in range(self.hparams.n_targets)
                    ]
                )
                self.class_heads = class_heads

        self.model_head = nn.Sequential(*modules)

        # check if optimizer supported
        assert self.hparams.optimizer in ["AdamW", "SGD", "RMSprop", "Adabelief", "Lion"], (
            "The selected optimizer is not supported. "
            "Select 'SGD', 'AdamW' or 'RMSprop' or 'Adabelief'!"
        )

        # mode ['single_cell' or 'bulk'] defined what is reported on the
        # progress bar
        self.report_bar_across_genes = True
        if self.hparams.mode == "bulk":
            self.report_bar_across_celltypes = True
            self.report_bar_matrx = False
        else:
            self.report_bar_across_celltypes = False
            self.report_bar_matrx = True

    def model_forward(self, x, task_id):
        if self.use_trunk:
            emb = self.model_trunk(x)
            emb = self.model_head(emb)
        else:
            emb = self.model_head(x)
    
        if self.hparams.use_logit_pars:
            loc = self.loc_head(emb)
            log_scale = self.log_scale_head[task_id](emb)
            y_hat = self.logit_pars.get_logits_from_logistic_pars(loc, log_scale, task_id)
        else:
            y_hat = self.class_heads[task_id](emb)
        
        return y_hat

    def forward(self, inputs):
        x, y, task_ids = inputs

        y_hat = torch.zeros_like(y)
        for task_id in range(self.hparams.n_targets):
            task_mask = task_ids == task_id
            if task_mask.sum() == 0:
                continue

            res = self.model_forward(x[task_mask], task_id)
            if res.ndim > 2:
                res = torch.argmax(res, dim=-1).float()
            y_hat[task_mask] = res
        return y_hat

    def training_step(self, batch_list, batch_idx):
        """PL training step definition"""
        # training_step defines the train loop.
        # it is independent of forward

        total_loss = 0.
        y_list, y_hat_list = [], []
        for i, batch in enumerate(batch_list):
            if batch is None:
                continue
            
            x, y = batch
            y_hat = self.model_forward(x, i)

            if self.hparams.target_is_log and not self.hparams.log_train:
                # transform back to exponentials
                y = torch.exp(y) - 1
                y_hat = torch.exp(y_hat) - 1

            if self.hparams.log_train and not self.hparams.target_is_log:
                # log transform expression data for correlation
                y = torch.log(y + 1)
                y_hat = torch.log(y_hat + 1)

            if self.hparams.loss_list[i] == "mse":
                loss = F.mse_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "poisson":
                loss = poisson_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "mae":
                loss = F.l1_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "ce":
                weight = torch.tensor([1] + [10 for _ in range(self.hparams.n_classes[i] - 1)]).float().to(y_hat.device)
                loss = F.cross_entropy(y_hat.view(-1, y_hat.shape[-1]), y.reshape(-1).long(), weight=weight)
                # pick the class with the highest probability
                y_hat = torch.argmax(y_hat, dim=-1).float()
            elif self.hparams.loss_list[i] == "poissonnll":
                loss = F.poisson_nll_loss(y_hat, y, log_input=False)
            elif self.hparams.loss_list[i] == "pearson":
                loss = self.pearson_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "pearson_poissonnll":
                loss = self.pearson_loss(y_hat, y) + F.poisson_nll_loss(y_hat, y, log_input=False)
            elif self.hparams.loss_list[i] == "pearson_l1":
                loss = self.pearson_loss(y_hat, y) + F.smooth_l1_loss(y_hat, y)
            elif self.hparams.loss_list[i] == 'mixture':
                l1 = self.pearson_loss(y_hat, y)
                l2 = F.poisson_nll_loss(y_hat, y, log_input=False)
                l3 = F.smooth_l1_loss(y_hat, y)
                loss = l1 + l2 + l3
            else:
                raise Exception("Select mse, poisson or poissonnll as loss hyperparam")

            # Logging to TensorBoard by default
            self.log(f"train_loss_{i}", loss, prog_bar=True, sync_dist=True)

            total_loss += loss
            y_list.append(y.detach().cpu())  # to avoid memory leak and memory blow up
            y_hat_list.append(y_hat.detach().cpu())  # to avoid memory leak and memory blow up

        total_loss /= len(batch_list)
        self.log(f"train_loss", total_loss, prog_bar=True, sync_dist=True)

        return {"loss": total_loss, "y": y_list, "y_hat": y_hat_list}

    def training_epoch_end(self, outs):
        """PL to run training set eval after epoch"""
        if not self.hparams.train_eval:
            return

        for i in range(len(outs[0]["y"])):
            y = torch.cat([x["y"][i] for x in outs])
            y_hat = torch.cat([x["y_hat"][i] for x in outs])

            if (
                self.hparams.log_validate
                and not self.hparams.log_train
                and not self.hparams.target_is_log
            ):
                # log transform expression data for correlation
                y = torch.log(y + 1)
                y_hat = torch.log(y_hat + 1)

            if self.hparams.std_validate:
                # calculate mean and std of observed and predicted targets
                # update class state
                self.target_means_obs = torch.mean(y, dim=0)
                self.target_stds_obs = torch.std(y, dim=0)
                self.target_means_pred = torch.mean(y_hat, dim=0)
                self.target_stds_pred = torch.std(y_hat, dim=0)
                # normalize output
                y = y - self.target_means_obs
                y = y / self.target_stds_obs
                y_hat = y_hat - self.target_means_pred
                y_hat = y_hat / self.target_stds_pred

            pc_across_tss = torch.stack(
                [pearson_corrcoef(y_hat[:, i], y[:, i]) for i in range(y_hat.shape[1])]
            )
            pc_across_tss = torch.nan_to_num(
                pc_across_tss
            )  # handle correlation NaN : default to 0
            mean_pc_across_tss = pc_across_tss.mean()

            pc_across_celltypes = torch.stack(
                [pearson_corrcoef(y_hat[i, :], y[i, :]) for i in range(y_hat.shape[0])]
            )
            pc_across_celltypes = torch.nan_to_num(
                pc_across_celltypes
            )  # handle correlation NaN : default to 0
            mean_pc_across_celltypes = pc_across_celltypes.mean()

            # calculate correlation of whole gene x cell(type) matrix
            pc_whole_matrix = pearson_corrcoef(torch.ravel(y_hat), torch.ravel(y))

            # outs is a list of whatever you returned in `validation_step`
            self.log(
                f"train_corr_across_tss_{i}",
                mean_pc_across_tss,
                prog_bar=self.report_bar_across_genes,
            )
            self.log(
                f"train_corr_across_celltypes_{i}",
                mean_pc_across_celltypes,
                prog_bar=self.report_bar_across_celltypes,
            )
            self.log(f"train_corr_{i}", pc_whole_matrix, prog_bar=self.report_bar_matrx)

    def validation_step(self, batch_list, batch_idx):
        """PL validation step definition"""
        y_list, y_hat_list = [], []
        val_loss = []
        for i, batch in enumerate(batch_list):
            if batch is None:
                continue

            x, y = batch
            y_hat = self.model_forward(x, i)

            if self.hparams.target_is_log and not self.hparams.log_validate:
                # transform back to exponentials
                y = torch.exp(y + 1)
                y_hat = torch.exp(y_hat + 1)

            if self.hparams.log_validate and not self.hparams.target_is_log:
                # log transform expression data for correlation
                y = torch.log(y + 1)
                y_hat = torch.log(y_hat + 1)

            if self.hparams.loss_list[i] == "mse":
                loss = F.mse_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "poisson":
                loss = poisson_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "mae":
                loss = F.l1_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "ce":
                weight = torch.tensor([1] + [10 for _ in range(self.hparams.n_classes[i] - 1)]).float().to(y_hat.device)
                loss = F.cross_entropy(y_hat.view(-1, y_hat.shape[-1]), y.reshape(-1).long(), weight=weight)
                # pick the class with the highest probability
                y_hat = torch.argmax(y_hat, dim=-1).float()
            elif self.hparams.loss_list[i] == "poissonnll":
                loss = F.poisson_nll_loss(y_hat, y, log_input=False)
            elif self.hparams.loss_list[i] == "pearson":
                loss = self.pearson_loss(y_hat, y)
            elif self.hparams.loss_list[i] == "pearson_poissonnll":
                loss = self.pearson_loss(y_hat, y) + F.poisson_nll_loss(y_hat, y, log_input=False)
            elif self.hparams.loss_list[i] == "pearson_l1":
                loss = self.pearson_loss(y_hat, y) + F.smooth_l1_loss(y_hat, y)
            elif self.hparams.loss_list[i] == 'mixture':
                l1 = self.pearson_loss(y_hat, y)
                l2 = F.poisson_nll_loss(y_hat, y, log_input=False)
                l3 = F.smooth_l1_loss(y_hat, y)
                loss = l1 + l2 + l3
            else:
                raise Exception("Select mse, poisson or poissonnll as loss hyperparam")

            val_loss.append(loss)
            y_list.append(y.detach().cpu())  # to avoid memory leak and memory blow up
            y_hat_list.append(y_hat.detach().cpu())  # to avoid memory leak and memory blow up

        return {"y": y_list, "y_hat": y_hat_list, "val_loss": val_loss}

    def validation_epoch_end(self, outs):
        """PL to run validation set eval after epoch"""

        total_mean_pc_across_tss = 0.
        total_pc_across_celltypes = 0.
        total_val_loss = 0.

        for i in range(len(outs[0]["y"])):  # iter all the tasks
            y = torch.cat([x["y"][i] for x in outs])
            y_hat = torch.cat([x["y_hat"][i] for x in outs])
            val_loss = torch.stack([x["val_loss"][i] for x in outs]).mean()

            total_val_loss += val_loss
            self.log(f"val_loss_{i}", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)

            if self.hparams.std_validate:
                # normalize output
                if hasattr(self, "target_means_obs"):
                    y = y - self.target_means_obs
                    y = y / self.target_stds_obs
                    y_hat = y_hat - self.target_means_pred
                    y_hat = y_hat / self.target_stds_pred
                else:
                    # for first epoch take mean and std of validation set
                    y = y - torch.mean(y, dim=0)
                    y = y / torch.std(y, dim=0)
                    y_hat = y_hat - torch.mean(y_hat, dim=0)
                    y_hat = y_hat / torch.std(y_hat, dim=0)

            pc_across_tss = torch.stack(
                [pearson_corrcoef(y_hat[:, i], y[:, i]) for i in range(y_hat.shape[1])]
            )
            pc_across_tss = torch.nan_to_num(
                pc_across_tss
            )  # handle correlation NaN : default to 0
            mean_pc_across_tss = pc_across_tss.mean()

            pc_across_celltypes = torch.stack(
                [pearson_corrcoef(y_hat[i, :], y[i, :]) for i in range(y_hat.shape[0])]
            )
            pc_across_celltypes = torch.nan_to_num(
                pc_across_celltypes
            )  # handle correl NaN : default to 0
            mean_pc_across_celltypes = pc_across_celltypes.mean()

            # calculate correlation of whole gene x cell(type) matrix
            pc_whole_matrix = pearson_corrcoef(torch.ravel(y_hat), torch.ravel(y))

            total_mean_pc_across_tss += mean_pc_across_tss
            total_pc_across_celltypes += mean_pc_across_celltypes
            # outs is a list of whatever you returned in `validation_step`
            self.log(
                f"valid_corr_across_tss_{i}",
                mean_pc_across_tss,
                prog_bar=self.report_bar_across_genes,
            )
            self.log(
                f"valid_corr_across_celltypes_{i}",
                mean_pc_across_celltypes,
                prog_bar=self.report_bar_across_celltypes,
            )
            self.log(f"valid_corr_{i}", pc_whole_matrix, prog_bar=self.report_bar_matrx)
            self.log(f"hp_metric_{i}", val_loss)

        self.log("val_loss", total_val_loss / len(outs[0]["y"]))
        self.log("valid_corr_across_tss", total_mean_pc_across_tss / len(outs[0]["y"]))
        self.log("valid_corr_across_celltypes", total_pc_across_celltypes / len(outs[0]["y"]))

    def test_step(self, batch, batch_idx):
        """PL test step definition"""
        """PL validation step definition"""
        x, y, task_id = batch
        y_hat = self.model_forward(x, task_id)
        return {"y": y, "y_hat": y_hat}

    # def test_epoch_end(self, outs):
    #     """PL to run validation set eval after epoch"""
    #     y = torch.cat([outs[i]["y"] for i in range(len(outs))])
    #     y_hat = torch.cat([outs[i]["y_hat"] for i in range(len(outs))])

    #     if self.hparams.log_validate:
    #         # log transform expression data for correlation
    #         y = torch.log(y + 1)
    #         y_hat = torch.log(y_hat + 1)

    #     if self.hparams.std_validate:
    #         # normalize output
    #         y = y - self.target_means_obs
    #         y = y / self.target_stds_obs
    #         y_hat = y_hat - self.target_means_pred
    #         y_hat = y_hat / self.target_stds_pred

    #     pc_across_tss = torch.stack(
    #         [pearson_corrcoef(y_hat[:, i], y[:, i]) for i in range(y_hat.shape[1])]
    #     )
    #     pc_across_tss = torch.nan_to_num(
    #         pc_across_tss
    #     )  # handle correlation NaN : default to 0
    #     mean_pc_across_tss = pc_across_tss.mean()

    #     pc_across_celltypes = torch.stack(
    #         [pearson_corrcoef(y_hat[i, :], y[i, :]) for i in range(y_hat.shape[0])]
    #     )
    #     pc_across_celltypes = torch.nan_to_num(
    #         pc_across_celltypes
    #     )  # handle correl NaN : default to 0
    #     mean_pc_across_celltypes = pc_across_celltypes.mean()

    #     # calculate correlation of whole gene x cell(type) matrix
    #     pc_whole_matrix = pearson_corrcoef(torch.ravel(y_hat), torch.ravel(y))

    #     # outs is a list of whatever you returned in `validation_step`
    #     self.log("test_corr_across_tss", mean_pc_across_tss)
    #     self.log("test_corr_across_celltypes", mean_pc_across_celltypes)
    #     self.log("test_corr", pc_whole_matrix)

    def configure_optimizers(self):
        """PL configure optimizer"""
        if self.hparams.optimizer == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "RMSprop":
            optimizer = optim.RMSprop(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.optimizer == 'Adabelief':
            optimizer = AdaBelief(self.parameters(), lr=self.hparams.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        
        elif self.hparams.optimizer == 'Lion':
            optimizer = Lion(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)


        if self.hparams.lr_schedule == "linear_warm_up":
            # mimicking the Enformer learning rate scheduler
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-04,
                    end_factor=1.0,
                    total_iters=self.hparams.lr_warmup_epochs,
                ),
                "name": "lr_logging",
            }

        elif self.hparams.lr_schedule == "linear_warm_up_cosine_decay":
            # linear warump with cosine decay
            lr_scheduler = {
                "scheduler": get_linear_warm_up_cosine_decay_scheduler(
                    optimizer,
                    lr_warmup_epochs=self.hparams.lr_warmup_epochs,
                    lr_max_epochs=self.hparams.lr_max_epochs,
                ),
                "name": "lr_logging",
            }
        elif self.hparams.lr_schedule == "constant":
            # constant learning rate
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda epoch: 1
                ),
                "name": "lr_logging",
            }
        elif self.hparams.lr_schedule == "reduce_on_plateau":
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.25,
                    patience=2,
                ),
                "monitor": "valid_corr_across_tss",
                "name": "lr_logging",
            }

        return [optimizer], [lr_scheduler]

