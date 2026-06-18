# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.

import logging
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import onnxruntime
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from data_pipeline import ArrayDataModule, Scaler
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim

from utils import (
    check_params,
    get_params,
    get_dynamic_nakashima_res,
    get_nakashima_matrix,
    torch_calc_dist_cluster,
    torch_dist_att_func,
)

logger = logging.getLogger(__name__)

# Switched to "high" to preserve deterministic accuracy in Physics-informed matrices
torch.set_float32_matmul_precision("high") 


class CustomLoss(nn.Module):
    """
    Custom loss function for evaluating the model's performance, incorporating MSE, RMSE, MAE,
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the custom loss value given the model output and target values.
        """
        mse_loss = F.mse_loss(output, target)
        rmse_loss = torch.sqrt(mse_loss)
        mae_loss = F.l1_loss(output, target)
        return rmse_loss, mae_loss


class NegLogLikelihoodLoss(nn.Module):
    """
    Custom loss function constructing Gaussian Negative Log Likelihood
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor, scale=1.0, lambda_reg=1.0) -> torch.Tensor:
        """
        find the best mean and std to maximize the likelihood
        """
        sigma = torch.exp(log_sigma)

        ## original nll
        loss = 0.5 * (log_sigma + (target - y_pred) ** 2 / sigma)

        reg = 0

        return scale * loss.mean() + reg


class ResidualBlock(nn.Module):
    def __init__(self, dim, use_BN=False, gate_init=0.1):
        super().__init__()
        self.use_BN = use_BN
        self.fc = nn.Linear(dim, dim)
        if use_BN:
            self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        out = self.fc(x)
        if self.use_BN:
            out = self.bn(out)
        return F.silu(x + out)


class Regressor(L.LightningModule):
    """
    A neural network model for predicting the dynamics of a tennis ball and racket interaction.
    """

    def __init__(self, cfg: DictConfig, scaler_input: Scaler, scaler_output: Scaler, cluster_centers: np.ndarray):
        """
        Args in cfg:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output.
        hidden_dims (list of int): List of dimensions for the hidden layers.
        learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()

        self.residual_model_bool: bool = cfg.residual_model_bool

        ## input ###
        self.input_types = cfg.input_types
        self.input_config = cfg.input_config

        self.hidden_dims: List[int] = cfg.hidden_dims
        self.vel_branch_hidden_dims: List[int] = cfg.vel_branch_hidden_dims
        self.spin_branch_hidden_dims: List[int] = cfg.spin_branch_hidden_dims
        self.other_branch_hidden_dims: List[int] = cfg.other_branch_hidden_dims
        self.conf_head_hidden_dims: List[int] = cfg.conf_head_hidden_dims
        self.vel_output_dim: int = cfg.output_dim  # same dimension for speed and spin
        self.spin_output_dim: int = cfg.output_dim
        self.learning_rate: float = cfg.learning_rate
        self.dropout_rate: float = cfg.dropout_rate

        self.att_type = 4

        ### Physics Informed Model smoothness
        self.physics = get_params()
        self.gradient_matrix: torch.Tensor = torch.Tensor(get_nakashima_matrix(self.physics))
        # Scaler previously fitted
        self.scaler_input: Scaler = scaler_input

        self.scaler_output: Scaler = scaler_output

        self.cluster_centers: torch.Tensor = torch.Tensor(cluster_centers)

        self.nakashima_type = cfg.nakashima_type

        # Build the encoders
        self.input_encoder_idx_map = {}
        self.encoder_list = nn.ModuleList()
        self.input_dim = 0
        shared_network_dim = 0
        for input_type in self.input_types:
            if input_type in self.input_config:
                dim_info = self.input_config[input_type]
                input_dim, layer_dim = dim_info["input_dim"], dim_info["layer_dim"]
                shared_network_dim += layer_dim
                self.input_dim += input_dim
                use_BN = False
                if input_dim != layer_dim:
                    if input_type == "bv" or input_type == "rv":
                        cur_branch_hidden_dims = self.vel_branch_hidden_dims
                        use_BN = True
                    elif input_type == "bs" or input_type == "rs":
                        cur_branch_hidden_dims = self.spin_branch_hidden_dims
                    else:
                        cur_branch_hidden_dims = self.other_branch_hidden_dims
                        use_BN = True

                    self.encoder_list.append(
                        self.build_general_branch(input_dim, layer_dim, [cur_branch_hidden_dims[-1]], use_BN=use_BN)
                    )

                    self.input_encoder_idx_map[input_type] = len(self.encoder_list) - 1
            else:
                assert "_" in input_type
                shared_network_dim += 1
                self.input_dim += 1

        print("\n*input_dim: ", self.input_dim)

        # Build the network
        self.shared_network: nn.Sequential = self.build_shared_network(shared_network_dim, use_BN=True)
        # Build the decoders
        self.vel_branch_decoder: nn.Sequential = self.build_general_branch(
            self.hidden_dims[-1], self.vel_output_dim, self.vel_branch_hidden_dims, use_BN=True
        )
        self.spin_branch_decoder: nn.Sequential = self.build_general_branch(
            self.hidden_dims[-1], self.spin_output_dim, self.spin_branch_hidden_dims, use_BN=False
        )
        # Build the confidence head
        self.vel_conf_head: nn.Sequential = self.build_general_branch(
            self.hidden_dims[-1], self.vel_output_dim, self.conf_head_hidden_dims, use_BN=True
        )
        self.spin_conf_head: nn.Sequential = self.build_general_branch(
            self.hidden_dims[-1], self.spin_output_dim, self.conf_head_hidden_dims, use_BN=False
        )

        # Custom Physics Loss
        self.criterion: CustomLoss = CustomLoss(cfg)

        self.conf_nll_criterion: NegLogLikelihoodLoss = NegLogLikelihoodLoss(cfg)

        self.model_name: str = cfg.model_name
        if self.residual_model_bool:
            self.model_name += "_" + self.nakashima_type + "_residual"

    def build_shared_network(self, current_dim, use_BN=False) -> nn.Sequential:
        """Build the Shared feature layers of the model"""
        layers = []
        # Create hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

            if (i + 1) % 2 == 0:
                layers.append(nn.Dropout(self.dropout_rate))

            layers.append(nn.ReLU())
            current_dim = hidden_dim
        return nn.Sequential(*layers)

    def build_general_branch(self, current_dim, output_dim, branch_hidden_dims, use_BN=False) -> nn.Sequential:
        """Build the general layers of the model for each input type"""
        layers = []

        # Create hidden layers for the current branch
        for i, hidden_dim in enumerate(branch_hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_BN:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if (i + 1) % 2 == 0:
                layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.SiLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))  # Output dimension of the current branch
        return nn.Sequential(*layers)

    def forward(self, model_input: torch.Tensor):
        """Forward step for Pytorch"""

        model_input_scaled = self.scaler_input.transform_torch(model_input)  # type: ignore

        if self.residual_model_bool:
            with torch.no_grad():

                nakashima_output = get_dynamic_nakashima_res(
                    self.gradient_matrix.to(device=self.device), model_input[:, :6], method=self.nakashima_type
                )

                test_min_dist = torch_calc_dist_cluster(
                    model_input_scaled[:, :6], self.cluster_centers.to(device=self.device)
                )
                att_factor = torch_dist_att_func(test_min_dist, type=self.att_type)  # in batch
                att_factor = torch.clamp(att_factor, min=0, max=1).view(model_input.shape[0], 1)
                ### done calc dist ###

        concat_branch = None
        cur_input_idx = 0
        for input_type in self.input_types:
            if input_type in self.input_config:
                dim_info = self.input_config[input_type]
                input_dim, layer_dim = dim_info["input_dim"], dim_info["layer_dim"]
            else:
                assert "_" in input_type
                input_dim = 1

            cur_output = model_input_scaled[:, cur_input_idx : cur_input_idx + input_dim]
            cur_output = torch.reshape(cur_output, (-1, input_dim))

            if input_type in self.input_encoder_idx_map:
                encoder_idx = self.input_encoder_idx_map[input_type]
                cur_branch = self.encoder_list[encoder_idx]
                cur_output = cur_branch(cur_output)

            if concat_branch is None:
                concat_branch = cur_output
            else:
                concat_branch = torch.cat((concat_branch, cur_output), 1)

            cur_input_idx += input_dim

        shared_output = self.shared_network(concat_branch)
        velocity_output = self.vel_branch_decoder(shared_output)
        spin_output = self.spin_branch_decoder(shared_output)

        vel_log_sigma = self.vel_conf_head(shared_output)
        spin_log_sigma = self.spin_conf_head(shared_output)

        if self.residual_model_bool:

            velocity_output_att = velocity_output * att_factor + nakashima_output[:, :3]
            spin_output_att = spin_output * att_factor + nakashima_output[:, 3:]

            velocity_output += nakashima_output[:, :3]
            spin_output += nakashima_output[:, 3:]

            return velocity_output, spin_output, velocity_output_att, spin_output_att, vel_log_sigma, spin_log_sigma

        else:  # No residual
            return velocity_output, spin_output, vel_log_sigma, spin_log_sigma

    def configure_optimizers(self):
        """Configure Optimizer"""
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)

    def _step(self, train_batch, batch_idx, stage):
        x, y = train_batch

        if self.residual_model_bool:
            y_hat1, y_hat2, _, _, vel_log_sigma_hat, spin_log_sigma_hat = self(x)
        else:
            y_hat1, y_hat2, vel_log_sigma_hat, spin_log_sigma_hat = self(x)

        vel_label = y[:, :3]
        spin_label = y[:, 3:]

        rmse_loss_vel, mae_loss_vel = self.criterion(output=y_hat1, target=vel_label)
        rmse_loss_spin, mae_loss_spin = self.criterion(output=y_hat2, target=spin_label, scale=0.01)

        ### NLL
        loss_vel_nll_conf = self.conf_nll_criterion(
            y_pred=y_hat1, log_sigma=vel_log_sigma_hat, target=vel_label, lambda_reg=10, scale=100
        )
        loss_spin_nll_conf = self.conf_nll_criterion(
            y_pred=y_hat2, log_sigma=spin_log_sigma_hat, target=spin_label, lambda_reg=0.1
        )

        total_loss = loss_vel_nll_conf + loss_spin_nll_conf

        values = {
            f"{stage}_loss": total_loss,
            f"{stage}_rmse_loss_vel": rmse_loss_vel,
            f"{stage}_mae_loss_vel": mae_loss_vel,
            f"{stage}_rmse_loss_spin": rmse_loss_spin,
            f"{stage}_mae_loss_spin": mae_loss_spin,
            f"{stage}_conf_nll_loss_vel": loss_vel_nll_conf,
            f"{stage}_log_sigma_vel": vel_log_sigma_hat.mean(),
            f"{stage}_conf_nll_loss_spin": loss_spin_nll_conf,
            f"{stage}_log_sigma_spin": spin_log_sigma_hat.mean(),
        }

        self.log_dict(values, on_epoch=True, prog_bar=True, logger=True)

        return total_loss  # Return the total loss for tracking

    def training_step(self, train_batch, batch_idx):
        """Override for training_step"""
        return self._step(train_batch, batch_idx, "train")

    def validation_step(self, val_batch, batch_idx):
        """Override for validation_step"""
        return self._step(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        """Override for step_step"""
        return self._step(test_batch, batch_idx, "test")

    def export_to_onnx(self):
        """Export the model to onnx configuration"""
        # Define a dummy input similar to what is used in the deploy method
        dummy_input = torch.randn((1, self.input_dim), device=self.device)

        # Define the file name for the ONNX model
        onnx_file_path = Path.cwd() / (self.model_name + ".onnx")

        output_names = ["Velocity_Output", "Spin_Output", "Log_Std", "Std"]
        if self.residual_model_bool:
            output_names.extend(["Velocity_Output_Att", "Spin_Output_Att"])

        
        torch.onnx.export(
            self,  # The model
            dummy_input,  # The dummy input
            onnx_file_path,  # The output file path
            export_params=True,  # Store the trained parameter weights inside the model
            opset_version=13,
            input_names=["Ball_Pre_State"],  # Names of the model's input variables
            output_names=output_names,  # Names of the model's output variables
        )
        print(f"Model exported to {onnx_file_path}")
        ## test if onnx is valid
        onnxruntime.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])


@hydra.main(version_base="1.1", config_path="../", config_name="proposed_model_config")  # hydra.job.chdir=True
def main(cfg: DictConfig):
    """
    Main function for training, validating, and testing the neural network model.
    """

    check_params(cfg)

    data_module = ArrayDataModule(cfg.neural_network)
    # Explicitly call setup to ensure that the dataset is preprocessed before training
    data_module.setup() 

    # ------- Initialize the neural network model -------
    nn_model = Regressor(
        cfg.neural_network, data_module.scaler_input, data_module.scaler_output, data_module.cluster_centers
    )

    trainer = L.Trainer(
        max_epochs=cfg.neural_network.epochs,
        max_steps=150000,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.neural_network.early_stopping_patience)],
    )

    trainer.fit(
        nn_model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader()
    )

    trainer.test(nn_model, dataloaders=data_module.test_dataloader())

    nn_model.export_to_onnx()


if __name__ == "__main__":
    main()  # pylint: disable=E1120