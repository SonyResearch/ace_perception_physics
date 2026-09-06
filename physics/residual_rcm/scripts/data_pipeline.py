# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.

from enum import Enum
from pathlib import Path
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split


from utils import get_params, get_nakashima_matrix


class Scaler(Enum):
    """
    Enum class for selecting different scaling strategies for data preprocessing.

    Supported scalers include:
    - MIN_MAX_SCALER: Scales features to a range between 0 and 1.
    - STANDARD_SCALER: Standardizes features by removing the mean and scaling to unit variance.
    - ROBUST_SCALER: Scales features using statistics that are robust to outliers (median and interquartile range).

    Methods:
        get_scaler(scaler_type): Returns an instance of the selected scaler with a custom inverse transform method.
        inverse_transform_torch(scaler, tensor): Performs inverse transformation of scaled
            tensors based on the scaler type.
    """

    MIN_MAX_SCALER = 1
    STANDARD_SCALER = 2
    ROBUST_SCALER = 3
    QUANTILE_TRANSFORMER_UNI = 4
    QUANTILE_TRANSFORMER_NORM = 5

    @classmethod
    def get_scaler(cls, scaler_type: "Scaler") -> "Scaler":
        """Map corresponding scikit scaler"""
        scaler_map = {
            cls.MIN_MAX_SCALER: MinMaxScaler(),
            cls.STANDARD_SCALER: StandardScaler(),
            cls.ROBUST_SCALER: RobustScaler(),
            cls.QUANTILE_TRANSFORMER_UNI: QuantileTransformer(n_quantiles=20, output_distribution="uniform"),
            cls.QUANTILE_TRANSFORMER_NORM: QuantileTransformer(n_quantiles=20, output_distribution="normal"),
        }
        scaler_instance = scaler_map[scaler_type]
        scaler_instance.transform_torch = lambda tensor: cls.transform_torch(scaler_instance, tensor)
        scaler_instance.inverse_transform_torch = lambda tensor: cls.inverse_transform_torch(scaler_instance, tensor)
        return scaler_instance

    @staticmethod
    def inverse_transform_torch(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Assign the appropriate inverse transform method based on the scaler type."""
        if isinstance(scaler, MinMaxScaler):
            return Scaler._inverse_transform_min_max(scaler, tensor)
        if isinstance(scaler, StandardScaler):
            return Scaler._inverse_transform_standard(scaler, tensor)
        if isinstance(scaler, RobustScaler):
            return Scaler._inverse_transform_robust(scaler, tensor)
        if isinstance(scaler, QuantileTransformer):
            return Scaler._inverse_transform_quantile(scaler, tensor)
        raise ValueError("Unknown scaler type")

    @staticmethod
    def transform_torch(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Assign the appropriate transform method based on the scaler type."""
        if isinstance(scaler, MinMaxScaler):
            return Scaler._transform_min_max(scaler, tensor)
        if isinstance(scaler, StandardScaler):
            return Scaler._transform_standard(scaler, tensor)
        if isinstance(scaler, RobustScaler):
            return Scaler._transform_robust(scaler, tensor)
        if isinstance(scaler, QuantileTransformer):
            return Scaler._transform_quantile(scaler, tensor)
        raise ValueError("Unknown scaler type")

    @staticmethod
    def _transform_min_max(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Apply MinMax scaling"""
        dtype = tensor.dtype
        device = tensor.device
        scale_ = torch.as_tensor(scaler.scale_, dtype=dtype, device=device)
        min_ = torch.as_tensor(scaler.min_, dtype=dtype, device=device)
        if scale_ is None or min_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'transform'.")
        return tensor * scale_ + min_

    @staticmethod
    def _transform_standard(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Standard scaling"""
        dtype = tensor.dtype
        device = tensor.device
        mean_ = torch.as_tensor(scaler.mean_, dtype=dtype, device=device)
        var_ = torch.as_tensor(scaler.var_, dtype=dtype, device=device)
        if mean_ is None or var_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'transform'.")
        return (tensor - mean_) / var_

    @staticmethod
    def _transform_robust(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Robust scaling"""
        dtype = tensor.dtype
        device = tensor.device
        center_ = torch.as_tensor(scaler.center_, dtype=dtype, device=device)
        scale_ = torch.as_tensor(scaler.scale_, dtype=dtype, device=device)
        if center_ is None or scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'transform'.")
        return (tensor - center_) / scale_

    @staticmethod
    def _transform_quantile(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Apply QuantileTransformer scaling (ONNX-compatible)"""
        dtype = tensor.dtype
        device = tensor.device
        quantiles = torch.as_tensor(scaler.quantiles_, dtype=dtype, device=device)
        references = torch.as_tensor(scaler.references_, dtype=dtype, device=device)
        if not hasattr(scaler, "quantiles_"):
            raise RuntimeError("Scaler has not been fitted yet.")

        n_features = quantiles.shape[1]
        n_quantiles = references.shape[0]
        result = torch.zeros_like(tensor)

        for feature_idx in range(n_features):
            x = tensor[:, feature_idx]
            x_quantiles = quantiles[:, feature_idx]

            comparisons = x.unsqueeze(1) >= x_quantiles.unsqueeze(0)
            indices = comparisons.sum(dim=1).long()
            indices = torch.clamp(indices, 1, n_quantiles - 1)

            x0 = x_quantiles[indices - 1]
            x1 = x_quantiles[indices]
            y0 = references[indices - 1]
            y1 = references[indices]

            weights = (x - x0) / torch.clamp(x1 - x0, min=1e-8)
            result[:, feature_idx] = y0 + weights * (y1 - y0)

        return torch.clamp(result, 0.0, 1.0)

    @staticmethod
    def _inverse_transform_min_max(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Undo MinMax scaling"""
        device = tensor.device
        scale_ = torch.tensor(scaler.scale_, device=device)
        min_ = torch.tensor(scaler.min_, device=device)
        if scale_ is None or min_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'inverse_transform'.")
        return (tensor - min_) / scale_

    @staticmethod
    def _inverse_transform_standard(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Undo Standard scaling"""
        device = tensor.device
        mean_ = torch.tensor(scaler.mean_, device=device)
        var_ = torch.tensor(scaler.var_, device=device)
        if mean_ is None or var_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'inverse_transform'.")
        return tensor * var_ + mean_

    @staticmethod
    def _inverse_transform_robust(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Undo Robust scaling"""
        device = tensor.device
        center_ = torch.tensor(scaler.center_, device=device)
        scale_ = torch.tensor(scaler.scale_, device=device)
        if center_ is None or scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'inverse_transform'.")
        return tensor * scale_ + center_

    @staticmethod
    def _inverse_transform_quantile(scaler, tensor: torch.Tensor) -> torch.Tensor:
        """Undo QuantileTransformer scaling (ONNX-compatible)"""
        dtype = tensor.dtype
        device = tensor.device
        quantiles = torch.as_tensor(scaler.quantiles_, dtype=dtype, device=device)
        references = torch.as_tensor(scaler.references_, dtype=dtype, device=device)

        if not hasattr(scaler, "quantiles_"):
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'inverse_transform'.")

        tensor_clipped = torch.clamp(tensor, 0.0, 1.0)
        n_features = quantiles.shape[1]
        n_quantiles = references.shape[0]
        result = torch.zeros_like(tensor)

        for feature_idx in range(n_features):
            x = tensor_clipped[:, feature_idx]
            y_values = quantiles[:, feature_idx]

            comparisons = x.unsqueeze(1) >= references.unsqueeze(0)
            indices = comparisons.sum(dim=1).long()
            indices = torch.clamp(indices, 1, n_quantiles - 1)

            x0 = references[indices - 1]
            x1 = references[indices]
            y0 = y_values[indices - 1]
            y1 = y_values[indices]

            weights = (x - x0) / torch.clamp(x1 - x0, min=1e-8)
            result[:, feature_idx] = y0 + weights * (y1 - y0)

        return result


class ArrayDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for handling input and output datasets, scaling,
    and splitting data into training, validation, and test sets.
    """

    ball_radius = 0.002  # [m]
    mass = 0.0027  # [kg]
    ball_inertia = 3.5964e-06  # [kg*m^2]

    def __init__(self, cfg: DictConfig, seed_splitting: int = 42):
        super().__init__()
        self.cfg = cfg
        self.dataset_path: Path = Path(cfg.dataset_path)
        self.batch_size: int = cfg.batch_size
        self.residual_model_bool: bool = cfg.residual_model_bool

        self.train_perc: float = cfg.train_perc
        self.test_perc: float = cfg.test_perc
        self.val_perc: float = cfg.val_perc
        self.seed_splitting: int = seed_splitting

        ## input ###
        self.input_types = cfg.input_types
        self.input_config = cfg.input_config

        # Corresponding scaler function present both on enum class and config file
        scaler_type = cfg.scaler_type
        self.scaler_input: Scaler = Scaler.get_scaler(Scaler(scaler_type))
        self.scaler_output: Scaler = Scaler.get_scaler(Scaler(scaler_type))

        # Only for residual model
        self.physics = get_params()
        self.nakashima_matrix = get_nakashima_matrix(self.physics)

        # Initialize placeholders to be filled later
        self.input_dataset: np.ndarray
        self.output_dataset: np.ndarray
        self.info_dataset: np.ndarray
        self.train_size: int
        self.val_size: int
        self.test_size: int
        self.train_dataset: TensorDataset
        self.val_dataset: TensorDataset
        self.test_dataset: TensorDataset

        self.cluster_centers = np.array([])

    def setup(self, stage=None):
        """Pre-processing of the dataset"""
        super().setup(stage)
        self.input_dataset, self.output_dataset, self.info_dataset = self._get_dataset_from_csv(plot_histograms=self.cfg.plot_histograms)

        data_tensor = torch.tensor(self.input_dataset, dtype=torch.float32)
        labels_tensor = torch.tensor(self.output_dataset, dtype=torch.float32)

        full_dataset = TensorDataset(data_tensor, labels_tensor)

        total_samples = len(self.input_dataset)
        self.train_size = int(self.train_perc * total_samples)
        
        self.val_size = int(self.val_perc * total_samples) 
        self.test_size = total_samples - self.train_size - self.val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [self.train_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(self.seed_splitting),
        )

        print(f"train: {len(self.train_dataset)}, val: {len(self.val_dataset)}, test: {len(self.test_dataset)}")

        # Fit Scalers & Clusters ONLY on Training Data
        train_indices = self.train_dataset.indices
        train_input_data = self.input_dataset[train_indices]
        train_output_data = self.output_dataset[train_indices]

        self.scaler_input.fit(train_input_data)
        self.scaler_output.fit(train_output_data)

        self.cluster_centers = self.fit_cluster(train_input_data)

    def fit_cluster(self, train_data: np.ndarray):
        """Fit KMeans clustering on the training data to find cluster centers."""
        X = self.scaler_input.transform(train_data)[:, :6]
        kmeans = KMeans(n_clusters=1000, random_state=0, n_init="auto").fit(X)

        return kmeans.cluster_centers_

    def _get_dataset_from_csv(self, start_date="2025-11-01", end_date="2027-01-01", plot_histograms=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the dataset from a CSV file.
        Error handling for FileNotFound and for Missing coloumn.
        """
        print("** loading csv dataset: ", self.dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"File not found: {self.dataset_path}")

        # Load CSV(s) into a DataFrame
        if self.dataset_path.is_dir():
            # Get all CSV files in the directory
            csv_list = [str(file) for file in Path(self.dataset_path).rglob("*.csv")]
            if not csv_list:
                raise FileNotFoundError(f"No CSV files found in directory: {self.dataset_path}")

            # Concatenate all the CSV files into one DataFrame
            data = pd.concat([pd.read_csv(file) for file in csv_list], ignore_index=True)
        else:
            data = pd.read_csv(self.dataset_path)

        # Filter out data by dates
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
        data = data[(data["date"] >= start_date) & (data["date"] < end_date)]
        print(f"Only considering data between: [{start_date}, {end_date})")

        ####### input cols ######
        before_ball_vel = [
            "Before_Vel_X",
            "Before_Vel_Y",
            "Before_Vel_Z",
        ]
        before_ball_spin = [
            "Before_Spin_X",
            "Before_Spin_Y",
            "Before_Spin_Z",
        ]

        dist_cols = ["Before_Dist_X", "Before_Dist_Y", "Before_Dist_Z"]

        before_ball_pos = ["Global_Ball_Pos_X", "Global_Ball_Pos_Y", "Global_Ball_Pos_Z"]
        ### potential input ####
        racket_spin_cols = [
            "Racket_Spin_X",
            "Racket_Spin_Y",
            "Racket_Spin_Z",
        ]
        racket_spin_cols_global = [
            "Global_Racket_Spin_X",
            "Global_Racket_Spin_Y",
            "Global_Racket_Spin_Z",
        ]
        dist_cols_global = [
            "Global_dx_pre",
            "Global_dy_pre",
            "Global_dz_pre",
        ]
        before_ball_vel_global = [
            "Global_Ball_Vel_Pre_X",
            "Global_Ball_Vel_Pre_Y",
            "Global_Ball_Vel_Pre_Z",
        ]
        before_ball_spin_global = [
            "Global_Ball_Spin_Pre_X",
            "Global_Ball_Spin_Pre_Y",
            "Global_Ball_Spin_Pre_Z",
        ]
        racket_velocity_cols_global = [
            "Global_Racket_Velocity_X",
            "Global_Racket_Velocity_Y",
            "Global_Racket_Velocity_Z",
        ]
        racket_velocity_cols = [
            "Racket_Velocity_X",
            "Racket_Velocity_Y",
            "Racket_Velocity_Z",
        ]
        orientation_cols = ["Orientation_X", "Orientation_Y", "Orientation_Z", "Orientation_W"]

        required_columns_output = [
            "After_Vel_X",
            "After_Vel_Y",
            "After_Vel_Z",
            "After_Spin_X",
            "After_Spin_Y",
            "After_Spin_Z",
        ]

        required_columns_info = [
            "Global_Ball_Pos_X",
            "Global_Ball_Pos_Y",
            "Global_Ball_Pos_Z",
            "Orientation_X",
            "Orientation_Y",
            "Orientation_Z",
            "Orientation_W",
            "Global_Ball_Vel_Pre_X",
            "Global_Ball_Vel_Pre_Y",
            "Global_Ball_Vel_Pre_Z",
            "Global_Ball_Spin_Pre_X",
            "Global_Ball_Spin_Pre_Y",
            "Global_Ball_Spin_Pre_Z",
            "Racket_Velocity_X",
            "Racket_Velocity_Y",
            "Racket_Velocity_Z",
            "Global_Ball_Vel_Post_X",
            "Global_Ball_Vel_Post_Y",
            "Global_Ball_Vel_Post_Z",
            "Global_Ball_Spin_Post_X",
            "Global_Ball_Spin_Post_Y",
            "Global_Ball_Spin_Post_Z",
        ]

        xyz_dict = {"x": 0, "y": 1, "z": 2, "w": 3}
        
        input_data_list = []
        
        for input_type in self.input_types:
            if input_type in self.input_config:
                dim_info = self.input_config[input_type]
                input_dim, layer_dim = dim_info["input_dim"], dim_info["layer_dim"]
            else:
                assert "_" in input_type

            if input_type.startswith("bv"):
                cur_input = data[before_ball_vel].to_numpy()
            elif input_type.startswith("bs"):
                cur_input = data[before_ball_spin].to_numpy()
            elif input_type.startswith("dist"):
                cur_input = data[dist_cols[1:]].to_numpy()
            elif input_type.startswith("bp"):
                cur_input = data[before_ball_pos].to_numpy()
            elif input_type.startswith("rv"):
                cur_input = data[racket_velocity_cols].to_numpy()
            elif input_type.startswith("rs"):
                cur_input = data[racket_spin_cols].to_numpy()
            elif input_type.startswith("grv"):
                cur_input = data[racket_velocity_cols_global].to_numpy()
            elif input_type.startswith("grs"):
                cur_input = data[racket_spin_cols_global].to_numpy()
            elif input_type.startswith("gbv"):
                cur_input = data[before_ball_vel_global].to_numpy()
            elif input_type.startswith("gbs"):
                cur_input = data[before_ball_spin_global].to_numpy()
            elif input_type.startswith("gdist"):
                cur_input = data[dist_cols_global].to_numpy()
            elif input_type.startswith("gwrx"):
                cur_input = np.cross(data[racket_spin_cols_global].to_numpy(), data[dist_cols_global].to_numpy())
            elif input_type.startswith("rq"):
                cur_input = data[orientation_cols].to_numpy()
            else:
                raise NotImplementedError(f"unknown input type: {input_type}")

            if "_" in input_type:
                idx = xyz_dict[input_type.split("_")[1]]
                if input_type.startswith("dist"):
                    idx -= 1
                cur_input = np.reshape(cur_input[:, idx], (-1, 1))
            else:
                assert input_dim == cur_input.shape[1]

            input_data_list.append(cur_input)

        # Concatenate all at once
        input_data = np.concatenate(input_data_list, axis=1)

        expected_output = data[required_columns_output].to_numpy()
        data_info = data[required_columns_info].to_numpy()

        ######### DONE #########
        print("\n data pipeline - input_data: ", input_data.shape)

        if plot_histograms:
            matplotlib.use("TkAgg")

            # Plot histograms of the input and output data to check the distribution.
            # Input data typically has dimension 8: bv:3, bs:3, dist:2
            # Output data typically has dimension 6: After_Vel: 3, After_Spin: 3
            input_shape = np.shape(input_data)[1]
            output_shape = np.shape(expected_output)[1]
            num_cols = max(input_shape, output_shape)
            fig, axs = plt.subplots(2, num_cols, sharex=False, sharey=True, constrained_layout=True)
            NBINS = 20

            # Inputs
            for idx in range(input_shape):
                ax = axs[0][idx]
                ax.hist(input_data[:, idx], bins=NBINS)
                ax.set_title(f"Input {idx}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency (Counts)")
                ax.minorticks_on()
                ax.grid("on", "both")

            # Outputs
            for idx in range(output_shape):
                ax = axs[1][idx]
                ax.hist(expected_output[:, idx], bins=NBINS)
                ax.set_title(f"Output {idx}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency (Counts)")
                ax.minorticks_on()
                ax.grid("on", "both")

            plt.show()

        return input_data, expected_output, data_info

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def transform_with_nakashima(self) -> np.ndarray:
        """
        Transforms the input dataset using the Nakashima transformation.
        """
        input_ball_state_dataset = self.input_dataset
        output_ball_state_dataset = self.output_dataset
        residual_output_dataset = []
        cnt = 0
        for ball_state, ball_state_output in zip(input_ball_state_dataset, output_ball_state_dataset):
            nakashima_output_ball_state = self.nakashima_transformation(ball_state)

            residual_output_dataset.append(ball_state_output - nakashima_output_ball_state)
        residual_output_dataset = np.vstack(residual_output_dataset)
        return residual_output_dataset

    def nakashima_transformation(self, input_array: np.ndarray) -> np.ndarray:
        """
        Applies a transformation to the input dataset state using the Nakashima model.
        """
        output_ball_state = np.dot(self.nakashima_matrix, input_array)
        return output_ball_state
