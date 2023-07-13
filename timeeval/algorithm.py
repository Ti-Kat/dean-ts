import argparse
import json
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from src.dean_controller import DeanTsController


@dataclass
class CustomParameters:
    random_state: int = 21  # Randomness seed for reproducible results

    # Ensemble specifications
    ensemble_size: int = 40  # Number of submodels to train
    reverse_window: bool = True  # Whether to assign the window scores of the submodels to the lag feature positions as well (default)

    tsd: bool = False  # Whether to apply STL time series decomposition as preprocessing step (only for UTS)
    period: int = 20  # Periodicity for STL (currently not automatically determined)

    combination_method: str = 'thresh'  # Score combination method, potential values: 'thresh', 'average', 'max', 'median', 'dean'
    thresh_value: int = 0  # Lower bound for Z-Score to be considered in thresh method

    feature_bagging: bool = True  # Whether to use feature bagging for MTS
    fb_range: (int, int) = (1, 3)  # Min and max number of features for feature bagging

    subsampling: str = 'none'  # Subsampling method, potential values: 'none', 'random', 'structured'
    rs_range: (int, int) = (1024, 8192)  # Min and max number of samples for random subsampling
    ss_r: [float] = field(default_factory=lambda: [0.2, 0.25, 0.5])  # Percentage splits of total data for structured subsampling (Has to fit ensemble_size and training data length)
    ss_m: [int] = field(default_factory=lambda: [2, 5, 5])  # Number of times each split percentage ss_r should process the entire time series (Has to fit ensemble_size and training data length)

    # Submodel specifications
    bias: bool = False  # Whether to allow learnable shift in hidden layers
    depth: int = 3  # Number of layers for each base detector network
    activation: str = 'relu'  # Which activation function to use in hidden layers
    lag_indices_count: int = 63  # Lag feature count per dimension for each base model
    look_back_range: (int, int) = (64, 512)  # Range of how many previous time steps are randomly taken into consideration for lag feature selection

    lr: float = 0.01  # Learning rate
    batch: int = 32  # Batch size


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return pd.read_csv(self.dataInput).to_numpy()

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def train(args: AlgorithmArgs):
    train_data = args.ts
    config = asdict(args.customParameters)
    controller = DeanTsController(config)
    controller.train(train_data)
    controller.save(args.modelOutput)


def execute(args: AlgorithmArgs):
    test_data = args.ts

    # Semi-supervised
    try:
        controller = DeanTsController.load(args.modelInput)
    # Unsupervised
    except FileNotFoundError:
        config = asdict(args.customParameters)
        controller = DeanTsController(config)
        controller.train(test_data)

    anomaly_scores = controller.predict(test_data)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
