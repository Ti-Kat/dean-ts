import argparse
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
import sys

from src.dean_controller import DeanTsController


@dataclass
class CustomParameters:
    seed: int = 21  # Randomness seed for reproducible results

    # Ensemble specifications
    ensemble_size: int = 50  # Number of lag models
    reverse_window: bool = True

    combination_method: str = 'thresh'  # Score combination method, options: 'thresh', 'average', 'max', 'median'
    thresh_value: int = 0  # Lower bound for Z-Score to be considered

    subsampling: str = 'none'  # Subsampling method, potential values: 'none', 'vs', 'structured'
    vs_lower: int = 500  # Min sample size for variable subsampling
    vs_upper: int = 5000  # Max sample size for variable subsampling

    feature_bagging: bool = True  # Whether to use feature bagging for MTS
    fb_lower: int = 1  # Min feature count for feature bagging
    fb_upper: int = 5  # Max feature count for feature bagging

    # Submodel specifications
    bias: bool = False  # Whether to allow learnable shift in hidden layers
    depth: int = 3  # Number of layers for each base detector network
    bag: int = 128  # Dimensionality for each base model (bag - 1 lag features will be chosen)
    look_back: int = 256  # How many previous time steps are taken into consideration for feature selection

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
    controller = DeanTsController.load(args.modelInput)
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
