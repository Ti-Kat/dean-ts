import argparse
from dataclasses import dataclass, asdict, field
import json
import numpy as np
import pandas as pd
import sys
from typing import List

from src.dean_controller import DeanTsController


@dataclass
class CustomParameters:
    linear_layer_shape: List[int] = field(default_factory=lambda: [100, 10])
    activation: str = "relu"
    split: float = 0.8
    anomaly_window_size: int = 20
    batch_size: int = 64
    test_batch_size: int = 256
    epochs: int = 1
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    n_neighbors: int = 12
    n_estimators: int = 3
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.iloc[:, 1:-1].values

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)

def train(args: AlgorithmArgs):
    data = args.ts
    parameters = asdict(args.customParameters)
    controller = DeanTsController(**parameters)
    controller.train()
    controller.save(args.modelOutput)

def execute(args: AlgorithmArgs):
    data = args.ts
    controller = DeanTsController.load(args.modelInput)
    anomaly_scores = controller.predict(data)
    anomaly_scores.tofile(args.dataOutput, sep="\n")



if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
