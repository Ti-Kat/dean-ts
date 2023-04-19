import argparse
import json
import sys

from dataclasses import dataclass


@dataclass
class CustomParameters:
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def main():
    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")


if __name__ == "__main__":
    main()
