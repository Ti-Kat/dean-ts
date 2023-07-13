# Evaluation using TimeEval

To evaluate DEAN-TS using [TimeEval](https://github.com/HPI-Information-Systems/TimeEval), an interface for integration using Docker is provided.
This integration is largely analogous to the algorithms already provided by the TimeEval developers
(see [TimeEval Algorithms](https://github.com/HPI-Information-Systems/TimeEval-algorithms) for step-by-step instructions).
The default configuration can be adjusted in the class `CustomParameters` within `algorithm.py`.

DEAN-TS first requires the base image `python3-base` which is available [here](https://github.com/HPI-Information-Systems/TimeEval-algorithms).

Then, the DEAN-TS image can be created by executing the following command in the main directory:

`docker build -t dean-ts:latest -f timeeval/Dockerfile .`

The DEAN-TS image should now be ready for use in an unsupervised or semi-supervised evaluation.

For details on the evaluation process, see the [TimeEval documentation](https://timeeval.readthedocs.io/en/latest/).
