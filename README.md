# DEAN-TS
Deep Ensemble Anomaly Detection on Time Series.

Usable either as standalone method or in combination with [TimeEval](https://github.com/HPI-Information-Systems/TimeEval) (WIP).

## TimeEval and GutenTAG
- [TimeEval](https://github.com/HPI-Information-Systems/TimeEval)
- [Time Eval Doc](https://timeeval.readthedocs.io/en/latest/)
- [TimeEval Algorithms](https://github.com/HPI-Information-Systems/TimeEval-algorithms)
- [TimeEval-GUI](https://github.com/HPI-Information-Systems/TimeEval-GUI)
- [GutenTAG](https://github.com/HPI-Information-Systems/gutentag)
- [GutenTAG Doc](https://github.com/HPI-Information-Systems/gutentag/blob/main/doc/index.md)
- [Evaluation Paper](https://hpi-information-systems.github.io/timeeval-evaluation-paper/)

## Usage

### Configuration
- Use `config/configuration.yaml` to specify meta information
- Use ``

### Running
- Use `main.py` to train an ensemble according to the configuration

## Building Docker image for TimeEval
- Run `docker build -t dean-ts:latest -f timeeval/Dockerfile .` in the main directory

## DEAN-TS Architecture

### DeanTsController
- storage_manager
- ensemble
- load_models()
- train()
- predict()

### DeanTsEnsemble
- submodels: dict [(level, index): model]
- data_train
- data_test
- config
- scores
- train_models()
- score()
- aggregate_results()

### DeanTsSubmodel
- model
- history
- lag_indices
- input_model_ids
- scores_window
- scores
- train()
- predict()
- preprocess_data()
- reverse_window()

### DeanTsStorageManager
- result_path
- storage_format (timeeval or extensive)
- store_model()
- load_model()

### DeanTsInterpreter
- WIP
