# DEAN-TS

This repository provides an implementation of the **Deep Ensemble Anomaly Detection for Time Series (DEAN-TS)** method that I developed as part of my master thesis.
It is based on [Deep Ensemble Anomaly Detection (DEAN)](https://github.com/KDD-OpenSource/DEAN) and, as the name suggests,
applies its concepts to time series.

## Example usage

### Preparation
- Install requirements according to `requirements.txt`.

### Configuration
- Specify parameterization of DEAN-TS in `config/configuration.yaml`.

### Running
- Specify the details of the execution and apply DEAN-TS by customizing `src/main.py` and then execute it.

## TimeEval / GutenTAG

A substantial part of the evaluation in my master thesis is performed with the benchmark tool [TimeEval](https://github.com/HPI-Information-Systems/TimeEval),
among others on synthetic datasets generated with the generation tool [GutenTAG](https://github.com/HPI-Information-Systems/gutentag) by the same developers.
In particular, DEAN-TS expects input formatting as defined there.
Some of the synthetic datasets used are included in `datasets/ecg` for easy experimentation with DEAN-TS.

Information on the integration of DEAN-TS into TimeEval can be found [here](https://github.com/Ti-Kat/dean-ts/blob/main/timeeval/README.md).

### Useful links: 
- [TimeEval](https://github.com/HPI-Information-Systems/TimeEval)
- [Time Eval Doc](https://timeeval.readthedocs.io/en/latest/)
- [TimeEval Algorithms](https://github.com/HPI-Information-Systems/TimeEval-algorithms)
- [GutenTAG](https://github.com/HPI-Information-Systems/gutentag)
- [GutenTAG Doc](https://github.com/HPI-Information-Systems/gutentag/blob/main/doc/index.md)
