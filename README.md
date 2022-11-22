# DEAN-TS
Deep Ensemble Anomaly Detection on Time Series.

### Usage

- Use `loaddata.py` to specify a dataset (currently MNIST).
- Use `main.py` to train an ensemble according to the attributes in `hyper.yaml`.
- Use `merge.py` to combine all trained sub-models into one ensemble score.

 Notice that this code does not include the autoencoder used in the original paper.

### Notes

The current version of the original paper is given in DEAN.pdf.
