import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from scikitplot.metrics import plot_roc, plot_precision_recall

from src.dean_controller import DeanTsController


def plot_result_interactive(controller: DeanTsController, test_data, dataset_name, score, runtime):
    title = f"""
    Dataset: {str(dataset_name)}
    <br>Score: {str(round(score, 4))} (AUC-ROC)
    <br>Runtime: {str(round(runtime, 2))} (s)
    """

    # Prepare data
    data = np.c_[test_data,
                 controller.ensemble.ensemble_score,
                 controller.ensemble.submodel_scores.T]

    # Determine number of submodels based on the data shape
    num_submodels = data.shape[1] - 3

    fig = px.line(data[:, 1:], labels=[''], title=title)

    # Update labels
    legend_labels = {
        '0': 'Time Series',
        '1': 'Ground Truth',
        '2': 'Ensemble'
    }

    for i in range(num_submodels):
        legend_labels[str(i + 3)] = f'Submodel {i}'

    for trace, label in zip(fig.data, legend_labels.values()):
        trace.name = label

    fig.update_layout(legend_title_text='', xaxis_title='Time step', yaxis_title='Value')

    fig.show()


def plot_roc_curve(y_true, y_scores, title=''):
    plot_roc(y_true=y_true,
             y_probas=np.c_[1 - y_scores, y_scores],
             plot_micro=False,
             plot_macro=False,
             classes_to_plot=[1],
             figsize=(8, 8))

    if title != '':
        plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true, y_scores, title=''):
    plot_precision_recall(y_true=y_true,
                          y_probas=np.c_[1 - y_scores, y_scores],
                          plot_micro=False,
                          classes_to_plot=[1],
                          figsize=(8, 8))

    if title != '':
        plt.title(title)

    plt.tight_layout()
    plt.show()
