import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dean_controller import DeanTsController


class DeanTsInterpreter:
    def __init__(self, controller: DeanTsController):
        self.controller: DeanTsController = controller
        self.config: dict = controller.config
        self.importance_df: pd.DataFrame | None = None

    def plot_importance_scores(self, start_index: int, end_index: int):
        """ Plots the importance score per feature combinations for a given range of time steps as bar plot.
        Depends on calling `compute_importance` beforehand.
        """
        # Compute average over specified range
        importance_scores_avg = self.importance_df['importance_scores'].apply(
            lambda x: np.mean(x[start_index:end_index]))

        # Remove "(" and ")", trailing "," and "'"
        tsd_component_combinations = [
            str(feature).replace('(', '').replace(')', '').rstrip(',').replace("'", '').replace(' ', '')
            for feature in self.importance_df['features']]

        positive_color = 'C2'  # Green
        negative_color = 'C3'  # Red
        colors = [positive_color if val >= 0 else negative_color for val in importance_scores_avg]

        # Plot the bar plot
        fig, ax = plt.subplots(figsize=(18, 6))
        bars = ax.bar(tsd_component_combinations, importance_scores_avg, color=colors)

        # Set the y-axis limit to include negative values below the x-axis
        ax.set_ylim(min(importance_scores_avg) - 0.2, max(importance_scores_avg) + 0.2)

        # Add a horizontal line at 0
        ax.axhline(0, color='black', linestyle='--')

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('TSD Component Combination', fontsize=16)
        plt.ylabel('Importance', fontsize=16)
        plt.title(f'Importance Scores for Time Range {start_index}-{end_index}', fontsize=16)

        # Add value labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=14)

        plt.tight_layout()
        plt.show()

    def compute_importance(self):
        """ Computes importance scores for each feature combination for each datapoint and stores it in a pandas df.
        Assumes that TSD was used.
        """
        df_data = []
        for i in range(self.controller.config['ensemble_size']):
            model = self.controller.ensemble.submodels[i]
            mapping = {0: 'T', 1: 'S', 2: 'R'}
            model_features = tuple(mapping[value] for value in np.sort(model.features))
            model_scores = self.controller.ensemble.submodel_scores[i]
            row_data = {'features': model_features,
                        'predictions': model_scores}
            df_data.append(row_data)
        df = pd.DataFrame(df_data)

        # Average predictions across all submodels
        averaged_predictions = np.vstack(df['predictions'].values).mean(axis=0)

        # Average predictions by all submodels with a given feature combination
        predictions_per_features = df.groupby('features')['predictions'].apply(
            lambda x: np.vstack(x.values).mean(axis=0))

        # Importance defined as deviation from overall consent by a given feature combination
        importance_scores_per_feature_comb = [predictions_per_features[x] - averaged_predictions for x in
                                              predictions_per_features.index]

        self.importance_df = pd.DataFrame({
            'features': predictions_per_features.index,
            'importance_scores': [importance_scores for importance_scores in importance_scores_per_feature_comb]
        })
