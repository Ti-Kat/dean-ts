import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

BASE_PATH = '../../datasets/tods/uts/'


def plot_csv_images():
    plt.clf()

    for filename in os.listdir(BASE_PATH):
        if not filename[-3:] == 'csv':
            continue

        path = os.path.join(BASE_PATH, filename)
        # Change file ending from .csv to .png
        img_path = path[:-3] + 'png'

        df = pd.read_csv(path)
        df.insert(loc=0, column='time', value=np.arange(len(df.index)))

        x = df['time'].to_numpy()
        y = df['value'].to_numpy()
        z = df['anomaly'].to_numpy()
        marked_positions = np.where(z == 1)[0]

        print(img_path)

        plt.plot(x, y, '-s', markevery=marked_positions, label='Anomalies')
        plt.legend()
        plt.savefig(img_path)
        plt.clf()


def insert_time_columns():
    for filename in os.listdir(BASE_PATH):
        if not filename[-3:] == 'csv':
            continue

        path = os.path.join(BASE_PATH, filename)

        df = pd.read_csv(path)

        if 'time' not in df.columns:
            df.insert(loc=0, column='time', value=np.arange(len(df.index)))

        df.to_csv(path, index=False)


insert_time_columns()
