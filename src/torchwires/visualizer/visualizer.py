import math
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from ..repo.repo import Repo


class Visualizer:
    SPLIT_COL = 'split'
    BATCH_COL = 'batch'
    EPOCH_COL = 'epoch'
    DEVICE_COL = 'device'

    @staticmethod
    def display_df(
            repo: Repo
    ):
        df = repo.observer.get_pandas()

        df.drop(columns=[Visualizer.BATCH_COL, Visualizer.DEVICE_COL], inplace=True)

        return df.groupby(
            [Visualizer.EPOCH_COL, Visualizer.SPLIT_COL]
        ).mean()

    @staticmethod
    def visualize_repo(
            repo: Repo,
            style_no: int = 12,
            n_cols: int = 3,
            cell_size: Tuple[float, float] = (5.0, 2.5),
            wspace: float = 1 / 3,
            hspace: float = 1 / 3,
    ):
        plt.style.use(
            plt.style.available[style_no]
        )

        df = repo.observer.get_pandas()

        all_splits = ['train', 'val']

        all_features = list(df.columns)

        features_2_draw = all_features.copy()
        features_2_draw.remove(Visualizer.EPOCH_COL)
        features_2_draw.remove(Visualizer.BATCH_COL)
        features_2_draw.remove(Visualizer.SPLIT_COL)
        features_2_draw.remove(Visualizer.DEVICE_COL)

        n_rows = math.ceil(len(features_2_draw) / n_cols)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, )

        axs = axs.reshape(n_rows, -1)

        fig.suptitle(
            f"Repo: {repo.repo_name}",
        )

        fig.set_size_inches(
            w=cell_size[0] * n_cols,
            h=cell_size[1] * n_rows,
        )

        fig.subplots_adjust(
            wspace=wspace,
            hspace=hspace,
        )

        for idx, feature in enumerate(features_2_draw):
            row_idx, col_idx = idx // n_cols, idx % n_cols

            for split in all_splits:
                split_df = df[df[Visualizer.SPLIT_COL] == split]

                grouped_df = split_df.groupby(Visualizer.EPOCH_COL)
                x_data = grouped_df[Visualizer.EPOCH_COL].mean()
                y_data = grouped_df[feature].mean()

                if np.all(np.isnan(y_data)):
                    continue

                axs[row_idx, col_idx].plot(
                    x_data, y_data,
                    label=f"{split}",
                )

            axs[row_idx, col_idx].set_xlabel(Visualizer.EPOCH_COL)
            axs[row_idx, col_idx].set_ylabel(feature)
            axs[row_idx, col_idx].legend()

        for idx in range(len(features_2_draw), n_cols * n_rows):
            row_idx, col_idx = idx // n_cols, idx % n_cols
            fig.delaxes(axs[row_idx, col_idx])

        plt.show()

    @staticmethod
    def visualize_comparison(
            repos: List[Repo],
            style_no: int = 12,
            cell_size: Tuple[float, float] = (7.0, 2.5),
            wspace: float = 1 / 5,
            hspace: float = 1 / 3,
    ):
        plt.style.use(
            plt.style.available[style_no]
        )

        all_splits = ['train', 'val']
        n_cols = len(all_splits)

        all_features = []

        for repo in repos:
            all_features.extend(repo.observer.get_pandas().columns.values.tolist())

        all_features = list(set(all_features))
        all_features.remove(Visualizer.EPOCH_COL)
        all_features.remove(Visualizer.BATCH_COL)
        all_features.remove(Visualizer.SPLIT_COL)
        all_features.remove(Visualizer.DEVICE_COL)

        n_rows = len(all_features)

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
        )

        fig.set_size_inches(
            w=cell_size[0] * n_cols,
            h=cell_size[1] * n_rows,
        )

        fig.subplots_adjust(
            wspace=wspace,
            hspace=hspace,
        )

        fig.suptitle(
            "Repos: " + ", ".join([r.repo_name for r in repos])
        )

        for row_idx, feature in enumerate(all_features):
            for col_idx, col in enumerate(all_splits):
                flag_del_cell = True

                for repo in repos:
                    df = repo.observer.get_pandas()
                    split_df = df[df[Visualizer.SPLIT_COL] == col]

                    if feature in df.columns:
                        grouped_df = split_df.groupby(Visualizer.EPOCH_COL)
                        x_data = grouped_df[Visualizer.EPOCH_COL].mean()
                        y_data = grouped_df[feature].mean()

                        if np.all(np.isnan(y_data)):
                            continue

                        flag_del_cell = False
                        axs[row_idx, col_idx].plot(
                            x_data, y_data,
                            label=f"{repo.repo_name}",
                        )

                if flag_del_cell:
                    fig.delaxes(axs[row_idx, col_idx])
                else:
                    axs[row_idx, col_idx].set_xlabel(Visualizer.EPOCH_COL)
                    axs[row_idx, col_idx].set_ylabel(f"{col} {feature}")
                    axs[row_idx, col_idx].legend()

        plt.show()
