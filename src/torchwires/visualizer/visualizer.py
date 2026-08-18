import math
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants.constants import LOADER_TRAIN_TYPE, LOADER_VAL_TYPE
from ..repo.repo import Repo
from ..state.base_state import BaseState


class Visualizer:

    @staticmethod
    def display_df(
            repo: Repo
    ):
        df = repo.history.return_as_df()

        split_col = BaseState.KEY_LOADER_TYPE
        batch_col = BaseState.KEY_BATCH_NO
        epoch_col = BaseState.KEY_EPOCH_NO

        df.drop(columns=[batch_col], inplace=True)

        return df.groupby(
            [epoch_col, split_col]
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

        df = repo.history.return_as_df()

        split_col = BaseState.KEY_LOADER_TYPE
        batch_col = BaseState.KEY_BATCH_NO
        epoch_col = BaseState.KEY_EPOCH_NO

        all_splits = [LOADER_TRAIN_TYPE, LOADER_VAL_TYPE]

        all_features = list(df.columns)

        features_2_draw = all_features.copy()
        features_2_draw.remove(split_col)
        features_2_draw.remove(batch_col)
        features_2_draw.remove(epoch_col)

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
                split_df = df[df[split_col] == split]

                grouped_df = split_df.groupby(epoch_col)
                x_data = grouped_df[epoch_col].mean()
                y_data = grouped_df[feature].mean()

                if np.all(np.isnan(y_data)):
                    continue

                axs[row_idx, col_idx].plot(
                    x_data, y_data,
                    label=f"{split}",
                )

            axs[row_idx, col_idx].set_xlabel(epoch_col)
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

        split_col = BaseState.KEY_LOADER_TYPE
        batch_col = BaseState.KEY_BATCH_NO
        epoch_col = BaseState.KEY_EPOCH_NO

        all_splits = [LOADER_TRAIN_TYPE, LOADER_VAL_TYPE]
        n_cols = len(all_splits)

        all_features = []

        for repo in repos:
            all_features.extend(repo.history.get_tracked_features())

        all_features = list(set(all_features))
        all_features.remove(split_col)
        all_features.remove(batch_col)
        all_features.remove(epoch_col)

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
                    df = repo.history.return_as_df()
                    split_df = df[df[split_col] == col]

                    if feature in df.columns:
                        grouped_df = split_df.groupby(epoch_col)
                        x_data = grouped_df[epoch_col].mean()
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
                    axs[row_idx, col_idx].set_xlabel(epoch_col)
                    axs[row_idx, col_idx].set_ylabel(f"{col} {feature}")
                    axs[row_idx, col_idx].legend()

        plt.show()
