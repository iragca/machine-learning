from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


class KNNPipeline:
    """A pipeline for evaluating K-Nearest Neighbors classifiers.

    This class handles the training, evaluation, and visualization of KNN models
    with support for multiple trials and grid search over different n_neighbors values.

    Attributes:
        features (pl.DataFrame): Input features for classification
        target (pl.Series): Target labels
        data_name (str): Name of the dataset being used
    """

    def __init__(self, features: pl.DataFrame, target: pl.Series, data_name: str):
        self.features = features
        self.target = target
        self.data_name = data_name

    def evaluate(
        self, 
        n_neighbors: int, 
        trials: int = 50, 
        test_size: float = 0.25, 
        _verbose: bool = True, 
        metric: str = "minkowski"
    ) -> Tuple[float, float]:
        """Evaluate KNN model with multiple trials"""
        train_scores, test_scores = [], []

        if _verbose:
            progress_bar = tqdm(
                range(trials), desc=f"KNN ({trials} trials)", unit="trial", leave=False
            )
            logger.info(f"Starting KNN ({trials} trials)")

        for _ in progress_bar if _verbose else range(trials):
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=test_size
            )
            model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric).fit(X_train, y_train)
            train_scores.append(model.score(X_train, y_train))
            test_scores.append(model.score(X_test, y_test))

        if _verbose:
            logger.info(f"Finished KNN ({trials} trials)")

        return np.mean(train_scores), np.mean(test_scores)

    def grid_search(
        self, n_neighbors: List[int], trials: int = 50, test_size: float = 0.25, plot: bool = True, metric: str = "minkowski"
    ) -> Tuple[List[float], List[float]]:
        """Perform grid search over different n_neighbors values"""
        logger.info(
            f"Starting Grid Search for {self.data_name}. "
            f"n_neighbors: {n_neighbors}, trials: {trials}"
        )

        train_means, test_means = [], []
        progress_bar = tqdm(
            n_neighbors, desc=f"Grid Search ({trials} trials)", unit="n_neighbors", leave=False
        )

        for n in progress_bar:
            train_mean, test_mean = self.evaluate(n, trials, test_size, _verbose=False, metric=metric)
            train_means.append(train_mean)
            test_means.append(test_mean)

        train_means, test_means = np.array(train_means), np.array(test_means)

        chart = None
        if plot:
            chart = self._plot_results(n_neighbors, train_means, test_means, trials)

        logger.info(
            f"Finished Grid Search for {self.data_name}\n "
            f"Train max accuracy: {train_means.max()}\n "
            f"Test max accuracy: {test_means.max()}"
        )
        return train_means, test_means, self.data_name, chart

    def _plot_results(
        self,
        n_neighbors: List[int],
        train_means: List[float],
        test_means: List[float],
        trials: int,
    ):
        """Plot the grid search results"""
        fig, ax = plt.subplots()

        sns.lineplot(ax=ax, x=n_neighbors, y=train_means, label="Train")
        sns.lineplot(ax=ax, x=n_neighbors, y=test_means, label="Test")

        ax.set_title(f"{self.data_name} KNN Performance ({trials} trials)")
        ax.set_xlabel("Number of Neighbors")
        ax.set_ylabel("Mean Accuracy")

        plt.legend()
        plt.grid(True)
        plt.show()

        return fig
