from pathlib import Path
from joblib import dump

import click
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state", default=42, type=int, show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler", default=True, type=bool, show_default=True,
)
@click.option(
    "--max-iter", default=100, type=int, show_default=True,
)
@click.option(
    "--n_init", default=10, type=int, show_default=True,
)
@click.option(
    "--n_clusters", default=7, type=int, show_default=True,
)
@click.option(
    "--use_variance_threshold", default=False, type=bool, show_default=True,
)
@click.option(
    "--use_random_fores_classifier", default=False, type=bool, show_default=True,
)
@click.option(
    "--use_sequential_feature_selector", default=False, type=bool, show_default=True,
)
@click.option(
    "--use_feature_reduction", default=False, type=bool, show_default=True,
)
@click.option(
    "--n_iter", default=100, type=int, show_default=True,
)
@click.option(
    "--threshold", default=0.8, type=float, show_default=True,
)
@click.option(
    "--n_neighbors", default=5, type=int, show_default=True,
)
@click.option(
    "--n_features_to_select", default=7, type=int, show_default=True,
)
@click.option(
    "--use_cross_val", default=True, type=bool, show_default=True,
)
@click.option(
    "--cv", default=5, type=int, show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    n_init: int,
    n_clusters: int,
    use_variance_threshold: bool,
    use_random_fores_classifier: bool,
    use_sequential_feature_selector: bool,
    use_feature_reduction: bool,
    n_iter: int,
    threshold: float,
    n_neighbors: int,
    n_features_to_select: int,
    use_cross_val: bool,
    cv: int,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path, random_state, test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(
            use_scaler,
            n_clusters,
            max_iter,
            n_init,
            random_state,
            use_variance_threshold,
            use_random_fores_classifier,
            use_sequential_feature_selector,
            use_feature_reduction,
            n_iter,
            threshold,
            n_neighbors,
            n_features_to_select,
            use_cross_val,
            cv,
        )

        pipeline.fit(features_train, target_train)

        if use_cross_val:
            results = cross_validate(pipeline, features_train, features_val, cv=cv)
            accuracy = -np.mean(results["test_score"])
        else:
            accuracy = accuracy_score(target_val, pipeline.predict(features_val))

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_init", n_init)
        mlflow.log_param("use_variance_threshold", use_variance_threshold)
        mlflow.log_param("use_random_fores_classifier", use_random_fores_classifier)
        mlflow.log_param(
            "use_sequential_feature_selector", use_sequential_feature_selector
        )
        mlflow.log_param("use_feature_reduction", use_feature_reduction)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("n_features_to_select", n_features_to_select)

        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
