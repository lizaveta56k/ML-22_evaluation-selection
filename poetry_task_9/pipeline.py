from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def create_pipeline(
    use_scaler: bool,
    n_clusters: int,
    max_iter: int,
    n_init: int,
    random_state: int,
    use_variance_threshold: bool,
    use_random_fores_classifier: bool,
    use_sequential_feature_selector: bool,
    use_feature_reduction: bool,
    n_iter: int,
    threshold: float,
    n_neighbors: int,
    n_features_to_select: int,
    use_mlp_classifier: bool,
    use_decision_tree_classifier: bool,
) -> Pipeline:
    pipeline_steps = []
    selector = SimpleImputer()

    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_mlp_classifier:
        classifier = MLPClassifier(alpha=1, max_iter=max_iter)
    elif use_decision_tree_classifier:
        classifier = DecisionTreeClassifier(random_state=random_state)
    else:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    if use_random_fores_classifier:
        selector = SelectFromModel(RandomForestClassifier(random_state=random_state))
    elif use_variance_threshold:
        selector = VarianceThreshold(threshold=threshold)
    elif use_sequential_feature_selector:
        selector = SequentialFeatureSelector(
            KNeighborsClassifier(n_neighbors=n_neighbors),
            n_features_to_select=n_features_to_select,
        )

    if (
        use_variance_threshold
        or use_random_fores_classifier
        or use_sequential_feature_selector
    ):
        pipeline_steps.append(("selector", selector))
    if use_feature_reduction:
        pipeline_steps.append(
            (
                "feature_reduction",
                TruncatedSVD(
                    n_components=n_features_to_select,
                    n_iter=n_iter,
                    random_state=random_state,
                ),
            )
        )
    pipeline_steps.append(
        (
            "classifier",
            classifier,
        )
    )

    return Pipeline(steps=pipeline_steps)
