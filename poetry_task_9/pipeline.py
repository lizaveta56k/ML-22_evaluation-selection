from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    use_scaler: bool, n_clusters: int, max_iter: int, n_init: int, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KMeans(
                random_state=random_state, n_clusters=n_clusters, max_iter=max_iter, n_init=n_init
            ),
        )
    )
    
    return Pipeline(steps=pipeline_steps)