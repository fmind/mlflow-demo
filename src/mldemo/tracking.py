"""Demo of MLflow Tracking: https://mlflow.org/docs/latest/tracking.html"""

# %% IMPORTS

import os

import mlflow
from sklearn import datasets, ensemble, metrics, model_selection

# %% CONFIGS

# - MLflow
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "tracking"

# - Model
MAX_DEPTH = 3
N_ESTIMATORS = 5

# - Others
CV = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# %% MLFLOW

# configure mlflow server/experiment
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog()  # enable auto logging

# %% DATASETS

X, y = datasets.load_wine(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# %% TRAINING

# create one mlflow run for the training
with mlflow.start_run(description="Training") as run:
    print(f"[START] Run ID: {run.info.run_id}")

    # - model
    model = ensemble.RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # - tags
    mlflow.set_tag("event", "MLOps Meetup")

    # - params
    mlflow.log_param("test_size", TEST_SIZE)

    # - metrics
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # - artifacts
    mlflow.log_artifact(os.path.abspath(__file__))

    print(f"[STOP] Run ID: {run.info.run_id}")


# %% TUNING

# create one mlflow run for each model
with mlflow.start_run(description="Tuning") as run:
    print(f"[START] Run ID: {run.info.run_id}")

    # - grid
    grid = {"max_depth": [3, 5, 7], "n_estimators": [5, 10, 15]}

    # - model
    model = ensemble.RandomForestClassifier(random_state=RANDOM_STATE)

    # - search
    search = model_selection.GridSearchCV(model, grid, cv=CV, verbose=1)
    search.fit(X_train, y_train)

    print(f"[STOP] Run ID: {run.info.run_id}")
