import mlflow

TRACKING_URL = "http://ec2-54-175-207-176.compute-1.amazonaws.com"


def log_to_mlflow(params, metrics, artifact_dir, artifacts,
                  experiment_name="default", run_name=None, tracking_url=TRACKING_URL):
    """Logs metrics, parameters and artifacts to MLflow

    Args:
        params (dict of {str: str}): input parameters to the model
        metrics (dict of {str: numeric}): metrics output from the model
        artifact_dir (str): local directory in which artifacts are stored
        artifacts (list[str]): list of files to be logged
        experiment_name (str): name of the MLflow experiment (default: "default")
        run_name (str): name of the MLflow run (default: None)
        tracking_url (str): MLflow tracking server URL (default: TRACKING_URL)

    Assumptions:
        The params and metrics dicts are flattened out (no nesting)

    """

    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(artifact_dir + artifact)
