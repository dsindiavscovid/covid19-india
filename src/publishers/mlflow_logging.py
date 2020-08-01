import datetime

import mlflow
import pandas as pd
from utils.util import make_clickable

TRACKING_URL = "http://ec2-54-175-207-176.compute-1.amazonaws.com"


def log_to_mlflow(params, metrics, artifact_dict, tags=None, experiment_name="default", run_name=None):
    """Logs metrics, parameters and artifacts to MLflow

    Args:
        params (dict of {str: str}): input parameters to the model
        metrics (dict of {str: numeric}): metrics output from the model
        artifact_dict (dict): file paths of artifacts
        tags ():
        experiment_name (str): name of the MLflow experiment (default: "default")
        run_name (str): name of the MLflow run (default: None)

    """

    mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact, path in artifact_dict.items():
            mlflow.log_artifact(path)
        if tags is not None:
            mlflow.set_tags(tags)


def get_previous_runs(experiment_name, region, interval=0):
    """Get links of previous runs for a region in the last n=interval days

    Args:
        experiment_name (str): name of the MLflow experiment
        region (str): name of region # list?
        interval (int, optional): number of days prior to current day to start search at

    Returns:
        pd.DataFrame: links to relevant runs
    """
    # TODO: Check for status failed

    start_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(interval)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    components = [TRACKING_URL, '#/experiments', experiment_id, 'runs']
    prefix = "/".join(components)

    query = "params.region = '{}'".format(region)
    runs_df = mlflow.search_runs(experiment_ids=experiment_id, filter_string=query)

    if runs_df.empty:
        return pd.DataFrame()

    runs_df = runs_df[runs_df['start_time'] >= start_date]
    run_ids = runs_df['run_id']

    links = ["/".join([prefix, run_id]) for run_id in run_ids]
    links_df = pd.DataFrame(
        {'Published on': runs_df['start_time'], 'Run name': runs_df['tags.mlflow.runName'], 'Link to run': links})
    links_df['Published on'] = links_df['Published on'].apply(lambda x: x.date())
    links_df.index += 1
    links_df = links_df.style.format({'Link to run': make_clickable})

    return links_df

