import pprint

import chevron
import pandas as pd
from configs.model_building_session_config import ModelBuildingSessionParams, ModelBuildingSessionMetrics, \
    ModelBuildingSessionOutputArtifacts
from utils.data_util import pydantic_to_dict


def create_report(params, metrics, artifact_dict, template_path='template.mustache', report_path='report.md'):
    if isinstance(params, ModelBuildingSessionParams):
        params = pydantic_to_dict(params)
    if isinstance(metrics, ModelBuildingSessionMetrics):
        metrics = pydantic_to_dict(metrics)
    if isinstance(artifact_dict, ModelBuildingSessionOutputArtifacts):
        artifact_dict = pydantic_to_dict(artifact_dict)

    params = {'_'.join(['params', k]): v for k, v in params.items()}
    metrics = {'_'.join(['metrics', k]): v for k, v in metrics.items()}
    artifact_dict = {'_'.join(['artifact_list', k]): v for k, v in artifact_dict.items()}

    template_inputs = dict()
    template_inputs.update(params)
    template_inputs.update(metrics)
    template_inputs.update(artifact_dict)

    for k, v in template_inputs.items():
        if isinstance(v, pd.DataFrame):
            template_inputs[k] = v.to_markdown()
        if isinstance(v, dict):
            template_inputs[k] = pprint.pformat(v, indent=4)

    with open(template_path, 'r') as f:
        report = chevron.render(f, template_inputs)

    f = open(report_path, 'w')
    f.write(report)
    f.close()
