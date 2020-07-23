import chevron


def create_report(params, metrics, artifact_list, template_path='template.mustache', report_path='report.md'):
    params = {'_'.join(['params', k]): v for k, v in params.items()}
    metrics = {'_'.join(['metrics', k]): v for k, v in metrics.items()}
    artifact_list = {'_'.join(['artifact_list', k]): v for k, v in artifact_list.items()}

    template_inputs = dict()
    template_inputs.update(params)
    template_inputs.update(metrics)
    template_inputs.update(artifact_list)

    with open(template_path, 'r') as f:
        report = chevron.render(f, template_inputs)

    f = open(report_path, 'w')
    f.write(report)
    f.close()
