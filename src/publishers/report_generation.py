import chevron


def create_report(params, metrics, artifact_list):
    params = {'_'.join(['param', k]): v for k, v in params.items()}
    metrics = {'_'.join(['metric', k]): v for k, v in metrics.items()}
    artifact_list = {'_'.join(['artifact', k]): v for k, v in artifact_list.items()}

    template_inputs = dict()
    template_inputs.update(params)
    template_inputs.update(metrics)
    template_inputs.update(artifact_list)

    with open('template.mustache', 'r') as f:
        report = chevron.render(f, template_inputs)

    f = open("report.md", "w")
    f.write(report)
    f.close()
