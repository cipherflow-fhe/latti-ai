"""Manual per-layer MaxPool configuration, applied before compiler lowering."""

import math

from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


def configure_maxpool(graph, settings):
    """Apply {layer_id: {alpha, upper_bound, B}} without changing calibration.

    Validate the entire request before mutating the graph. Missing layers and
    unsupported operators are errors rather than silently ignored settings.
    """
    if not isinstance(settings, dict):
        raise ValueError('MaxPool configuration must be an object keyed by layer ID')
    nodes = {node.layer_id: node for node in graph.dag.nodes
             if getattr(node, 'layer_type', None) == 'maxpool2d'}
    pending = []
    for layer_id, values in settings.items():
        if layer_id not in nodes:
            raise ValueError(f'Unknown original MaxPool layer: {layer_id}')
        if not isinstance(values, dict) or set(values) != {'alpha', 'upper_bound', 'B'}:
            raise ValueError(f'{layer_id}: specify exactly alpha, upper_bound and B')
        get_maxpool_profile(values['alpha'])
        checked = {'alpha': values['alpha']}
        for name in ('upper_bound', 'B'):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{layer_id}: {name} must be positive and finite')
            checked[name] = float(value)
        node = nodes[layer_id]
        if (list(node.kernel_shape), list(node.stride), list(node.padding)) != ([3, 3], [2, 2], [1, 1]):
            raise ValueError(f'{layer_id}: configurable staged MaxPool requires kernel=3, stride=2, padding=1')
        pending.append((node, checked))
    for node, values in pending:
        for name, value in values.items():
            setattr(node, name, value)
