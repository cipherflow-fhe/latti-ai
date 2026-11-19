import sys
import json
import importlib.util
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo))
import numpy as np
import torch
import onnx
import h5py
from onnx import helper, numpy_helper, TensorProto
from training.nn_tools import eval_fn_hat_for_aespa as numeric
from training.nn_tools import export as current
from training.nn_tools import feature_mat_h5 as feature

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
if not root.is_relative_to('/tmp'):
    raise ValueError('Numerical evidence must be under /tmp')
(root / 'oracle/nn_tools').mkdir(parents=True, exist_ok=True)
for name in ['eval_fn_hat_for_aespa', 'export', 'feature_mat_h5']:
    (root / 'oracle/nn_tools' / f'{name}.py').write_bytes(
        subprocess.check_output(['git', 'show', 'HEAD:training/nn_tools/' + name + '.py'], cwd=repo)
    )


def oracle(name):
    spec = importlib.util.spec_from_file_location(
        'training.nn_tools._oracle_' + name, root / 'oracle/nn_tools' / f'{name}.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


old_numeric = oracle('eval_fn_hat_for_aespa')
old_export = oracle('export')
old_feature = oracle('feature_mat_h5')
report = {'provenance': 'isolated original torch path oracle', 'atol': 1e-12, 'rtol': 1e-12, 'activation': {}, 'h5': {}}
for name, cls in [('relu', torch.nn.ReLU), ('relu6', torch.nn.ReLU6), ('silu', torch.nn.SiLU), ('gelu', torch.nn.GELU)]:
    x = np.linspace(-20, 20, 4001, dtype=np.float64)
    y = cls()(torch.from_numpy(x)).numpy()
    actual = numeric.activation_function(name)(x)
    np.testing.assert_allclose(actual, y, atol=1e-12, rtol=1e-12)
    errors = {}
    for degree in [0, 2, 4, 6]:
        expected = old_numeric.get_hermite_coeffs_for_module(cls, degree, tol=1e-8, limit=20)
        got = numeric.get_hermite_coeffs_for_activation(name, degree, tol=1e-8, limit=20)
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(numeric.get_hermite_coeffs_for_module(cls, degree), expected)
        errors[str(degree)] = float(np.max(np.abs(np.array(got) - expected)))
    report['activation'][name] = {'values_max_abs': float(np.max(np.abs(actual - y))), 'hermite_max_abs': errors}
    for kind in ['ordinary', 'feature_mat']:
        dest = root / 'numeric-fixtures' / name / kind
        dest.mkdir(parents=True, exist_ok=True)
        source = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        'RangeNormPoly2d' if kind == 'ordinary' else 'PolyActRNPoly',
                        ['x', 'act.rangenorm.running_max'] if kind == 'ordinary' else ['x'],
                        ['y'],
                        name='act',
                        domain='latti',
                        degree=4,
                        activation=name,
                        upper_bound=2.0,
                        eps=0.003,
                    )
                ],
                'fixture',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 2, 2])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 2, 2])],
                [numpy_helper.from_array(np.array([2.5, 4.0], dtype=np.float32), 'act.rangenorm.running_max')],
            )
        )
        onnx.save(source, dest / 'source.onnx')
        layer = {
            'type': 'polyact' if kind == 'ordinary' else 'pcmpoly',
            'weight_path': 'act.weight',
            'degree': 4,
            'feature_input': ['x'],
        }
        (dest / 'ct.json').write_text(
            json.dumps({'layer': {'act': layer}, 'feature': {'x': {'shape': [3, 2], 'data_type': 'feature_mat'}}})
        )
        fn = current.export_h5_from_onnx if kind == 'ordinary' else feature.export_feature_mat_h5_from_onnx
        old_fn = old_export.export_h5_from_onnx if kind == 'ordinary' else old_feature.export_feature_mat_h5_from_onnx
        if kind == 'feature_mat' and name == 'relu6':
            continue  # original FeatureMat supports only ReLU/SiLU/GELU
        old_fn(str(dest / 'source.onnx'), str(dest / 'ct.json'), str(dest / 'old.h5'), verbose=False)
        fn(str(dest / 'source.onnx'), str(dest / 'ct.json'), str(dest / 'new.h5'), verbose=False)
        with h5py.File(dest / 'old.h5') as old, h5py.File(dest / 'new.h5') as new:
            assert set(old) == set(new) == {'act.weight'}
            a, b = old['act.weight'], new['act.weight']
            assert a.shape == b.shape and a.dtype == b.dtype == np.float64
            np.testing.assert_allclose(a[...], b[...], atol=1e-12, rtol=1e-12)
            report['h5'][f'{name}/{kind}'] = {
                'dataset': 'act.weight',
                'shape': a.shape,
                'dtype': str(a.dtype),
                'max_abs': float(np.max(np.abs(a[...] - b[...]))),
            }
# Public lazy imports still expose all original training names.
import training.nn_tools as nn_tools

for name in nn_tools.__all__:
    getattr(nn_tools, name)
(root / 'numeric-comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
