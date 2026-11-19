"""Small named fixtures for failure injection; outputs only under the given /tmp root."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

root = Path(sys.argv[1]).resolve()
if not root.is_relative_to('/tmp'):
    raise ValueError('Failure fixtures must be generated under /tmp')
repo = Path(__file__).resolve().parents[4]
root.mkdir(parents=True, exist_ok=True)


def value(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 512])


weight = numpy_helper.from_array(np.ones((1, 512), dtype=np.float32), 'weight')
nodes = [
    helper.make_node('Add', ['input', 'weight'], ['crossing']),
    helper.make_node('Mul', ['crossing', 'crossing'], ['output']),
]
model = helper.make_model(
    helper.make_graph(nodes, 'generic', [value('input')], [value('crossing'), value('output')], [weight]),
    opset_imports=[helper.make_opsetid('', 17)],
    ir_version=8,
)
onnx.save_model(model, root / 'model.onnx', save_as_external_data=True, location='weights.data', size_threshold=0)
np.savetxt(root / 'input.csv', np.arange(512, dtype=np.float32), delimiter=',')
(root / 'inputs.json').write_text(
    json.dumps({'inputs': [{'name': 'input', 'csv': 'input.csv', 'shape': [1, 512], 'axes': ['N', 'F']}]})
)
for name, replacement in {
    'cycle': [helper.make_node('Add', ['output', 'weight'], ['crossing']), nodes[1]],
    'forwarded': [nodes[0], helper.make_node('Add', ['crossing', 'input'], ['output'])],
    'identity': [helper.make_node('Identity', ['input'], ['crossing']), nodes[1]],
    'reshape': [helper.make_node('Reshape', ['input', 'shape'], ['crossing']), nodes[1]],
    'transpose': [helper.make_node('Transpose', ['input'], ['crossing'], perm=[0, 1]), nodes[1]],
}.items():
    candidate = onnx.ModelProto()
    candidate.CopyFrom(model)
    del candidate.graph.node[:]
    candidate.graph.node.extend(replacement)
    if name == 'reshape':
        candidate.graph.initializer.append(numpy_helper.from_array(np.array([1, 512], dtype=np.int64), 'shape'))
    onnx.save(candidate, root / (name + '.onnx'))
branch = helper.make_graph([helper.make_node('Identity', ['input'], ['branch'])], 'branch', [], [value('branch')])
control = onnx.ModelProto()
control.CopyFrom(model)
control.graph.node[0].CopyFrom(
    helper.make_node('If', ['condition'], ['crossing'], then_branch=branch, else_branch=branch)
)
control.graph.initializer.append(numpy_helper.from_array(np.array(True), 'condition'))
onnx.save(control, root / 'control.onnx')

# A small parameter role tests the real materializer/Go protocol without large keys.
echo_profile = json.loads((repo / 'inference/lattisense/frontend/parameter.json').read_text())['CKKS']['12']
echo_profile.update(poly_modulus_degree=4096, ring_type=1, n_slots=4096)
(root / 'echo-parameters.json').write_text(json.dumps({'p': echo_profile}))
task = root / 'parameter-task'
shutil.rmtree(task, ignore_errors=True)
profile = json.loads((repo / 'inference/lattisense/frontend/parameter.json').read_text())['CKKS']['15']
profile.update(poly_modulus_degree=32768, ring_type=1, n_slots=32768, param_name='PN15Default')
feature = dict(dim=0, channel=4, pack_num=4, skip=8192, level=1, ckks_parameter_id='p')
config = dict(
    evaluation_keys=[],
    use_btp=False,
    pack_style='multiplexed',
    block_shape=[32, 32],
    deployment_mode='server_provisioned_runner',
    input_mode='plaintext',
    parameter_mode='encrypted_offline',
    decryptor='provisioner',
    task_input_id=['input'],
    task_output_id=['output'],
    task_input_param={'input': feature},
    task_output_param={'output': feature},
)
graph = {
    'feature': {'input': feature, 'output': feature},
    'layer': {
        'scale': {
            'type': 'mult_scalar',
            'feature_input': ['input'],
            'feature_output': ['output'],
            'weight_path': 'scale.weight',
        }
    },
}
signature = {
    'algorithm': 'CKKS',
    'encrypted_parameters': [dict(id='mult_scalar_scale', type='ct', size=[1], level=1, scale=2**40)],
    'online': [
        dict(id='input', type='pt', phase='in', size=[1], level=0),
        dict(id='output', type='ct', phase='out', size=[1], level=1),
    ],
}
for role in ['runner', 'provisioner']:
    directory = task / role
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in {
        'task_config.json': config,
        'ckks_parameter.json': {'p': profile},
        'nn_layers_ct_0.json': graph,
        'task_signature.json': signature,
        'key_requirements.json': {},
        'mega_ag.json': {},
        'compiled_mega_ag.json': {},
        'fhe_parameter.json': {},
    }.items():
        (directory / name).write_text(json.dumps(data))
with h5py.File(task / 'provisioner/model_parameters.h5', 'w') as weights:
    weights['scale.weight'] = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
files = {
    str(file.relative_to(task)): {'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    for file in task.rglob('*')
    if file.is_file()
}
(task / 'artifact-manifest.json').write_text(
    json.dumps(
        dict(
            schema='latti-ai.browser-model-artifact.v1',
            kind='task',
            virtualRoot='/latti/jobs/11111111-1111-1111-1111-111111111111',
            entry='.',
            deploymentMode='server_provisioned_runner',
            files=files,
        )
    )
)
# Real task artifacts have an explicit directory entry.
(task / 'task').mkdir(exist_ok=True)
for role in ['runner', 'provisioner']:
    (task / role).rename(task / 'task' / role)
manifest = json.loads((task / 'artifact-manifest.json').read_text())
manifest['entry'] = 'task'
manifest['files'] = {'task/' + name: data for name, data in files.items()}
(task / 'artifact-manifest.json').write_text(json.dumps(manifest))
