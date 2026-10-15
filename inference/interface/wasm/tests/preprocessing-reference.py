"""Regenerate small browser fixtures from train.py's actual evaluation transform."""

import ast
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

repo = Path(__file__).resolve().parents[4]
source = repo / 'examples/test_cifar10/train.py'
loader = next(
    node
    for node in ast.parse(source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'get_cifar10_loaders'
)
scope = {'transforms': transforms, 'input_shape': [3, 32, 32]}
for node in loader.body:
    if isinstance(node, ast.Assign) and node.targets[0].id in ('normalize', 'test_transform'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
transform = scope['test_transform']
fixtures = []
for name, width, height in [
    ('rgb.png', 32, 32),
    ('up.png', 13, 19),
    ('down.png', 67, 59),
    ('wide.png', 65, 17),
    ('solid.png', 1, 1),
    ('jpeg.jpg', 32, 32),
]:
    y, x = np.indices((height, width))
    pixels = np.stack(
        ((3 * x + 11 * y + 37) % 256, (17 * x + 7 * y + 91) % 256, (13 * x + 19 * y + 5) % 256), axis=2
    ).astype(np.uint8)
    image = Image.fromarray(pixels)
    encoded = io.BytesIO()
    image.save(encoded, format='JPEG' if name.endswith('.jpg') else 'PNG')
    decoded = Image.open(io.BytesIO(encoded.getvalue())).convert('RGB')
    fixtures.append(
        {
            'name': name,
            'width': width,
            'height': height,
            'image': base64.b64encode(encoded.getvalue()).decode(),
            'rgb': base64.b64encode(transform.transforms[0](decoded).tobytes()).decode(),
            'values': base64.b64encode(transform(decoded).numpy().astype('<f4').tobytes()).decode(),
        }
    )
output = Path(__file__).parent / 'fixtures/preprocessing.json'
output.parent.mkdir(exist_ok=True)
output.write_text(
    json.dumps({'source': 'train.py:get_cifar10_loaders.test_transform', 'fixtures': fixtures}, separators=(',', ':'))
    + '\n'
)
print(output)
