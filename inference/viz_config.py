# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import graphviz
import json
import sys

if len(sys.argv) < 2:
    print('Usage: python viz_config.py <path_to_json>')
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    nn_config = json.load(f)

layers_to_remove = list()
features_to_remove = list()
for layer_id, layer in nn_config['layer'].items():
    if layer['type'] == 'batchnorm':
        feature_in_id = layer['feature_in'][0]
        feature_out_id = layer['feature_out'][0]
        for pre_layer_id, pre_layer in nn_config['layer'].items():
            if feature_in_id in pre_layer['feature_out']:
                pre_layer['feature_out'].remove(feature_in_id)
                pre_layer['feature_out'].append(feature_out_id)
        layers_to_remove.append(layer_id)
        features_to_remove.append(feature_in_id)
for x in layers_to_remove:
    del nn_config['layer'][x]
for x in features_to_remove:
    del nn_config['feature'][x]

graph = graphviz.Graph()
graph.graph_attr['ranksep'] = '0.25'

for feature_id, feature in nn_config['feature'].items():
    shape_str = str(feature['n_channel'])
    if feature['dim'] > 0:
        shape = feature['shape']
        for s in shape:
            shape_str += '*'
            shape_str += str(s)
    label = f'{feature_id} {shape_str}'
    graph.node(name=feature_id, label=label)

for layer_id, layer in nn_config['layer'].items():
    if 'relu' in layer['type'] or 'mpc' in layer['type']:
        graph.node(name=layer_id, label=f'{layer_id}: relu', shape='box', style='filled', fillcolor='#40e0d0')
    elif 'conv2d' not in layer['type']:
        graph.node(name=layer_id, label=f'{layer_id}: {layer["type"]}', shape='box')
    else:
        shape_str = (
            f'{layer["n_channel_out"]}*{layer["n_channel_in"]}*{layer["kernel_shape"][0]}*{layer["kernel_shape"][1]}'
        )
        graph.node(name=layer_id, label=f'{layer_id}: {layer["type"]} {shape_str}', shape='box')
    for feature_in_id in layer['feature_in']:
        graph.edge(feature_in_id, layer_id)
    for feature_out_id in layer['feature_out']:
        graph.edge(layer_id, feature_out_id)

graph.render('viz/graph.gv', format='pdf', view=False)
graph.render('viz/graph.gv', format='png', view=False)
