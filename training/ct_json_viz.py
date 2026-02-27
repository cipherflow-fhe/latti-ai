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
import argparse
import os


def main(input_filename, output_folder, output_name='graph.gv'):
    ct_json_filename = input_filename
    with open(ct_json_filename, 'r') as f:
        ct_json = json.load(f)

    graph = graphviz.Graph()

    for feature_id, feature_p in ct_json['feature'].items():
        label_str = f'{feature_p["channel"]}'
        if 'shape' in feature_p.keys():
            label_str += f', {feature_p["shape"]}'
        label_str += f', lv{feature_p["level"]}'
        feature_scale = float(feature_p['scale'])
        if abs(feature_scale - 1.0) > 0.00001:
            label_str += f', scale:{feature_p["scale"]}'
        graph.node(name=feature_id, label=label_str, shape='box')

    for layer_id, layer_p in ct_json['layer'].items():
        is_mpc_layer = layer_p['type'] in ('relu2d', 'maxpool', 'bootstrapping', 'mpc_refresh')
        if is_mpc_layer:
            graph.node(name=layer_id, label=f'{layer_p["type"]}', style='filled', fillcolor='cornflowerblue')
        elif layer_p['type'] == 'mult_scalar':
            graph.node(name=layer_id, label=f'{layer_p["type"]}', style='filled', fillcolor='yellow')
        elif layer_p['type'] == 'drop_level':
            graph.node(name=layer_id, label=f'{layer_p["type"]}', style='filled', fillcolor='red')
        else:
            if 'weight_scale' in layer_p.keys():
                graph.node(name=layer_id, label=f'{layer_id}, scale:{layer_p["weight_scale"]}')
            else:
                graph.node(name=layer_id, label=f'{layer_id}')
        for input_id in layer_p['feature_input']:
            graph.edge(input_id, layer_id)
        for output_id in layer_p['feature_output']:
            graph.edge(layer_id, output_id)

    output_path = os.path.join(output_folder, output_name)
    graph.render(output_path, format='pdf', view=False)
    print(f'Graph saved to {output_path}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize CT JSON graph')
    parser.add_argument('input_filename', help='Path to input CT JSON file')
    parser.add_argument('-o', '--output', default='graph.gv', help='Output filename (default: graph.gv)')
    args = parser.parse_args()

    input_filename = os.path.abspath(args.input_filename)
    output_folder = os.path.dirname(input_filename)

    main(input_filename, output_folder, args.output)
