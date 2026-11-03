"""Cross-language checks for the optional high-precision Bootstrap preset."""
import json
from pathlib import Path
import re
import unittest
import subprocess
import sys
import tempfile

from training.model_compiler.components import N16QP1547H192H32, EXPERIMENTAL_N16S50H192H32
from inference.lattisense.frontend.custom_task import CkksBtpParam


class PrecisionParametersTest(unittest.TestCase):
    def test_experimental50_chain_consistency(self):
        from sympy import isprime
        compiler = EXPERIMENTAL_N16S50H192H32
        frontend = CkksBtpParam.create_experimental50_param()
        self.assertEqual(compiler.q, frontend.q)
        self.assertEqual(compiler.p, frontend.p)
        self.assertEqual(frontend.scale, 1 << 50)
        self.assertEqual(frontend.btp_output_level, compiler.max_level)
        self.assertEqual(compiler.max_level, 5)
        self.assertEqual(compiler.q[6:], N16QP1547H192H32.q[6:])
        self.assertEqual(len(set(compiler.q + compiler.p)), len(compiler.q + compiler.p))
        self.assertTrue(all(isprime(q) and q % (2 * compiler.poly_modulus_degree) == 1
                            for q in compiler.q + compiler.p))
        source = (Path(__file__).resolve().parents[1] /
                  'inference/lattisense/fhe_ops_lib/lattigo/go_sdk/bootstrap.go').read_text()
        residual = source.split('copy(literal.SchemeParams.Q[1:6], []uint64{')[1].split('}')[0]
        self.assertEqual([int(x.strip()) for x in residual.split(',')], compiler.q[1:6])

    def test_go_preset_matches_compiler_and_frontend(self):
        source = (Path(__file__).resolve().parents[1] /
            'inference/lattisense/fhe_ops_lib/lattigo/ckks/bootstrapping/default_params.go').read_text()
        source = source.split('N16QP1547H192H32 = defaultParametersLiteral{')[1]
        compiler = N16QP1547H192H32
        frontend = CkksBtpParam.create_precision45_param()
        for name in ('q', 'p'):
            literal = source.split(name.upper() + ': []uint64{')[1].split('}')[0]
            expected = [int(x, 16) for x in re.findall(r'0x[0-9a-f]+', literal)]
            self.assertEqual(getattr(compiler, name), expected)
            self.assertEqual(getattr(frontend, name), expected)
        self.assertEqual(compiler.max_level, frontend.btp_output_level)
        self.assertEqual(frontend.btp_output_level, 5)
        self.assertEqual(frontend.scale, 1 << compiler.log_default_scale)
        self.assertEqual(frontend.stc_params.scaling_factor, [[q] for q in compiler.q[6:9]])
        self.assertEqual(frontend.cts_params.scaling_factor, [[q] for q in compiler.q[20:24]])
        self.assertEqual(frontend.eval_mod_params.arcsine_deg, 7)

    def test_generated_stage_arguments_and_bootstrap_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run([sys.executable, 'scripts/prepare_maxpool_ckks_test.py',
                '--output', str(root), '--channels', '1', '--alpha', '14',
                '--precision-bits', '45'], check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parents[1])
            graph = json.loads((root/'server/nn_layers_ct_0.json').read_text())
            for feature in graph['feature'].values():
                self.assertGreaterEqual(feature['level'], 0)
                self.assertLessEqual(feature['level'], 5)
            manifest = json.loads((root/'server/maxpool_stage_parameters.json').read_text())
            dag = json.loads((root/'server/mega_ag.json').read_text())
            data = {node['id']: node for node in dag['data'].values()}
            count = 0
            for stage in manifest.values():
                for name, spec in stage.items():
                    if 'scale' not in spec:
                        continue
                    count += 1
                    self.assertEqual(spec['encoding'], 'full_rns')
                    for i in range(spec['count']):
                        node = data[f'{name}_{i}']
                        self.assertEqual(node['type'], 'pt')
                        self.assertEqual(node['level'], spec['level'])
            self.assertGreater(count, 0)
            for op in dag['compute'].values():
                if op['type'] == 'bootstrap':
                    self.assertEqual(dag['data'][str(op['outputs'][0])]['level'], 5)

    def test_legacy_default_unchanged(self):
        param = CkksBtpParam.create_default_param()
        self.assertEqual(param.scale, 1 << 40)
        self.assertEqual(param.btp_output_level, 9)


if __name__ == '__main__':
    unittest.main()
