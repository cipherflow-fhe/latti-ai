# Recheck retained full logits, commands, monitor samples, and binary hashes.
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
root = repo / 'docs/report'
sys.path.insert(0, str(repo / 'tools'))
from benchmark_server_provisioned_runner import read_logits, metrics

provenance = json.loads((root / 'source_provenance.json').read_text())
results = {}
for model, size in [('resnet20', 10), ('mobilenet', 1000)]:
    report = json.loads((root / f'{model}_runtime_lazy.json').read_text())
    assert report['acceptance_passed']
    assert report['source_sha256'] == provenance['source_sha256']
    for path_key, hash_key in [('onnx', 'onnx_sha256'), ('input_csv', 'input_csv_sha256')]:
        assert hashlib.sha256(Path(report[path_key]).read_bytes()).hexdigest() == report[hash_key]
    reference = read_logits(root / 'raw' / f'{model}-plaintext.csv')
    assert len(reference) == size
    baseline = max(s['metrics']['max_abs_error'] for s in report['modes']['legacy']['samples'])
    assert report['modes']['legacy']['ckks_parameter'] == report['modes']['compressed']['ckks_parameter']
    summary = {}
    for mode, result in report['modes'].items():
        samples = result['samples']
        assert len(samples) == 4 and sum(s['warmup'] for s in samples) == 1
        assert result['seed_stride'] == ([1, 1] if mode == 'legacy' else [4, 4])
        assert result['structure']['canonical_load_nodes'] == result['bundle']['ciphertext_count']
        assert result['structure']['public_masks_absent']
        assert result['structure']['shape_level_scale_layout_match']
        assert sum(x['payload_bytes'] for x in result['bundle']['arguments'].values()) == result['encrypted_parameter_payload_bytes']
        for sample in samples:
            assert len(read_logits(sample['logits_csv'])) == size
            actual = metrics(reference, read_logits(sample['logits_csv']))
            assert actual == sample['metrics']
            assert actual['max_abs_error'] <= .01 and actual['rmse'] <= .001
            assert actual['cosine_similarity'] >= .9999 and actual['top1_match']
            if mode == 'compressed':
                assert actual['max_abs_error'] <= 2 * baseline + 1e-6
            assert sample['cache']['canonical_loads'] == result['bundle']['ciphertext_count']
            assert sample['cache']['cache_peak_bytes'] <= sample['cache']['cache_capacity_bytes'] == 64 * 1024**2
            for monitor in sample['monitors'].values():
                with Path(monitor).open() as stream:
                    assert list(csv.DictReader(stream))
        for command in result['commands']:
            assert command['returncode'] == 0
            assert Path(command['log']).is_file(), command['log']
        measured = [s for s in samples if not s['warmup']]
        for field in ('runner_wall_ms', 'decrypt_wall_ms', 'e2e_fhe_wall_ms', 'cpu_peak_rss_bytes', 'gpu_peak_used_bytes'):
            assert result[field] == statistics.median(s[field] for s in measured)
        assert result['gpu_core_ms'] == statistics.median(s['gpu_core_ms'] for s in measured)
        summary[mode] = {'complete_logits': size, 'gpu_samples': len(samples), 'all_gates_passed': True,
                         'canonical_load_nodes': result['bundle']['ciphertext_count']}
    results[model] = summary
for path, expected in json.loads((root / 'raw/build_artifacts.json').read_text()).items():
    assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected['sha256']
print(json.dumps(results, indent=2))
