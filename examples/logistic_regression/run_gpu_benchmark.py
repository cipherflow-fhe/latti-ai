"""Run a real CKKS task and sample its process GPU memory via nvidia-smi."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('task', type=Path)
    args = parser.parse_args()
    environment = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip()
    started = datetime.now(timezone.utc).isoformat()
    samples = []
    with (args.task / 'runtime.log').open('w') as log:
        process = subprocess.Popen([str(args.executable.resolve()), str(args.task.resolve())],
                                   stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'],
                                   text=True, capture_output=True)
            if query.returncode == 0:
                memory = 0
                found = False
                for line in query.stdout.splitlines():
                    fields = line.split(',')
                    if len(fields) == 2 and fields[0].strip() == str(process.pid):
                        try:
                            memory += int(fields[1].strip()) * 1024**2
                            found = True
                        except ValueError:
                            pass
                if found:
                    samples.append(memory)
            time.sleep(0.2)
    metrics = dict(started_utc=started, gpu_environment=environment,
                   gpu_process_sampled_peak_bytes=max(samples, default=None),
                   gpu_memory_sample_count=len(samples),
                   gpu_memory_sampling='nvidia-smi process memory; 0.2 s plus query latency; sampled peak, not exact allocation high-water mark',
                   exit_code=process.returncode)
    (args.task / 'resource_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    if process.returncode:
        print((args.task / 'runtime.log').read_text()[-6000:])
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
