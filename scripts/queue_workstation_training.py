"""Run the two prepared families sequentially after a GPU becomes available.

Does not terminate existing processes. A free GPU has no compute applications
and less than 1 GiB allocated for three consecutive 30-second observations.
This is advisory scheduling, not an exclusive cluster reservation.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def query(fields, kind):
    output = subprocess.check_output([
        'nvidia-smi', '--query-' + kind + '=' + fields,
        '--format=csv,noheader,nounits'], text=True)
    return list(csv.reader(output.splitlines(), skipinitialspace=True))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, help='Explicitly assigned GPU; permits existing processes with at least 16 GiB free')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from experiments.train_families import PairedDataset, validate_splits
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    split_path = root / 'splits/m3fd_grouped_seed42.json'
    splits = validate_splits(json.loads(split_path.read_text()))
    for split in ['train', 'val']:
        for stem in splits[split]:
            for folder, suffix in [('Vis', '.png'), ('Ir', '.png'), ('labels_yolo', '.txt')]:
                if not (args.data_root / folder / (stem + suffix)).is_file():
                    raise FileNotFoundError((split, folder, stem))
        PairedDataset(args.data_root, splits[split])[0]
    status_path = root / 'queue_status.json'

    def status(**values):
        values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      pid=os.getpid(), source_root=str(root))
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2))
        temporary.replace(status_path)
        print(json.dumps(values), flush=True)

    (root / 'logs').mkdir(exist_ok=True)
    (root / 'runs').mkdir(exist_ok=True)
    for family in ['fasterrcnn', 'retinanet']:
        observations = {}
        while True:
            busy = {row[0] for row in query('gpu_uuid,pid', 'compute-apps') if row}
            devices = query('index,uuid,memory.used,memory.total', 'gpu')
            if args.gpu_index is None:
                free = {row[1] for row in devices if row[1] not in busy and int(row[2]) < 1024}
            else:
                free = {row[1] for row in devices if int(row[0]) == args.gpu_index
                        and int(row[3]) - int(row[2]) >= 16384}
            observations = {gpu: observations.get(gpu, 0) + 1 for gpu in free}
            ready = sorted(gpu for gpu, count in observations.items()
                           if count >= (3 if args.gpu_index is None else 1))
            if ready:
                gpu = ready[0]
                break
            status(state='waiting_for_free_gpu', family=family,
                   free_observations=observations)
            time.sleep(30)
        output = root / 'runs' / family
        command = [sys.executable, '-u', '-m', 'experiments.train_families',
                   '--family', family, '--data-root', str(args.data_root),
                   '--splits', str(split_path), '--output', str(output),
                   '--device', 'cuda:0', '--epochs', '100', '--batch-size', '4',
                   '--workers', '2', '--max-hours', '2']
        status(state='training', family=family, gpu_uuid=gpu, command=command)
        with (root / 'logs' / (family + '.log')).open('x') as log:
            result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(state='failed', family=family, returncode=result.returncode)
            return result.returncode
    status(state='completed', families=['fasterrcnn', 'retinanet'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
