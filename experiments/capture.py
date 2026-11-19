"""Record an ordered, repeatable FLIR preview stream without moving the camera."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=180)
    p.add_argument('--fps', type=float, default=8)
    p.add_argument('--server', default='http://127.0.0.1:8080')
    p.add_argument('--environment', default='indoor_lab')
    a = p.parse_args()
    if a.frames < 1 or a.fps <= 0:
        p.error('frames and fps must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    def read(sensor):
        with urllib.request.urlopen(a.server + '/api/frame/fusion/' + sensor, timeout=3) as response:
            data = response.read()
        return data, time.monotonic()
    frames = []
    origin = time.monotonic()
    with ThreadPoolExecutor(2) as pool:
        for i in range(a.frames):
            time.sleep(max(0, origin + i / a.fps - time.monotonic()))
            pair = list(pool.map(read, ['rgb', 'thermal']))
            shapes = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape for data, _ in pair]
            if shapes[0] != shapes[1]:
                raise ValueError('calibrated frame dimensions differ')
            entry = {'id': i, 'sequence': a.output.name, 'scene': 'unannotated',
                     'environment': a.environment, 'timestamp_s': max(t for _, t in pair) - origin,
                     'host_skew_ms': abs(pair[0][1] - pair[1][1]) * 1000}
            for sensor, (data, _) in zip(['rgb', 'thermal'], pair):
                filename = f'{sensor}_{i:06d}.jpg'
                (a.output / filename).write_bytes(data)
                entry[sensor] = filename
                entry[sensor + '_sha256'] = hashlib.sha256(data).hexdigest()
            frames.append(entry)
    manifest = {'schema_version': 1, 'dataset': 'FLIR indoor preview pilot',
                'created': datetime.now().isoformat(), 'timestamp_kind': 'host_HTTP_arrival',
                'registered': True, 'labels': None, 'frames': frames,
                'limitations': ['JPEG previews', 'independent HTTP endpoints', 'no ground truth',
                                'single indoor environment', 'no controlled natural scene transitions']}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(a.output / 'manifest.json')


if __name__ == '__main__':
    main()
