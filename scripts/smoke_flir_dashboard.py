"""Validate a running FLIR dashboard and save measured live-camera evidence."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:5000'


def api(path, payload=None):
    req = urllib.request.Request(BASE + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def measure(policy, view, confidence):
    initial = api('/api/control', {'running': True, 'policy': policy,
                                   'view': view, 'confidence': confidence})
    last_timestamp = initial.get('latest', {}).get('timestamp')
    rows = []
    deadline = time.monotonic() + 60
    while len(rows) < 20 and time.monotonic() < deadline:
        status = api('/api/status')
        if status['error']:
            raise RuntimeError(status['error'])
        row = status['latest']
        if row.get('timestamp') and row['timestamp'] != last_timestamp:
            last_timestamp = row['timestamp']
            if row['policy'] == policy and row['view'] == view and row['confidence'] == confidence:
                rows.append(row)
        time.sleep(.15)
    assert len(rows) == 20, f'Only received {len(rows)} processed frames'
    assert status['sources']['rgb']['connected'] and status['sources']['thermal']['connected']
    if policy == 'fixed576':
        assert all(row['resolution'] == 576 for row in rows)
    return {'policy': policy, 'view': view, 'confidence': confidence,
            'fps': status['fps'], 'mean_inference_ms': status['mean_inference_ms'],
            'source_shapes': [status['sources'][name]['shape'] for name in ['rgb', 'thermal']],
            'host_arrival_skew_ms': status['sources']['skew_ms'], 'frames': rows}


def main():
    with urllib.request.urlopen(BASE, timeout=5) as response:
        assert b'FLIR' in response.read()
    for asset in ['/static/styles.css', '/static/app.js']:
        with urllib.request.urlopen(BASE + asset, timeout=5) as response:
            assert response.status == 200
    results = [measure('sequence', 'split', .25), measure('fixed576', 'thermal', .4)]
    snapshot = api('/api/snapshot', {})
    assert Path(snapshot['image']).stat().st_size > 1000
    assert json.loads(Path(snapshot['metadata']).read_text())['view'] == 'thermal'
    stopped = api('/api/control', {'running': False})
    assert not stopped['running']
    frame = stopped['latest']['frame']
    time.sleep(.5)
    assert api('/api/status')['latest']['frame'] == frame
    results.append(measure('sequence', 'rgb', .25))
    try:
        api('/api/control', {'policy': 'invalid'})
        raise AssertionError('Invalid policy accepted')
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
    with urllib.request.urlopen(BASE + '/stream.mjpg', timeout=10) as response:
        assert b'Content-Type: image/jpeg' in response.read(1024)
    api('/api/control', {'view': 'split', 'policy': 'sequence', 'confidence': .25})
    report = {'tested_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'checks': ['page and assets', 'live dynamic inference', 'fixed576 and confidence',
                         'RGB/thermal/split', 'snapshot and metadata', 'stop and restart',
                         'invalid control rejection', 'MJPEG output'],
              'measurements': results, 'snapshot': snapshot}
    output = Path(__file__).resolve().parents[1] / 'captures' / 'flir_live_validation.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(output), 'checks': report['checks'],
                      'measurements': [{k: v for k, v in r.items() if k != 'frames'} for r in results]}, indent=2))


if __name__ == '__main__':
    main()
