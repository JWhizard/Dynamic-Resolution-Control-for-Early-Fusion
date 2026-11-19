"""Frozen-checkpoint, common-COCO evaluation on the grouped M3FD test split.

No fitting or model selection. Uses original native YOLO-label coordinates,
as in the repository evaluator. Each detector retains its trained preprocessing.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch

from experiments.detector_families import Detector, NAMES
from experiments.train_families import validate_splits
from fusion.preprocess import detect, load_model


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def annotations(text, image_id, width, height, first_id):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError('malformed annotation')
        cls, cx, cy, bw, bh = values
        if cls != int(cls) or not 0 <= cls < len(NAMES) or bw <= 0 or bh <= 0:
            raise ValueError('invalid class or box')
        w, h = bw * width, bh * height
        rows.append(dict(id=first_id + len(rows), image_id=image_id,
                         category_id=int(cls) + 1, iscrowd=0,
                         bbox=[cx * width - w / 2, cy * height - h / 2, w, h],
                         area=w * h))
    return rows


def score(gt_dict, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(images=gt_dict['images'], annotations=[],
                              categories=gt_dict['categories'])
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    result = {n: float(x) if x >= 0 else None for n, x in zip(names, ev.stats)}
    result['AP_per_class'] = {}
    for index, category in enumerate(ev.params.catIds):
        name=gt.cats[category]['name']
        values = ev.eval['precision'][:, :, index, 0, -1]
        values = values[values >= 0]
        result['AP_per_class'][name] = float(values.mean()) if len(values) else None
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('CUDA GPU required for this workstation evaluation')
    split_path = Path(plan['splits'])
    if sha256(split_path) != plan['split_sha256']:
        raise ValueError('split hash mismatch')
    splits = validate_splits(json.loads(split_path.read_text()))
    ids = splits['test']
    if len(ids) != plan['test_images'] or len(set(ids)) != len(ids):
        raise ValueError('test size mismatch or duplicate IDs')
    for model in plan['models']:
        if sha256(model['checkpoint']) != model['sha256']:
            raise ValueError('checkpoint hash mismatch: ' + model['name'])
        if model['family'] != 'yolo':
            metadata = json.loads(Path(model['training_metadata']).read_text())
            if metadata['split_sha256'] != plan['split_sha256'] or metadata['test_data_loaded']:
                raise ValueError('training split mismatch or training accessed test data')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(args.plan.read_text())
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    data = Path(plan['data_root'])
    gt = dict(info={}, images=[], annotations=[],
              categories=[dict(id=i+1, name=n) for i, n in enumerate(NAMES)])
    inputs = []
    for index, stem in enumerate(ids):
        paths = [data / folder / (stem + ext) for folder, ext in
                 [('Vis', '.png'), ('Ir', '.png'), ('labels_yolo', '.txt')]]
        rgb = cv2.imread(str(paths[0]))
        ir = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None or rgb.shape[:2] != ir.shape:
            raise ValueError('missing/mismatched pair ' + stem)
        h, w = ir.shape
        gt['images'].append(dict(id=index, file_name=stem + '.png', width=w, height=h))
        gt['annotations'].extend(annotations(paths[2].read_text(), index, w, h,
                                            len(gt['annotations']) + 1))
        inputs.append(dict(id=index, stem=stem, width=w, height=h,
                           files=[dict(path=str(p), sha256=sha256(p)) for p in paths]))
    (args.output / 'ground_truth.json').write_text(json.dumps(gt))
    (args.output / 'input_hashes.json').write_text(json.dumps(inputs, indent=2))
    results = []
    import torchvision
    import ultralytics
    environment = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                       ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(args.device), python=platform.python_version(),
                       source_sha256=sha256(__file__), plan_sha256=sha256(args.plan),
                       started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2))
    for entry in plan['models']:
        print('Starting', entry['name'], flush=True)
        start = time.monotonic()
        if entry['family'] == 'yolo':
            model = load_model(entry['checkpoint'], args.device)
            names = model.names
            names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
            if names != NAMES or model.model[0].conv.in_channels != 4:
                raise ValueError('YOLO class/channel contract differs')
        else:
            model = Detector(entry['checkpoint'], args.device)
            if model.metadata['family'] != entry['family']:
                raise ValueError('family mismatch')
        predictions = []
        for index, stem in enumerate(ids):
            rgb = cv2.imread(str(data / 'Vis' / (stem + '.png')))
            ir = cv2.imread(str(data / 'Ir' / (stem + '.png')), cv2.IMREAD_GRAYSCALE)
            frame = np.ascontiguousarray(np.dstack([rgb, ir]))
            if entry['family'] == 'yolo':
                detections = detect(model, torch.from_numpy(frame).to(args.device),
                                    plan['resolution'], conf=plan['confidence']).cpu().numpy()
            else:
                detections = model.infer(frame, plan['resolution'])
            if not np.isfinite(detections).all():
                raise ValueError('nonfinite predictions')
            for x1, y1, x2, y2, confidence, cls in detections.tolist():
                if int(cls) != cls or not 0 <= cls < len(NAMES):
                    raise ValueError('invalid predicted class')
                if confidence <= plan['confidence']:
                    continue
                predictions.append(dict(image_id=index, category_id=int(cls) + 1,
                                        bbox=[x1, y1, x2-x1, y2-y1], score=confidence))
            if (index + 1) % 100 == 0:
                print(entry['name'], index + 1, '/', len(ids), flush=True)
        path = args.output / (entry['name'] + '_predictions.json')
        path.write_text(json.dumps(predictions))
        metrics = score(gt, predictions)
        row = dict(name=entry['name'], family=entry['family'], images=len(ids),
                   predictions=len(predictions), resolution=plan['resolution'],
                   checkpoint_sha256=entry['sha256'], metrics=metrics,
                   wall_seconds=time.monotonic()-start)
        results.append(row)
        (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    (args.output / 'COMPLETED.json').write_text(json.dumps(dict(
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        models=len(results), test_images=len(ids)), indent=2))


if __name__ == '__main__':
    main()
