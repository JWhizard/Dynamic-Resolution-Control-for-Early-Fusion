"""Verify completed full-sequence cases and export compact per-sequence metrics.

This summarizes saved scores; it does not recompute pooled COCO AP.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from experiments.evaluate_families import sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = args.raw/'runs'
    plan = json.loads((runs/'plan.json').read_text())
    completed = json.loads((runs/'COMPLETED.json').read_text())
    entries = json.loads((runs/'summary.json').read_text())
    expected = Counter(tuple(sorted(j.items())) for j in plan['jobs'])
    actual = Counter(tuple(sorted(e['job'].items())) for e in entries)
    if expected != actual or completed['jobs'] != len(entries):
        raise ValueError('Job matrix incomplete or duplicated')
    sequences = {s['sequence']:s for s in plan['sequences']}
    rows = []
    totals = Counter()
    for entry in entries:
        job = entry['job']
        sequence = sequences[job['sequence']]
        result = runs/entry['result']
        raw = json.loads(result.read_text())
        scores = json.loads((runs/entry['scores_file']).read_text())
        if scores != entry['accuracy'] or scores['provenance']['result_sha256'] != sha256(result):
            raise ValueError('Saved score/result mismatch: '+str(result))
        if (raw['manifest_sha256'] != sequence['manifest_sha256'] or
            scores['provenance']['annotations_sha256'] != sequence['annotations_sha256']):
            raise ValueError('Input/annotation provenance mismatch')
        if raw['input_frames'] != sequence['frames'] or raw['processed_frames']+raw['dropped_frames'] != raw['input_frames']:
            raise ValueError('Frame accounting mismatch')
        summary = raw['summary']
        if raw['power_errors'] or summary['energy']['named_rail_sum_joules'] is None:
            raise ValueError('Power measurement missing or failed')
        for key in ['input_frames','processed_frames','dropped_frames']:
            if raw[key] != entry[key]:raise ValueError('Summary count mismatch')
            totals[key] += raw[key]
        rows.append(dict(**job, environment=sequence['environment'],
            source_frames=raw['input_frames'],processed_frames=raw['processed_frames'],
            dropped_frames=raw['dropped_frames'],
            AP_missing_empty=scores['overall']['AP'],AP_causal=scores['causal_time_aligned']['AP'],
            AP50_causal=scores['causal_time_aligned']['AP50'],
            mean_service_ms=summary['service_ms']['mean'],p95_service_ms=summary['service_ms']['p95'],
            p99_service_ms=summary['service_ms']['p99'],mean_controller_ms=summary['controller_ms']['mean'],
            budget_miss_fraction=summary['budget_miss_fraction'],
            named_rail_J=summary['energy']['named_rail_sum_joules'],
            J_per_source_frame=summary['energy']['named_rail_sum_joules']/raw['input_frames'],
            J_per_completed_inference=summary['energy']['named_rail_sum_joules']/raw['processed_frames']))
    if (totals['input_frames'] != completed['input_frame_evaluations'] or
        totals['processed_frames'] != completed['processed_frames'] or
        totals['dropped_frames'] != completed['dropped_frames']):
        raise ValueError('Completion totals mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'cases.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for source, name in [(runs/'plan.json','plan.json'),(runs/'COMPLETED.json','COMPLETED.json'),
        (args.raw/'protocol_before_scores.json','protocol.json'),(args.raw/'evaluation/inventory.json','data_inventory.json')]:
        shutil.copy2(source,args.output/name)
    validation = dict(verified_cases=len(rows),unique_pairs=sum(s['frames'] for s in sequences.values()),
        **dict(totals),result_score_hashes_match=True,source_accounting_valid=True,
        saved_score_summaries_match=True,annotation_manifest_hashes_match=True,power_read_errors=0)
    (args.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (args.output/'REPORT.md').write_text(f'''# Full-sequence KAIST replay completion — November 19, 2026

All {len(rows)} planned cases completed: three frozen detectors, six full
day/night sequences, eight fixed/adaptive configurations under CPU preprocessing
thread changes and one fixed576 constant-thread control per detector/sequence.
There are {validation['unique_pairs']:,} unique paired images, replayed across
{totals['input_frames']:,} source-frame evaluations; {totals['processed_frames']:,}
were processed and {totals['dropped_frames']:,} were skipped by latest-due-input replay.

`cases.csv` contains per-sequence causal and skipped-empty COCO person AP,
service latency, controller cost, budget misses and named-rail energy. AP values
are fractions. These are saved scores, not newly recomputed pooled COCO metrics.
`validation.json` records verified result hashes, annotation/manifest hash links,
saved-score consistency, source-frame accounting and power read errors.

The study extends previously evaluated 600-frame prefixes. It is frozen
M3FD-to-KAIST transfer, not a new untouched confirmatory test or official KAIST
benchmark score. Labels use person class mapping and crowd-ignore handling.
Separate day/night recordings do not establish natural environment transitions.
There is one run per case. Named-rail energy includes background host activity.
The 100 ms service budget differs from the 50 ms source arrival interval.

Run-level outputs, predictions and traces remain at `{args.raw}`.
Source snapshots and the launch protocol were preserved there before scoring.
Reproduce this export with `python -m experiments.summarize_long_kaist
--raw /path/to/study --output /NEW/summary` (on one shell line).
Broader pooled, worst-sequence and transition analysis remains follow-up work.
''')
    print(json.dumps(validation,indent=2))


if __name__ == '__main__':
    main()
