"""Exploratory transitions from ground-truth person presence, never fed to policy.

These are object-presence changes, not independently annotated day/weather cuts.
At least ten consecutive source frames define a retained state segment.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def presence_events(frames,annotations,minimum=10):
    counts=Counter(a['image_id'] for a in annotations if not a['iscrowd'])
    segments=[];start=0
    while start<len(frames):
        state=counts[frames[start]['id']]>0;end=start+1
        while end<len(frames) and (counts[frames[end]['id']]>0)==state and frames[end]['sequence']==frames[start]['sequence']:end+=1
        if end-start>=minimum:segments.append((start,end,state,frames[start]['sequence']))
        start=end
    events=[];previous=None
    for start,end,state,sequence in segments:
        if previous and previous[3]==sequence and previous[2]!=state:
            events.append({'frame_index':start,'timestamp_s':frames[start]['timestamp_s'],
                           'from':'person-present' if previous[2] else 'no-person',
                           'to':'person-present' if state else 'no-person','stable_segment_frames':end-start})
        previous=(start,end,state,sequence)
    return events


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--raw',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());frames=manifest['frames'][:1200]
    gt=json.loads((a.manifest.parent/manifest['coco_annotations']).read_text())
    events=presence_events(frames,gt['annotations']);observations={}
    for path in a.raw.glob('kaist_*.json'):
        result=json.loads(path.read_text())
        if 'rows' not in result:continue
        rows=result['rows'];by_event=[]
        for event in events:
            index=event['frame_index'];before=[r for r in rows if index-20<=r['frame_index']<index]
            after=[r for r in rows if index<=r['frame_index']<index+20]
            if not before or not after:continue
            changed=next((r for r in after if r['resolution']!=before[-1]['resolution']),None)
            by_event.append({'event':event,'before_resolution_mix':dict(Counter(r['resolution'] for r in before)),
                             'after_resolution_mix':dict(Counter(r['resolution'] for r in after)),
                             'before_mean_service_ms':sum(r['service_ms'] for r in before)/len(before),
                             'after_mean_service_ms':sum(r['service_ms'] for r in after)/len(after),
                             'first_switch_source_delay_ms':(frames[changed['frame_index']]['timestamp_s']-event['timestamp_s'])*1000 if changed else None})
        observations[path.stem]=by_event
    record={'definition':'Post-hoc person-present/no-person transitions from supplied labels; state persists >=10 frames. Labels are not controller inputs.',
            'limitations':['One night preview prefix; these are not day/night/weather transitions.','One-second windows are correlated diagnostic observations; no transition-window AP or significance claim.'],
            'events':events,'observations':observations}
    a.output.write_text(json.dumps(record,indent=2));print('Content transitions:',len(events),a.output)


if __name__=='__main__':main()
