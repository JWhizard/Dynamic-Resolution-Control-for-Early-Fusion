"""Select dense held-out KAIST sequence prefixes before inspecting labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def select(test,day,night,train,length=600):
    test,day,night,train=[set(v) for v in [test,day,night,train]]
    if test&train or day&night or day|night!=test:
        raise ValueError('invalid official train/test/day/night partition')
    groups=defaultdict(list)
    for name in test:groups[str(Path(name).parent)].append(name)
    selected=[]
    for number in range(6,12):
        group=min(g for g in groups if Path(g).parts[0]==f'set{number:02d}')
        rows=sorted(groups[group],key=lambda s:int(Path(s).name[1:]))
        if length is not None:
            if length<=0:raise ValueError('length must be positive or None for full sequences')
            rows=rows[:length]
            if len(rows)!=length:raise ValueError('sequence too short')
        indices=[int(Path(x).name[1:]) for x in rows]
        if any(b-a!=1 for a,b in zip(indices,indices[1:])):
            raise ValueError('sparse frames cannot be treated as native20Hz video')
        environment='day' if set(rows)<=day else 'night' if set(rows)<=night else None
        if environment is None:raise ValueError('mixed/unknown environment')
        selected.append(dict(sequence=group,environment=environment,frames=rows,
                             start_frame=indices[0],end_frame=indices[-1]))
    return selected


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=600)
    p.add_argument('--full-sequences',action='store_true',help='Use every dense frame in each selected sequence')
    a=p.parse_args()
    if a.output.exists():p.error('preserve selection')
    def lines(name):return (a.root/'imageSets'/(name+'.txt')).read_text().splitlines()
    selected=select(lines('test-all-01'),lines('test-day-01'),lines('test-night-01'),lines('train-all-01'),None if a.full_sequences else a.frames)
    result=dict(protocol=('Full native dense sequence' if a.full_sequences else f'First{a.frames} native dense frames')+' of the lexicographically first sequence in each official test set06–11; no label-based selection.',
        selection_length=None if a.full_sequences else a.frames,
        source_root=str(a.root),fps=20,sequence_count=len(selected),
        unique_frames=sum(len(s['frames']) for s in selected),sequences=selected,
        limitations=['Six predeclared sequence prefixes, not the full official test benchmark.','Native20Hz timestamps reconstructed from frame indices, not independently verified hardware timestamps.'])
    a.output.write_text(json.dumps(result,indent=2))
    print('Selected',len(selected),'sequences /',result['unique_frames'],'pairs')


if __name__=='__main__':main()
