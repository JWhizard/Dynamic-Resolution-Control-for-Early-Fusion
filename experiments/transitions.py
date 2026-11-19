"""Describe adaptation after resources, annotated scenes, or corruptions change."""
import argparse
import json
from pathlib import Path


def transitions(rows,field,consecutive=5):
    events=[]
    for index in range(1,len(rows)):
        if rows[index][field]==rows[index-1][field]:continue
        # Censor at next condition change or independent video boundary.
        end=index+1
        while end<len(rows) and rows[end][field]==rows[index][field] and rows[end]['sequence']==rows[index]['sequence']:end+=1
        window=rows[index:end]
        settle=None
        for k in range(len(window)-consecutive+1):
            if all(not r['budget_miss'] for r in window[k:k+consecutive]):
                settle=(window[k+consecutive-1]['end']-window[0]['start'])*1000;break
        changed=next((r for r in window if r['resolution']!=rows[index-1]['resolution']),None)
        events.append({'frame_index':rows[index]['frame_index'],'from':rows[index-1][field],
                       'to':rows[index][field],'budget_settled':settle is not None,'budget_settle_ms':settle,
                       'time_to_first_resolution_change_ms':(changed['end']-window[0]['start'])*1000 if changed else None,
                       'observed_horizon_ms':(window[-1]['end']-window[0]['start'])*1000,'observed_frames':len(window)})
    return events


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    output={}
    for path in a.directory.glob('*.json'):
        r=json.loads(path.read_text())
        if isinstance(r,dict) and 'rows' in r and 'arguments' in r:
            output[path.stem]={field:transitions(r['rows'],field) for field in ['phase','scene','corruption']}
    dest=a.directory/'transition_response.json';dest.write_text(json.dumps(output,indent=2));print(dest)


if __name__=='__main__':main()
