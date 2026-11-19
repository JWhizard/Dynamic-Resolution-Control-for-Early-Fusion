"""Causal latest-completed-box alignment at each recorded input arrival.

No future prediction, interpolation or tracking. Independent sequences cannot
reuse one another's predictions. Requires same native image coordinates.
"""
import bisect


def align_predictions(frames,rows):
    if not rows:return []
    first=rows[0];origin=first['end']-first['output_age_ms']/1000-(frames[first['frame_index']]['timestamp_s']-frames[0]['timestamp_s'])
    completed=[r['end']-origin for r in rows]
    output=[]
    for frame in frames:
        tick=frame['timestamp_s']-frames[0]['timestamp_s']
        index=bisect.bisect_right(completed,tick)-1
        if index<0:continue
        row=rows[index]
        if row['sequence']!=frame['sequence']:continue
        output.append({'image_id':frame['id'],'detections':row['detections'],
                       'source_frame_index':row['frame_index']})
    return output
