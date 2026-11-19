"""Import paired registered constant-rate RGB/IR videos as an ordered manifest.

Matching frame indices are assumed synchronized. Equal FPS/dimensions do not
prove registration or sensor synchronization; provide already verified inputs.
"""
import argparse
import json
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--rgb',required=True,type=Path)
    p.add_argument('--thermal',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--sequence',required=True)
    p.add_argument('--environment',required=True)
    p.add_argument('--max-frames',type=int,default=300)
    a=p.parse_args()
    if a.max_frames<1 or a.output.exists():p.error('positive frame limit and new output directory required')
    caps=[cv2.VideoCapture(str(path)) for path in [a.rgb,a.thermal]]
    try:
        if not all(c.isOpened() for c in caps):raise ValueError('both videos must open')
        rates=[c.get(cv2.CAP_PROP_FPS) for c in caps]
        if min(rates)<=0 or abs(rates[0]-rates[1])>.01:raise ValueError('constant matching FPS required; resynchronize first')
        a.output.mkdir(parents=True)
        frames=[]
        for i in range(a.max_frames):
            reads=[c.read() for c in caps]
            if not any(ok for ok,_ in reads):break
            if not all(ok for ok,_ in reads):raise ValueError('videos have different lengths')
            images=[im for _,im in reads]
            if images[0].shape[:2]!=images[1].shape[:2]:raise ValueError('register/crop inputs before import')
            row={'id':i,'sequence':a.sequence,'scene':'unannotated','environment':a.environment,
                 'timestamp_s':i/rates[0]}
            for name,im in zip(['rgb','thermal'],images):
                filename=f'{name}_{i:06d}.png'
                if not cv2.imwrite(str(a.output/filename),im):raise OSError('frame write failed')
                row[name]=filename
            frames.append(row)
        if not frames:raise ValueError('empty videos')
        manifest={'schema_version':1,'dataset':a.sequence,'labels':None,'registered':'user_must_verify',
                  'timestamp_kind':'constant_FPS_frame_index_assumed_alignment','source_videos':[str(a.rgb),str(a.thermal)],
                  'frames':frames,'limitations':['Frame indices assumed synchronized; no hardware timestamp verification.',
                                                'Registration must be verified independently.']}
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(a.output/'manifest.json')
    finally:
        for cap in caps:cap.release()


if __name__=='__main__':main()
