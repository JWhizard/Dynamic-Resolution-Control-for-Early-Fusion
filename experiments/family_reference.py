"""Generate original PyTorch predictions from frozen same-input arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.detector_families import Detector
from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('preserve existing reference')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    d=Detector(a.weights,'cpu');inputs=np.load(a.inputs,allow_pickle=False);rows=[]
    for key in inputs.files:
        for res in [480,544,576,640]:
            rows.append(dict(key=key,resolution=res,detections=d.infer(inputs[key],res).tolist()))
        print('Reference completed',key,flush=True)
    result=dict(checkpoint_sha256=sha256(a.weights),inputs_sha256=sha256(a.inputs),
                family=d.metadata['family'],torch=torch.__version__,execution_device='workstation CPU',
                source_sha256=sha256(__file__),rows=rows)
    a.output.write_text(json.dumps(result))


if __name__=='__main__':main()
