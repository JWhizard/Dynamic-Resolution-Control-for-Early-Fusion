"""Build full-model FP32 TensorRT engines on the target Jetson GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--export',type=Path,required=True)
    a=p.parse_args();root=a.export
    if (root/'model.engine').exists():p.error('preserve the existing engine')
    metadata=json.loads((root/'export.json').read_text())
    command=['/usr/src/tensorrt/bin/trtexec','--onnx='+str(root/'full.onnx'),
             '--saveEngine='+str(root/'model.engine'),
             '--minShapes=images:1x4x256x320','--optShapes=images:1x4x384x576',
             '--maxShapes=images:1x4x640x640','--memPoolSize=workspace:1024',
             '--builderOptimizationLevel=2','--noTF32','--skipInference']
    start=time.monotonic()
    with (root/'build.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    metadata.update(command=command,build_seconds=time.monotonic()-start,
                    returncode=result.returncode,precision='fp32_no_tf32')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps(metadata,indent=2))
        raise RuntimeError('TensorRT build failed; inspect '+str(root/'build.log'))
    metadata.update(engine='model.engine',engine_sha256=sha256(root/'model.engine'),
                    backend='full TensorRT CNN/heads/proposals/ROI/NMS; CPU transform/native coordinate restoration')
    (root/'engine.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
