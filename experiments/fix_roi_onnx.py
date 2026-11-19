"""Make known classifier dimensions explicit for empty Faster R-CNN ROI batches.

Replacing [N,-1] with [N,28] and [N,-1,4] with [N,7,4] preserves
the seven-class bbox coder while removing ambiguous zero-element reshapes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    model=onnx.load(str(a.export/'full.onnx'));nodes={n.name:n for n in model.graph.node}
    for name,value in [('/roi_heads/Constant_2',28),('/roi_heads/Constant_37',7)]:
        node=nodes[name]
        attr=next(x for x in node.attribute if x.name=='value')
        if numpy_helper.to_array(attr.t).tolist()!=[-1]:raise ValueError('unexpected exported graph')
        attr.t.CopyFrom(numpy_helper.from_array(np.array([value],np.int64)))
    onnx.checker.check_model(model);onnx.save(model,str(a.output/'full.onnx'))
    metadata=json.loads((a.export/'export.json').read_text())
    metadata.update(original_onnx_sha256=metadata['onnx_sha256'],
        onnx_sha256=sha256(a.output/'full.onnx'),
        graph_fix='Known seven-class bbox coder dimensions for zero-element ROI reshape; no weights/threshold changes',
        graph_fix_source_sha256=sha256(__file__))
    (a.output/'export.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
