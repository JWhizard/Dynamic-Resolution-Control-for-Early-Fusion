"""Export complete Torchvision detection graph after its exact CPU transform.

CNN, proposal/ROI heads, and NMS are in ONNX. Native image preprocessing and
final box-coordinate restoration remain outside the graph, as in the YOLO path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torchvision.models.detection.image_list import ImageList
from experiments.detector_families import Detector


def export_batched_nms(boxes,scores,labels,iou_threshold):
    """Class-aware coordinate NMS without a data-dependent empty If branch.

    A negative-score dummy guarantees a valid ReduceMax input. Removing its
    index preserves original positive-score candidates, including empty input.
    """
    from torchvision.ops import nms
    count=boxes.shape[0]
    padded=torch.cat([boxes,boxes.new_zeros((1,4))],dim=0)
    values=torch.cat([scores,scores.new_full((1,),-1)],dim=0)
    classes=torch.cat([labels,labels.new_zeros((1,))],dim=0)
    offsets=classes.to(padded)*(padded.max()+1)
    keep=nms(padded+offsets[:,None],values,iou_threshold)
    return keep[keep<count]


class FullGraph(torch.nn.Module):
    def __init__(self,model,family):
        super().__init__();self.model=model;self.family=family

    def forward(self,images,image_size):
        sizes=[(image_size[0],image_size[1])]
        batch=ImageList(images,sizes)
        features=self.model.backbone(images)
        if self.family=='fasterrcnn':
            proposals,_=self.model.rpn(batch,features)
            detections,_=self.model.roi_heads(features,proposals,sizes)
        else:
            features=list(features.values())
            outputs=self.model.head(features)
            anchors=self.model.anchor_generator(batch,features)
            counts=[f.shape[-2]*f.shape[-1]*9 for f in features]
            split={k:list(v.split(counts,dim=1)) for k,v in outputs.items()}
            anchor_split=[list(a.split(counts)) for a in anchors]
            detections=self.model.postprocess_detections(split,anchor_split,sizes)
        d=detections[0]
        keep=(d['labels']>0)&(d['scores']>.001)
        result=torch.cat([d['boxes'][keep],d['scores'][keep,None],
                          (d['labels'][keep,None]-1).float()],dim=1)
        # Fixed-size output avoids host-side allocation for data-dependent NMS.
        # Zero-score rows are discarded after GPU inference, preserving all detections.
        return torch.cat([result,result.new_zeros((300,6))],dim=0)[:300].reshape(300,6)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    detector=Detector(a.weights,'cpu');model=detector.model
    graph=FullGraph(model,detector.metadata['family']).eval()
    x=torch.zeros(1,4,384,576);size=torch.tensor([360,576],dtype=torch.int64)
    path=a.output/'full.onnx'
    from torchvision.ops import boxes as box_ops
    original_nms=box_ops.batched_nms
    try:
        box_ops.batched_nms=export_batched_nms
        with torch.inference_mode():
            torch.onnx.export(graph,(x,size),str(path),input_names=['images','image_size'],
                              output_names=['detections'],opset_version=17,dynamo=False,
                              dynamic_axes={'images':{2:'height',3:'width'}})
    finally:
        box_ops.batched_nms=original_nms
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    record=dict(family=detector.metadata['family'],checkpoint=str(a.weights),
                checkpoint_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                export_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input='Torchvision CPU-normalized/resized/padded BGRIR; actual resized image_size',
                output='resized-image xyxy/score/zero-based class; CPU restoration to native coordinates')
    (a.output/'export.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
