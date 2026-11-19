"""Four-channel Torchvision detector builders and the shared native-box adapter.

These are training/integration tools, not trained model results. Actual training
is restricted to a CUDA device by train_families.py. Input order is B,G,R,IR.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

NAMES=['Bus','Car','Lamp','Motorcycle','People','Truck']


def build(family,pretrained=False):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn,retinanet_resnet50_fpn
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights,RetinaNet_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    if family=='fasterrcnn':
        model=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,weights_backbone=None)
        model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,7)
        model.roi_heads.score_thresh=.001;model.roi_heads.detections_per_img=300
    elif family=='retinanet':
        model=retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None,weights_backbone=None)
        model.head.classification_head=RetinaNetClassificationHead(model.backbone.out_channels,model.anchor_generator.num_anchors_per_location()[0],7)
        model.score_thresh=.001;model.detections_per_img=300
    else:raise ValueError('family must be fasterrcnn or retinanet')
    old=model.backbone.body.conv1
    stem=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    with torch.no_grad():
        stem.weight[:,:3].copy_(old.weight[:,[2,1,0]])
        stem.weight[:,3].zero_()
    model.backbone.body.conv1=stem
    model.transform.image_mean=[.406,.456,.485,.5]
    model.transform.image_std=[.225,.224,.229,.25]
    return model


class Detector:
    def __init__(self,weights,device):
        saved=torch.load(Path(weights),map_location='cpu',weights_only=True)
        if saved.get('channel_order')!='B,G,R,IR' or saved.get('class_names')!=NAMES:raise ValueError('checkpoint channel/class contract differs')
        self.model=build(saved['family'],False);self.model.load_state_dict(saved['model_state']);self.model.to(device).eval()
        self.device=torch.device(device);self.metadata={'family':saved['family'],'channel_order':saved['channel_order'],'class_names':NAMES,'device':str(device)}

    @torch.inference_mode()
    def infer(self,frame,resolution):
        self.model.transform.min_size=(resolution,);self.model.transform.max_size=resolution
        x=torch.from_numpy(np.ascontiguousarray(frame)).permute(2,0,1).to(self.device).float()/255
        result=self.model([x])[0];keep=(result['labels']>0)&(result['scores']>.001)
        return torch.cat([result['boxes'][keep],result['scores'][keep,None],(result['labels'][keep,None]-1).float()],1).cpu().numpy()


def create(weights,device):return Detector(weights,device)
