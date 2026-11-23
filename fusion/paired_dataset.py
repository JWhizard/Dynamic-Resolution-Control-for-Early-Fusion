"""On-demand B,G,R,IR loading for paired JPEG datasets without TIFF copies."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .trainer import FusionDataset, FusionTrainer


class PairedFusionDataset(FusionDataset):
    def load_image(self, i, rect_mode=True):
        # Keep the stock RGB resize/buffer bookkeeping. Only the returned array is
        # extended: mutating self.ims would append an extra IR channel on reuse.
        rgb, original, resized = super().load_image(i, rect_mode)
        path=Path(self.im_files[i])
        thermal=path.parent.parent.parent/'thermal'/path.parent.name/path.name
        ir=cv2.imread(str(thermal),cv2.IMREAD_GRAYSCALE)
        if ir is None or ir.shape!=tuple(original):
            raise ValueError('Missing or mismatched paired thermal image: '+str(thermal))
        if ir.shape!=tuple(resized):
            # The pinned Ultralytics8.4.14 RGB loader uses INTER_LINEAR for
            # both up/downsampling. Match it for the paired thermal channel.
            ir=cv2.resize(ir,(resized[1],resized[0]),interpolation=cv2.INTER_LINEAR)
        if rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Visible source must have three BGR channels')
        return np.ascontiguousarray(np.dstack([rgb,ir])),original,resized


class PairedFusionTrainer(FusionTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        cfg=self.args
        return PairedFusionDataset(img_path=img_path,imgsz=cfg.imgsz,batch_size=batch,
            augment=mode=='train',hyp=cfg,rect=cfg.rect or mode=='val',cache=False,
            single_cls=cfg.single_cls or False,stride=max(int(unwrap_model(self.model).stride.max()),32),
            pad=0.0 if mode=='train' else .5,prefix=colorstr(f'{mode}: '),task=cfg.task,
            classes=cfg.classes,data=self.data,fraction=cfg.fraction if mode=='train' else 1.0)


def make_paired_trainer(sizes):
    return type('PairedFusionTrainerMS',(PairedFusionTrainer,),{'ms_sizes':tuple(sizes)})
