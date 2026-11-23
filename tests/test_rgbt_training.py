import cv2
import numpy as np
from ultralytics.cfg import get_cfg

from experiments.prepare_rgbt_training import prefix,choose_split
from experiments.evaluate_rgbt_development import feedback,person_recall
from fusion.paired_dataset import PairedFusionDataset


def test_capture_prefixes_are_indivisible_and_split_is_reproducible():
    assert prefix('DJI_0028_3')==prefix('DJI_0028_5')
    assert prefix('DJI_0028_3')!=prefix('DJI_0029_3')
    groups={f'g{i}':dict(frames=100+i,classes=[1+i]*7) for i in range(12)}
    split=choose_split(groups,trials=100)
    assert split==choose_split(groups,trials=100)
    flattened=[g for values in split.values() for g in values]
    assert len(flattened)==len(set(flattened))==len(groups)
    assert set(flattened)==set(groups)


def test_paired_resize_and_rgb_cache_never_mix_or_duplicate_the_ir_channel(tmp_path):
    root=tmp_path/'data'
    for directory in ['images/train','thermal/train','labels/train']:(root/directory).mkdir(parents=True)
    rgb=np.zeros((80,100,3),dtype=np.uint8);rgb[:]=[20,80,170]
    ir=np.arange(8000,dtype=np.uint16).reshape(80,100).astype(np.uint8)
    cv2.imwrite(str(root/'images/train/seq__00000.jpg'),rgb)
    original_rgb=tmp_path/'visible_original.jpg'
    (root/'images/train/seq__00000.jpg').rename(original_rgb)
    (root/'images/train/seq__00000.jpg').symlink_to(original_rgb)
    cv2.imwrite(str(root/'thermal/train/seq__00000.jpg'),ir)
    (root/'labels/train/seq__00000.txt').write_text('0 .5 .5 .2 .2\n')
    hyp=get_cfg();hyp.mosaic=0;hyp.mixup=0;hyp.copy_paste=0
    dataset=PairedFusionDataset(img_path=str(root/'images/train'),imgsz=32,batch_size=1,
        augment=True,hyp=hyp,rect=False,cache=False,data=dict(channels=4,names={i:str(i) for i in range(7)},nc=7))
    first,original,resized=dataset.load_image(0);again,_,_=dataset.load_image(0)
    expected=cv2.resize(cv2.imread(str(root/'thermal/train/seq__00000.jpg'),0),(resized[1],resized[0]),interpolation=cv2.INTER_LINEAR)
    assert first.shape[2]==again.shape[2]==4
    assert np.array_equal(first[...,3],expected)
    assert np.array_equal(first,again)
    assert dataset.ims[0].shape[2]==3


def test_target_pedestrian_and_bus_feedback_matches_source_controller_class_contract():
    det=np.array([[0,0,10,10,.9,c] for c in range(7)],dtype=float)
    mapped=feedback(det)
    assert list(mapped[:,5])==[1,4,0]  # car, person, bus in target order
    assert np.array_equal(det[:,5],np.arange(7))


def test_person_recall_obeys_confidence_threshold_with_raw_target_category_three():
    gt=dict(info={},images=[dict(id=1,width=40,height=40)],categories=[dict(id=3,name='pedestrian')],
        annotations=[dict(id=1,image_id=1,category_id=3,bbox=[1,1,10,10],area=100,iscrowd=0)])
    pred=dict(image_id=1,category_id=3,bbox=[1,1,10,10],score=.25)
    assert person_recall(gt,[pred])['small_person_recall50_conf25']==1
    assert person_recall(gt,[dict(pred,score=.24)])['person_recall50_conf25']==0
