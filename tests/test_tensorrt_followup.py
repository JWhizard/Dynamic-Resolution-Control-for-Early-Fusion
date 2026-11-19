import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.validate_tensorrt import match
from experiments.resources import start_workers,stop_workers


def test_detection_comparison_matches_permuted_rows():
    reference=np.array([[0,0,10,10,.9,4],[20,20,40,40,.8,1]])
    assert match(reference,reference[::-1])['passed']
    bad=reference.copy();bad[0,5]=0
    assert not match(reference,bad)['passed']
    missing=reference[:1]
    assert not match(reference,missing)['passed']


def test_owned_cpu_worker_is_ready_and_stops():
    cores=sorted(os.sched_getaffinity(0))
    workers=start_workers('cpu',1,cores)
    try:assert workers[0].poll() is None
    finally:stop_workers(workers)
    assert workers[0].poll() is not None


def test_endurance_cannot_be_scored_as_accuracy(tmp_path,monkeypatch):
    from experiments.score import main
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    result=tmp_path/'result.json';result.write_text(json.dumps({'arguments':{'cycles':2}}))
    monkeypatch.setattr(sys,'argv',['score','--manifest',str(manifest),'--result',str(result),'--output',str(tmp_path/'score.json')])
    with pytest.raises(ValueError,match='endurance cycles'):main()


def test_streaming_never_uses_future_or_other_sequence_predictions():
    from experiments.streaming import align_predictions
    frames=[{'id':i,'sequence':'a' if i<3 else 'b','timestamp_s':i*.05} for i in range(4)]
    rows=[{'frame_index':0,'sequence':'a','end':100.06,'output_age_ms':60.,'detections':[[0,0,1,1,.9,4]]}]
    aligned=align_predictions(frames,rows)
    assert [r['image_id'] for r in aligned]==[2]
    assert aligned[0]['source_frame_index']==0


def test_guarded_budget_cap_and_hysteresis(tmp_path):
    from experiments.policies import Policy
    controller=tmp_path/'controller.json'
    controller.write_text(json.dumps({'tree':{'leaf':640},'actions':[480,544,576,640],
                                      'default_res':640,'interval':1,'reset_on_manifest_group':False}))
    p=Policy('guarded',budget_ms=50,controller=controller)
    p.cost={480:20,544:30,576:40,640:60}
    assert p.choose('frame-0')==576
    p.update('frame-0',np.empty((0,6)),576,40)
    p.cost[640]=40
    assert p.choose('frame-1')==576
    for i in range(3):p.update(f'frame-{i+1}',np.empty((0,6)),576,40)
    assert p.choose('frame-5')==640
    p.cost={480:30,544:60,576:80,640:100}
    assert p.choose('frame-6')==480


def test_guarded_scene_and_resource_ablations(tmp_path):
    from experiments.policies import Policy
    controller=tmp_path/'controller.json'
    controller.write_text(json.dumps({'tree':{'leaf':480},'actions':[480,544,576,640],
                                      'default_res':480,'interval':1,'reset_on_manifest_group':False}))
    no_scene=Policy('guarded-no-scene',budget_ms=50,controller=controller)
    no_scene.cost={480:10,544:15,576:20,640:25}
    assert no_scene.sequence is None and no_scene.choose('frame-0')==640
    no_resource=Policy('guarded-no-resource',budget_ms=1,controller=controller)
    no_resource.cost={480:999,544:999,576:999,640:999}
    assert no_resource.choose('frame-0')==480
    no_resource.update('frame-0',np.empty((0,6)),480,999)
    assert no_resource.cost[480]==999


def test_additional_family_four_channel_construction():
    import torch
    from experiments.detector_families import build
    for family in ['fasterrcnn','retinanet']:
        model=build(family,pretrained=False)
        assert model.backbone.body.conv1.in_channels==4
        assert torch.count_nonzero(model.backbone.body.conv1.weight[:,3])==0
        assert model.transform.image_mean==[.406,.456,.485,.5]
        if family=='fasterrcnn':assert model.roi_heads.box_predictor.cls_score.out_features==7
        else:assert model.head.classification_head.num_classes==7


def test_training_dataset_channel_and_label_contract(tmp_path):
    import cv2
    from experiments.train_families import PairedDataset
    for directory in ['Vis','Ir','labels_yolo']:(tmp_path/directory).mkdir()
    image=np.full((32,32,3),[10,20,30],np.uint8)
    cv2.imwrite(str(tmp_path/'Vis/00000.png'),image)
    cv2.imwrite(str(tmp_path/'Ir/00000.png'),np.full((32,32),80,np.uint8))
    (tmp_path/'labels_yolo/00000.txt').write_text('4 0.5 0.5 0.5 0.5\n')
    frame,target=PairedDataset(tmp_path,['00000'])[0]
    assert np.allclose(frame[:,0,0].numpy(),np.array([10,20,30,80])/255)
    assert target['labels'].tolist()==[5]
    assert target['boxes'].tolist()==[[8,8,24,24]]


def test_training_rejects_cpu_before_creating_output(tmp_path,monkeypatch):
    from experiments.train_families import main
    output=tmp_path/'training'
    monkeypatch.setattr(sys,'argv',['train','--family','retinanet','--data-root',str(tmp_path),'--device','cpu','--output',str(output)])
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==2 and not output.exists()


def test_group_split_validation_rejects_shared_groups():
    from experiments.train_families import validate_splits
    spec={'splits':{'train':['a'],'val':['b'],'test':['c']},'groups':{'a':'g0','b':'g1','c':'g2'}}
    assert validate_splits(spec)==spec['splits']
    spec['groups']['b']='g0'
    with pytest.raises(ValueError,match='group leakage'):validate_splits(spec)


def test_content_transitions_require_stability_and_do_not_cross_sequences():
    from experiments.content_transitions import presence_events
    frames=[{'id':i,'sequence':'a','timestamp_s':i/20} for i in range(36)]
    anns=[{'image_id':i,'iscrowd':0} for i in range(12,24)]
    assert [e['frame_index'] for e in presence_events(frames,anns)]==[12,24]
    for row in frames[24:]:row['sequence']='b'
    assert [e['frame_index'] for e in presence_events(frames,anns)]==[12]
