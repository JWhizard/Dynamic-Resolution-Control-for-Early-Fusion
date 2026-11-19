import pytest
import xml.etree.ElementTree as ET
from experiments.kaist_selection import select
from experiments.prepare_kaist_video import convert_objects


def example_splits():
    day=[f'set{s:02d}/V000/I{i:05d}' for s in range(6,9) for i in range(3)]
    night=[f'set{s:02d}/V000/I{i:05d}' for s in range(9,12) for i in range(3)]
    return day+night,day,night,['set00/V000/I00000']


def test_selection_is_dense_balanced_and_never_crosses_train():
    args=example_splits();chosen=select(*args,length=2)
    assert len(chosen)==6
    assert [x['environment'] for x in chosen]==['day']*3+['night']*3
    assert all(x['start_frame']==0 and x['end_frame']==1 for x in chosen)
    with pytest.raises(ValueError,match='partition'):select(args[0],args[1],args[2],[args[0][0]],2)


def test_sparse_detection_benchmark_cannot_be_called_twenty_hz_video():
    args=example_splits();all_frames=[x for x in args[0] if not x.endswith('I00001')]
    day=[x for x in all_frames if int(x[3:5])<9];night=[x for x in all_frames if int(x[3:5])>=9]
    with pytest.raises(ValueError,match='sparse'):select(all_frames,day,night,args[3],2)


def test_full_sequence_selection_preserves_variable_lengths():
    test,day,night,train=example_splits()
    day=day+['set06/V000/I00003']
    chosen=select(day+night,day,night,train,length=None)
    assert [len(x['frames']) for x in chosen]==[4,3,3,3,3,3]
    assert chosen[0]['end_frame']==3
    with pytest.raises(ValueError,match='positive'):select(test,*example_splits()[1:],length=0)


def test_only_clear_person_labels_are_scored_and_other_humans_ignored():
    root=ET.Element('annotation')
    for name,difficult in [('person',0),('people',0),('person?',0),('cyclist',0),('person',1)]:
        obj=ET.SubElement(root,'object');ET.SubElement(obj,'name').text=name
        ET.SubElement(obj,'difficult').text=str(difficult);box=ET.SubElement(obj,'bndbox')
        for k,v in [('x',-5),('y',10),('w',20),('h',40)]:ET.SubElement(box,k).text=str(v)
    rows=convert_objects(root,12,640,512,1)
    assert [x['iscrowd'] for x in rows]==[0,1,1,1,1]
    assert rows[0]['bbox']==[0,10,15,40]
    assert {x['image_id'] for x in rows}=={12}


def test_person_scoring_drops_vehicle_classes_and_keeps_global_source_ids():
    from experiments.report_kaist_video import person_predictions
    rows=[dict(image_id=1203,detections=[[0,0,10,10,.9,0],[2,3,12,23,.2,4]])]
    assert person_predictions(rows)==[dict(image_id=1203,category_id=1,bbox=[2,3,10,20],score=.2)]


def test_recovery_is_not_reported_for_an_infeasible_service_budget():
    from experiments.report_kaist_video import restore_recovery
    rows=[dict(phase=2,start=20+i*.2,end=20.2+i*.2,service_ms=200) for i in range(4)]
    assert restore_recovery(rows,100) is None
    for i in range(1,4):rows[i]['service_ms']=80
    assert restore_recovery(rows,100)==pytest.approx(800)


def test_pooled_gt_subsets_do_not_reassign_independent_sequence_ids():
    from experiments.report_kaist_video import subset_gt
    gt=dict(images=[dict(id=0),dict(id=600)],annotations=[dict(id=1,image_id=0),dict(id=2,image_id=600)],categories=[dict(id=1,name='person')])
    subset=subset_gt(gt,[600])
    assert subset['images']==[dict(id=600)]
    assert subset['annotations']==[dict(id=2,image_id=600)]
