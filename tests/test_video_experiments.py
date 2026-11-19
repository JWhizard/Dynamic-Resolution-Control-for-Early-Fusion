import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.power import RailSampler
from experiments.policies import Policy
from experiments.run import perturb, stats


def test_power_units_and_nonoverlapping_rails(tmp_path):
    hw = tmp_path/'hwmon0';hw.mkdir()
    for name,value in {'name':'ina3221','in1_label':'VDD_CPU_CV','in1_input':'20000',
                       'curr1_input':'500','in7_label':'sum of shunt voltages','in7_input':'1200'}.items():
        (hw/name).write_text(value)
    sampler = RailSampler(tmp_path)
    assert sampler.sample()['watts'] == {'VDD_CPU_CV':10.0}
    sampler.samples = [{'time':0.,'watts':{'VDD_CPU_CV':10.}}, {'time':2.,'watts':{'VDD_CPU_CV':20.}}]
    assert sampler.energy(.5,1.5)['VDD_CPU_CV'] == pytest.approx(15.)
    assert sampler.energy(-1,1)['VDD_CPU_CV'] is None


def test_aimd_downshifts_and_requires_sustained_headroom():
    p=Policy('aimd',budget_ms=100)
    assert p.choose('frame-0')==576
    p.update('frame-0',[],576,150)
    assert p.choose('frame-1')==544
    for i in range(3):p.update(f'frame-{i}',[],544,50)
    assert p.choose('frame-4')==544
    p.update('frame-4',[],544,50)
    assert p.choose('frame-5')==576


def test_deadline_exploration_and_contention_response():
    p=Policy('deadline',budget_ms=120)
    for i,r in enumerate([480,544,576,640]):
        assert p.choose(f'frame-{i}')==r
        p.update(f'frame-{i}',[],r,60*(r/480)**2)
    assert p.choose('frame-4')==640
    for i in range(20):
        r=p.choose(f'frame-{i+4}');p.update(f'frame-{i+4}',[],r,300*(r/480)**2)
    assert p.choose('frame-30')==480


def test_seeded_random_and_switch_pattern():
    a,b=Policy('random',seed=42),Policy('random',seed=42)
    assert [a.choose(str(i)) for i in range(20)]==[b.choose(str(i)) for i in range(20)]
    p=Policy('switch')
    assert [p.choose(f'frame-{i}') for i in range(5)]==[480,544,576,640,480]


def test_corruption_inputs_and_p99():
    rgb=np.full((32,32,3),100,np.uint8);ir=rgb.copy()
    dark,_=perturb(rgb,ir,'rgb_dark');assert dark[0,0,0]==20
    _,drop=perturb(rgb,ir,'ir_dropout');assert drop.max()==0
    assert rgb.min()==100 and ir.min()==100
    assert stats([1,2,3])['p99']==pytest.approx(2.98)


def test_transition_settling_is_censored_at_next_change():
    from experiments.transitions import transitions
    rows=[{'phase':0 if i<2 else 1,'sequence':'a','resolution':480 if i<3 else 640,
           'budget_miss':i<3,'start':float(i),'end':i+.1,'frame_index':i} for i in range(9)]
    event=transitions(rows,'phase')[0]
    assert event['frame_index']==2 and event['budget_settled']
    assert event['budget_settle_ms']==pytest.approx(5100)
    assert event['time_to_first_resolution_change_ms']==pytest.approx(1100)
    short=transitions(rows[:5],'phase')[0]
    assert not short['budget_settled'] and short['budget_settle_ms'] is None


def test_unlabeled_data_cannot_produce_accuracy(tmp_path,monkeypatch):
    import json
    from experiments.score import main
    manifest=tmp_path/'manifest.json';result=tmp_path/'result.json';output=tmp_path/'accuracy.json'
    manifest.write_text(json.dumps({'frames':[]}));result.write_text('{}')
    monkeypatch.setattr(sys,'argv',['score','--manifest',str(manifest),'--result',str(result),'--output',str(output)])
    with pytest.raises(ValueError,match='Labeled COCO annotations'):
        main()
    assert not output.exists()


def test_paired_video_import_preserves_order(tmp_path,monkeypatch):
    import cv2,json
    from experiments.import_video import main
    videos=[]
    for sensor in ['rgb','thermal']:
        filename=tmp_path/(sensor+'.avi')
        writer=cv2.VideoWriter(str(filename),cv2.VideoWriter_fourcc(*'MJPG'),8,(32,24))
        assert writer.isOpened()
        for value in [40,80,120]:writer.write(np.full((24,32,3),value,np.uint8))
        writer.release();videos.append(filename)
    output=tmp_path/'imported'
    monkeypatch.setattr(sys,'argv',['import_video','--rgb',str(videos[0]),'--thermal',str(videos[1]),
                                  '--output',str(output),'--sequence','unit-fixture','--environment','synthetic-unit-fixture'])
    main()
    manifest=json.loads((output/'manifest.json').read_text())
    assert [f['timestamp_s'] for f in manifest['frames']]==[0,.125,.25]
    assert [f['id'] for f in manifest['frames']]==[0,1,2]
    assert cv2.imread(str(output/manifest['frames'][2]['rgb'])).mean()>110


def test_coco_scorer_known_perfect_and_empty_predictions(tmp_path,monkeypatch):
    pytest.importorskip('pycocotools')
    import cv2,json,hashlib
    from experiments.score import main
    cv2.imwrite(str(tmp_path/'frame.png'),np.zeros((16,16,3),np.uint8))
    gt={'info':{'description':'synthetic unit fixture'},'images':[{'id':7,'width':16,'height':16,'file_name':'frame.png'}],
        'categories':[{'id':1,'name':'person'}],
        'annotations':[{'id':1,'image_id':7,'category_id':1,'bbox':[1,1,8,8],'area':64,'iscrowd':0}]}
    (tmp_path/'gt.json').write_text(json.dumps(gt))
    manifest={'coco_annotations':'gt.json','class_to_category':{'0':1},
              'frames':[{'id':7,'rgb':'frame.png','environment':'synthetic unit fixture'}]}
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    result={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'input_frames':1,
            'rows':[{'image_id':7,'detections':[[1,1,9,9,.99,0]]}]}
    result_path=tmp_path/'result.json';output=tmp_path/'accuracy.json'
    monkeypatch.setattr(sys,'argv',['score','--manifest',str(manifest_path),'--result',str(result_path),'--output',str(output)])
    result_path.write_text(json.dumps(result));main()
    assert json.loads(output.read_text())['overall']['AP']==pytest.approx(1.)
    result['rows'][0]['detections']=[];result_path.write_text(json.dumps(result));main()
    assert json.loads(output.read_text())['overall']['AP']==pytest.approx(0.)
