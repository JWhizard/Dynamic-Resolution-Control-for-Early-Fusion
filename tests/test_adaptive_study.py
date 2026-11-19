import json
import numpy as np
import pytest
from experiments.adaptive_study import configure_policy, summary


def test_scene_history_resets_at_group_boundary(tmp_path):
    path=tmp_path/'controller.json'
    path.write_text(json.dumps(dict(tree={'leaf':480},interval=1,default_res=576,
                                   actions=[480,544,576,640],safety_small_people=False)))
    p=configure_policy('sequence',25,path,{'0':'a','1':'a','2':'b'}, {})
    assert p.choose('0')==576
    p.update('0',np.empty((0,6)),576,10)
    assert p.choose('1')==480
    p.update('1',np.empty((0,6)),480,10)
    assert p.choose('2')==576


def test_guarded_uses_pretest_profiles_and_summary_counts(tmp_path):
    path=tmp_path/'controller.json'
    path.write_text(json.dumps(dict(tree={'leaf':640},default_res=576)))
    p=configure_policy('guarded',25,path,{'0':'a'}, {480:10,544:15,576:30,640:40})
    assert p.choose('0')==544
    assert summary([1,2,3])['n']==3


def make_report_case(root):
    row=dict(model='example',policy='sequence',frames=2,resolution_mix={'576':2},
             power_read_errors=[],named_rail_sum_J_per_frame=1,rail_joules={'rail':2})
    (root/'COMPLETED.json').write_text(json.dumps(dict(jobs=1,processed_frames=2)))
    (root/'plan.json').write_text(json.dumps(dict(device='cpu')))
    (root/'summary.json').write_text(json.dumps([row]))
    raw=root/'example_sequence.json'
    raw.write_text(json.dumps(dict(rows=[{'image_id':0},{'image_id':1}])))
    return row,raw


def test_report_rejects_repeated_images_instead_of_counting_as_independent(tmp_path):
    from experiments.report_adaptive import inspect
    _,raw=make_report_case(tmp_path)
    assert len(inspect(tmp_path,1,2)[0])==1
    raw.write_text(json.dumps(dict(rows=[{'image_id':0},{'image_id':0}])))
    with pytest.raises(AssertionError):inspect(tmp_path,1,2)


def test_report_rejects_uncovered_energy_window(tmp_path):
    from experiments.report_adaptive import inspect
    row,_=make_report_case(tmp_path)
    row['rail_joules']['rail']=None
    (tmp_path/'summary.json').write_text(json.dumps([row]))
    with pytest.raises(AssertionError):inspect(tmp_path,1,2)
