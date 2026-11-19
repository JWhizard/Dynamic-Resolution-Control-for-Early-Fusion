import numpy as np
import pytest
import torch
from torchvision.ops import batched_nms
from experiments.export_family_full import export_batched_nms


@pytest.mark.parametrize('count',[0,1,50,1200])
def test_export_nms_preserves_empty_and_multiclass_candidates(count):
    generator=torch.Generator().manual_seed(18)
    start=torch.rand(count,2,generator=generator)*400
    size=torch.rand(count,2,generator=generator)*90+1
    boxes=torch.cat([start,start+size],dim=1)
    scores=torch.rand(count,generator=generator)*.999+.001
    classes=torch.randint(0,7,(count,),generator=generator)
    assert torch.equal(export_batched_nms(boxes,scores,classes,.5),batched_nms(boxes,scores,classes,.5))


def test_family_preprocess_matches_training_model_transform():
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from experiments.detector_families import build
    model=build('retinanet',False).eval()
    transform=GeneralizedRCNNTransform(576,576,[.406,.456,.485,.5],[.225,.224,.229,.25]).eval()
    frame=torch.from_numpy(np.random.default_rng(1).integers(0,256,(4,405,711),dtype=np.uint8)).float()/255
    for res in [480,544,576,640]:
        model.transform.min_size=(res,);model.transform.max_size=res
        transform.min_size=(res,);transform.max_size=res
        expected,_=model.transform([frame],None);actual,_=transform([frame],None)
        assert expected.image_sizes==actual.image_sizes
        assert torch.equal(expected.tensors,actual.tensors)


def test_zero_area_diagnostic_uses_coordinates_without_relaxing_positive_iou():
    from experiments.validate_family_trt import match_degenerate
    ref=np.array([[10,20,10,40,.05,4],[0,0,20,20,.8,1]],dtype=float)
    assert match_degenerate(ref,ref.copy(),.01)['passed']
    changed=ref.copy();changed[0,1]+=.1
    assert not match_degenerate(ref,changed,.01)['passed']
    changed=ref.copy();changed[1,:4]=[50,50,70,70]
    assert not match_degenerate(ref,changed,.01)['passed']


def test_benchmark_rejects_failed_gate_before_creating_output(tmp_path,monkeypatch):
    import json,sys
    from experiments.benchmark_family_trt import main
    from experiments.evaluate_families import sha256
    engine=tmp_path/'model.engine';engine.write_bytes(b'fixture')
    config=tmp_path/'engine.json';config.write_text(json.dumps(dict(engine=engine.name,engine_sha256=sha256(engine))))
    gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(passed=False)))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(models=[dict(engine=str(config),validation=str(gate))])))
    output=tmp_path/'output';monkeypatch.setattr(sys,'argv',['benchmark','--plan',str(plan),'--output',str(output)])
    with pytest.raises(ValueError,match='failed numerical gate'):main()
    assert not output.exists()


def test_zero_area_numerical_match_requires_class_and_coordinates():
    from experiments.validate_family_trt import match_degenerate
    a=np.array([[10,0,10,20,.02,4]],dtype=np.float32)
    assert match_degenerate(a,a.copy(),.01)['passed']
    b=a.copy();b[0,3]+=.02
    assert not match_degenerate(a,b,.01)['passed']
    b=a.copy();b[0,5]=3
    assert not match_degenerate(a,b,.01)['passed']
