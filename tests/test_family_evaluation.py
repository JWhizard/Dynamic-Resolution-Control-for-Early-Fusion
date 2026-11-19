import pytest
from experiments.evaluate_families import annotations, score
from experiments.detector_families import NAMES


def test_annotations_use_actual_native_size_and_retain_original_coordinates():
    rows = annotations('4 .5 .5 .2 .4\n', 7, 400, 280, 1)
    assert rows[0]['category_id'] == 5
    assert rows[0]['bbox'] == [160, 84, 80, 112]
    assert rows[0]['area'] == 8960
    assert annotations('0 0 0 .5 .5', 0, 100, 100, 1)[0]['bbox'][0] == -25
    with pytest.raises(ValueError):
        annotations('6 .5 .5 .2 .2', 0, 100, 100, 1)


def test_common_coco_score_perfect_person_small_box_and_empty_predictions():
    gt = dict(info={}, images=[dict(id=0, width=100, height=100)],
              annotations=annotations('4 .5 .5 .2 .2', 0, 100, 100, 1),
              categories=[dict(id=i+1, name=n) for i, n in enumerate(NAMES)])
    prediction = dict(image_id=0, category_id=5, bbox=[40, 40, 20, 20], score=.9)
    result = score(gt, [prediction])
    assert result['AP'] == pytest.approx(1)
    assert result['AP_small'] == pytest.approx(1)
    assert result['AP_per_class']['People'] == pytest.approx(1)
    assert result['AP_per_class']['Car'] is None
    assert score(gt, [])['AP'] == 0
    prediction['category_id'] = 2
    assert score(gt, [prediction])['AP'] == 0


def test_coco_single_category_does_not_assume_six_classes_or_consecutive_ids():
    gt=dict(info={},images=[dict(id=0,width=100,height=100)],
            annotations=[dict(id=1,image_id=0,category_id=17,bbox=[10,10,20,20],area=400,iscrowd=0)],
            categories=[dict(id=17,name='person')])
    result=score(gt,[dict(image_id=0,category_id=17,bbox=[10,10,20,20],score=.9)])
    assert result['AP']==pytest.approx(1)
    assert result['AP_per_class']=={'person':pytest.approx(1)}
