from experiments.compare_rgbt_tiny import common_gt, common_predictions, annotation_stats, TINY_MAP


def test_common_categories_keep_negative_images_and_never_map_cyclist_to_motorcycle():
    gt=dict(images=[dict(id=1),dict(id=2)],categories=[dict(id=i,name=n) for i,n in enumerate(['ship','car','cyclist','pedestrian','bus','drone','plane'])],
        annotations=[dict(id=i,image_id=1,category_id=i,bbox=[1,2,10,10],area=100,iscrowd=0) for i in range(7)])
    mapped=common_gt(gt,TINY_MAP)
    assert len(mapped['images'])==2
    assert {a['category_id'] for a in mapped['annotations']}=={1,2,5}
    assert [a['id'] for a in mapped['annotations']]==[1,3,4]
    assert annotation_stats(mapped)['boxes']==3
    assert len(common_predictions([dict(category_id=i) for i in range(1,8)]))==3


def test_ignore_conversion_preserves_native_geometry():
    gt=dict(images=[dict(id=1)],categories=[dict(id=3,name='pedestrian')],
        annotations=[dict(id=1,image_id=1,category_id=3,bbox=[-1,2,10,10],area=100,iscrowd=0,ignore=1)])
    mapped=common_gt(gt,TINY_MAP)
    assert mapped['annotations'][0]['iscrowd']==1
    assert mapped['annotations'][0]['bbox']==[-1,2,10,10]
    assert gt['annotations'][0]['iscrowd']==0


def test_shared_class_coco_evaluator_handles_noncontiguous_category_ids():
    from experiments.evaluate_families import score
    gt=dict(info={},images=[dict(id=1,width=40,height=40)],
        categories=[dict(id=i,name=n) for i,n in [(1,'bus'),(2,'car'),(5,'person')]],
        annotations=[dict(id=i,image_id=1,category_id=i,bbox=[i,1,10,10],area=100,iscrowd=0) for i in [1,2,5]])
    predictions=[dict(image_id=1,category_id=i,bbox=[i,1,10,10],score=.99) for i in [1,2,5]]
    result=score(gt,predictions)
    assert result['AP']>.99
    assert set(result['AP_per_class'])=={'bus','car','person'}
