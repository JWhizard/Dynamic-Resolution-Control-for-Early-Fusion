import pytest

from experiments.analyze_long_kaist import aggregate_cases, pair_cases, paired_bootstrap


def case(sequence,n,processed,ap,service,energy):
    return dict(sequence=sequence,source_frames=n,processed_frames=processed,
        AP_causal=ap,AP_missing_empty=ap/2,AP50_causal=ap,AP_small_causal=ap,
        person_recall50_conf25=ap,initial_unavailable_frames=1,
        mean_service_ms=service,mean_controller_ms=1,mean_completion_age_ms=service+10,
        p95_completion_age_ms=service+20,budget_misses=0,named_rail_J=energy,
        dropped_frames=n-processed,switches=1)


def test_unequal_sequences_do_not_weight_accuracy_like_cost_or_drops():
    result=aggregate_cases([case('a',100,100,.8,10,50),case('b',300,100,.2,30,150)])
    assert result['macro_causal_AP']==pytest.approx(.5)
    assert result['mean_service_ms']==pytest.approx(20)
    assert result['dropped_fraction']==pytest.approx(.5)
    assert result['J_per_source_frame']==pytest.approx(.5)
    assert result['J_per_completed_inference']==pytest.approx(1)


def test_pairing_rejects_missing_or_duplicate_sequence_units():
    a=case('a',100,100,.8,10,50);b=case('b',100,100,.8,10,50)
    with pytest.raises(ValueError,match='same unique sequences'):pair_cases([a],[b])
    with pytest.raises(ValueError,match='same unique sequences'):pair_cases([a,a],[a,a])
    assert pair_cases([b,a],[a,b])==[(a,a),(b,b)]


def test_identical_policies_have_zero_paired_uncertainty_and_reordering_is_safe():
    a=case('a',100,100,.8,10,50);b=case('b',300,100,.2,30,150)
    result=paired_bootstrap([a,b],[b,a])
    assert set(result.values())=={0.0}
