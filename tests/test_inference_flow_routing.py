from app.flows.inference import uses_in_process_cohort_features


def test_cohort_models_keep_raw_extracts_inside_one_prefect_task() -> None:
    assert uses_in_process_cohort_features("cohort")
    assert uses_in_process_cohort_features("cohort_tier_lift")
    assert uses_in_process_cohort_features("playerscore")


def test_other_models_keep_separate_extract_and_feature_tasks() -> None:
    assert not uses_in_process_cohort_features("predicted_fills")
    assert not uses_in_process_cohort_features("ShoeAdvantage")
