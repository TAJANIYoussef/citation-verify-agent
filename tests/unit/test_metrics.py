"""Unit tests for eval/metrics.py — pure Python, no Ollama."""
import pytest
from citverif.eval.metrics import compute_metrics, format_metrics_table, ALL_VERDICTS


def test_perfect_predictions():
    labels = ["supported", "unsupported", "misleading", "partially_supported", "unverifiable"]
    m = compute_metrics(labels, labels)
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.accuracy == pytest.approx(1.0)
    assert m.hallucination_rate == pytest.approx(0.0)


def test_all_wrong_predictions():
    expected  = ["supported",   "supported",   "unsupported"]
    predicted = ["unsupported", "unsupported", "supported"]
    m = compute_metrics(expected, predicted)
    assert m.accuracy == pytest.approx(0.0)


def test_hallucination_rate_counts_false_supported():
    expected  = ["unsupported", "misleading", "supported"]
    predicted = ["supported",   "supported",  "supported"]
    m = compute_metrics(expected, predicted)
    # 2 out of 3 are false-positive "supported"
    assert m.hallucination_rate == pytest.approx(2 / 3)


def test_hallucination_rate_zero_when_correct():
    expected  = ["supported", "unsupported"]
    predicted = ["supported", "unsupported"]
    m = compute_metrics(expected, predicted)
    assert m.hallucination_rate == pytest.approx(0.0)


def test_per_class_f1_supported():
    expected  = ["supported", "supported", "unsupported"]
    predicted = ["supported", "unsupported", "unsupported"]
    m = compute_metrics(expected, predicted)
    # supported: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1=0.667
    assert m.per_class["supported"].f1 == pytest.approx(2/3, abs=0.01)
    assert m.per_class["supported"].support == 2


def test_macro_f1_averages_all_five_classes():
    # Only "supported" appears — other 4 classes have F1=0
    expected  = ["supported", "supported"]
    predicted = ["supported", "supported"]
    m = compute_metrics(expected, predicted)
    # supported F1=1.0, others=0.0 → macro = 1/5 = 0.2
    assert m.macro_f1 == pytest.approx(0.2)


def test_resolution_rate():
    expected  = ["supported", "supported", "unverifiable"]
    predicted = ["supported", "supported", "unverifiable"]
    flags = [True, True, False]
    m = compute_metrics(expected, predicted, resolved_flags=flags)
    assert m.resolution_rate == pytest.approx(2 / 3)


def test_mean_latency():
    expected  = ["supported"]
    predicted = ["supported"]
    m = compute_metrics(expected, predicted, latencies_s=[4.2])
    assert m.mean_latency_s == pytest.approx(4.2)


def test_empty_input_returns_zero_metrics():
    m = compute_metrics([], [])
    assert m.macro_f1 == 0.0
    assert m.total == 0


def test_format_metrics_table_contains_all_verdicts():
    expected  = ["supported", "unsupported"]
    predicted = ["supported", "unsupported"]
    m = compute_metrics(expected, predicted)
    table = format_metrics_table(m, mode="react")
    for v in ALL_VERDICTS:
        assert v in table
    assert "Macro-F1" in table
    assert "Hallucination" in table


def test_mismatched_lengths_raises():
    with pytest.raises(AssertionError):
        compute_metrics(["supported"], ["supported", "unsupported"])
