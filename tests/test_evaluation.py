import pytest

from training.pipeline.evaluation import is_promotable


# ── 기준 충족 ─────────────────────────────────────────────────────────────────

def test_is_promotable_true_when_both_criteria_met():
    assert is_promotable({"test_accuracy": 0.85, "cum_lift_10": 2.0}) is True


def test_is_promotable_true_at_exact_boundary():
    """경계값(accuracy=0.80, lift=1.5)에서 True여야 한다."""
    assert is_promotable({"test_accuracy": 0.80, "cum_lift_10": 1.5}) is True


# ── 기준 미달 ─────────────────────────────────────────────────────────────────

def test_is_promotable_false_when_accuracy_below_threshold():
    assert is_promotable({"test_accuracy": 0.79, "cum_lift_10": 2.0}) is False


def test_is_promotable_false_when_lift_below_threshold():
    assert is_promotable({"test_accuracy": 0.85, "cum_lift_10": 1.49}) is False


def test_is_promotable_false_when_both_below_threshold():
    assert is_promotable({"test_accuracy": 0.70, "cum_lift_10": 1.0}) is False


# ── 커스텀 임계값 ─────────────────────────────────────────────────────────────

def test_is_promotable_custom_thresholds_true():
    assert is_promotable(
        {"test_accuracy": 0.75, "cum_lift_10": 1.2},
        min_accuracy=0.70,
        min_lift=1.0,
    ) is True


def test_is_promotable_custom_thresholds_false():
    assert is_promotable(
        {"test_accuracy": 0.85, "cum_lift_10": 2.0},
        min_accuracy=0.90,
        min_lift=3.0,
    ) is False
