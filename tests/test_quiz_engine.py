"""
test_quiz_engine.py — Unit tests for quiz_engine.py
Aligned with the 96-feature vector (Ahlem's feature_engineering.py).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import quiz_engine as qe


# ─────────────────────────────────────────────────────────────────────────────
# MockModel
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_model_shape():
    model = qe.MockModel(seed=42)
    X = np.zeros((1, 96))          # must be 96 features now
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_vector_length():
    pool = qe._generate_fake_question_pool()
    q = pool[0]
    features = qe.build_features([], q)
    assert isinstance(features, np.ndarray)
    assert features.shape == (96,), f"Expected 96, got {features.shape[0]}"
    assert len(qe.FEATURE_NAMES) == 96


def test_feature_vector_no_nan():
    pool = qe._generate_fake_question_pool()
    q = pool[5]
    history = [
        {
            "question_id": "q_001",
            "topic_id":    0,
            "topic":       qe.TOPIC_NAMES[0],
            "difficulty":  0.2,
            "correct":     1,
            "time":        0.05,
            "hint_count":  0,
            "attempt_count": 1,
        }
    ]
    features = qe.build_features(history, q)
    assert not np.any(np.isnan(features)), "Feature vector contains NaN values"


def test_feature_vector_empty_history():
    """With empty history, all count/rate features should be 0; avg_attempts defaults to 1.0."""
    pool = qe._generate_fake_question_pool()
    q = pool[0]
    features = qe.build_features([], q)
    # Each topic block is 12 features: [easy_c, easy_w, med_c, med_w, hard_c, hard_w,
    #   rate, total, avg_time, recent_weighted, avg_hints, avg_attempts]
    # Indices 0-9 and 10 (easy..avg_hints) should be 0; index 11 (avg_attempts) == 1.0
    for tid_idx in range(qe.N_TOPICS):
        base = tid_idx * 12
        # indices 0-10 (count/rate/time/hint slots) must all be 0
        assert np.all(features[base:base + 11] == 0.0), (
            f"Topic {tid_idx}: expected zeros for count/rate features, "
            f"got {features[base:base+11]}"
        )
        # index 11: avg_attempts defaults to 1.0
        assert features[base + 11] == 1.0, (
            f"Topic {tid_idx}: expected avg_attempts=1.0, got {features[base+11]}"
        )


def test_feature_vector_topic_onehot():
    """One-hot section should have exactly one 1 at the candidate topic position."""
    pool = qe._generate_fake_question_pool()
    for q in pool[:7]:   # one per topic
        features = qe.build_features([], q)
        onehot = features[89:]           # last 7 elements
        assert onehot.sum() == 1.0, f"One-hot sum should be 1, got {onehot.sum()}"
        assert onehot[q["topic_id"]] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# History record keys (aligned with Ahlem)
# ─────────────────────────────────────────────────────────────────────────────

def test_simulate_answer_keys():
    pool = qe._generate_fake_question_pool()
    q = pool[0]
    record = qe.simulate_answer(q, student_level=0.5)
    required_keys = {"question_id", "topic_id", "difficulty", "correct",
                     "time", "hint_count", "attempt_count"}
    assert required_keys.issubset(record.keys()), (
        f"Missing keys: {required_keys - record.keys()}"
    )
    assert isinstance(record["topic_id"], int)
    assert record["correct"] in (0, 1)
    assert isinstance(record["time"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_computation():
    history = (
        [{"topic_id": 0}] * 4    # topic 0: 4 questions → conf = 1.0
        + [{"topic_id": 1}] * 2  # topic 1: 2 questions → conf = 0.5
    )
    # Topics 2–6: 0 questions → conf = 0.0 each
    # Mean of [1.0, 0.5, 0, 0, 0, 0, 0] = 1.5/7 ≈ 0.2143
    # Engine uses min(c/3, 1.0) divisor:
    # topic0: 4 questions → min(4/3, 1) = 1.0
    # topic1: 2 questions → min(2/3, 1) ≈ 0.6667
    # topics 2–6: 0 → 0.0
    # Mean = (1.0 + 0.6667 + 0*5) / 7 ≈ 0.2381
    conf = qe.compute_confidence(history, qe.TOPIC_IDS)
    expected = (min(4/3.0, 1.0) + min(2/3.0, 1.0) + 0.0 * 5) / 7
    assert abs(conf - expected) < 1e-6, f"Expected {expected:.4f}, got {conf:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Validation errors
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_strategy_raises_error():
    model = qe.MockModel(seed=42)
    pool = qe._generate_fake_question_pool()
    with pytest.raises(ValueError, match="Invalid strategy"):
        qe.run_quiz(model, pool, "magic_strategy")


def test_missing_committee_raises_error():
    model = qe.MockModel(seed=42)
    pool = qe._generate_fake_question_pool()
    with pytest.raises(ValueError, match="requires exactly 5 committee models"):
        qe.run_quiz(model, pool, "committee", committee=None)


def test_empty_question_pool_raises_error():
    model = qe.MockModel(seed=42)
    with pytest.raises(ValueError, match="must not be empty"):
        qe.run_quiz(model, [], "random")


# ─────────────────────────────────────────────────────────────────────────────
# Full quiz run
# ─────────────────────────────────────────────────────────────────────────────

def test_run_quiz_returns_correct_format():
    model = qe.MockModel(seed=42)
    pool = qe._generate_fake_question_pool()
    result = qe.run_quiz(model, pool, "random", seed=42)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(qe.TOPIC_IDS)

    for topic_id, info in result.items():
        assert "level" in info
        assert "score" in info
        assert "topic_name" in info
        assert info["level"] in ["weak", "medium", "strong", "unknown"]
        if info["score"] is not None:
            assert isinstance(info["score"], float)
            assert 0.0 <= info["score"] <= 1.0


def test_run_quiz_with_trace_keys():
    model = qe.MockModel(seed=42)
    pool = qe._generate_fake_question_pool()
    out = qe.run_quiz_with_trace(model, pool, "uncertainty", seed=42)

    assert "result" in out
    assert "history" in out
    assert "questions_asked" in out
    assert "confidence_history" in out
    assert isinstance(out["questions_asked"], int)
    assert out["questions_asked"] <= qe.MAX_QUESTIONS


def test_committee_strategy_runs():
    committee = [qe.MockModel(seed=i) for i in range(5)]
    pool = qe._generate_fake_question_pool()
    result = qe.run_quiz(
        model=qe.MockModel(seed=99),
        question_pool=pool,
        strategy_name="committee",
        committee=committee,
        seed=7,
    )
    assert set(result.keys()) == set(qe.TOPIC_IDS)


def test_entropy_strategy_runs():
    model = qe.MockModel(seed=42)
    pool = qe._generate_fake_question_pool()
    result = qe.run_quiz(model, pool, "entropy", seed=42)
    assert set(result.keys()) == set(qe.TOPIC_IDS)
