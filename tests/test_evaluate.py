import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import evaluate as ev


def test_simulate_answer_binary_output():
    rng = np.random.default_rng(42)
    outputs = [ev.simulate_answer(0.6, 0.4, rng) for _ in range(100)]
    assert all(x in (0, 1) for x in outputs)


def test_answer_oracle_uses_real_then_fallback():
    student_df = pd.DataFrame(
        [
            {
                "user_id": 1,
                "problem_id": 100,
                "correct": 1,
                "topic_id": 0,
                "difficulty": 0.2,
                "response_time_norm": 0.1,
                "hint_count": 0,
                "attempt_count": 1,
            }
        ]
    )
    rng = np.random.default_rng(7)
    answer_fn = ev.build_student_answer_oracle(student_df, student_level=0.5, rng=rng)

    known = answer_fn({"question_id": "100", "topic_id": 0, "difficulty": 0.2})
    assert known["correct"] == 1

    unknown = answer_fn({"question_id": "999", "topic_id": 0, "difficulty": 0.2})
    assert unknown["correct"] in (0, 1)
    assert unknown["question_id"] == "999"


def test_run_mock_step1_simulation_outputs_expected_shape():
    test_df = pd.DataFrame(
        [
            {
                "user_id": 1,
                "problem_id": 10,
                "correct": 1,
                "topic_id": 0,
                "difficulty": 0.2,
                "skill_name": "A",
                "response_time_norm": 0.1,
                "hint_count": 0,
                "attempt_count": 1,
            },
            {
                "user_id": 1,
                "problem_id": 11,
                "correct": 0,
                "topic_id": 1,
                "difficulty": 0.8,
                "skill_name": "B",
                "response_time_norm": 0.2,
                "hint_count": 1,
                "attempt_count": 2,
            },
            {
                "user_id": 2,
                "problem_id": 10,
                "correct": 0,
                "topic_id": 0,
                "difficulty": 0.2,
                "skill_name": "A",
                "response_time_norm": 0.3,
                "hint_count": 0,
                "attempt_count": 1,
            },
            {
                "user_id": 2,
                "problem_id": 11,
                "correct": 1,
                "topic_id": 1,
                "difficulty": 0.8,
                "skill_name": "B",
                "response_time_norm": 0.4,
                "hint_count": 0,
                "attempt_count": 1,
            },
        ]
    )

    pool_df = pd.DataFrame(
        [
            {"problem_id": 10, "difficulty": 0.2, "topic_id": 0, "question_id": "10"},
            {"problem_id": 20, "difficulty": 0.5, "topic_id": 0, "question_id": "20"},
            {"problem_id": 11, "difficulty": 0.8, "topic_id": 1, "question_id": "11"},
            {"problem_id": 21, "difficulty": 0.4, "topic_id": 1, "question_id": "21"},
        ]
    )

    config = ev.SimulationConfig(max_questions=3, seed=10)
    trace_df, curve_df = ev.run_mock_step1_simulation(
        test_df=test_df,
        question_pool_df=pool_df,
        config=config,
        strategies=["uncertainty", "random"],
    )

    assert not trace_df.empty
    assert not curve_df.empty
    assert set(trace_df["strategy"].unique()) == {"uncertainty", "random"}
    assert trace_df["question_number"].min() == 1
    assert trace_df["question_number"].max() == 3
    assert np.all((curve_df["accuracy_pct"] >= 0) & (curve_df["accuracy_pct"] <= 100))
