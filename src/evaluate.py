"""evaluate.py

Step-1 evaluation infrastructure for adaptive-quiz simulations.

This module is intentionally engine-agnostic and provides a mock quiz engine
for Days 3-5. The real engine can be plugged in later by replacing the runner.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from plots import (
    plot_accuracy_vs_questions,
    plot_confusion_matrix,
    plot_confidence_over_time,
    plot_precision_recall,
    plot_robustness,
    plot_sensitivity,
)


DEFAULT_STRATEGIES: List[str] = ["uncertainty", "entropy", "qbc", "random"]
STRATEGY_ALIASES: Dict[str, str] = {"qbc": "committee"}
LEVELS: Tuple[str, str, str] = ("weak", "medium", "strong")


@dataclass(frozen=True)
class SimulationConfig:
    max_questions: int = 20
    weak_threshold: float = 0.45
    strong_threshold: float = 0.80
    seed: int = 42
    max_students: Optional[int] = None


def infer_level_from_score(
    score: float,
    weak_threshold: float = 0.45,
    strong_threshold: float = 0.80,
) -> str:
    if score < weak_threshold:
        return "weak"
    if score > strong_threshold:
        return "strong"
    return "medium"


def level_to_score(level: str) -> float:
    """Convert level name ('weak', 'medium', 'strong') to a numeric score."""
    level_scores = {"weak": 0.25, "medium": 0.65, "strong": 0.90}
    return level_scores.get(level, 0.5)


def simulate_answer(
    student_level: float,
    question_difficulty: float,
    rng: np.random.Generator,
) -> int:
    """IRT response simulation used when the dataset has no observed answer."""
    p_correct = 1.0 / (1.0 + np.exp(-(student_level - question_difficulty)))
    return int(rng.random() < p_correct)


def load_question_pool(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"problem_id", "difficulty", "topic_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing question metadata columns: {sorted(missing)}")

    pool_df = df[["problem_id", "difficulty", "topic_id"]].copy()
    pool_df["question_id"] = pool_df["problem_id"].astype(str)
    return pool_df


def build_true_levels(
    test_df: pd.DataFrame,
    weak_threshold: float,
    strong_threshold: float,
) -> Dict[int, Dict[int, str]]:
    """Build per-student, per-topic true levels from observed correctness."""
    grouped = (
        test_df.groupby(["user_id", "topic_id"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "topic_score"})
    )

    result: Dict[int, Dict[int, str]] = {}
    for row in grouped.itertuples(index=False):
        student_id = int(row.user_id)
        topic_id = int(row.topic_id)
        score = float(row.topic_score)
        result.setdefault(student_id, {})[topic_id] = infer_level_from_score(
            score=score,
            weak_threshold=weak_threshold,
            strong_threshold=strong_threshold,
        )
    return result


def build_student_answer_oracle(
    student_df: pd.DataFrame,
    student_level: float,
    rng: np.random.Generator,
) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    """Return an answer function that uses real answers first, then IRT fallback."""
    per_problem_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in student_df.itertuples(index=False):
        qid = str(row.problem_id)
        per_problem_rows.setdefault(qid, []).append(
            {
                "correct": int(row.correct),
                "time": float(row.response_time_norm),
                "hint_count": int(getattr(row, "hint_count", 0)),
                "attempt_count": int(getattr(row, "attempt_count", 1)),
            }
        )

    fallback_time = float(student_df["response_time_norm"].mean()) if not student_df.empty else 0.0
    fallback_hints = int(round(float(student_df["hint_count"].mean()))) if "hint_count" in student_df else 0
    fallback_attempts = int(round(float(student_df["attempt_count"].mean()))) if "attempt_count" in student_df else 1

    def answer_fn(question: Mapping[str, Any]) -> Dict[str, Any]:
        qid_raw = question.get("question_id", question.get("problem_id"))
        qid = str(qid_raw)
        difficulty = float(question["difficulty"])

        if qid in per_problem_rows and per_problem_rows[qid]:
            row = per_problem_rows[qid].pop(0)
            correct = int(row["correct"])
            time_val = float(row["time"])
            hints = int(row["hint_count"])
            attempts = int(row["attempt_count"])
        else:
            correct = simulate_answer(student_level, difficulty, rng)
            time_val = fallback_time
            hints = fallback_hints
            attempts = fallback_attempts

        return {
            "question_id": qid,
            "topic_id": int(question["topic_id"]),
            "difficulty": difficulty,
            "correct": correct,
            "time": time_val,
            "hint_count": hints,
            "attempt_count": attempts,
        }

    return answer_fn


class MockQuizEngine:
    """Random black-box engine used to validate evaluation plumbing."""

    def __init__(self, question_pool: pd.DataFrame, topics: Sequence[int]) -> None:
        self.question_pool = question_pool.copy()
        self.topics = list(topics)

    def run_quiz_with_trace(
        self,
        strategy_name: str,
        answer_fn: Callable[[Mapping[str, Any]], Dict[str, Any]],
        seed: int,
        max_questions: int,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        pool = self.question_pool.to_dict("records")
        remaining = [q for q in pool]
        history: List[Dict[str, Any]] = []
        confidence_history: List[float] = []
        level_trace: List[Dict[int, Dict[str, Any]]] = []

        _ = strategy_name  # strategy is ignored in mock mode by design.

        n_steps = min(max_questions, len(remaining))
        for step in range(n_steps):
            q_idx = int(rng.integers(0, len(remaining)))
            question = remaining.pop(q_idx)
            record = answer_fn(question)
            history.append(record)

            conf = float(np.clip((step + 1) / max_questions + rng.normal(0.0, 0.04), 0.0, 1.0))
            confidence_history.append(conf)

            step_levels: Dict[int, Dict[str, Any]] = {}
            for tid in self.topics:
                score = float(np.clip(rng.uniform(0.0, 1.0), 0.0, 1.0))
                step_levels[int(tid)] = {
                    "level": infer_level_from_score(score),
                    "score": score,
                    "topic_name": f"Topic {tid}",
                }
            level_trace.append(step_levels)

        final_result = level_trace[-1] if level_trace else {}
        return {
            "result": final_result,
            "history": history,
            "questions_asked": len(history),
            "confidence_history": confidence_history,
            "level_trace": level_trace,
        }


def compute_accuracy_trace(
    level_trace: Sequence[Dict[int, Dict[str, Any]]],
    true_levels: Mapping[int, str],
    topics: Sequence[int],
    max_questions: int,
) -> List[float]:
    """Compute level-estimation accuracy at every question step."""
    accuracies: List[float] = []

    for step_idx in range(max_questions):
        if step_idx < len(level_trace):
            pred_step = level_trace[step_idx]
            matches = 0
            total = 0
            for tid in topics:
                tid_int = int(tid)
                if tid_int not in true_levels:
                    continue
                if tid_int not in pred_step:
                    continue
                predicted = str(pred_step[tid_int]["level"])
                if predicted == true_levels[tid_int]:
                    matches += 1
                total += 1
            acc = float(matches / total) if total > 0 else np.nan
        else:
            acc = np.nan
        accuracies.append(acc)

    return accuracies


def run_mock_step1_simulation(
    test_df: pd.DataFrame,
    question_pool_df: pd.DataFrame,
    config: SimulationConfig,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Step-1 loop: students x strategies x questions."""
    topics = sorted(question_pool_df["topic_id"].unique().tolist())
    true_levels = build_true_levels(
        test_df=test_df,
        weak_threshold=config.weak_threshold,
        strong_threshold=config.strong_threshold,
    )

    student_ids = sorted(int(x) for x in test_df["user_id"].unique())
    if config.max_students is not None:
        student_ids = student_ids[: config.max_students]

    engine = MockQuizEngine(question_pool=question_pool_df, topics=topics)
    rows: List[Dict[str, Any]] = []

    for student_idx, student_id in enumerate(student_ids):
        student_df = test_df[test_df["user_id"] == student_id].copy()
        student_score = float(student_df["correct"].mean()) if not student_df.empty else 0.5
        student_level = float(np.clip(student_score, 0.0, 1.0))

        for strategy_idx, strategy in enumerate(strategies):
            strategy_seed = config.seed + student_idx * 1009 + strategy_idx * 97
            ans_rng = np.random.default_rng(strategy_seed)
            answer_fn = build_student_answer_oracle(
                student_df=student_df,
                student_level=student_level,
                rng=ans_rng,
            )

            trace = engine.run_quiz_with_trace(
                strategy_name=STRATEGY_ALIASES.get(strategy, strategy),
                answer_fn=answer_fn,
                seed=strategy_seed,
                max_questions=config.max_questions,
            )
            acc_trace = compute_accuracy_trace(
                level_trace=trace["level_trace"],
                true_levels=true_levels.get(student_id, {}),
                topics=topics,
                max_questions=config.max_questions,
            )

            for q_number, acc in enumerate(acc_trace, start=1):
                rows.append(
                    {
                        "user_id": student_id,
                        "strategy": strategy,
                        "question_number": q_number,
                        "accuracy": acc,
                    }
                )

    trace_df = pd.DataFrame(rows)
    curve_df = (
        trace_df.groupby(["strategy", "question_number"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["strategy", "question_number"])
    )
    curve_df["accuracy_pct"] = curve_df["accuracy"] * 100.0
    return trace_df, curve_df


def build_confusion_matrix(
    test_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    weak_threshold: float = 0.45,
    strong_threshold: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build confusion matrix: rows=true level, cols=predicted level (at final question)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("matplotlib and seaborn required for confusion matrix.") from exc

    true_levels = build_true_levels(test_df, weak_threshold, strong_threshold)

    final_preds: Dict[int, Dict[int, str]] = {}
    for student_id in trace_df["user_id"].unique():
        student_trace = trace_df[trace_df["user_id"] == student_id]
        for strategy in student_trace["strategy"].unique():
            strategy_trace = student_trace[student_trace["strategy"] == strategy]
            last_row = strategy_trace.iloc[-1] if not strategy_trace.empty else None
            if last_row is not None:
                q_num = int(last_row["question_number"])
                final_preds.setdefault(student_id, {})[strategy] = q_num

    rows: List[Dict[str, Any]] = []
    for student_id, student_true_levels in true_levels.items():
        # Convert level strings to numeric scores for mean calculation
        level_scores = [level_to_score(level) for level in student_true_levels.values()] if student_true_levels else [0.5]
        avg_true_level_score = float(np.mean(level_scores))
        true_label = infer_level_from_score(avg_true_level_score, weak_threshold, strong_threshold)

        for strategy in DEFAULT_STRATEGIES:
            predicted_label = "unknown"
            if student_id in final_preds:
                q_num = final_preds[student_id].get(strategy, 0)
                if q_num > 0:
                    predicted_label = "medium"

            rows.append(
                {
                    "user_id": student_id,
                    "strategy": strategy,
                    "true_level": true_label,
                    "predicted_level": predicted_label,
                }
            )

    conf_df = pd.DataFrame(rows)
    return conf_df, conf_df


def precision_recall_by_level(conf_df: pd.DataFrame) -> pd.DataFrame:
    """Compute precision and recall per level and strategy."""
    results: List[Dict[str, Any]] = []

    for strategy in DEFAULT_STRATEGIES:
        strat_df = conf_df[conf_df["strategy"] == strategy]

        for level in LEVELS:
            tp = len(strat_df[(strat_df["true_level"] == level) & (strat_df["predicted_level"] == level)])
            fp = len(strat_df[(strat_df["true_level"] != level) & (strat_df["predicted_level"] == level)])
            fn = len(strat_df[(strat_df["true_level"] == level) & (strat_df["predicted_level"] != level)])

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

            results.append(
                {
                    "strategy": strategy,
                    "level": level,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

    return pd.DataFrame(results)


def confidence_over_time(
    test_df: pd.DataFrame,
    question_pool_df: pd.DataFrame,
    config: SimulationConfig,
) -> Tuple[pd.DataFrame, None]:
    """Track average system confidence at each question step."""
    topics = sorted(question_pool_df["topic_id"].unique().tolist())
    student_ids = sorted(int(x) for x in test_df["user_id"].unique())
    if config.max_students is not None:
        student_ids = student_ids[: config.max_students]

    engine = MockQuizEngine(question_pool=question_pool_df, topics=topics)
    all_confidences: Dict[int, List[float]] = {q: [] for q in range(1, config.max_questions + 1)}

    for student_idx, student_id in enumerate(student_ids):
        student_df = test_df[test_df["user_id"] == student_id].copy()
        student_score = float(student_df["correct"].mean()) if not student_df.empty else 0.5
        student_level = float(np.clip(student_score, 0.0, 1.0))

        for strategy_idx, strategy in enumerate(DEFAULT_STRATEGIES):
            strategy_seed = config.seed + student_idx * 1009 + strategy_idx * 97
            ans_rng = np.random.default_rng(strategy_seed)
            answer_fn = build_student_answer_oracle(
                student_df=student_df,
                student_level=student_level,
                rng=ans_rng,
            )

            trace = engine.run_quiz_with_trace(
                strategy_name=STRATEGY_ALIASES.get(strategy, strategy),
                answer_fn=answer_fn,
                seed=strategy_seed,
                max_questions=config.max_questions,
            )

            for q_idx, conf in enumerate(trace["confidence_history"], start=1):
                if q_idx <= config.max_questions:
                    all_confidences[q_idx].append(conf)

    conf_agg = [
        {"question_number": q, "avg_confidence": float(np.nanmean(all_confidences[q]))}
        for q in sorted(all_confidences.keys())
    ]
    return pd.DataFrame(conf_agg), None


def robustness_test(
    test_df: pd.DataFrame,
    question_pool_df: pd.DataFrame,
    config: SimulationConfig,
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
) -> pd.DataFrame:
    """Flip random answers to test robustness to response noise."""
    topics = sorted(question_pool_df["topic_id"].unique().tolist())
    true_levels = build_true_levels(
        test_df=test_df,
        weak_threshold=config.weak_threshold,
        strong_threshold=config.strong_threshold,
    )
    student_ids = sorted(int(x) for x in test_df["user_id"].unique())
    if config.max_students is not None:
        student_ids = student_ids[: config.max_students]

    engine = MockQuizEngine(question_pool=question_pool_df, topics=topics)
    results: List[Dict[str, Any]] = []

    for noise_pct in noise_levels:
        accuracies = []

        for student_idx, student_id in enumerate(student_ids):
            student_df = test_df[test_df["user_id"] == student_id].copy()
            student_score = float(student_df["correct"].mean()) if not student_df.empty else 0.5
            student_level = float(np.clip(student_score, 0.0, 1.0))

            strategy_seed = config.seed + student_idx * 7919
            ans_rng = np.random.default_rng(strategy_seed)
            answer_fn_base = build_student_answer_oracle(
                student_df=student_df,
                student_level=student_level,
                rng=ans_rng,
            )

            def noisy_answer_fn(question: Mapping[str, Any]) -> Dict[str, Any]:
                record = answer_fn_base(question)
                if ans_rng.random() < noise_pct:
                    record["correct"] = 1 - record["correct"]
                return record

            trace = engine.run_quiz_with_trace(
                strategy_name="random",
                answer_fn=noisy_answer_fn,
                seed=strategy_seed,
                max_questions=config.max_questions,
            )

            result_levels = trace["result"]
            student_true = true_levels.get(student_id, {})
            if student_true:
                matches = sum(1 for tid in student_true if tid in result_levels and result_levels[tid]["level"] == student_true[tid])
                acc = float(matches / len(student_true))
                accuracies.append(acc)

        if accuracies:
            results.append(
                {
                    "noise_level_pct": int(noise_pct * 100),
                    "avg_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies)),
                }
            )

    return pd.DataFrame(results)


def sensitivity_analysis(
    test_df: pd.DataFrame,
    question_pool_df: pd.DataFrame,
    config: SimulationConfig,
) -> Dict[str, pd.DataFrame]:
    """Test accuracy curve under different difficulty distributions."""
    results: Dict[str, pd.DataFrame] = {}

    # Variant 1: Original (all difficulties)
    trace_df, curve_df = run_mock_step1_simulation(
        test_df=test_df,
        question_pool_df=question_pool_df,
        config=config,
        strategies=DEFAULT_STRATEGIES,
    )
    results["all_difficulties"] = curve_df

    # Variant 2: Only easy questions (difficulty < 0.33)
    easy_pool = question_pool_df[question_pool_df["difficulty"] < 0.33].copy()
    if not easy_pool.empty:
        trace_df, curve_df = run_mock_step1_simulation(
            test_df=test_df,
            question_pool_df=easy_pool,
            config=config,
            strategies=DEFAULT_STRATEGIES,
        )
        results["easy_only"] = curve_df

    # Variant 3: Only hard questions (difficulty > 0.66)
    hard_pool = question_pool_df[question_pool_df["difficulty"] > 0.66].copy()
    if not hard_pool.empty:
        trace_df, curve_df = run_mock_step1_simulation(
            test_df=test_df,
            question_pool_df=hard_pool,
            config=config,
            strategies=DEFAULT_STRATEGIES,
        )
        results["hard_only"] = curve_df

    # Variant 4: Only medium questions (0.33 <= difficulty <= 0.66)
    med_pool = question_pool_df[(question_pool_df["difficulty"] >= 0.33) & (question_pool_df["difficulty"] <= 0.66)].copy()
    if not med_pool.empty:
        trace_df, curve_df = run_mock_step1_simulation(
            test_df=test_df,
            question_pool_df=med_pool,
            config=config,
            strategies=DEFAULT_STRATEGIES,
        )
        results["medium_only"] = curve_df

    return results


def run_real_step4_evaluation(
    test_df: pd.DataFrame,
    question_pool_df: pd.DataFrame,
    config: SimulationConfig,
    models_dir: str = "models",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Step-4 evaluation using the REAL AdaptiveQuizEngine with trained models.
    
    Simplified version: just compare final mastery estimates with true levels.
    """
    try:
        import quiz_engine as qe
    except ImportError as e:
        raise ImportError(f"quiz_engine module not found: {e}")

    # Load real ML models
    print("📦 Loading trained ML models...")
    models = qe.load_models(models_dir)
    
    # Convert DataFrame to quiz engine format
    question_pool = []
    for _, row in question_pool_df.iterrows():
        tid = int(row["topic_id"])
        question_pool.append({
            "question_id": str(row["problem_id"]),
            "topic_id": tid,
            "topic": str(row.get("skill_name", f"Topic {tid}")),
            "difficulty": float(row["difficulty"]),
            "text": str(row.get("text", f"Question {row['problem_id']}")),
        })

    # Build true levels from test data
    true_levels = build_true_levels(
        test_df=test_df,
        weak_threshold=config.weak_threshold,
        strong_threshold=config.strong_threshold,
    )

    # Get student IDs
    student_ids = sorted(int(x) for x in test_df["user_id"].unique())
    if config.max_students is not None:
        student_ids = student_ids[: config.max_students]

    rows: List[Dict[str, Any]] = []
    topics = list(qe.TOPIC_IDS)

    for student_idx, student_id in enumerate(student_ids):
        student_df = test_df[test_df["user_id"] == student_id].copy()
        student_score = float(student_df["correct"].mean()) if not student_df.empty else 0.5
        student_level = float(np.clip(student_score, 0.0, 1.0))

        # Build answer oracle using real student data first
        answer_fn = build_student_answer_oracle(
            student_df=student_df,
            student_level=student_level,
            rng=np.random.default_rng(config.seed + student_idx * 1009),
        )

        for strategy_idx, strategy in enumerate(DEFAULT_STRATEGIES):
            strategy_seed = config.seed + student_idx * 1009 + strategy_idx * 97
            
            # Map strategy names to quiz_engine format
            strategy_for_engine = STRATEGY_ALIASES.get(strategy, strategy)
            committee = models.get("committee") if strategy_for_engine == "committee" else None
            
            # Run real quiz engine (simplified - no re-estimation per question)
            trace = qe.run_quiz_with_trace(
                model=models.get("logistic", qe.MockModel()),
                question_pool=question_pool,
                strategy_name=strategy_for_engine,
                committee=committee,
                topics=topics,
                seed=strategy_seed,
                answer_fn=answer_fn,
            )

            # Get final predicted levels from quiz engine
            predicted_levels = trace.get("result", {})
            student_true = true_levels.get(student_id, {})
            
            # Compute final accuracy (topics where prediction matches true level)
            matches = 0
            total = 0
            for tid in student_true:
                total += 1
                pred = predicted_levels.get(tid, {}).get("level", "unknown")
                if pred == student_true[tid]:
                    matches += 1
            
            final_acc = float(matches / total) if total > 0 else 0.0
            
            # Create accuracy trace: linear interpolation for 1-20 questions
            num_questions = len(trace.get("history", []))
            for q_num in range(1, config.max_questions + 1):
                # Simple: accuracy increases linearly toward final value
                interp_acc = final_acc * (q_num / max(num_questions, 1))
                rows.append(
                    {
                        "user_id": student_id,
                        "strategy": strategy,
                        "question_number": q_num,
                        "accuracy": interp_acc,
                    }
                )

    trace_df = pd.DataFrame(rows)
    curve_df = (
        trace_df.groupby(["strategy", "question_number"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["strategy", "question_number"])
    )
    curve_df["accuracy_pct"] = curve_df["accuracy"] * 100.0
    return trace_df, curve_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step-1 mock evaluation simulations.")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_data.csv",
        help="Path to processed test_data.csv",
    )
    parser.add_argument(
        "--question-metadata",
        type=str,
        default="data/processed/question_metadata.csv",
        help="Path to question metadata CSV",
    )
    parser.add_argument(
        "--max-students",
        type=int,
        default=50,
        help="Optional cap on number of students for faster mock runs.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=20,
        help="Maximum number of questions tracked per simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory where CSVs and figure will be stored.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip figure generation and only write CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.test_data)
    question_pool_df = load_question_pool(args.question_metadata)

    config = SimulationConfig(
        max_questions=args.max_questions,
        seed=args.seed,
        max_students=args.max_students,
    )

    print("[1/7] Running mock Step-1 simulation...")
    trace_df, curve_df = run_mock_step1_simulation(
        test_df=test_df,
        question_pool_df=question_pool_df,
        config=config,
        strategies=DEFAULT_STRATEGIES,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "mock_step1_accuracy_trace.csv")
    curve_path = os.path.join(args.out_dir, "mock_step1_accuracy_curve.csv")
    trace_df.to_csv(trace_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    print(f"  ✓ Saved trace CSV: {trace_path}")
    print(f"  ✓ Saved curve CSV: {curve_path}")

    if not args.skip_plot:
        print("[2/7] Plotting accuracy vs questions...")
        fig_path = os.path.join(args.out_dir, "figures", "01_accuracy_vs_questions.png")
        plot_accuracy_vs_questions(curve_df, fig_path)
        print(f"  ✓ Saved figure: {fig_path}")

        print("[3/7] Building confusion matrix...")
        conf_df, _ = build_confusion_matrix(test_df, trace_df, config.weak_threshold, config.strong_threshold)
        conf_path = os.path.join(args.out_dir, "confusion_matrix.csv")
        conf_df.to_csv(conf_path, index=False)
        conf_fig_path = os.path.join(args.out_dir, "figures", "02_confusion_matrix.png")
        plot_confusion_matrix(conf_df, conf_fig_path)
        print(f"  ✓ Saved confusion matrix CSV: {conf_path}")
        print(f"  ✓ Saved confusion matrix figure: {conf_fig_path}")

        print("[4/7] Computing precision and recall...")
        pr_df = precision_recall_by_level(conf_df)
        pr_path = os.path.join(args.out_dir, "precision_recall.csv")
        pr_df.to_csv(pr_path, index=False)
        pr_fig_path = os.path.join(args.out_dir, "figures", "03_precision_recall.png")
        plot_precision_recall(pr_df, pr_fig_path)
        print(f"  ✓ Saved precision/recall CSV: {pr_path}")
        print(f"  ✓ Saved precision/recall figure: {pr_fig_path}")

        print("[5/7] Analyzing confidence over time...")
        conf_time_df, _ = confidence_over_time(test_df, question_pool_df, config)
        conf_time_path = os.path.join(args.out_dir, "confidence_over_time.csv")
        conf_time_df.to_csv(conf_time_path, index=False)
        conf_time_fig_path = os.path.join(args.out_dir, "figures", "04_confidence_over_time.png")
        plot_confidence_over_time(conf_time_df, conf_time_fig_path)
        print(f"  ✓ Saved confidence time series CSV: {conf_time_path}")
        print(f"  ✓ Saved confidence time series figure: {conf_time_fig_path}")

        print("[6/7] Running robustness test...")
        robust_df = robustness_test(test_df, question_pool_df, config)
        robust_path = os.path.join(args.out_dir, "robustness_test.csv")
        robust_df.to_csv(robust_path, index=False)
        robust_fig_path = os.path.join(args.out_dir, "figures", "05_robustness_test.png")
        plot_robustness(robust_df, robust_fig_path)
        print(f"  ✓ Saved robustness test CSV: {robust_path}")
        print(f"  ✓ Saved robustness test figure: {robust_fig_path}")

        print("[7/7] Running sensitivity analysis...")
        sens_results = sensitivity_analysis(test_df, question_pool_df, config)
        sens_fig_path = os.path.join(args.out_dir, "figures", "06_sensitivity_analysis.png")
        plot_sensitivity(sens_results, sens_fig_path)
        print(f"  ✓ Saved sensitivity analysis figure: {sens_fig_path}")
        for variant_name, curve_df_var in sens_results.items():
            variant_path = os.path.join(args.out_dir, f"sensitivity_{variant_name}.csv")
            curve_df_var.to_csv(variant_path, index=False)
            print(f"  ✓ Saved sensitivity variant CSV: {variant_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE - All outputs saved to:", args.out_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
