"""
quiz_engine.py — Adaptive Quiz Engine
======================================

This module implements an adaptive quiz system aligned with Ahlem's 96-feature vector layout.
Author : Alaa (Part 3 — Quiz Engine & Query Strategies)
Integration: Thresholds Narrowed to 0.25/0.35 to match conservative model scores.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Only import joblib if available
try:
    import joblib
except ImportError:
    joblib = None

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Real ASSISTments topic IDs (0-6)
N_TOPICS: int = 7
TOPIC_IDS: List[int] = list(range(N_TOPICS))

TOPIC_NAMES: Dict[int, str] = {
    0: "Fraction/Decimal/Percent Conversion",
    1: "Finding Percents",
    2: "Ordering Fractions",
    3: "Equation Solving (Multi-Step)",
    4: "Equation Solving (Simple)",
    5: "Proportion/Fraction/Decimal/Percent",
    6: "Subtraction Whole Numbers",
}

# Thresholds updated to match your models' conservative probability range
DIFFICULTY_EASY_MAX: float   = 0.33
DIFFICULTY_MEDIUM_MIN: float = 0.33
DIFFICULTY_MEDIUM_MAX: float = 0.66

MAX_QUESTIONS: int       = 20
CONFIDENCE_THRESHOLD: float = 0.80
LEVEL_THRESHOLDS: Dict[str, float] = {"weak": 0.45, "strong": 0.80}

VALID_STRATEGIES: List[str] = ["uncertainty", "entropy", "committee", "random"]

# ── Feature-vector column names (96 total) ─────────────────────────────────
_PER_TOPIC_FEATURE_NAMES: List[str] = [
    "easy_correct", "easy_wrong", "med_correct", "med_wrong",
    "hard_correct", "hard_wrong", "success_rate", "total",
    "avg_time", "recent_weighted", "avg_hints", "avg_attempts",
]

FEATURE_NAMES: List[str] = (
    [f"topic{t}_{feat}" for t in TOPIC_IDS for feat in _PER_TOPIC_FEATURE_NAMES]
    + ["candidate_difficulty", "global_success_x_diff", "global_success", "global_avg_time", "total_history"]
    + [f"topic_onehot_{t}" for t in TOPIC_IDS]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES & MOCK MODEL
# ═══════════════════════════════════════════════════════════════════════════════

Question = Dict[str, Any]
AnswerRecord = Dict[str, Any]
StudentHistory = List[AnswerRecord]

class MockModel:
    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        p_correct = self._rng.uniform(0.15, 0.55, size=X.shape[0]) # Mock conservative model
        return np.column_stack([1.0 - p_correct, p_correct])

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(models_dir: str) -> Dict[str, Any]:
    model_specs = {
        "logistic": "logistic_model.pkl",
        "naive_bayes": "naive_bayes_model.pkl",
        "decision_tree": "decision_tree_model.pkl",
        "random_forest": "random_forest_model.pkl",
    }
    models = {}
    for key, filename in model_specs.items():
        path = os.path.join(models_dir, filename)
        if os.path.isfile(path) and joblib is not None:
            models[key] = joblib.load(path)
        else:
            models[key] = MockModel(seed=hash(key) % 2**31)
    
    committee = []
    for i in range(5):
        path = os.path.join(models_dir, f"committee_model_{i}.pkl")
        if os.path.isfile(path) and joblib is not None:
            committee.append(joblib.load(path))
        else:
            committee.append(MockModel(seed=100+i))
    models["committee"] = committee
    return models

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION POOL
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_fake_question_pool() -> List[Question]:
    pool = []
    rng = np.random.RandomState(0)
    for tid in TOPIC_IDS:
        for idx in range(10):
            diff = round(rng.uniform(0.1, 0.9), 2)
            pool.append({
                "question_id": f"q_{tid}_{idx}",
                "topic_id": tid,
                "topic": TOPIC_NAMES[tid],
                "difficulty": diff,
                "text": f"Question {idx} for {TOPIC_NAMES[tid]}",
            })
    return pool

def load_question_pool(metadata_path: str) -> List[Question]:
    if os.path.isfile(metadata_path):
        df = pd.read_csv(metadata_path)
        pool = []
        for _, row in df.iterrows():
            tid = int(row["topic_id"])
            pool.append({
                "question_id": str(row["problem_id"]),
                "topic_id": tid,
                "topic": str(row.get("skill_name", f"Topic {tid}")),
                "difficulty": float(row["difficulty"]),
                "text": str(row.get("text", f"Question {row['problem_id']}")),
            })
        return pool
    return _generate_fake_question_pool()

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (Fixed to 96 Features Always)
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(
    history: StudentHistory,
    candidate_question: Question
) -> np.ndarray:
    """Build a 96-element feature vector matching Ahlem's architecture."""
    # Architecture is FIXED to 7 topics for model compatibility
    arch_topics = TOPIC_IDS
    features = []
    
    total_h = len(history)
    global_success = sum(h["correct"] for h in history) / total_h if total_h > 0 else 0.0
    global_avg_time = float(np.mean([h["time"] for h in history])) if total_h > 0 else 0.0

    for tid in arch_topics:
        t_data = [h for h in history if h["topic_id"] == tid]
        total = len(t_data)
        
        easy_c = sum(1 for h in t_data if h["difficulty"] < DIFFICULTY_EASY_MAX and h["correct"] == 1)
        easy_w = sum(1 for h in t_data if h["difficulty"] < DIFFICULTY_EASY_MAX and h["correct"] == 0)
        med_c  = sum(1 for h in t_data if DIFFICULTY_MEDIUM_MIN <= h["difficulty"] <= DIFFICULTY_MEDIUM_MAX and h["correct"] == 1)
        med_w  = sum(1 for h in t_data if DIFFICULTY_MEDIUM_MIN <= h["difficulty"] <= DIFFICULTY_MEDIUM_MAX and h["correct"] == 0)
        hard_c = sum(1 for h in t_data if h["difficulty"] > DIFFICULTY_MEDIUM_MAX and h["correct"] == 1)
        hard_w = sum(1 for h in t_data if h["difficulty"] > DIFFICULTY_MEDIUM_MAX and h["correct"] == 0)
        
        rate = sum(h["correct"] for h in t_data) / total if total > 0 else 0.0
        avg_t = float(np.mean([h["time"] for h in t_data])) if total > 0 else global_avg_time
        
        recent_w = global_success
        if total > 0:
            weights = np.linspace(1, 2, total)
            recent_w = float(np.average([h["correct"] for h in t_data], weights=weights))
            
        avg_h = float(np.mean([h["hint_count"] for h in t_data])) if total > 0 else 0.0
        avg_a = float(np.mean([h["attempt_count"] for h in t_data])) if total > 0 else 1.0

        features.extend([easy_c, easy_w, med_c, med_w, hard_c, hard_w, rate, total, avg_t, recent_w, avg_h, avg_a])

    cand_diff = float(candidate_question["difficulty"])
    features.append(cand_diff)
    features.append(global_success * cand_diff)
    features.append(global_success)
    features.append(global_avg_time)
    features.append(float(total_h))

    # One-hot
    cand_tid = int(candidate_question["topic_id"])
    onehot = [0.0] * N_TOPICS
    if cand_tid in arch_topics:
        onehot[arch_topics.index(cand_tid)] = 1.0
    features.extend(onehot)

    return np.array(features, dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confidence(history: StudentHistory, topics: List[int]) -> float:
    if not topics: return 0.0
    counts = {t: 0 for t in topics}
    for r in history:
        tid = r.get("topic_id")
        if tid in counts: counts[tid] += 1
    # require 3 questions per topic instead of 4 to hit confidence faster
    return float(np.mean([min(c/3.0, 1.0) for c in counts.values()]))

def estimate_levels(history, model, topics):
    """
    Estimate mastery for each topic by probing the model with a 
    standard 'Medium Difficulty' (0.5) question for each topic.
    """
    res = {}
    for tid in topics:
        # Determine current mastery by asking the model:
        # 'If we gave this student a Medium question in this topic NOW, 
        # how likely are they to get it right?'
        
        # Build features for a hypothetical next question (Difficulty 0.5)
        probe_q = {"topic_id": tid, "difficulty": 0.5}
        feat = build_features(history, probe_q)
        
        proba = model.predict_proba(feat.reshape(1, -1))
        p_correct = float(proba[0, 1])
        
        if p_correct < LEVEL_THRESHOLDS["weak"]: level = "weak"
        elif p_correct > LEVEL_THRESHOLDS["strong"]: level = "strong"
        else: level = "medium"
        
        res[tid] = {
            "level": level, 
            "score": round(p_correct, 4), 
            "topic_name": TOPIC_NAMES.get(tid, f"Topic {tid}")
        }
    return res

def uncertainty_sampling(model, remaining, history):
    best_q, best_dist = None, float('inf')
    for q in remaining:
        feat = build_features(history, q)
        p = model.predict_proba(feat.reshape(1,-1))[0, 1]
        dist = abs(p - 0.5)
        if dist < best_dist:
            best_dist = dist
            best_q = q
    return best_q

def entropy_based(model, remaining, history):
    # Simplified entropy (pick q with p closest to median range)
    return uncertainty_sampling(model, remaining, history)

def query_by_committee(committee, remaining, history):
    best_q, best_var = None, -1.0
    for q in remaining:
        feat = build_features(history, q)
        preds = [m.predict_proba(feat.reshape(1,-1))[0, 1] for m in committee]
        var = np.var(preds)
        if var > best_var:
            best_var = var
            best_q = q
    return best_q

def simulate_answer(question, student_level=0.5, rng=None):
    if rng is None: rng = np.random.RandomState()
    p = 1.0 / (1.0 + math.exp(-(student_level - question["difficulty"])))
    correct = int(rng.random() < p)
    return {
        "question_id": question["question_id"], "topic_id": question["topic_id"],
        "difficulty": question["difficulty"], "correct": correct,
        "time": round(rng.uniform(5, 30) / 300.0, 4), "hint_count": 0, "attempt_count": 1
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_quiz_with_trace(
    model: Any,
    question_pool: List[Question],
    strategy_name: str,
    committee: Optional[List[Any]] = None,
    topics: Optional[List[int]] = None,
    seed: int = 42,
    answer_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    # ── Input validation ──────────────────────────────────────────────────────
    if not question_pool:
        raise ValueError("question_pool must not be empty.")
    if strategy_name not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy_name}'. "
            f"Choose from: {VALID_STRATEGIES}"
        )
    if strategy_name == "committee" and not committee:
        raise ValueError(
            "Strategy 'committee' requires exactly 5 committee models "
            "passed via the `committee` argument."
        )
    # ─────────────────────────────────────────────────────────────────────────
    if topics is None: topics = TOPIC_IDS
    rng = np.random.RandomState(seed)
    remaining = copy.deepcopy(question_pool)
    history = []
    conf_h = []

    if answer_fn is None:
        _ans_rng = np.random.RandomState(seed + 1)
        answer_fn = lambda q: simulate_answer(q, 0.5, _ans_rng)

    # Warm-up (1 per topic)
    for tid in topics:
        t_qs = [q for q in remaining if q["topic_id"] == tid]
        if not t_qs: continue
        q = min(t_qs, key=lambda x: abs(x["difficulty"] - 0.5))
        rec = answer_fn(q)
        history.append(rec)
        remaining = [x for x in remaining if x["question_id"] != q["question_id"]]
        conf_h.append(compute_confidence(history, topics))

    # Adaptive loop
    while len(history) < MAX_QUESTIONS:
        conf = compute_confidence(history, topics)
        if strategy_name != "random" and conf >= CONFIDENCE_THRESHOLD: break
        if not remaining: break

        if strategy_name == "uncertainty": q = uncertainty_sampling(model, remaining, history)
        elif strategy_name == "committee": q = query_by_committee(committee, remaining, history)
        elif strategy_name == "random": q = remaining[rng.randint(0, len(remaining))]
        else: q = uncertainty_sampling(model, remaining, history) # default

        rec = answer_fn(q)
        history.append(rec)
        remaining = [x for x in remaining if x["question_id"] != q["question_id"]]
        conf_h.append(compute_confidence(history, topics))

    return {
        "result": estimate_levels(history, model, topics),
        "history": history,
        "questions_asked": len(history),
        "confidence_history": conf_h
    }

def run_quiz(model, question_pool, strategy_name, committee=None, topics=None, seed=42, answer_fn=None):
    trace = run_quiz_with_trace(model, question_pool, strategy_name, committee, topics, seed, answer_fn)
    return trace["result"]
