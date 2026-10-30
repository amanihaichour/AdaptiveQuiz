import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quiz_engine as qe

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP & MODELS
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Adaptive Quiz API", description="Powered by Logistic Regression (96 Features)")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model/pool state
MODELS = {}
QUESTION_POOL = []

@app.on_event("startup")
def startup_event():
    global MODELS, QUESTION_POOL
    # 1. Load ML models
    models_dir = os.path.join(os.getcwd(), "models")
    MODELS = qe.load_models(models_dir)
    
    # 2. Load Question Bank
    bank_path = os.path.join(os.getcwd(), "data", "processed", "full_question_bank.json")
    if os.path.isfile(bank_path):
        with open(bank_path, "r") as f:
            QUESTION_POOL = json.load(f)

    # 3. Serve Frontend
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class HistoryItem(BaseModel):
    question_id: str
    topic_id: int
    difficulty: float
    correct: int
    time: float
    hint_count: int = 0
    attempt_count: int = 1

class QuizState(BaseModel):
    history: List[HistoryItem]
    topics_of_interest: List[int]
    strategy_name: str = "uncertainty"

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/topics")
def get_topics():
    return [{"id": k, "name": v} for k, v in qe.TOPIC_NAMES.items()]

@app.post("/api/next-question")
def get_next_question(state: QuizState):
    """
    Given the current student history, use the ML model to pick the 
    most informative NEXT question.
    """
    history_dicts = [h.dict() for h in state.history]
    
    # Filter remaining questions
    asked_ids = {h.question_id for h in state.history}
    remaining = [q for q in QUESTION_POOL 
                 if q["problem_id"] not in asked_ids
                 and q["topic_id"] in state.topics_of_interest]
    
    if not remaining:
        return {"finished": True}

    # Initial Phase (1 per topic)
    current_topic_coverage = {tid: sum(1 for h in history_dicts if h["topic_id"] == tid) for tid in state.topics_of_interest}
    next_tid = next((tid for tid, count in current_topic_coverage.items() if count == 0), None)
    
    selected_question = None
    if next_tid is not None:
        # Warm-up phase: Pick a mid-difficulty question for a new topic
        t_qs = [q for q in remaining if q["topic_id"] == next_tid]
        selected_question = min(t_qs, key=lambda x: abs(x["difficulty"] - 0.5)) if t_qs else remaining[0]
    else:
        # Adaptive phase: Use Uncertainty Sampling
        # Ensure we use the best available model (logistic or committee)
        model = MODELS.get("logistic")
        if state.strategy_name == "committee":
            selected_question = qe.query_by_committee(MODELS.get("committee"), remaining, history_dicts)
        else:
            selected_question = qe.uncertainty_sampling(model, remaining, history_dicts)

    return {
        "finished": False,
        "question": selected_question,
        "confidence": qe.compute_confidence(history_dicts, state.topics_of_interest)
    }

@app.post("/api/mastery")
def get_mastery(state: QuizState):
    """
    Uses the ML model to estimate the student's mastery levels for each topic.
    """
    history_dicts = [h.dict() for h in state.history]
    results = qe.estimate_levels(history_dicts, MODELS.get("logistic"), state.topics_of_interest)
    return {"mastery": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
