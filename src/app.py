import os
import sys
import uuid
import math
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Ensure src is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quiz_engine as qe

app = Flask(__name__)
CORS(app)

print("Loading models and question pool...")
import json

try:
    MODELS = qe.load_models("models")
except Exception as e:
    print(f"Warning: Could not load real models, using mock. {e}")
    MODELS = {}
    
MAIN_MODEL = MODELS.get("logistic", qe.MockModel())

if all(f"committee_{i}" in MODELS for i in range(5)):
    COMMITTEE = [MODELS[f"committee_{i}"] for i in range(5)]
else:
    COMMITTEE = [qe.MockModel() for _ in range(5)]

# Load real questions
TOPIC_MAP = {"algebra": 0, "probability": 1, "calculus": 2, "statistics": 3, "geometry": 4}
with open("src/ui_questions.json", "r", encoding="utf-8") as f:
    RAW_POOL = json.load(f)

POOL = []
for i, q in enumerate(RAW_POOL):
    correct_letter = q.get("correct", "A")
    correct_idx = {"A":0, "B":1, "C":2, "D":3}.get(correct_letter, 0)
    opts = q.get("options", {"A":"1","B":"2","C":"3","D":"4"})
    
    POOL.append({
        "question_id": i + 1,
        "topic_id": TOPIC_MAP.get(q["topic"].lower(), 0),
        "difficulty": q["difficulty"],
        "text": q["text"],
        "options": [opts.get("A"), opts.get("B"), opts.get("C"), opts.get("D")],
        "correct_idx": correct_idx,
        "explanation": q.get("explanation", "")
    })

sessions = {}

@app.route("/")
def index():
    # Serve the new connected HTML file
    return send_file("../outputs/adaptive_quiz_connected.html")

@app.route("/api/start", methods=["POST"])
def start_quiz():
    data = request.json or {}
    strategy = data.get("strategy", "uncertainty")
    session_id = str(uuid.uuid4())
    
    # Initialize state
    sessions[session_id] = {
        "strategy": strategy,
        "history": [],
        "remaining": list(POOL),
        "topics": qe.TOPIC_IDS
    }
    
    return jsonify({"session_id": session_id, "status": "started"})

@app.route("/api/next", methods=["GET"])
def get_next():
    session_id = request.args.get("session_id")
    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
        
    state = sessions[session_id]
    strategy = state["strategy"]
    history = state["history"]
    remaining = state["remaining"]
    topics = state["topics"]
    
    if not remaining:
        return jsonify({"finished": True, "reason": "pool_empty"})

    def compute_demo_conf(hist):
        if not hist: return 0.0
        n = len(hist)
        # Base: each question contributes 0.1 confidence (stops ~Q9 base)
        base_c = n * 0.10
        # Bonus certainty from model predictions diverging from 0.5
        levs = qe.estimate_levels(hist, MAIN_MODEL, topics)
        avg_cert = sum(abs(v["score"] - 0.5) * 2 for v in levs.values()) / len(topics)
        return min(1.0, base_c + (avg_cert * 0.35))

    conf = compute_demo_conf(history)
    # Must answer at least 5 questions before early stopping kicks in
    MIN_QUESTIONS = 5
    if strategy != "random" and len(history) >= MIN_QUESTIONS and conf >= 0.85:
        return jsonify({"finished": True, "reason": "confidence_reached"})
    
    # Warm-up check: Ensure we have at least 1 question per topic
    history_tids = {r["topic_id"] for r in history}
    missing_topics = [t for t in topics if t not in history_tids]
    
    if missing_topics:
        # Ask a warm-up question for the first missing topic
        tid = missing_topics[0]
        t_qs = [q for q in remaining if q["topic_id"] == tid]
        if t_qs:
            q = min(t_qs, key=lambda x: abs(x["difficulty"] - 0.5))
        else:
            q = remaining[0]
    else:
        # Standard active learning sampling
        if strategy == "uncertainty": 
            q = qe.uncertainty_sampling(MAIN_MODEL, remaining, history)
        elif strategy == "committee": 
            q = qe.query_by_committee(COMMITTEE, remaining, history)
        elif strategy == "random": 
            q = remaining[np.random.randint(0, len(remaining))]
        else: 
            q = qe.uncertainty_sampling(MAIN_MODEL, remaining, history)
    
    return jsonify({
        "finished": False,
        "question": {
            "id": q["question_id"],
            "topic_id": list(TOPIC_MAP.keys())[q["topic_id"]], # Return string name for UI
            "difficulty": q["difficulty"],
            "text": q["text"],
            "options": q["options"],
            "correct_idx": q["correct_idx"],
            "explanation": q["explanation"]
        }
    })

@app.route("/api/answer", methods=["POST"])
def submit_answer():
    data = request.json or {}
    session_id = data.get("session_id")
    q_id = data.get("question_id")
    correct = data.get("correct")
    time_taken = data.get("time", 15.0)
    
    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
        
    state = sessions[session_id]
    
    # find question in remaining
    q = next((x for x in state["remaining"] if x["question_id"] == q_id), None)
    if not q:
        return jsonify({"error": "Question not found"}), 400
        
    # remove from remaining
    state["remaining"] = [x for x in state["remaining"] if x["question_id"] != q_id]
    
    # build answer record
    rec = {
        "question_id": q["question_id"],
        "topic_id": q["topic_id"],
        "difficulty": q["difficulty"],
        "correct": int(correct),
        "time": round(time_taken / 300.0, 4), 
        "hint_count": 0,
        "attempt_count": 1
    }
    state["history"].append(rec)
    
    def compute_demo_conf(hist):
        if not hist: return 0.0
        base_c = len(hist) * 0.06
        levs = qe.estimate_levels(hist, MAIN_MODEL, state["topics"])
        avg_cert = sum(abs(v["score"] - 0.5)*2 for v in levs.values()) / len(state["topics"])
        return min(1.0, base_c + (avg_cert * 0.4))

    # Compute current estimates
    scaled_conf = min(1.0, compute_demo_conf(state["history"]) / 0.85)
    
    levels = qe.estimate_levels(state["history"], MAIN_MODEL, state["topics"])
    
    return jsonify({
        "confidence": scaled_conf,
        "levels": levels,
        "history_len": len(state["history"])
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
