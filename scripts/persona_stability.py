import sys
import os
import joblib
import numpy as np

# Add src to our path
sys.path.insert(0, os.path.abspath('src'))
import quiz_engine as qe

def run_n_tests(model, pool, topics, ability, label, n_tests=20):
    """
    Runs N quiz simulations for a student type and averages the results.
    """
    all_avg_scores = []
    
    for i in range(n_tests):
        test_seed = 42 + i
        rng = np.random.RandomState(test_seed)
        
        def student_ans(q):
            is_correct = 1 if rng.random() < ability else 0
            return {
                'question_id': q['question_id'], 'topic_id': q['topic_id'],
                'difficulty': q['difficulty'], 'correct': is_correct,
                'time': 0.05, 'hint_count': 0, 'attempt_count': 1
            }
        
        # Run quiz (random)
        res = qe.run_quiz_with_trace(model, pool, "random", topics=topics, answer_fn=student_ans, seed=test_seed)
        
        # Grand average across topics for this persona
        quiz_avg = np.mean([res['result'][tid]['score'] for tid in topics])
        all_avg_scores.append(quiz_avg)
    
    grand_avg = np.mean(all_avg_scores)
    
    if grand_avg < qe.LEVEL_THRESHOLDS["weak"]: label_out = "WEAK"
    elif grand_avg > qe.LEVEL_THRESHOLDS["strong"]: label_out = "STRONG"
    else: label_out = "MEDIUM"
    
    print(f"{label:15s} | Ability: {ability:.2f} | Avg Score: {grand_avg:.4f} | Final Level: {label_out}")
    return grand_avg

def main():
    print("================================================================")
    print("      DEEP STABILITY TEST - FULL SKILL SPECTRUM (20 Trials)")
    print("================================================================\n")
    print(f"Current Thresholds -> Weak (< {qe.LEVEL_THRESHOLDS['weak']}) | Strong (> {qe.LEVEL_THRESHOLDS['strong']})\n")
    
    # 1. Load real model
    try:
        models = qe.load_models('models/')
        model = models['logistic']
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup pool/topics
    pool = qe._generate_fake_question_pool()
    topics = [0, 4, 6]

    # 3. Test 7 personas to map the full curve
    personas = [
        (0.10, "FAILING"),
        (0.30, "STRUGGLING"),
        (0.50, "BORDERLINE"),
        (0.60, "AVERAGE"),
        (0.75, "ADVANCED"),
        (0.90, "HIGH-ADV"),
        (1.00, "PERFECT"),
    ]

    print(f"{'PERSONA':15s} | {'ABILITY':10s} | {'AVG SCORE':10s} | {'LABEL'}")
    print("-" * 65)
    for ability, label in personas:
        run_n_tests(model, pool, topics, ability, label, n_tests=20)

if __name__ == "__main__":
    main()
