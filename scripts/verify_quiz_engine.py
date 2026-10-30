import sys
import os
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath('src'))
import quiz_engine as qe

def main():
    model = joblib.load('models/logistic_model.pkl')
    # Using real metadata for realism
    pool = qe.load_question_pool('data/processed/question_metadata.csv')
    
    # We focus on 3 topics: 0 (Conversion), 4 (Simple Equ), 6 (Subtraction)
    topics = [0, 4, 6]

    # CASE 1: FAILING STUDENT (0 correct)
    def fail_ans(q):
        return {
            'question_id': q['question_id'], 'topic_id': q['topic_id'], 'difficulty': q['difficulty'],
            'correct': 0, 'time': 0.05, 'hint_count': 1, 'attempt_count': 2
        }

    # CASE 2: PERFECT STUDENT (100% correct)
    def perf_ans(q):
        return {
            'question_id': q['question_id'], 'topic_id': q['topic_id'], 'difficulty': q['difficulty'],
            'correct': 1, 'time': 0.05, 'hint_count': 0, 'attempt_count': 1
        }

    print("============================================================")
    print("        QUIZ ENGINE - MASTERY & EFFICIENCY TEST")
    print("============================================================\n")

    for label, ans_fn in [('FAILING', fail_ans), ('PERFECT', perf_ans)]:
        trace = qe.run_quiz_with_trace(
            model, pool, 'uncertainty', topics=topics, answer_fn=ans_fn, seed=1
        )
        
        print(f"[{label} STUDENT RESULTS]")
        print(f"  Questions Asked : {trace['questions_asked']}")
        print(f"  Final Confidence: {trace['confidence_history'][-1]:.2f}")
        
        # Check topic levels
        for tid, info in trace['result'].items():
            print(f"  - {info['topic_name']:35s}: {info['level']:8s} (Score: {info['score']})")
        print("-" * 60)

    # FINAL VERIFICATION
    # Failing should have score < 0.55 (Weak)
    # Perfect should have score > 0.67 (Strong)
    
    print("\n[VERIFICATION SUMMARY]")
    f_res = qe.run_quiz(model, pool, 'uncertainty', topics=topics, answer_fn=fail_ans, seed=1)
    p_res = qe.run_quiz(model, pool, 'uncertainty', topics=topics, answer_fn=perf_ans, seed=1)
    
    f_score = np.mean([v['score'] for v in f_res.values()])
    p_score = np.mean([v['score'] for v in p_res.values()])
    
    if f_score < 0.55 and p_score > 0.67:
        print("  ✓ SUCCESS: All Mastery Thresholds Correctly Distinguished!")
    else:
        print("  ✗ FAILURE: Thresholds need more tuning.")
        print(f"    Failing Avg Score: {f_score:.4f} (Expected < 0.55)")
        print(f"    Perfect Avg Score: {p_score:.4f} (Expected > 0.67)")

if __name__ == "__main__":
    main()
