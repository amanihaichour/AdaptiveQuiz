import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


# ======================
# 1. Load Processed Data
# ======================
train_df = pd.read_csv("../data/processed/train_data.csv")
test_df = pd.read_csv("../data/processed/test_data.csv")


# ======================
# 2. Feature Engineering Function
# ======================
def build_features(history, candidate_question, topics):
    features = []

    total_history = len(history)

    # Global features
    global_success = (
        sum(h['correct'] for h in history) / total_history
        if total_history > 0 else 0
    )

    global_avg_time = (
        np.mean([h['time'] for h in history])
        if total_history > 0 else 0
    )

    for topic in topics:
        topic_data = [h for h in history if h['topic_id'] == topic]
        total = len(topic_data)

        # Difficulty-based counts
        easy_correct = sum(1 for h in topic_data if h['difficulty'] < 0.33 and h['correct'] == 1)
        easy_wrong   = sum(1 for h in topic_data if h['difficulty'] < 0.33 and h['correct'] == 0)

        med_correct = sum(1 for h in topic_data if 0.33 <= h['difficulty'] <= 0.66 and h['correct'] == 1)
        med_wrong   = sum(1 for h in topic_data if 0.33 <= h['difficulty'] <= 0.66 and h['correct'] == 0)

        hard_correct = sum(1 for h in topic_data if h['difficulty'] > 0.66 and h['correct'] == 1)
        hard_wrong   = sum(1 for h in topic_data if h['difficulty'] > 0.66 and h['correct'] == 0)

        # Statistics
        success_rate = (
            sum(h['correct'] for h in topic_data) / total
            if total > 0 else 0
        )

        avg_time = (
            np.mean([h['time'] for h in topic_data])
            if total > 0 else 0
        )

        # Recent weighted performance
        if total > 0:
            weights = np.linspace(1, 2, total)
            recent_weighted = np.average(
                [h['correct'] for h in topic_data],
                weights=weights
            )
        else:
            recent_weighted = 0

        # Behavioral features
        avg_hints = (
            np.mean([h['hint_count'] for h in topic_data])
            if total > 0 else 0
        )

        avg_attempts = (
            np.mean([h['attempt_count'] for h in topic_data])
            if total > 0 else 0
        )

        features.extend([
            easy_correct, easy_wrong,
            med_correct, med_wrong,
            hard_correct, hard_wrong,
            success_rate, total, avg_time,
            recent_weighted,
            avg_hints, avg_attempts
        ])

    # Candidate features
    features.append(candidate_question['difficulty'])

    # Interaction feature
    features.append(global_success * candidate_question['difficulty'])

    # Global stats
    features.extend([
        global_success,
        global_avg_time,
        total_history
    ])

    # One-hot encoding of topic
    topic_vector = [0] * len(topics)
    topic_vector[candidate_question['topic_id']] = 1
    features.extend(topic_vector)

    return np.array(features)


# ======================
# 3. Build Dataset
# ======================
topics = sorted(train_df['topic_id'].unique())

X = []
y = []
groups = []

for student_id, student_data in train_df.groupby('user_id'):
    student_data = student_data.sort_index()
    history = []

    for _, row in student_data.iterrows():
        candidate = {
            'difficulty': row['difficulty'],
            'topic_id': row['topic_id']
        }

        features = build_features(history, candidate, topics)

        X.append(features)
        y.append(row['correct'])
        groups.append(student_id)

        # Update history
        history.append({
            'topic_id': row['topic_id'],
            'difficulty': row['difficulty'],
            'correct': row['correct'],
            'time': row['response_time_norm'],
            'hint_count': row.get('hint_count', 0),
            'attempt_count': row.get('attempt_count', 1)
        })

X = np.array(X)
y = np.array(y)
groups = np.array(groups)


# ======================
# 4. Save Outputs
# ======================
os.makedirs("../outputs", exist_ok=True)

np.save("../outputs/X_train.npy", X)
np.save("../outputs/y_train.npy", y)
np.save("../outputs/groups_train.npy", groups)