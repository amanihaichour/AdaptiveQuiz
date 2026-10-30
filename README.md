# ASSISTments Adaptive Quiz

An adaptive quiz that estimates a student's mastery of each math topic (**weak / medium / strong**) with as few questions as possible. A classifier trained on the [ASSISTments 2009–2010 Skill Builder](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) dataset predicts the chance that a student answers a question correctly. Active-learning strategies then use that prediction to pick the most informative next question.

## Quick start

Requires Python 3.10–3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_bank.py        # writes data/processed/full_question_bank.json
python src/main.py                 # serves API + UI on http://127.0.0.1:8000
```

Open http://127.0.0.1:8000, choose the topics you want to be tested on, and answer questions. The quiz ends after 20 questions, or earlier if the question bank runs out. It then shows a mastery level for each topic.

Run every command from the repository root.

## How it works

1. **Features (96 per prediction):** for each of the 7 topics, 12 statistics from the student's history so far (accuracy, average difficulty, time, hints, attempts, …), plus 5 global features and a one-hot vector for the candidate question's topic. See `build_features` in [src/quiz_engine.py](src/quiz_engine.py).
2. **Model:** a scikit-learn pipeline that predicts `P(correct | history, question)`. Logistic regression is the main model. A committee of 5 bootstrapped models is used for query-by-committee.
3. **Choosing the next question:**
   - *Warm-up:* one medium-difficulty question per selected topic.
   - *Adaptive:* `uncertainty` picks the question whose predicted probability is closest to 0.5, `entropy` is currently an alias for `uncertainty`, `committee` maximizes disagreement between committee models, and `random` is the baseline.
4. **Mastery estimate:** for each topic, the model scores a hypothetical medium-difficulty (0.5) question given the student's history. The score maps to a level: `< 0.45` → weak, `> 0.80` → strong, otherwise medium. These thresholds were calibrated by simulating students of known ability ([scripts/persona_stability.py](scripts/persona_stability.py)).

### Topics

| ID | Topic |
|----|-------|
| 0 | Fraction/Decimal/Percent Conversion |
| 1 | Finding Percents |
| 2 | Ordering Fractions |
| 3 | Equation Solving (Multi-Step) |
| 4 | Equation Solving (Simple) |
| 5 | Proportion/Fraction/Decimal/Percent |
| 6 | Subtraction Whole Numbers |

## Project layout

```
├── src/
│   ├── main.py                 # FastAPI server (API + static frontend)
│   ├── quiz_engine.py          # features, query strategies, mastery estimation
│   ├── generate_bank.py        # builds the UI question bank (JSON)
│   ├── feature_engineering.py  # builds training matrices from processed data
│   ├── train_models.py         # trains models + committee, learning curve
│   ├── evaluate.py             # simulation-based evaluation of strategies
│   └── plots.py                # evaluation plots
├── frontend/                   # static HTML/CSS/JS client
├── data/
│   ├── raw/                    # original ASSISTments CSV
│   └── processed/              # train/test splits + question metadata
├── models/                     # pre-trained .pkl models (used by the server)
├── notebooks/                  # EDA → training → demo → evaluation (01–06)
├── scripts/                    # sanity/calibration scripts
├── tests/                      # pytest suite
└── outputs/figures/            # generated figures
```

## API

All endpoints are stateless. The client sends its full answer history with each request.

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET  | `/api/topics` | – | `[{id, name}]` |
| POST | `/api/next-question` | `QuizState` | `{finished, question, confidence}` |
| POST | `/api/mastery` | `QuizState` | `{mastery: {topic_id: {level, score, topic_name}}}` |

```json
// QuizState
{
  "history": [
    {"question_id": "q3_4", "topic_id": 3, "difficulty": 0.4, "correct": 1, "time": 0.05}
  ],
  "topics_of_interest": [3, 4],
  "strategy_name": "uncertainty"     // or "committee"
}
```

`time` is the normalized response time: seconds / 300.

## Retraining

The committed models are enough to run the app. To rebuild them from the processed data:

```bash
python src/feature_engineering.py   # → outputs/{X,y,groups}_train.npy
python src/train_models.py          # → models/*.pkl, outputs/figures/learning_curve.png
```

Cross-validation uses `GroupKFold` grouped by student, so no student appears in both the training and validation folds. The raw-to-processed preprocessing step lives in [notebooks/01_eda_and_preprocessing.ipynb](notebooks/01_eda_and_preprocessing.ipynb).

## Tests and evaluation

```bash
pytest                                   # unit tests
python scripts/persona_stability.py      # simulated students across abilities → level check
python scripts/verify_quiz_engine.py     # weak vs. strong student sanity check
python src/evaluate.py --max-students 50 # strategy comparison → outputs/
```

## License

MIT. See [LICENSE](LICENSE).
