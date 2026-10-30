/**
 * app.js - Adaptive Math Hub
 * ML-Engine Integration Frontend
 */

const API_BASE = "http://127.0.0.1:8000";

let state = {
    history: [],
    topics_of_interest: [],
    current_q: null,
    start_time: null,
    finished: false,
    confidence: 0
};

// --- Initialization ---
window.onload = async () => {
    fetchTopics();
};

async function fetchTopics() {
    try {
        const response = await fetch(`${API_BASE}/api/topics`);
        const topics = await response.json();
        
        const container = document.getElementById("topic-selector");
        container.innerHTML = "";
        
        topics.forEach(t => {
            const btn = document.createElement("button");
            btn.className = "option-btn";
            btn.innerText = t.name;
            btn.onclick = () => {
                btn.classList.toggle("selected");
                updateTopicList(t.id);
            };
            container.appendChild(btn);
        });
        
        document.getElementById("begin-quiz").classList.remove("hidden");
    } catch (err) {
        logReasoning("Backend offline! Make sure main.py is running on port 8000.");
    }
}

function updateTopicList(topicId) {
    const idx = state.topics_of_interest.indexOf(topicId);
    if (idx > -1) state.topics_of_interest.splice(idx, 1);
    else state.topics_of_interest.push(topicId);
}

// --- Quiz Flow ---
document.getElementById("begin-quiz").onclick = async () => {
    if (state.topics_of_interest.length === 0) {
        alert("Please select at least one topic.");
        return;
    }
    document.getElementById("setup-screen").classList.add("hidden");
    document.getElementById("quiz-container").classList.remove("hidden");
    getNextQuestion();
};

async function getNextQuestion() {
    try {
        const response = await fetch(`${API_BASE}/api/next-question`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                history: state.history,
                topics_of_interest: state.topics_of_interest,
                strategy_name: "uncertainty"
            })
        });
        
        const data = await response.json();
        if (data.finished || state.history.length >= 20) {
            finishQuiz();
            return;
        }

        renderQuestion(data.question);
        updateStats(data.confidence);
        logReasoning(`ML Engine: Picking topic ${data.question.topic_id} (Diff: ${data.question.difficulty}) for maximum information gain.`);
    } catch (err) {
        logReasoning("Error fetching next question.");
    }
}

function renderQuestion(q) {
    state.current_q = q;
    state.start_time = Date.now();
    
    document.getElementById("topic-display").innerText = q.skill_name || "Math Topic";
    document.getElementById("diff-display").innerText = q.difficulty.toFixed(2);
    document.getElementById("question-text").innerText = q.text || "No text available.";
    
    const optionsGrid = document.getElementById("options-grid");
    optionsGrid.innerHTML = "";
    
    q.options.forEach(opt => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.innerText = opt;
        btn.onclick = () => submitAnswer(opt);
        optionsGrid.appendChild(btn);
    });
}

function submitAnswer(chosen) {
    const timeTaken = (Date.now() - state.start_time) / 300000; // Normalized for ML model
    const isCorrect = chosen === state.current_q.answer ? 1 : 0;
    
    logReasoning(`Student answered ${isCorrect ? "Correctly" : "Incorrectly"} in ${((timeTaken*300000)/1000).toFixed(1)}s.`);
    
    state.history.push({
        question_id: state.current_q.problem_id,
        topic_id: state.current_q.topic_id,
        difficulty: state.current_q.difficulty,
        correct: isCorrect,
        time: timeTaken,
        hint_count: 0,
        attempt_count: 1
    });
    
    fetchMastery();
    getNextQuestion();
}

async function fetchMastery() {
    try {
        const response = await fetch(`${API_BASE}/api/mastery`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                history: state.history,
                topics_of_interest: state.topics_of_interest
            })
        });
        
        const data = await response.json();
        renderMastery(data.mastery);
    } catch (err) {}
}

function renderMastery(mastery) {
    const list = document.getElementById("mastery-list");
    list.innerHTML = "";
    
    Object.values(mastery).forEach(m => {
        const item = document.createElement("div");
        item.className = "topic-level";
        const badgeClass = `level-${m.level}`;
        item.innerHTML = `
            <span>${m.topic_name}</span>
            <span class="level-badge ${badgeClass}">${m.level}</span>
        `;
        list.appendChild(item);
    });
}

function updateStats(confidence) {
    const fill = document.getElementById("conf-fill");
    fill.style.width = `${Math.min(confidence*100, 100)}%`;
    document.getElementById("conf-text").innerText = `Engine Certainty: ${(confidence*100).toFixed(0)}%`;
}

function finishQuiz() {
    document.getElementById("quiz-container").classList.add("hidden");
    document.getElementById("results-screen").classList.remove("hidden");
    logReasoning("Quiz Finished. Model confidence threshold reached.");
}

function logReasoning(msg) {
    const trace = document.getElementById("reasoning-trace");
    const p = document.createElement("p");
    p.innerText = `> ${msg}`;
    trace.prepend(p);
}
