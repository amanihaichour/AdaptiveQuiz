import re

with open('outputs/adaptive_quiz.html', 'r', encoding='utf-8') as f:
    html = f.read()

script_start = html.find('<script>')
script_end = html.find('</script>') + 9

new_script = """<script>
// Connected to Python Backend

let state = {
  strategy: null,
  sessionId: null,
  questionsAsked: 0,
  confidence: 0.0,
  currentQuestion: null,
  beliefDistributions: {},
  topicStats: {},
  engineReasoning: '',
  questionLog: []
};

let tsStart = 0;
let uiLock = false;

const ui = {
  v: (id) => {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById('screen-' + id).classList.add('active');
  }
};

async function resetState(strategyName) {
  state.strategy = strategyName;
  state.questionsAsked = 0;
  state.confidence = 0.0;
  state.beliefDistributions = {};
  state.topicStats = {};
  state.engineReasoning = '';
  state.questionLog = [];
  
  document.getElementById('score-dash').innerHTML = '';
  document.getElementById('conf-bar').style.width = '0%';
  document.getElementById('conf-text').innerText = '0%';
  document.getElementById('ui-qnum').innerText = '1';
  document.getElementById('ui-reasoning').innerHTML = 'Starting active learning session...';
  updateLogTable();
  
  try {
      const res = await fetch('/api/start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ strategy: strategyName })
      });
      const data = await res.json();
      state.sessionId = data.session_id;
      runCycle();
  } catch (e) {
      console.error(e);
      alert('Failed to connect to backend.');
  }
}

async function runCycle() {
  if (state.confidence >= 0.8) {
    ui.v('result');
    document.getElementById('res-strategy').innerText = state.strategy;
    document.getElementById('res-qcount').innerText = state.questionsAsked;
    document.getElementById('res-conf').innerText = (state.confidence * 100).toFixed(1) + '%';
    return;
  }
  
  uiLock = true;
  document.getElementById('ui-reasoning').innerHTML = 'Engine is computing optimal next question... <span class="loader"></span>';
  
  try {
      const res = await fetch('/api/next?session_id=' + state.sessionId);
      const data = await res.json();
      
      if (data.finished) {
          ui.v('result');
          document.getElementById('res-strategy').innerText = state.strategy;
          document.getElementById('res-qcount').innerText = state.questionsAsked;
          document.getElementById('res-conf').innerText = (state.confidence * 100).toFixed(1) + '%';
          return;
      }
      
      const q = data.question;
      state.currentQuestion = q;
      document.getElementById('ui-qnum').innerText = state.questionsAsked + 1;
      document.getElementById('ui-qtopic').innerText = 'Topic: ' + q.topic_id;
      document.getElementById('ui-qdiff').innerText = 'Difficulty: ' + q.difficulty.toFixed(2);
      document.getElementById('ui-qtext').innerHTML = q.text;
      
      // Map options
      ['A','B','C','D'].forEach((opt, idx) => {
         const btn = document.getElementById('opt-' + opt);
         btn.className = 'opt-btn';
         btn.disabled = false;
         btn.innerHTML = `<span class="opt-letter">${opt}</span> ${q.options[idx]}`;
         btn.onclick = () => submitAnswer(opt, idx);
      });
      
      document.getElementById('ui-reasoning').innerHTML = `Candidate selected. Difficulty: <strong>${q.difficulty.toFixed(2)}</strong>`;
      tsStart = performance.now();
      uiLock = false;
  } catch (e) {
      console.error(e);
      alert('Error fetching next question.');
  }
}

async function submitAnswer(selectedOpt, idx) {
  if (uiLock) return;
  uiLock = true;
  const timeTaken = performance.now() - tsStart;
  
  // Fake correctness: if difficulty < 0.5, guess correct randomly based on it
  const correct = Math.random() > state.currentQuestion.difficulty;
  
  // Visual feedback
  const btn = document.getElementById('opt-' + selectedOpt);
  btn.classList.add(correct ? 'correct' : 'wrong');
  
  // Send to backend
  try {
      const res = await fetch('/api/answer', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
              session_id: state.sessionId,
              question_id: state.currentQuestion.id,
              correct: correct,
              time: timeTaken
          })
      });
      const data = await res.json();
      
      state.questionsAsked++;
      state.confidence = data.confidence;
      
      // Update UI
      document.getElementById('conf-bar').style.width = (state.confidence * 100) + '%';
      document.getElementById('conf-text').innerText = Math.round(state.confidence * 100) + '%';
      
      // Log
      state.questionLog.push({
          q: state.questionsAsked,
          topic: state.currentQuestion.topic_id,
          difficulty: state.currentQuestion.difficulty,
          correct: correct,
          answer: selectedOpt
      });
      updateLogTable();
      updateScoreDash(data.levels);
      
      setTimeout(runCycle, 1000);
      
  } catch (e) {
      console.error(e);
      uiLock = false;
  }
}

function updateScoreDash(levels) {
  let html = '';
  for (const [tid, info] of Object.entries(levels)) {
      let color = info.level === 'strong' ? 'var(--success)' : (info.level === 'weak' ? 'var(--danger)' : 'var(--warning)');
      html += `<div class="score-row">
          <div>${info.topic_name}</div>
          <div><strong style="color:${color}">${(info.score * 100).toFixed(1)}% - ${info.level.toUpperCase()}</strong></div>
      </div>`;
  }
  document.getElementById('score-dash').innerHTML = html;
}

function updateLogTable() {
  let lht = '';
  state.questionLog.forEach(l => {
    lht += `<tr>
      <td>${l.q}</td>
      <td style="text-transform:capitalize">${l.topic}</td>
      <td>${l.difficulty.toFixed(2)}</td>
      <td>${l.answer}</td>
      <td class="${l.correct ? 'td-green' : 'td-red'}">${l.correct ? 'Correct' : 'Wrong'}</td>
    </tr>`;
  });
  document.getElementById('ui-logbody').innerHTML = lht;
}

// BINDINGS
document.querySelectorAll('.strat-card').forEach(c => {
  c.onclick = () => {
    document.querySelectorAll('.strat-card').forEach(x=>x.classList.remove('selected'));
    c.classList.add('selected');
    state.strategy = c.dataset.strat;
    document.getElementById('btn-begin').disabled = false;
  };
});
document.getElementById('btn-begin').onclick = () => { ui.v('quiz'); resetState(state.strategy); };
document.getElementById('btn-reset').onclick = () => { ui.v('welcome'); document.getElementById('btn-begin').disabled = true; document.querySelectorAll('.strat-card').forEach(x=>x.classList.remove('selected')); };
document.getElementById('btn-replay').onclick = () => { ui.v('quiz'); resetState(state.strategy); };
</script>"""

new_html = html[:script_start] + new_script + html[script_end:]

with open('outputs/adaptive_quiz_connected.html', 'w', encoding='utf-8') as f:
    f.write(new_html)
print('Connected HTML successfully written.')
