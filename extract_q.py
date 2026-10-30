import json
import re

with open('outputs/adaptive_quiz.html', 'r', encoding='utf-8') as f:
    html = f.read()

m = re.search(r'const QUESTIONS = (\[.*?\]);', html, flags=re.DOTALL)
if m:
    q_str = m.group(1)
    
    # remove comments
    q_str = re.sub(r'//.*', '', q_str)
    
    # fix keys: only match keys preceded by { or , and optional whitespace
    q_str = re.sub(r'(?<=[{,])\s*([a-zA-Z0-9_]+):', r' "\1":', q_str)
    
    try:
        data = json.loads(q_str)
        with open('src/ui_questions.json', 'w', encoding='utf-8') as out:
            json.dump(data, out, indent=2)
        print('Saved ui_questions.json!')
    except Exception as e:
        print('JSON Error:', e)
        with open('src/ui_questions_raw.json', 'w', encoding='utf-8') as out:
            out.write(q_str)
