import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json

DATA_DIR = r'd:\科研项目\量子\co_qcr_experiment\data'
RESPONSES_DIR = os.path.join(DATA_DIR, 'responses')
PROGRESS_FILE = os.path.join(DATA_DIR, 'progress_total.json')

def migrate_old_responses():
    progress = {
        'date': '2026-04-09',
        'openrouter_completed': {
            'z-ai/glm-4.5-air:free': [],
            'minimax/minimax-m2.5:free': []
        },
        'zhipu_completed': [],
        'total_calls': 0
    }
    
    if not os.path.exists(RESPONSES_DIR):
        return progress
    
    for filename in os.listdir(RESPONSES_DIR):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(RESPONSES_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qid = data.get('question_id', '')
            model = data.get('model', '')
            model_short = data.get('model_short', '')
            
            if 'glm' in model.lower() or model_short == 'glm':
                progress['openrouter_completed']['z-ai/glm-4.5-air:free'].append({
                    'question_id': qid,
                    'file': filename,
                    'timestamp': data.get('timestamp', '')
                })
                progress['total_calls'] += 1
            
            elif 'minimax' in model.lower() or 'minimax' in model_short.lower():
                progress['openrouter_completed']['minimax/minimax-m2.5:free'].append({
                    'question_id': qid,
                    'file': filename,
                    'timestamp': data.get('timestamp', '')
                })
                progress['total_calls'] += 1
            
            elif 'zhipu' in model_short.lower() or 'zhipu' in data.get('source', '').lower():
                progress['zhipu_completed'].append({
                    'question_id': qid,
                    'file': filename,
                    'timestamp': data.get('timestamp', '')
                })
                progress['total_calls'] += 1
        
        except Exception as e:
            print(f"[警告] 处理文件 {filename} 失败: {e}")
    
    return progress

if __name__ == "__main__":
    progress = migrate_old_responses()
    
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    
    print(f"[完成] 迁移进度:")
    print(f"  OpenRouter GLM: {len(progress['openrouter_completed'].get('z-ai/glm-4.5-air:free', []))} 次")
    print(f"  OpenRouter Minimax: {len(progress['openrouter_completed'].get('minimax/minimax-m2.5:free', []))} 次")
    print(f"  智谱AI: {len(progress['zhipu_completed'])} 次")
    print(f"  总计: {progress['total_calls']} 次")
