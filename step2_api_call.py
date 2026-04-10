import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import time
import requests
from datetime import datetime

DATA_DIR = r'd:\科研项目\量子\co_qcr_experiment\data'
OPENROUTER_KEY_FILE = r'd:\科研项目\量子\key.txt'
ZHIPU_KEY_FILE = r'd:\科研项目\量子\zipukey.txt'
QUESTIONS_FILE = os.path.join(DATA_DIR, 'sampled_questions.json')
RESPONSES_DIR = os.path.join(DATA_DIR, 'responses')

OPENROUTER_MODELS = [
    'z-ai/glm-4.5-air:free',
    'minimax/minimax-m2.5:free',
]

OPENROUTER_CALLS_PER_MODEL = 50
ZHIPU_CALLS = 50
ZHIPU_MODEL = 'glm-4.7-flash'

OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
ZHIPU_API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

def load_api_key(key_file):
    with open(key_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_questions():
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_progress_file():
    return os.path.join(DATA_DIR, 'progress_total.json')

def load_progress():
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'openrouter_completed': {},
        'zhipu_completed': [],
        'total_calls': 0
    }

def save_progress(progress):
    progress_file = get_progress_file()
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def call_openrouter(api_key, model, question, question_id):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/co-qcr-experiment',
        'X-OpenRouter-Title': 'CO-QCR Experiment'
    }
    
    system_prompt = """请仔细思考并回答以下问题。请使用"逐步思考"的方式，在回答中展示你的推理过程。
格式要求：
1. 首先列出你的思考步骤
2. 然后给出最终答案
请用中文回答。"""

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ],
        'temperature': 0.7,
        'max_tokens': 2048
    }
    
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'content': result['choices'][0]['message']['content'],
                'model': result.get('model', model),
                'usage': result.get('usage', {}),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def call_zhipu(api_key, question, question_id):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    system_prompt = """请仔细思考并回答以下问题。请使用"逐步思考"的方式，在回答中展示你的推理过程。
格式要求：
1. 首先列出你的思考步骤
2. 然后给出最终答案
请用中文回答。"""

    payload = {
        'model': ZHIPU_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ],
        'temperature': 0.7,
        'max_tokens': 2048
    }
    
    try:
        response = requests.post(
            ZHIPU_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'content': result['choices'][0]['message']['content'],
                'model': ZHIPU_MODEL,
                'usage': result.get('usage', {}),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_openrouter_calls(progress, questions, api_key):
    print(f"\n{'='*60}")
    print("  OpenRouter API 调用")
    print(f"  目标: 每个模型 {OPENROUTER_CALLS_PER_MODEL} 次")
    print(f"{'='*60}\n")
    
    model_short_names = {
        'z-ai/glm-4.5-air:free': 'glm_openrouter',
        'minimax/minimax-m2.5:free': 'minimax'
    }
    
    for model in OPENROUTER_MODELS:
        model_short = model_short_names.get(model, model.split('/')[-1])
        
        if model not in progress['openrouter_completed']:
            progress['openrouter_completed'][model] = []
        
        completed_count = len(progress['openrouter_completed'][model])
        
        if completed_count >= OPENROUTER_CALLS_PER_MODEL:
            print(f"[跳过] {model_short}: 已完成 {completed_count}/{OPENROUTER_CALLS_PER_MODEL} 次")
            continue
        
        print(f"\n[模型] {model_short}")
        print(f"  已完成: {completed_count}/{OPENROUTER_CALLS_PER_MODEL}")
        
        calls_needed = OPENROUTER_CALLS_PER_MODEL - completed_count
        
        for call_idx in range(calls_needed):
            q_idx = (completed_count + call_idx) % len(questions)
            question = questions[q_idx]
            
            print(f"  调用 {completed_count + call_idx + 1}/{OPENROUTER_CALLS_PER_MODEL}: 问题 {question['id'][:8]}...")
            
            result = call_openrouter(api_key, model, question['question'], question['id'])
            
            if result['success']:
                response_data = {
                    'question_id': question['id'],
                    'question': question['question'],
                    'model': model,
                    'model_short': model_short,
                    'source': 'openrouter',
                    'call_idx': completed_count + call_idx,
                    'response': result['content'],
                    'usage': result['usage'],
                    'timestamp': result['timestamp']
                }
                
                response_file = os.path.join(
                    RESPONSES_DIR,
                    f"openrouter_{model_short}_q{question['id']}_{completed_count + call_idx}.json"
                )
                with open(response_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)
                
                progress['openrouter_completed'][model].append({
                    'question_id': question['id'],
                    'file': os.path.basename(response_file),
                    'timestamp': result['timestamp']
                })
                progress['total_calls'] += 1
                
                print(f"    ✓ 成功 (tokens: {result['usage'].get('total_tokens', 'N/A')})")
            else:
                print(f"    ✗ 失败: {result['error'][:100]}")
            
            save_progress(progress)
            time.sleep(3)
    
    return progress

def run_zhipu_calls(progress, questions, api_key):
    print(f"\n{'='*60}")
    print("  智谱AI API 调用")
    print(f"  模型: {ZHIPU_MODEL}")
    print(f"  目标: {ZHIPU_CALLS} 次")
    print(f"{'='*60}\n")
    
    completed_count = len(progress['zhipu_completed'])
    
    if completed_count >= ZHIPU_CALLS:
        print(f"[跳过] 智谱AI: 已完成 {completed_count}/{ZHIPU_CALLS} 次")
        return progress
    
    print(f"已完成: {completed_count}/{ZHIPU_CALLS}")
    
    calls_needed = ZHIPU_CALLS - completed_count
    
    for call_idx in range(calls_needed):
        q_idx = (completed_count + call_idx) % len(questions)
        question = questions[q_idx]
        
        print(f"  调用 {completed_count + call_idx + 1}/{ZHIPU_CALLS}: 问题 {question['id'][:8]}...")
        
        result = call_zhipu(api_key, question['question'], question['id'])
        
        if result['success']:
            response_data = {
                'question_id': question['id'],
                'question': question['question'],
                'model': ZHIPU_MODEL,
                'model_short': 'zhipu_glm',
                'source': 'zhipu',
                'call_idx': completed_count + call_idx,
                'response': result['content'],
                'usage': result['usage'],
                'timestamp': result['timestamp']
            }
            
            response_file = os.path.join(
                RESPONSES_DIR,
                f"zhipu_glm_q{question['id']}_{completed_count + call_idx}.json"
            )
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            
            progress['zhipu_completed'].append({
                'question_id': question['id'],
                'file': os.path.basename(response_file),
                'timestamp': result['timestamp']
            })
            progress['total_calls'] += 1
            
            print(f"    ✓ 成功 (tokens: {result['usage'].get('total_tokens', 'N/A')})")
        else:
            print(f"    ✗ 失败: {result['error'][:100]}")
        
        save_progress(progress)
        time.sleep(1)
    
    return progress

def check_status():
    progress = load_progress()
    
    print("\n" + "="*60)
    print("  CO-QCR 数据收集状态")
    print("="*60)
    
    print("\n[OpenRouter]")
    for model in OPENROUTER_MODELS:
        completed = len(progress.get('openrouter_completed', {}).get(model, []))
        model_short = model.split('/')[-1].replace(':free', '')
        status = "✓ 完成" if completed >= OPENROUTER_CALLS_PER_MODEL else f"进行中"
        print(f"  {model_short}: {completed}/{OPENROUTER_CALLS_PER_MODEL} 次 [{status}]")
    
    print("\n[智谱AI]")
    zhipu_completed = len(progress.get('zhipu_completed', []))
    status = "✓ 完成" if zhipu_completed >= ZHIPU_CALLS else "进行中"
    print(f"  {ZHIPU_MODEL}: {zhipu_completed}/{ZHIPU_CALLS} 次 [{status}]")
    
    print(f"\n[总计] {progress.get('total_calls', 0)} 次调用")
    print("="*60 + "\n")

def run_all_calls():
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    
    openrouter_key = load_api_key(OPENROUTER_KEY_FILE)
    zhipu_key = load_api_key(ZHIPU_KEY_FILE)
    questions = load_questions()
    progress = load_progress()
    
    print(f"\n{'='*70}")
    print("  CO-QCR CoT 数据收集")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  问题数: {len(questions)}")
    print(f"{'='*70}")
    
    progress = run_openrouter_calls(progress, questions, openrouter_key)
    progress = run_zhipu_calls(progress, questions, zhipu_key)
    
    print(f"\n{'='*70}")
    print("  数据收集完成")
    print(f"  总调用: {progress['total_calls']} 次")
    print(f"{'='*70}\n")
    
    return progress

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true', help='仅检查状态')
    parser.add_argument('--openrouter', action='store_true', help='仅运行OpenRouter')
    parser.add_argument('--zhipu', action='store_true', help='仅运行智谱AI')
    args = parser.parse_args()
    
    if args.status:
        check_status()
    else:
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        
        openrouter_key = load_api_key(OPENROUTER_KEY_FILE)
        zhipu_key = load_api_key(ZHIPU_KEY_FILE)
        questions = load_questions()
        progress = load_progress()
        
        print(f"\n{'='*70}")
        print("  CO-QCR CoT 数据收集")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  问题数: {len(questions)}")
        print(f"{'='*70}")
        
        if args.openrouter:
            progress = run_openrouter_calls(progress, questions, openrouter_key)
        elif args.zhipu:
            progress = run_zhipu_calls(progress, questions, zhipu_key)
        else:
            progress = run_openrouter_calls(progress, questions, openrouter_key)
            progress = run_zhipu_calls(progress, questions, zhipu_key)
        
        print(f"\n{'='*70}")
        print("  数据收集完成")
        print(f"  总调用: {progress['total_calls']} 次")
        print(f"{'='*70}\n")
