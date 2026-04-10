import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import argparse
from datetime import datetime

DATA_DIR = r'd:\科研项目\量子\co_qcr_experiment\data'
RESPONSES_DIR = os.path.join(DATA_DIR, 'responses')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

def check_step_completed(step_name):
    checks = {
        'download': os.path.exists(os.path.join(DATA_DIR, 'sampled_questions.json')),
        'api': check_api_completed(),
        'test': os.path.exists(os.path.join(RESULTS_DIR, 'comparison_results.json'))
    }
    return checks.get(step_name, False)

def check_api_completed():
    progress_file = os.path.join(DATA_DIR, 'progress_total.json')
    if not os.path.exists(progress_file):
        return False
    
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress = json.load(f)
    
    openrouter_done = all(
        len(progress.get('openrouter_completed', {}).get(model, [])) >= 50
        for model in ['z-ai/glm-4.5-air:free', 'minimax/minimax-m2.5:free']
    )
    
    zhipu_done = len(progress.get('zhipu_completed', [])) >= 50
    
    return openrouter_done and zhipu_done

def count_responses():
    if not os.path.exists(RESPONSES_DIR):
        return 0
    return len([f for f in os.listdir(RESPONSES_DIR) if f.endswith('.json')])

def run_step(step_name):
    print(f"\n{'='*60}")
    print(f"  执行步骤: {step_name}")
    print(f"{'='*60}\n")
    
    if step_name == 'download':
        from step1_download_dataset import download_and_sample
        download_and_sample()
    
    elif step_name == 'api':
        from step2_api_call import run_all_calls
        run_all_calls()
    
    elif step_name == 'test':
        from step4_quantum_test import run_comparison
        run_comparison()
    
    elif step_name == 'visualize':
        from step5_visualize import generate_all_figures
        generate_all_figures()

def main():
    parser = argparse.ArgumentParser(description='CO-QCR实验主控脚本')
    parser.add_argument('--step', type=str, 
                       choices=['download', 'api', 'test', 'visualize', 'all'],
                       default='all', help='执行的步骤')
    parser.add_argument('--status', action='store_true', help='显示当前状态')
    args = parser.parse_args()
    
    if args.status:
        print("\n" + "="*60)
        print("  CO-QCR 实验状态")
        print("="*60)
        
        steps = [
            ('download', '下载数据集'),
            ('api', 'API调用'),
            ('test', '量子测试对比'),
        ]
        
        for step_key, step_name in steps:
            completed = check_step_completed(step_key)
            status = "✓ 完成" if completed else "○ 未完成"
            
            if step_key == 'api':
                count = count_responses()
                print(f"  {status}  {step_name}: {count} 个响应")
            else:
                print(f"  {status}  {step_name}")
        
        print("="*60 + "\n")
        return
    
    print("\n" + "="*70)
    print("  CO-QCR (Causal-Ordered Quantum Combinatorial Reasoning)")
    print("  实验流程控制")
    print("="*70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    if args.step == 'all':
        steps_to_run = []
        
        if not check_step_completed('download'):
            steps_to_run.append('download')
        else:
            print("[跳过] 数据集已下载")
        
        if not check_step_completed('api'):
            steps_to_run.append('api')
        else:
            print("[跳过] API调用已完成")
        
        if not check_step_completed('test'):
            steps_to_run.append('test')
        else:
            print("[跳过] 量子测试已完成")
        
        steps_to_run.append('visualize')
        
        for step in steps_to_run:
            run_step(step)
    
    else:
        run_step(args.step)
    
    print("\n" + "="*70)
    print("  流程执行完毕")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
