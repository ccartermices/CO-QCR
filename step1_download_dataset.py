import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import random
from modelscope.msdatasets import MsDataset

OUTPUT_DIR = r'd:\科研项目\量子\co_qcr_experiment\data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sampled_questions.json')

def download_and_sample(n_samples=20, seed=42):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"[跳过] 数据集已存在: {OUTPUT_FILE}")
        print(f"[跳过] 已有 {len(existing_data)} 条问题")
        return existing_data
    
    print("[步骤1] 下载 Chinese-SimpleQA 数据集...")
    
    try:
        ds = MsDataset.load(
            'AI-ModelScope/Chinese-SimpleQA',
            subset_name='default',
            split='train'
        )
        
        all_questions = []
        for item in ds:
            question_data = {
                'id': item.get('id', item.get('question_id', len(all_questions))),
                'question': item.get('question', item.get('prompt', '')),
                'answer': item.get('answer', item.get('response', '')),
                'category': item.get('category', item.get('type', 'unknown'))
            }
            if question_data['question']:
                all_questions.append(question_data)
        
        print(f"[信息] 数据集共 {len(all_questions)} 条问题")
        
    except Exception as e:
        print(f"[警告] 使用modelscope下载失败: {e}")
        print("[信息] 使用备用简单问题集...")
        all_questions = generate_backup_questions()
    
    random.seed(seed)
    if len(all_questions) > n_samples:
        sampled = random.sample(all_questions, n_samples)
    else:
        sampled = all_questions
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    
    print(f"[完成] 已保存 {len(sampled)} 条问题到: {OUTPUT_FILE}")
    
    print("\n[预览] 前5条问题:")
    for i, q in enumerate(sampled[:5]):
        print(f"  {i+1}. {q['question'][:50]}...")
    
    return sampled


def generate_backup_questions():
    backup = [
        {"id": 1, "question": "什么是量子纠缠？请解释其原理和应用。", "answer": "", "category": "科学"},
        {"id": 2, "question": "请解释机器学习中的过拟合问题及其解决方法。", "answer": "", "category": "技术"},
        {"id": 3, "question": "什么是区块链技术？它有哪些主要应用场景？", "answer": "", "category": "技术"},
        {"id": 4, "question": "请解释相对论的基本原理及其对现代物理学的影响。", "answer": "", "category": "科学"},
        {"id": 5, "question": "什么是深度学习？它与传统机器学习有什么区别？", "answer": "", "category": "技术"},
        {"id": 6, "question": "请解释气候变化的主要原因及其对环境的影响。", "answer": "", "category": "环境"},
        {"id": 7, "question": "什么是基因编辑技术CRISPR？它有哪些应用和伦理问题？", "answer": "", "category": "生物"},
        {"id": 8, "question": "请解释量子计算的基本原理及其潜在应用。", "answer": "", "category": "科学"},
        {"id": 9, "question": "什么是神经网络？请解释其工作原理。", "answer": "", "category": "技术"},
        {"id": 10, "question": "请解释黑洞的形成过程及其特性。", "answer": "", "category": "科学"},
        {"id": 11, "question": "什么是自然语言处理？它有哪些主要应用？", "answer": "", "category": "技术"},
        {"id": 12, "question": "请解释DNA复制的过程及其重要性。", "answer": "", "category": "生物"},
        {"id": 13, "question": "什么是可再生能源？请列举几种主要类型。", "answer": "", "category": "环境"},
        {"id": 14, "question": "请解释密码学中的公钥加密原理。", "answer": "", "category": "技术"},
        {"id": 15, "question": "什么是人工智能？它的发展历程是怎样的？", "answer": "", "category": "技术"},
        {"id": 16, "question": "请解释光合作用的过程及其对地球生态系统的重要性。", "answer": "", "category": "生物"},
        {"id": 17, "question": "什么是大数据？它有哪些主要特征和应用？", "answer": "", "category": "技术"},
        {"id": 18, "question": "请解释进化论的基本原理及其证据。", "answer": "", "category": "科学"},
        {"id": 19, "question": "什么是云计算？它有哪些服务模式？", "answer": "", "category": "技术"},
        {"id": 20, "question": "请解释暗物质和暗能量的概念及其在宇宙学中的意义。", "answer": "", "category": "科学"},
    ]
    return backup


if __name__ == "__main__":
    download_and_sample()
