import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import numpy as np
from collections import defaultdict
import pennylane as qml
from pennylane import numpy as pnp

DATA_DIR = r'd:\科研项目\量子\co_qcr_experiment\data'
RESPONSES_DIR = os.path.join(DATA_DIR, 'responses')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

TOP_K = 5


class QCR_LLM_Solver:
    """原始QCR-LLM方法（无因果约束）"""
    
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=2048)
        self.n_layers = 2
    
    def compute_hubo_coefficients(self, fragments_data):
        n = min(len(fragments_data['fragments']), self.n_qubits)
        popularity = fragments_data['fragment_popularity']
        co_occurrence = fragments_data['fragment_co_occurrence']
        
        w1 = np.zeros(self.n_qubits)
        w2 = np.zeros((self.n_qubits, self.n_qubits))
        
        for i, frag in enumerate(fragments_data['fragments'][:n]):
            w1[i] = -popularity.get(frag, 0.5)
        
        for i, f1 in enumerate(fragments_data['fragments'][:n]):
            for j, f2 in enumerate(fragments_data['fragments'][:n]):
                if i < j:
                    co_occur = co_occurrence.get(f1, {}).get(f2, 0)
                    w2[i, j] = -co_occur * 0.5
                    w2[j, i] = w2[i, j]
        
        return w1, w2
    
    def solve(self, fragments_data):
        w1, w2 = self.compute_hubo_coefficients(fragments_data)
        
        coeffs = []
        obs = []
        for i in range(self.n_qubits):
            coeffs.append(float(w1[i]) / 2)
            obs.append(qml.PauliZ(i))
        
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if abs(w2[i, j]) > 1e-10:
                    coeffs.append(float(w2[i, j]) / 4)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        H = qml.Hamiltonian(coeffs, obs)
        
        @qml.qnode(self.dev)
        def circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits * 2 + i * 2], wires=i)
                    qml.RZ(params[layer * self.n_qubits * 2 + i * 2 + 1], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(H)
        
        n_params = self.n_layers * self.n_qubits * 2
        params = pnp.array(np.random.uniform(0, np.pi, n_params), requires_grad=True)
        
        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        
        energies = []
        for _ in range(30):
            params, energy = optimizer.step_and_cost(circuit, params)
            energies.append(float(energy))
        
        @qml.qnode(self.dev)
        def measure_circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits * 2 + i * 2], wires=i)
                    qml.RZ(params[layer * self.n_qubits * 2 + i * 2 + 1], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        expectations = measure_circuit(params)
        probs = [(1 - exp) / 2 for exp in expectations]
        
        return probs, energies


class CO_QCR_Solver:
    """CO-QCR方法（带因果约束）"""
    
    def __init__(self, n_qubits=8, causal_pairs=None):
        self.n_qubits = n_qubits
        self.causal_pairs = causal_pairs or [(0, 1), (2, 3), (4, 5)]
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=2048)
        self.n_layers = 2
        self.causal_penalty = 1.5
    
    def detect_causal_pairs(self, fragments):
        n = min(len(fragments), self.n_qubits)
        pairs = []
        
        causal_keywords = [
            ('因为', '所以'),
            ('由于', '导致'),
            ('首先', '然后'),
            ('前提', '结论'),
            ('原因', '结果'),
        ]
        
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = fragments[i], fragments[j]
                for kw1, kw2 in causal_keywords:
                    if kw1 in f1 and kw2 in f2:
                        pairs.append((i, j))
                        break
        
        if not pairs:
            pairs = [(0, 1), (2, 3)][:max(1, n//3)]
        
        return pairs
    
    def compute_hubo_coefficients(self, fragments_data):
        n = min(len(fragments_data['fragments']), self.n_qubits)
        popularity = fragments_data['fragment_popularity']
        co_occurrence = fragments_data['fragment_co_occurrence']
        
        w1 = np.zeros(self.n_qubits)
        w2 = np.zeros((self.n_qubits, self.n_qubits))
        
        for i, frag in enumerate(fragments_data['fragments'][:n]):
            w1[i] = -popularity.get(frag, 0.5)
        
        for i, f1 in enumerate(fragments_data['fragments'][:n]):
            for j, f2 in enumerate(fragments_data['fragments'][:n]):
                if i < j:
                    co_occur = co_occurrence.get(f1, {}).get(f2, 0)
                    w2[i, j] = -co_occur * 0.5
                    w2[j, i] = w2[i, j]
        
        for (i, j) in self.causal_pairs:
            if i < self.n_qubits and j < self.n_qubits:
                w2[i, j] += self.causal_penalty
                w2[j, i] += self.causal_penalty
        
        return w1, w2
    
    def solve(self, fragments_data):
        self.causal_pairs = self.detect_causal_pairs(fragments_data['fragments'])
        w1, w2 = self.compute_hubo_coefficients(fragments_data)
        
        coeffs = []
        obs = []
        for i in range(self.n_qubits):
            coeffs.append(float(w1[i]) / 2)
            obs.append(qml.PauliZ(i))
        
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if abs(w2[i, j]) > 1e-10:
                    coeffs.append(float(w2[i, j]) / 4)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        H = qml.Hamiltonian(coeffs, obs)
        
        @qml.qnode(self.dev)
        def circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits * 2 + i * 2], wires=i)
                    qml.RZ(params[layer * self.n_qubits * 2 + i * 2 + 1], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            for (i, j) in self.causal_pairs:
                if i < self.n_qubits and j < self.n_qubits:
                    qml.CZ(wires=[i, j])
            
            return qml.expval(H)
        
        n_params = self.n_layers * self.n_qubits * 2
        params = pnp.array(np.random.uniform(0, np.pi, n_params), requires_grad=True)
        
        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        
        energies = []
        for _ in range(30):
            params, energy = optimizer.step_and_cost(circuit, params)
            energies.append(float(energy))
        
        @qml.qnode(self.dev)
        def measure_circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits * 2 + i * 2], wires=i)
                    qml.RZ(params[layer * self.n_qubits * 2 + i * 2 + 1], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            for (i, j) in self.causal_pairs:
                if i < self.n_qubits and j < self.n_qubits:
                    qml.CZ(wires=[i, j])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        expectations = measure_circuit(params)
        probs = [(1 - exp) / 2 for exp in expectations]
        
        return probs, energies
    
    def apply_causal_correction(self, probs, threshold=0.5):
        corrected = probs.copy()
        
        for (i, j) in self.causal_pairs:
            if i < len(probs) and j < len(probs):
                if corrected[j] >= threshold and corrected[i] < threshold:
                    corrected[i] = threshold + 0.1
        
        return corrected


def extract_reasoning_fragments(response_text):
    fragments = []
    
    if response_text is None or not isinstance(response_text, str):
        return fragments
    
    patterns = [
        r'首先[，,]([^。]+)',
        r'然后[，,]([^。]+)',
        r'接着[，,]([^。]+)',
        r'最后[，,]([^。]+)',
        r'因此[，,]([^。]+)',
        r'所以[，,]([^。]+)',
        r'因为([^，,]+)',
        r'步骤\s*\d+[：:]\s*([^。\n]+)',
        r'\d+[\.、]\s*([^。\n]+)',
        r'第一[，,]([^。]+)',
        r'第二[，,]([^。]+)',
        r'第三[，,]([^。]+)',
    ]
    
    import re
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        for match in matches:
            fragment = match.strip()
            if len(fragment) > 5 and len(fragment) < 200:
                fragments.append(fragment)
    
    sentences = re.split(r'[。！？\n]', response_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 200:
            if sentence not in fragments:
                if any(kw in sentence for kw in ['因为', '所以', '因此', '首先', '然后', '但是', '然而', '由于', '导致', '意味着', '表明', '说明']):
                    fragments.append(sentence)
    
    return fragments[:15]


def compute_fragment_stats(all_fragments):
    fragment_count = defaultdict(int)
    fragment_co_occurrence = defaultdict(lambda: defaultdict(int))
    
    for fragments in all_fragments:
        for frag in fragments:
            fragment_count[frag] += 1
        
        for i, f1 in enumerate(fragments):
            for f2 in fragments[i+1:]:
                fragment_co_occurrence[f1][f2] += 1
                fragment_co_occurrence[f2][f1] += 1
    
    total = len(all_fragments)
    popularity = {frag: count / total for frag, count in fragment_count.items()}
    
    return fragment_count, popularity, fragment_co_occurrence


def evaluate_coherence(selected_fragments, popularity, co_occurrence):
    if not selected_fragments:
        return 0.0
    
    scores = []
    for frag in selected_fragments:
        scores.append(popularity.get(frag, 0.5))
    
    pairwise_scores = []
    for i, f1 in enumerate(selected_fragments):
        for f2 in selected_fragments[i+1:]:
            co = co_occurrence.get(f1, {}).get(f2, 0)
            pairwise_scores.append(co)
    
    avg_pop = np.mean(scores) if scores else 0
    avg_co = np.mean(pairwise_scores) if pairwise_scores else 0
    
    return 0.6 * avg_pop + 0.4 * avg_co


def evaluate_causal_consistency(selected_indices, causal_pairs):
    if not causal_pairs or not selected_indices:
        return 1.0
    
    violations = 0
    for (i, j) in causal_pairs:
        if j in selected_indices and i not in selected_indices:
            violations += 1
    
    return 1.0 - (violations / len(causal_pairs)) if causal_pairs else 1.0


def load_responses():
    responses = []
    
    if not os.path.exists(RESPONSES_DIR):
        return responses
    
    for filename in os.listdir(RESPONSES_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(RESPONSES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    responses.append(data)
            except:
                continue
    
    return responses


def group_responses_by_question(responses):
    grouped = defaultdict(list)
    for resp in responses:
        qid = resp.get('question_id', 'unknown')
        grouped[qid].append(resp)
    return grouped


def run_comparison():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("  加载CoT响应数据")
    print(f"{'='*60}\n")
    
    responses = load_responses()
    print(f"[信息] 加载了 {len(responses)} 个响应")
    
    if len(responses) < 10:
        print("[警告] 响应数量不足，请先运行API调用收集数据")
        return
    
    grouped = group_responses_by_question(responses)
    print(f"[信息] 涉及 {len(grouped)} 个问题")
    
    print(f"\n{'='*60}")
    print("  QCR-LLM vs CO-QCR 对比实验")
    print(f"  Top-K选择: {TOP_K}")
    print(f"{'='*60}\n")
    
    qcr_solver = QCR_LLM_Solver(n_qubits=8)
    co_qcr_solver = CO_QCR_Solver(n_qubits=8)
    
    results = []
    
    for qid, question_responses in grouped.items():
        if len(question_responses) < 2:
            print(f"[跳过] 问题 {qid[:8]}...: 响应数不足 ({len(question_responses)})")
            continue
        
        top_k_responses = question_responses[:TOP_K]
        
        all_fragments = []
        for resp in top_k_responses:
            text = resp.get('response', '')
            fragments = extract_reasoning_fragments(text)
            if fragments:
                all_fragments.append(fragments)
        
        if not all_fragments:
            print(f"[跳过] 问题 {qid[:8]}...: 无法提取推理片段")
            continue
        
        unique_fragments = list(set([f for frags in all_fragments for f in frags]))[:15]
        
        if len(unique_fragments) < 3:
            print(f"[跳过] 问题 {qid[:8]}...: 唯一片段数不足 ({len(unique_fragments)})")
            continue
        
        print(f"\n[处理] 问题 {qid[:8]}... ({len(top_k_responses)} 个响应, {len(unique_fragments)} 个唯一片段)")
        
        fragment_count, popularity, co_occurrence = compute_fragment_stats(all_fragments)
        
        fragments_data = {
            'fragments': unique_fragments,
            'fragment_count': fragment_count,
            'fragment_popularity': popularity,
            'fragment_co_occurrence': co_occurrence
        }
        
        qcr_probs, qcr_energies = qcr_solver.solve(fragments_data)
        co_qcr_probs, co_qcr_energies = co_qcr_solver.solve(fragments_data)
        co_qcr_corrected = co_qcr_solver.apply_causal_correction(co_qcr_probs)
        
        threshold = 0.5
        qcr_selected_idx = [i for i, p in enumerate(qcr_probs) if p >= threshold]
        co_qcr_selected_idx = [i for i, p in enumerate(co_qcr_corrected) if p >= threshold]
        
        qcr_selected_frags = [unique_fragments[i] for i in qcr_selected_idx if i < len(unique_fragments)]
        co_qcr_selected_frags = [unique_fragments[i] for i in co_qcr_selected_idx if i < len(unique_fragments)]
        
        qcr_coherence = evaluate_coherence(qcr_selected_frags, popularity, co_occurrence)
        co_qcr_coherence = evaluate_coherence(co_qcr_selected_frags, popularity, co_occurrence)
        
        qcr_causal = evaluate_causal_consistency(qcr_selected_idx, co_qcr_solver.causal_pairs)
        co_qcr_causal = evaluate_causal_consistency(co_qcr_selected_idx, co_qcr_solver.causal_pairs)
        
        best_qcr_response_idx = np.argmax(qcr_probs) if qcr_probs else 0
        best_co_qcr_response_idx = np.argmax(co_qcr_corrected) if co_qcr_corrected else 0
        
        result = {
            'question_id': qid,
            'n_responses': len(top_k_responses),
            'n_fragments': len(unique_fragments),
            'causal_pairs': co_qcr_solver.causal_pairs,
            'qcr': {
                'probs': [float(p) for p in qcr_probs],
                'selected_indices': qcr_selected_idx,
                'n_selected': len(qcr_selected_idx),
                'coherence': float(qcr_coherence),
                'causal_consistency': float(qcr_causal),
                'final_energy': float(qcr_energies[-1]) if qcr_energies else 0,
                'best_response_idx': int(best_qcr_response_idx)
            },
            'co_qcr': {
                'probs': [float(p) for p in co_qcr_corrected],
                'selected_indices': co_qcr_selected_idx,
                'n_selected': len(co_qcr_selected_idx),
                'coherence': float(co_qcr_coherence),
                'causal_consistency': float(co_qcr_causal),
                'final_energy': float(co_qcr_energies[-1]) if co_qcr_energies else 0,
                'best_response_idx': int(best_co_qcr_response_idx)
            }
        }
        
        results.append(result)
        
        print(f"  QCR-LLM: 选中 {len(qcr_selected_idx)} 片段, 连贯性={qcr_coherence:.3f}, 因果一致性={qcr_causal:.3f}")
        print(f"  CO-QCR:   选中 {len(co_qcr_selected_idx)} 片段, 连贯性={co_qcr_coherence:.3f}, 因果一致性={co_qcr_causal:.3f}")
    
    results_file = os.path.join(RESULTS_DIR, 'comparison_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("  实验结果汇总")
    print(f"{'='*60}")
    
    if results:
        qcr_coherence_avg = np.mean([r['qcr']['coherence'] for r in results])
        co_qcr_coherence_avg = np.mean([r['co_qcr']['coherence'] for r in results])
        qcr_causal_avg = np.mean([r['qcr']['causal_consistency'] for r in results])
        co_qcr_causal_avg = np.mean([r['co_qcr']['causal_consistency'] for r in results])
        
        print(f"\n  平均连贯性:")
        print(f"    QCR-LLM: {qcr_coherence_avg:.4f}")
        print(f"    CO-QCR:  {co_qcr_coherence_avg:.4f}")
        if qcr_coherence_avg > 0:
            print(f"    提升:    {(co_qcr_coherence_avg - qcr_coherence_avg) / qcr_coherence_avg * 100:.2f}%")
        
        print(f"\n  平均因果一致性:")
        print(f"    QCR-LLM: {qcr_causal_avg:.4f}")
        print(f"    CO-QCR:  {co_qcr_causal_avg:.4f}")
        if qcr_causal_avg > 0:
            print(f"    提升:    {(co_qcr_causal_avg - qcr_causal_avg) / qcr_causal_avg * 100:.2f}%")
        
        print(f"\n  处理问题数: {len(results)}")
    
    print(f"\n[完成] 结果保存到: {results_file}")
    
    return results


if __name__ == "__main__":
    run_comparison()
