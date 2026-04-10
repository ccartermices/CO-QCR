import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r'd:\科研项目\量子\co_qcr_experiment\data'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures')

def load_results():
    results_file = os.path.join(RESULTS_DIR, 'comparison_results.json')
    if not os.path.exists(results_file):
        print("[错误] 未找到结果文件")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_comparison_bar(results):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    qcr_coherence = [r['qcr']['coherence'] for r in results]
    co_qcr_coherence = [r['co_qcr']['coherence'] for r in results]
    qcr_causal = [r['qcr']['causal_consistency'] for r in results]
    co_qcr_causal = [r['co_qcr']['causal_consistency'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(results))
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, qcr_coherence, width, label='QCR-LLM', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, co_qcr_coherence, width, label='CO-QCR', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Question ID', fontsize=11)
    ax1.set_ylabel('Coherence Score', fontsize=11)
    ax1.set_title('Reasoning Coherence Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['question_id'] for r in results], rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    ax2.bar(x - width/2, qcr_causal, width, label='QCR-LLM', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, co_qcr_causal, width, label='CO-QCR', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Question ID', fontsize=11)
    ax2.set_ylabel('Causal Consistency Score', fontsize=11)
    ax2.set_title('Causal Consistency Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['question_id'] for r in results], rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, 'comparison_bar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[保存] 柱状图: {save_path}")

def plot_summary_table(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    qcr_coherence_avg = np.mean([r['qcr']['coherence'] for r in results])
    co_qcr_coherence_avg = np.mean([r['co_qcr']['coherence'] for r in results])
    qcr_causal_avg = np.mean([r['qcr']['causal_consistency'] for r in results])
    co_qcr_causal_avg = np.mean([r['co_qcr']['causal_consistency'] for r in results])
    qcr_selected_avg = np.mean([r['qcr']['n_selected'] for r in results])
    co_qcr_selected_avg = np.mean([r['co_qcr']['n_selected'] for r in results])
    
    coherence_improve = (co_qcr_coherence_avg - qcr_coherence_avg) / qcr_coherence_avg * 100 if qcr_coherence_avg > 0 else 0
    causal_improve = (co_qcr_causal_avg - qcr_causal_avg) / qcr_causal_avg * 100 if qcr_causal_avg > 0 else 0
    
    table_data = [
        ['Metric', 'QCR-LLM', 'CO-QCR', 'Improvement'],
        ['Coherence Score', f'{qcr_coherence_avg:.4f}', f'{co_qcr_coherence_avg:.4f}', f'+{coherence_improve:.2f}%'],
        ['Causal Consistency', f'{qcr_causal_avg:.4f}', f'{co_qcr_causal_avg:.4f}', f'+{causal_improve:.2f}%'],
        ['Avg. Selected Fragments', f'{qcr_selected_avg:.1f}', f'{co_qcr_selected_avg:.1f}', '-'],
        ['Total Questions', str(len(results)), str(len(results)), '-'],
    ]
    
    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, 5):
        table[(i, 3)].set_facecolor('#27ae60')
        table[(i, 3)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('QCR-LLM vs CO-QCR Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    save_path = os.path.join(FIGURES_DIR, 'summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[保存] 汇总表: {save_path}")

def plot_energy_convergence(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if results and 'qcr' in results[0]:
        qcr_energies = [r['qcr'].get('final_energy', 0) for r in results]
        co_qcr_energies = [r['co_qcr'].get('final_energy', 0) for r in results]
        
        x = np.arange(len(results))
        ax.plot(x, qcr_energies, 'o-', label='QCR-LLM', color='#3498db', linewidth=2, markersize=8)
        ax.plot(x, co_qcr_energies, 's-', label='CO-QCR', color='#e74c3c', linewidth=2, markersize=8)
        
        ax.set_xlabel('Question ID', fontsize=11)
        ax.set_ylabel('Final Energy', fontsize=11)
        ax.set_title('Energy Minimization Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(FIGURES_DIR, 'energy_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[保存] 能量对比图: {save_path}")

def plot_radar_chart(results):
    qcr_coherence_avg = np.mean([r['qcr']['coherence'] for r in results])
    co_qcr_coherence_avg = np.mean([r['co_qcr']['coherence'] for r in results])
    qcr_causal_avg = np.mean([r['qcr']['causal_consistency'] for r in results])
    co_qcr_causal_avg = np.mean([r['co_qcr']['causal_consistency'] for r in results])
    
    qcr_diversity = 1 - np.mean([r['qcr']['n_selected'] / max(r['n_fragments'], 1) for r in results])
    co_qcr_diversity = 1 - np.mean([r['co_qcr']['n_selected'] / max(r['n_fragments'], 1) for r in results])
    
    categories = ['Coherence', 'Causal\nConsistency', 'Fragment\nDiversity']
    
    qcr_values = [qcr_coherence_avg, qcr_causal_avg, qcr_diversity]
    co_qcr_values = [co_qcr_coherence_avg, co_qcr_causal_avg, co_qcr_diversity]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    qcr_values += qcr_values[:1]
    co_qcr_values += co_qcr_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.fill(angles, qcr_values, color='#3498db', alpha=0.25)
    ax.plot(angles, qcr_values, 'o-', color='#3498db', linewidth=2, label='QCR-LLM')
    
    ax.fill(angles, co_qcr_values, color='#e74c3c', alpha=0.25)
    ax.plot(angles, co_qcr_values, 's-', color='#e74c3c', linewidth=2, label='CO-QCR')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    ax.set_title('Multi-dimensional Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, 'radar_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[保存] 雷达图: {save_path}")

def plot_improvement_heatmap(results):
    n_questions = len(results)
    improvements = np.zeros((n_questions, 2))
    
    for i, r in enumerate(results):
        qcr_coh = r['qcr']['coherence']
        co_qcr_coh = r['co_qcr']['coherence']
        improvements[i, 0] = (co_qcr_coh - qcr_coh) / max(qcr_coh, 0.01) * 100
        
        qcr_causal = r['qcr']['causal_consistency']
        co_qcr_causal = r['co_qcr']['causal_consistency']
        improvements[i, 1] = (co_qcr_causal - qcr_causal) / max(qcr_causal, 0.01) * 100
    
    fig, ax = plt.subplots(figsize=(8, max(6, n_questions * 0.4)))
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=50)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Coherence\nImprovement (%)', 'Causal Consistency\nImprovement (%)'])
    ax.set_yticks(range(n_questions))
    ax.set_yticklabels([f"Q{r['question_id']}" for r in results])
    
    for i in range(n_questions):
        for j in range(2):
            text = ax.text(j, i, f'{improvements[i, j]:.1f}%',
                          ha='center', va='center', fontsize=9,
                          color='white' if abs(improvements[i, j]) > 15 else 'black')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Improvement (%)', fontsize=10)
    
    ax.set_title('CO-QCR Improvement over QCR-LLM', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, 'improvement_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[保存] 改进热力图: {save_path}")

def generate_all_figures():
    results = load_results()
    if not results:
        print("[错误] 无法加载结果数据")
        return
    
    print(f"\n{'='*60}")
    print("  生成可视化图表")
    print(f"{'='*60}\n")
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("[生成] 柱状对比图...")
    plot_comparison_bar(results)
    
    print("[生成] 汇总表格...")
    plot_summary_table(results)
    
    print("[生成] 能量收敛图...")
    plot_energy_convergence(results)
    
    print("[生成] 雷达图...")
    plot_radar_chart(results)
    
    print("[生成] 改进热力图...")
    plot_improvement_heatmap(results)
    
    print(f"\n{'='*60}")
    print("  所有图表已生成")
    print(f"  保存目录: {FIGURES_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_all_figures()
