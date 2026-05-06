"""
融合策略优化版 CPU 测试
改进权重计算，增加动态范围
"""

import os
import numpy as np
import time

# 强制 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("=" * 60)
print("PoseCertain 融合策略 CPU 优化测试")
print("=" * 60)

# 测试配置
n_samples = 1000
np.random.seed(42)

print(f"\n=== 测试配置 ===")
print(f"样本数：{n_samples}")
print(f"运行模式：CPU Only (优化版)")

# 生成更真实的测试数据
loc_gt = np.random.uniform(-10, 10, (n_samples, 3))

# 策略 A: 回归 (稳定但有小偏差)
loc_pred_a = loc_gt + np.random.normal(0.03, 0.015, (n_samples, 3))

# 策略 B: 贝叶斯 (无偏但方差大，且方差估计准确)
noise_b = np.random.normal(0, 0.10, (n_samples, 3))
loc_pred_b = loc_gt + noise_b
# 方差估计与真实误差相关
loc_var_b = np.square(noise_b) + np.random.uniform(0.001, 0.02, (n_samples, 3))

print(f"\n=== 数据特性 ===")
print(f"策略 A 误差均值：{np.mean(np.abs(loc_pred_a - loc_gt)):.4f}")
print(f"策略 B 误差均值：{np.mean(np.abs(loc_pred_b - loc_gt)):.4f}")
print(f"策略 B 方差均值：{np.mean(loc_var_b):.4f}")

def optimized_fusion(pred_a, pred_b, var_b, weight_min=0.0, weight_max=1.0, epsilon=1e-6):
    """
    优化的融合函数
    
    改进点:
    1. 使用方差倒数加权
    2. 考虑预测一致性
    3. 自适应权重范围
    """
    # 计算基础权重 (方差倒数)
    inv_var = 1.0 / (var_b + epsilon)
    
    # 归一化到 [0, 1]
    weights = inv_var / (inv_var.max() + epsilon)
    
    # 计算预测一致性
    diff = np.abs(pred_a - pred_b)
    consistency = np.exp(-diff / 0.1)  # 一致性高时权重高
    
    # 组合权重
    weights = weights * consistency
    
    # 限制范围
    weights = np.clip(weights, weight_min, weight_max)
    
    # 融合
    fused_pred = weights * pred_b + (1 - weights) * pred_a
    
    return fused_pred, weights

# 测试不同配置
configs = [
    {'name': '基础融合', 'weight_min': 0.1, 'weight_max': 0.9},
    {'name': '宽范围融合', 'weight_min': 0.0, 'weight_max': 1.0},
    {'name': '保守融合', 'weight_min': 0.3, 'weight_max': 0.7},
    {'name': '激进融合', 'weight_min': 0.0, 'weight_max': 0.95},
]

print("\n" + "=" * 60)
print("测试结果对比")
print("=" * 60)

results = {}
for config in configs:
    start = time.time()
    fused, weights = optimized_fusion(
        loc_pred_a, loc_pred_b, loc_var_b,
        weight_min=config['weight_min'],
        weight_max=config['weight_max']
    )
    elapsed = time.time() - start
    
    error_fused = np.mean(np.linalg.norm(fused - loc_gt, axis=1))
    error_a = np.mean(np.linalg.norm(loc_pred_a - loc_gt, axis=1))
    error_b = np.mean(np.linalg.norm(loc_pred_b - loc_gt, axis=1))
    
    improvement_vs_a = (error_a - error_fused) / error_a * 100
    improvement_vs_b = (error_b - error_fused) / error_b * 100
    
    results[config['name']] = {
        'error': error_fused,
        'improvement_a': improvement_vs_a,
        'improvement_b': improvement_vs_b,
        'avg_weight': np.mean(weights),
        'time': elapsed * 1000
    }
    
    print(f"\n{config['name']}:")
    print(f"  融合误差：{error_fused:.6f} m")
    print(f"  vs 策略 A: {improvement_vs_a:+.2f}%")
    print(f"  vs 策略 B: {improvement_vs_b:+.2f}%")
    print(f"  平均权重：{np.mean(weights):.4f}")
    print(f"  耗时：{elapsed*1000:.2f} ms")

# 找出最佳配置
best_config = max(results.items(), key=lambda x: x[1]['improvement_b'])
print("\n" + "=" * 60)
print(f"🏆 最佳配置：{best_config[0]}")
print(f"   提升：{best_config[1]['improvement_b']:+.2f}% vs 策略 B")
print(f"   融合误差：{best_config[1]['error']:.6f} m")
print("=" * 60)

# 保存最佳结果
with open('cpu_fusion_optimized_results.txt', 'w') as f:
    f.write("PoseCertain 融合策略 CPU 优化测试结果\n")
    f.write("=" * 60 + "\n\n")
    
    for name, data in results.items():
        f.write(f"{name}:\n")
        f.write(f"  融合误差：{data['error']:.6f} m\n")
        f.write(f"  vs 策略 A: {data['improvement_a']:+.2f}%\n")
        f.write(f"  vs 策略 B: {data['improvement_b']:+.2f}%\n")
        f.write(f"  平均权重：{data['avg_weight']:.4f}\n")
        f.write(f"  耗时：{data['time']:.2f} ms\n\n")
    
    f.write(f"\n🏆 最佳配置：{best_config[0]}\n")
    f.write(f"   提升：{best_config[1]['improvement_b']:+.2f}% vs 策略 B\n")

print(f"\n结果已保存到：cpu_fusion_optimized_results.txt")
