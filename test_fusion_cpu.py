"""
融合策略 CPU 测试脚本 - 不使用 GPU
专门用于 CPU 环境的快速验证
"""

import os
import numpy as np

# 强制使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 禁用 TensorFlow GPU (如果安装了)
try:
    import tensorflow as tf
    # 禁用 GPU
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'CUDA')
    print("✅ TensorFlow GPU 已禁用，使用 CPU 模式")
except ImportError:
    print("⚠️  未安装 TensorFlow，使用纯 NumPy 模式")
    tf = None

print("=" * 60)
print("PoseCertain 融合策略 CPU 测试")
print("=" * 60)

# 测试配置
n_samples = 1000
np.random.seed(42)

print(f"\n=== 测试配置 ===")
print(f"样本数：{n_samples}")
print(f"运行模式：CPU Only")
print(f"设备：{os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}")

# 生成测试数据
loc_gt = np.random.uniform(-10, 10, (n_samples, 3))
loc_pred_a = loc_gt + np.random.normal(0.05, 0.02, (n_samples, 3))
loc_pred_b = loc_gt + np.random.normal(0, 0.08, (n_samples, 3))
loc_var_b = np.random.uniform(0.01, 0.15, (n_samples, 3))

print(f"\n=== 数据生成 ===")
print(f"策略 A 误差均值：{np.mean(np.abs(loc_pred_a - loc_gt)):.4f}")
print(f"策略 B 误差均值：{np.mean(np.abs(loc_pred_b - loc_gt)):.4f}")

def uncertainty_fusion_cpu(pred_a, pred_b, var_b, weight_min=0.1, weight_max=0.9, epsilon=1e-6):
    """
    CPU 优化的融合函数 (纯 NumPy 实现)
    
    参数:
        pred_a: 策略 A 预测 (N, 3)
        pred_b: 策略 B 预测 (N, 3)
        var_b: 策略 B 方差 (N, 3)
        weight_min: 最小权重
        weight_max: 最大权重
        epsilon: 数值稳定性
    
    返回:
        fused_pred: 融合预测
        weights: 融合权重
    """
    # 计算权重：w = 1 / (σ² + ε)
    inv_var = 1.0 / (var_b + epsilon)
    
    # 归一化：w_bayes = σ² / (σ² + 1)
    weights = inv_var / (inv_var + 1.0)
    
    # 限制范围
    weights = np.clip(weights, weight_min, weight_max)
    
    # 融合
    fused_pred = weights * pred_b + (1 - weights) * pred_a
    
    return fused_pred, weights

# 运行融合
print("\n=== 运行融合 ===")
import time
start_time = time.time()
fused_pred, fusion_weights = uncertainty_fusion_cpu(
    loc_pred_a, 
    loc_pred_b, 
    loc_var_b,
    weight_min=0.1,
    weight_max=0.9
)
end_time = time.time()

print(f"融合完成！耗时：{(end_time - start_time)*1000:.2f} ms")
print(f"平均每个样本：{(end_time - start_time)*1000/n_samples:.4f} ms")

# 计算误差
error_a = np.mean(np.linalg.norm(loc_pred_a - loc_gt, axis=1))
error_b = np.mean(np.linalg.norm(loc_pred_b - loc_gt, axis=1))
error_fused = np.mean(np.linalg.norm(fused_pred - loc_gt, axis=1))

print(f"\n=== 融合结果 ===")
print(f"策略 A 位置误差：{error_a:.6f} m")
print(f"策略 B 位置误差：{error_b:.6f} m")
print(f"融合策略误差：  {error_fused:.6f} m")

if error_a > 0:
    improvement_a = (error_a - error_fused) / error_a * 100
    print(f"\n相对提升:")
    print(f"  融合 vs 策略 A: {improvement_a:+.2f}%")
    
if error_b > 0:
    improvement_b = (error_b - error_fused) / error_b * 100
    print(f"  融合 vs 策略 B: {improvement_b:+.2f}%")

# 统计权重
print(f"\n=== 融合权重统计 ===")
print(f"平均权重 (贝叶斯): {np.mean(fusion_weights):.4f}")
print(f"权重范围：[{np.min(fusion_weights):.4f}, {np.max(fusion_weights):.4f}]")
print(f"权重标准差：{np.std(fusion_weights):.4f}")

# 按不确定性分组
avg_var = np.mean(loc_var_b, axis=1)
low_unc_mask = avg_var < np.percentile(avg_var, 50)
high_unc_mask = avg_var > np.percentile(avg_var, 50)

print(f"\n按不确定性分组:")
print(f"  低不确定性样本:")
print(f"    数量：{np.sum(low_unc_mask)}")
print(f"    平均权重：{np.mean(fusion_weights[low_unc_mask]):.4f}")
print(f"    融合误差：{np.mean(np.linalg.norm(fused_pred[low_unc_mask] - loc_gt[low_unc_mask], axis=1)):.6f}")
print(f"  高不确定性样本:")
print(f"    数量：{np.sum(high_unc_mask)}")
print(f"    平均权重：{np.mean(fusion_weights[high_unc_mask]):.4f}")
print(f"    融合误差：{np.mean(np.linalg.norm(fused_pred[high_unc_mask] - loc_gt[high_unc_mask], axis=1)):.6f}")

# 性能总结
print("\n" + "=" * 60)
print("CPU 测试完成!")
print("=" * 60)

print("\n=== 性能总结 ===")
if error_a > error_fused:
    print(f"✅ 融合策略优于策略 A: {(error_a - error_fused) / error_a * 100:.2f}%")
else:
    print(f"⚠️  融合策略不如策略 A: {(error_fused - error_a) / error_a * 100:.2f}%")

if error_b > error_fused:
    print(f"✅ 融合策略优于策略 B: {(error_b - error_fused) / error_b * 100:.2f}%")
else:
    print(f"⚠️  融合策略不如策略 B: {(error_fused - error_b) / error_b * 100:.2f}%")

# 保存结果
results = {
    'error_a': error_a,
    'error_b': error_b,
    'error_fused': error_fused,
    'improvement_a': (error_a - error_fused) / error_a * 100 if error_a > 0 else 0,
    'improvement_b': (error_b - error_fused) / error_b * 100 if error_b > 0 else 0,
    'avg_weight': np.mean(fusion_weights),
    'time_ms': (end_time - start_time) * 1000
}

print(f"\n=== 运行统计 ===")
print(f"总耗时：{results['time_ms']:.2f} ms")
print(f"样本数：{n_samples}")
print(f"吞吐量：{n_samples / results['time_ms'] * 1000:.0f} samples/sec")

# 导出结果
output_file = "cpu_fusion_results.txt"
with open(output_file, 'w') as f:
    f.write("PoseCertain 融合策略 CPU 测试结果\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"样本数：{n_samples}\n")
    f.write(f"策略 A 误差：{error_a:.6f} m\n")
    f.write(f"策略 B 误差：{error_b:.6f} m\n")
    f.write(f"融合误差：  {error_fused:.6f} m\n")
    f.write(f"\n提升:\n")
    f.write(f"  vs 策略 A: {results['improvement_a']:+.2f}%\n")
    f.write(f"  vs 策略 B: {results['improvement_b']:+.2f}%\n")
    f.write(f"\n权重统计:\n")
    f.write(f"  平均：{results['avg_weight']:.4f}\n")
    f.write(f"  范围：[{np.min(fusion_weights):.4f}, {np.max(fusion_weights):.4f}]\n")
    f.write(f"\n性能:\n")
    f.write(f"  耗时：{results['time_ms']:.2f} ms\n")
    f.write(f"  吞吐量：{n_samples / results['time_ms'] * 1000:.0f} samples/sec\n")

print(f"\n结果已保存到：{output_file}")
