"""
融合策略简化测试 - 不需要 TensorFlow

直接测试融合算法的核心逻辑
"""

import numpy as np

print("=" * 60)
print("PoseCertain 融合策略简化测试")
print("=" * 60)

# 创建测试数据
n_samples = 1000
np.random.seed(42)

# 模拟真实位置
loc_gt = np.random.uniform(-10, 10, (n_samples, 3))

# 策略 A: 回归预测 (稳定但有偏)
loc_pred_a = loc_gt + np.random.normal(0.05, 0.02, (n_samples, 3))

# 策略 B: 贝叶斯预测 (无偏但方差大)
loc_pred_b = loc_gt + np.random.normal(0, 0.08, (n_samples, 3))

# 策略 B 的方差估计
loc_var_b = np.random.uniform(0.01, 0.15, (n_samples, 3))

print("\n=== 测试数据 ===")
print(f"样本数：{n_samples}")
print(f"策略 A 误差均值：{np.mean(np.abs(loc_pred_a - loc_gt)):.4f}")
print(f"策略 B 误差均值：{np.mean(np.abs(loc_pred_b - loc_gt)):.4f}")

# 融合函数
def uncertainty_weighted_fusion(pred_a, pred_b, var_b, weight_min=0.1, weight_max=0.9, epsilon=1e-6):
    """
    基于不确定性的加权融合
    
    参数:
        pred_a: 策略 A 预测
        pred_b: 策略 B 预测
        var_b: 策略 B 方差
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

# 应用融合
print("\n=== 应用融合 ===")
fused_pred, fusion_weights = uncertainty_weighted_fusion(
    loc_pred_a, 
    loc_pred_b, 
    loc_var_b,
    weight_min=0.1,
    weight_max=0.9
)

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

# 统计融合权重
print(f"\n=== 融合权重统计 ===")
print(f"平均权重 (贝叶斯): {np.mean(fusion_weights):.4f}")
print(f"权重范围：[{np.min(fusion_weights):.4f}, {np.max(fusion_weights):.4f}]")

# 按不确定性分组
avg_var = np.mean(loc_var_b, axis=1)
low_unc_mask = avg_var < np.percentile(avg_var, 50)
high_unc_mask = avg_var > np.percentile(avg_var, 50)

print(f"\n按不确定性分组:")
print(f"  低不确定性样本 (贝叶斯权重高):")
print(f"    数量：{np.sum(low_unc_mask)}")
print(f"    平均权重：{np.mean(fusion_weights[low_unc_mask]):.4f}")
print(f"    融合误差：{np.mean(np.linalg.norm(fused_pred[low_unc_mask] - loc_gt[low_unc_mask], axis=1)):.6f}")
print(f"  高不确定性样本 (贝叶斯权重低):")
print(f"    数量：{np.sum(high_unc_mask)}")
print(f"    平均权重：{np.mean(fusion_weights[high_unc_mask]):.4f}")
print(f"    融合误差：{np.mean(np.linalg.norm(fused_pred[high_unc_mask] - loc_gt[high_unc_mask], axis=1)):.6f}")

# 可视化前 10 个样本
print("\n=== 前 10 个样本详情 ===")
print(f"{'样本':>4} | {'GT':>8} | {'策略 A':>8} | {'策略 B':>8} | {'融合':>8} | {'权重':>6}")
print("-" * 70)
for i in range(10):
    gt_norm = np.linalg.norm(loc_gt[i])
    a_norm = np.linalg.norm(loc_pred_a[i])
    b_norm = np.linalg.norm(loc_pred_b[i])
    f_norm = np.linalg.norm(fused_pred[i])
    w = fusion_weights[i, 0]
    print(f"{i:4d} | {gt_norm:8.4f} | {a_norm:8.4f} | {b_norm:8.4f} | {f_norm:8.4f} | {w:6.4f}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

# 性能总结
print("\n=== 性能总结 ===")
if error_a > error_fused:
    print(f"✅ 融合策略优于策略 A: {(error_a - error_fused) / error_a * 100:.2f}%")
else:
    print(f"⚠️  融合策略不如策略 A: {(error_fused - error_a) / error_a * 100:.2f}%")

if error_b > error_fused:
    print(f"✅ 融合策略优于策略 B: {(error_b - error_fused) / error_b * 100:.2f}%")
else:
    print(f"⚠️  融合策略不如策略 B: {(error_fused - error_b) / error_b * 100:.2f}%")

print(f"\n平均贝叶斯权重：{np.mean(fusion_weights):.4f}")
print(f"权重标准差：{np.std(fusion_weights):.4f}")
