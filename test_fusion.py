"""
融合策略测试脚本 - 快速验证融合效果

由于数据集可能不完整，这个脚本会：
1. 创建一个简化的测试环境
2. 模拟策略 A 和策略 B 的输出
3. 应用融合层
4. 展示融合效果
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ursonet.nn.fusion_layer import UncertaintyFusion
import tensorflow as tf

print("=" * 60)
print("PoseCertain 融合策略测试")
print("=" * 60)

# 创建测试数据
n_samples = 100
np.random.seed(42)

# 模拟真实位置
loc_gt = np.random.uniform(-10, 10, (n_samples, 3))

# 策略 A: 回归预测 (稳定但有偏)
loc_pred_a = loc_gt + np.random.normal(0.05, 0.02, (n_samples, 3))

# 策略 B: 贝叶斯预测 (无偏但方差大)
loc_pred_b = loc_gt + np.random.normal(0, 0.08, (n_samples, 3))

# 策略 B 的方差估计
loc_var_b = np.random.uniform(0.01, 0.15, (n_samples, 3))

print("\n=== 测试数据生成 ===")
print(f"样本数：{n_samples}")
print(f"策略 A 误差均值：{np.mean(np.abs(loc_pred_a - loc_gt)):.4f}")
print(f"策略 B 误差均值：{np.mean(np.abs(loc_pred_b - loc_gt)):.4f}")

# 创建融合层
fusion_layer = UncertaintyFusion(
    weight_min=0.1,
    weight_max=0.9,
    fusion_mode='uncertainty_weighted'
)

# 应用融合
print("\n=== 应用融合 ===")
loc_pred_a_tf = tf.constant(loc_pred_a.astype(np.float32))
loc_pred_b_tf = tf.constant(loc_pred_b.astype(np.float32))
loc_var_b_tf = tf.constant(loc_var_b.astype(np.float32))

fused_pred, fusion_weights = fusion_layer([
    loc_pred_a_tf,
    loc_pred_b_tf,
    loc_var_b_tf
])

# 转换为 numpy
fused_pred_np = fused_pred.numpy()
fusion_weights_np = fusion_weights.numpy()

# 计算误差
error_a = np.mean(np.linalg.norm(loc_pred_a - loc_gt, axis=1))
error_b = np.mean(np.linalg.norm(loc_pred_b - loc_gt, axis=1))
error_fused = np.mean(np.linalg.norm(fused_pred_np - loc_gt, axis=1))

print(f"\n=== 融合结果 ===")
print(f"策略 A 位置误差：{error_a:.6f} m")
print(f"策略 B 位置误差：{error_b:.6f} m")
print(f"融合策略误差：  {error_fused:.6f} m")
print(f"\n相对提升:")
print(f"  融合 vs 策略 A: {(error_a - error_fused) / error_a * 100:.2f}%")
print(f"  融合 vs 策略 B: {(error_b - error_fused) / error_b * 100:.2f}%")

# 统计融合权重
print(f"\n=== 融合权重统计 ===")
print(f"平均权重 (贝叶斯): {np.mean(fusion_weights_np):.4f}")
print(f"权重范围: [{np.min(fusion_weights_np):.4f}, {np.max(fusion_weights_np):.4f}]")

# 按不确定性分组
avg_var = np.mean(loc_var_b, axis=1)
low_unc_mask = avg_var < np.percentile(avg_var, 50)
high_unc_mask = avg_var > np.percentile(avg_var, 50)

print(f"\n按不确定性分组:")
print(f"  低不确定性样本 (贝叶斯权重高):")
print(f"    数量：{np.sum(low_unc_mask)}")
print(f"    平均权重：{np.mean(fusion_weights_np[low_unc_mask]):.4f}")
print(f"  高不确定性样本 (贝叶斯权重低):")
print(f"    数量：{np.sum(high_unc_mask)}")
print(f"    平均权重：{np.mean(fusion_weights_np[high_unc_mask]):.4f}")

# 可视化
print("\n=== 前 10 个样本详情 ===")
print(f"{'样本':>4} | {'GT':>8} | {'策略 A':>8} | {'策略 B':>8} | {'融合':>8} | {'权重':>6}")
print("-" * 70)
for i in range(10):
    gt_norm = np.linalg.norm(loc_gt[i])
    a_norm = np.linalg.norm(loc_pred_a[i])
    b_norm = np.linalg.norm(loc_pred_b[i])
    f_norm = np.linalg.norm(fused_pred_np[i])
    w = fusion_weights_np[i, 0]
    print(f"{i:4d} | {gt_norm:8.4f} | {a_norm:8.4f} | {b_norm:8.4f} | {f_norm:8.4f} | {w:6.4f}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

# 保存结果
results = {
    'error_a': error_a,
    'error_b': error_b,
    'error_fused': error_fused,
    'improvement_a': (error_a - error_fused) / error_a * 100,
    'improvement_b': (error_b - error_fused) / error_b * 100,
    'avg_weight': np.mean(fusion_weights_np)
}

print("\n=== 性能总结 ===")
print(f"融合策略相比策略 A 提升：{results['improvement_a']:.2f}%")
print(f"融合策略相比策略 B 提升：{results['improvement_b']:.2f}%")
print(f"平均贝叶斯权重：{results['avg_weight']:.4f}")

if results['improvement_a'] > 0 and results['improvement_b'] > 0:
    print("\n✅ 融合策略有效！两个基准都得到提升")
else:
    print("\n⚠️  融合策略需要调参")
