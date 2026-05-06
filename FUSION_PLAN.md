# PoseCertain 策略融合实施计划

## 📋 现状分析

### 当前策略对比

| 特性 | 策略 A (回归) | 策略 B (贝叶斯) | 融合策略 (建议) |
|------|--------------|----------------|----------------|
| **位置输出** | (x, y, z) | (μ, σ²) | 加权融合 |
| **损失函数** | MSE | Gaussian NLL | 联合损失 |
| **不确定性** | ❌ 无 | ✅ 认知 + 数据 | ✅ 更准确 |
| **OOD 检测** | 弱 | 强 | 更强 |
| **推理速度** | 快 | 中 (MC 采样) | 中 |
| **稳定性** | 高 | 中 | 最高 |

### 代码中的关键配置

```python
# pose_estimator.py - Args 类
class Args:
    # 策略 A: 直接回归
    bayesian_loc = False  # 默认关闭
    regress_loc = True
    
    # 策略 B: 贝叶斯回归
    # bayesian_loc = True  # 需要手动开启
    
    # MC Dropout (两个策略都可以用)
    mc_dropout = True
    mc_samples = 10
    dropout_rate = 0.2
```

---

## 🎯 融合方案实施

### 方案 1: 不确定性加权融合 (推荐)

#### 架构图

```
Input Image
    ↓
ResNet50 Backbone
    ↓
共享特征层 (C6)
    ↓
    ├──────────────┬──────────────┐
    ↓              ↓              ↓
回归分支       贝叶斯分支     朝向分支
(3D)          (3D + 3D)      (32³ bins)
    ↓              ↓              ↓
  pred_A        pred_B        ori_pmf
  (x,y,z)    (μ,σ²_x,σ²_y,σ²_z)   ↓
    ↓              ↓          quat_weighted_avg
    └────┬─────────┘              ↓
         ↓                        ↓
  Uncertainty Fusion         Orientation
  w = 1/(σ²+ε)               (保持不变)
         ↓
    fused_pred = w*pred_A + (1-w)*pred_B
```

#### 实施步骤

**Step 1: 修改 config.py**

```python
# 在 config.py 中添加新配置
class Config:
    # ... 现有配置 ...
    
    # 融合策略配置
    FUSE_LOC_STRATEGIES = False  # 默认关闭，向后兼容
    FUSION_MODE = 'uncertainty_weighted'  # 'uncertainty_weighted', 'learned', 'simple_avg'
    FUSION_WEIGHT_MIN = 0.1  # 最小融合权重
    FUSION_WEIGHT_MAX = 0.9  # 最大融合权重
```

**Step 2: 修改 heads_loc.py**

```python
# 在 ursonet/nn/heads_loc.py 中添加融合层
def build_loc_graph(feature_map, config, nr_features):
    # ... 现有代码 ...
    
    # 双分支输出
    if config.REGRESS_LOC:
        if config.BAYESIAN_LOC:
            # 贝叶斯分支
            loc_mu = KL.Dense(3, activation='linear', name="loc_mu")(x)
            loc_logvar = KL.Dense(3, activation='linear', name="loc_logvar")(x)
            
            # 融合分支（如果启用）
            if getattr(config, "FUSE_LOC_STRATEGIES", False):
                # 同时输出回归分支
                loc_reg = KL.Dense(3, activation='linear', name="loc_reg")(x)
                # 输出：[mu, logvar, reg_pred]
                return [loc_mu, loc_logvar, loc_reg]
            else:
                return [loc_mu, loc_logvar]
        else:
            # 纯回归分支
            loc = KL.Dense(3, activation='linear', name="loc_final")(x)
            return loc
    # ... 其余代码 ...
```

**Step 3: 添加融合层**

```python
# 新建 ursonet/nn/fusion_layer.py
import tensorflow.keras.layers as KL
from tensorflow.keras import backend as K

class UncertaintyFusion(KL.Layer):
    """
    基于不确定性的融合层
    
    输入:
        - regression_pred: 回归分支预测 (N, 3)
        - bayesian_pred: 贝叶斯分支预测 (N, 3)
        - bayesian_var: 贝叶斯分支方差 (N, 3)
    
    输出:
        - fused_pred: 融合后的预测 (N, 3)
        - fusion_weights: 融合权重 (N, 3)
    """
    def __init__(self, weight_min=0.1, weight_max=0.9, epsilon=1e-6, **kwargs):
        super(UncertaintyFusion, self).__init__(**kwargs)
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.epsilon = epsilon
    
    def call(self, inputs):
        reg_pred, bayes_pred, bayes_var = inputs
        
        # 计算权重：方差的倒数（不确定性越低权重越高）
        # w_bayes = 1 / (σ² + ε)
        inv_var = 1.0 / (bayes_var + self.epsilon)
        
        # 归一化到 [0, 1] 范围
        # w_reg = σ² / (σ² + σ²_reg) ≈ σ² / (σ² + constant)
        # 简化：w_bayes = 1 / (σ² + 1)
        w_bayes = inv_var / (inv_var + 1.0)
        
        # 限制权重范围
        w_bayes = K.clip(w_bayes, self.weight_min, self.weight_max)
        
        # 融合
        fused_pred = w_bayes * bayes_pred + (1 - w_bayes) * reg_pred
        
        return fused_pred, w_bayes
    
    def get_config(self):
        config = {
            'weight_min': self.weight_min,
            'weight_max': self.weight_max,
            'epsilon': self.epsilon
        }
        base_config = super(UncertaintyFusion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

**Step 4: 修改 model.py 中的模型构建**

```python
# 在 ursonet/model.py 中
from ursonet.nn.fusion_layer import UncertaintyFusion

# ... 在 build 方法中 ...

# 构建位置图
loc_outputs = build_loc_graph(C6, config, config.BRANCH_SIZE)

if config.FUSE_LOC_STRATEGIES and config.BAYESIAN_LOC:
    # 解包三个输出
    loc_mu, loc_logvar, loc_reg = loc_outputs
    
    # 计算方差
    loc_var = K.exp(loc_logvar)
    
    # 融合层
    fusion_layer = UncertaintyFusion(
        weight_min=config.FUSION_WEIGHT_MIN,
        weight_max=config.FUSION_WEIGHT_MAX
    )
    loc_fused, fusion_weights = fusion_layer([loc_reg, loc_mu, loc_var])
    
    # 输出融合结果
    loc = [loc_fused, fusion_weights]
else:
    loc = loc_outputs
```

**Step 5: 修改损失函数**

```python
# 在 losses.py 或 model.py 中
def loc_loss(y_true, y_pred, config):
    """
    位置损失函数（支持融合策略）
    """
    if config.BAYESIAN_LOC:
        # 贝叶斯损失：Gaussian NLL
        mu, logvar = y_pred
        var = K.exp(logvar)
        
        # NLL Loss: 0.5 * (log(σ²) + (x-μ)²/σ²)
        nll = 0.5 * (logvar + K.square(y_true - mu) / (var + config.LOC_EPS))
        
        return K.mean(nll)
    else:
        # 标准 MSE 损失
        return K.mean(K.square(y_true - y_pred))
```

---

### 方案 2: 简单平均融合 (基线)

```python
# 在 pose_estimator.py 中添加简单融合函数
def simple_fusion(pred_a, pred_b, weight=0.5):
    """
    简单加权平均融合
    
    Args:
        pred_a: 策略 A 预测
        pred_b: 策略 B 预测
        weight: 融合权重 (0-1)
    
    Returns:
        fused_pred: 融合预测
    """
    return weight * pred_a + (1 - weight) * pred_b

# 使用示例
# fused_loc = simple_fusion(loc_reg, loc_mu, weight=0.5)
```

---

## 📊 实验设计

### 实验 1: 基线对比

| 模型 | 配置 | 预期位置误差 |
|------|------|-------------|
| Baseline-A | `regress_loc=True, bayesian_loc=False` | 0.05m |
| Baseline-B | `regress_loc=True, bayesian_loc=True` | 0.06m |
| **Fusion** | `fuse_loc_strategies=True` | **0.045m** |

### 实验 2: 融合权重消融

| 融合模式 | weight_min | weight_max | 预期效果 |
|---------|-----------|-----------|---------|
| 简单平均 | 0.5 | 0.5 | 基线 |
| 动态权重 | 0.1 | 0.9 | 最佳 |
| 偏向回归 | 0.2 | 0.4 | 保守 |
| 偏向贝叶斯 | 0.6 | 0.8 | 激进 |

### 实验 3: OOD 检测能力

| 场景 | Baseline-A | Baseline-B | Fusion |
|------|-----------|-----------|--------|
| 正常光照 | 0.65 | 0.73 | **0.82** |
| 强阴影 | 0.58 | 0.69 | **0.79** |
| 部分遮挡 | 0.61 | 0.71 | **0.80** |

---

## 🚀 实施时间表

### 第 1 周：环境准备
- [x] 分析现有代码
- [ ] 配置训练环境
- [ ] 下载 SPEED 数据集
- [ ] 运行基线实验

### 第 2 周：实现融合层
- [ ] 实现 `UncertaintyFusion` 层
- [ ] 修改 `heads_loc.py`
- [ ] 修改 `model.py`
- [ ] 单元测试

### 第 3 周：训练与调优
- [ ] 训练融合模型
- [ ] 超参数调优
- [ ] 消融实验

### 第 4 周：评估与报告
- [ ] 完整评估
- [ ] 可视化结果
- [ ] 撰写报告

---

## 📈 预期成果

1. **代码贡献**:
   - 融合层实现 (`fusion_layer.py`)
   - 配置选项扩展
   - 训练/推理脚本更新

2. **性能提升**:
   - 位置精度：+10%
   - OOD 检测：+15%
   - 不确定性校准：显著改善

3. **文档输出**:
   - 技术报告
   - 使用示例
   - 最佳实践指南

---

**创建时间**: 2026-05-06  
**项目**: PoseCertain  
**状态**: 📝 计划完成，等待实施
