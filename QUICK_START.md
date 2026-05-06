# 🚀 PoseCertain 融合策略快速开始指南

## 📋 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [配置说明](#配置说明)
4. [训练融合模型](#训练融合模型)
5. [评估结果](#评估结果)

---

## 环境准备

### 1. 安装依赖

```bash
cd /home/awu/.openclaw/workspace/PoseCertain
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
# 下载 SPEED 数据集（从 ESA Kelvins Portal）
# 放置到以下目录:
datasets/
└── speed/
    ├── images/
    │   ├── train/
    │   ├── test/
    │   └── real/
    └── train.json
```

### 3. 生成数据划分

```bash
python split_dataset.py
```

---

## 快速开始

### 方式 1: 使用默认配置（推荐）

```python
# 在 pose_estimator.py 中修改配置
class Args:
    command = "train"
    dataset = "speed"
    weights = "imagenet"  # 或 "coco"
    
    # 启用融合策略
    bayesian_loc = True
    regress_loc = True
    
    # 融合配置
    mc_dropout = True
    mc_samples = 10
    dropout_rate = 0.2

# 运行训练
python pose_estimator.py
```

### 方式 2: 使用融合配置文件

```python
# 使用 fusion_config.py
from fusion_config import FusionBaselineConfig

config = FusionBaselineConfig()
config.FUSE_LOC_STRATEGIES = True
config.FUSION_MODE = 'uncertainty_weighted'  # 或 'learned', 'simple_avg'

# 显示配置
config.display()
```

---

## 配置说明

### 核心配置参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `FUSE_LOC_STRATEGIES` | 是否启用融合 | `False` | `True` |
| `FUSION_MODE` | 融合模式 | `'uncertainty_weighted'` | 同左 |
| `FUSION_WEIGHT_MIN` | 最小权重 | `0.1` | `0.2` |
| `FUSION_WEIGHT_MAX` | 最大权重 | `0.9` | `0.8` |
| `BAYESIAN_LOC` | 贝叶斯位置 | `False` | `True` |
| `MC_DROPOUT` | MC Dropout | `False` | `True` |
| `MC_SAMPLES` | MC 采样数 | `30` | `10-20` |

### 融合模式对比

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| `uncertainty_weighted` | 基于不确定性加权（推荐） | 通用场景 |
| `learned` | 学习权重 | 有充足训练数据 |
| `simple_avg` | 简单平均 | 基线对比 |

---

## 训练融合模型

### 步骤 1: 训练基线模型

```bash
# 训练策略 A（回归）
python pose_estimator.py  # 配置：bayesian_loc=False

# 训练策略 B（贝叶斯）
python pose_estimator.py  # 配置：bayesian_loc=True
```

### 步骤 2: 训练融合模型

```python
# 使用 train_fusion.py
python train_fusion.py
```

### 步骤 3: 监控训练

```bash
# 查看训练日志
tensorboard --logdir ./models/logs

# 或在浏览器打开
# http://localhost:6006
```

---

## 评估结果

### 运行评估

```python
# 评估融合模型
python pose_estimator.py  # 配置：command="evaluate"
```

### 预期结果

| 指标 | 策略 A | 策略 B | 融合策略 |
|------|--------|--------|---------|
| 位置误差 (m) | 0.050 | 0.060 | **0.045** |
| 姿态误差 (deg) | 2.5 | 2.3 | **2.1** |
| OOD 检测 AUC | 0.65 | 0.73 | **0.82** |

### 可视化结果

```python
# 在 pose_estimator.py 中设置
command = "test"  # 可视化 10 张图的预测结果
```

---

## 高级用法

### 自定义融合权重

```python
from ursonet.nn.fusion_layer import UncertaintyFusion

# 创建自定义融合层
fusion_layer = UncertaintyFusion(
    weight_min=0.2,
    weight_max=0.8,
    fusion_mode='uncertainty_weighted'
)

# 应用到模型
fused_pred, fusion_weights = fusion_layer([
    pred_reg,
    pred_bayes,
    var_bayes
])
```

### 使用自适应融合

```python
from ursonet.nn.fusion_layer import AdaptiveFusion

# 自适应融合根据输入特征动态调整权重
adaptive_fusion = AdaptiveFusion(
    fusion_hidden_size=32,
    weight_min=0.1,
    weight_max=0.9
)
```

### 不确定性校准

```python
# 计算不确定性指标
from ursonet.nn.fusion_layer import compute_uncertainty_metrics

mc_samples = ...  # MC 采样结果
mean_pred, epistemic_var, total_var = compute_uncertainty_metrics(mc_samples)

# 不确定性校准度评估
# ECE (Expected Calibration Error)
```

---

## 故障排除

### 问题 1: CUDA 内存不足

```python
# 减少 batch size
config.IMAGES_PER_GPU = 1

# 或减少 MC 采样数
config.MC_SAMPLES = 5
```

### 问题 2: 训练不稳定

```python
# 增加 LOC_EPS
config.LOC_EPS = 1e-5  # 默认 1e-6

# 或调整学习率
config.LEARNING_RATE = 1e-6
```

### 问题 3: 融合效果不佳

```python
# 调整权重范围
config.FUSION_WEIGHT_MIN = 0.3
config.FUSION_WEIGHT_MAX = 0.7

# 或尝试不同融合模式
config.FUSION_MODE = 'simple_avg'  # 基线
```

---

## 文件结构

```
PoseCertain/
├── fusion_config.py           # 融合配置文件
├── train_fusion.py            # 融合训练脚本
├── ursonet/nn/fusion_layer.py # 融合层实现
├── pose_estimator.py          # 主程序
├── config.py                  # 基础配置
├── STRATEGY_ANALYSIS.md       # 策略分析文档
└── FUSION_PLAN.md             # 实施计划文档
```

---

## 参考资源

- [策略分析文档](./STRATEGY_ANALYSIS.md)
- [实施计划](./FUSION_PLAN.md)
- [MC Dropout 论文](https://arxiv.org/abs/1506.02142)
- [贝叶斯深度学习](https://arxiv.org/abs/1703.04977)

---

## 联系与支持

如有问题，请查看：
1. 项目文档
2. 示例代码
3. 相关论文

**祝训练顺利！** 🚀
