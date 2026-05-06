"""
不确定性融合层 - 用于融合策略 A（回归）和策略 B（贝叶斯）

基于不确定性加权融合：
- 回归分支提供稳定性
- 贝叶斯分支提供不确定性估计
- 动态权重：不确定性越低，权重越高
"""

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K


class UncertaintyFusion(KL.Layer):
    """
    基于不确定性的融合层
    
    融合公式:
        fused = w * pred_bayes + (1 - w) * pred_reg
        w = 1 / (σ² + ε)  # 贝叶斯权重（不确定性越低权重越高）
    
    输入:
        - regression_pred: 回归分支预测 (N, 3)
        - bayesian_pred: 贝叶斯分支预测 (N, 3) 
        - bayesian_var: 贝叶斯分支方差 (N, 3)
    
    输出:
        - fused_pred: 融合后的预测 (N, 3)
        - fusion_weights: 融合权重 (N, 3)
    """
    
    def __init__(self, 
                 weight_min=0.1, 
                 weight_max=0.9, 
                 epsilon=1e-6,
                 fusion_mode='uncertainty_weighted',
                 **kwargs):
        """
        参数:
            weight_min: 最小融合权重
            weight_max: 最大融合权重
            epsilon: 数值稳定性参数
            fusion_mode: 融合模式
                - 'uncertainty_weighted': 基于不确定性加权（推荐）
                - 'learned': 学习权重（需要额外参数）
                - 'simple_avg': 简单平均
        """
        super(UncertaintyFusion, self).__init__(**kwargs)
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.epsilon = epsilon
        self.fusion_mode = fusion_mode
        
        # 仅在学习模式下需要
        if fusion_mode == 'learned':
            self.learned_weight = self.add_weight(
                name='learned_weight',
                shape=(1,),
                initializer='constant',
                trainable=True
            )
    
    def call(self, inputs):
        regression_pred, bayesian_pred, bayesian_var = inputs
        
        if self.fusion_mode == 'simple_avg':
            # 简单平均融合
            fused_pred = 0.5 * regression_pred + 0.5 * bayesian_pred
            fusion_weights = tf.ones_like(bayesian_pred) * 0.5
            
        elif self.fusion_mode == 'learned':
            # 学习权重融合
            w = tf.sigmoid(self.learned_weight)
            fused_pred = w * bayesian_pred + (1 - w) * regression_pred
            fusion_weights = tf.ones_like(bayesian_pred) * w
            
        else:  # 'uncertainty_weighted' (推荐)
            # 基于不确定性的加权融合
            # 计算贝叶斯权重：w = 1 / (σ² + ε)
            inv_var = 1.0 / (bayesian_var + self.epsilon)
            
            # 归一化：w_bayes = σ² / (σ² + 1) = 1 / (σ² + 1)
            # 这样当 σ² → 0 时，w → 1；当 σ² → ∞ 时，w → 0
            w_bayes = inv_var / (inv_var + 1.0)
            
            # 限制权重范围
            w_bayes = tf.clip_by_value(w_bayes, self.weight_min, self.weight_max)
            
            # 融合
            fused_pred = w_bayes * bayesian_pred + (1 - w_bayes) * regression_pred
            fusion_weights = w_bayes
        
        return fused_pred, fusion_weights
    
    def get_config(self):
        config = {
            'weight_min': float(self.weight_min),
            'weight_max': float(self.weight_max),
            'epsilon': float(self.epsilon),
            'fusion_mode': self.fusion_mode
        }
        base_config = super(UncertaintyFusion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaptiveFusion(KL.Layer):
    """
    自适应融合层 - 根据输入特征动态调整融合权重
    
    使用一个小型神经网络根据输入特征预测融合权重
    """
    
    def __init__(self, 
                 fusion_hidden_size=32,
                 weight_min=0.1, 
                 weight_max=0.9,
                 **kwargs):
        super(AdaptiveFusion, self).__init__(**kwargs)
        self.fusion_hidden_size = fusion_hidden_size
        self.weight_min = weight_min
        self.weight_max = weight_max
        
        # 权重预测网络
        self.weight_net = tf.keras.Sequential([
            KL.Dense(fusion_hidden_size, activation='relu'),
            KL.Dense(3),  # 输出 3 个权重（对应 x, y, z）
            KL.Activation('sigmoid')  # 限制在 [0, 1]
        ], name='weight_prediction_net')
    
    def call(self, inputs):
        regression_pred, bayesian_pred, bayesian_var, features = inputs
        
        # 根据特征预测权重
        predicted_weights = self.weight_net(features)
        
        # 限制权重范围
        predicted_weights = tf.clip_by_value(
            predicted_weights, 
            self.weight_min, 
            self.weight_max
        )
        
        # 融合
        fused_pred = predicted_weights * bayesian_pred + \
                     (1 - predicted_weights) * regression_pred
        
        return fused_pred, predicted_weights
    
    def get_config(self):
        config = {
            'fusion_hidden_size': self.fusion_hidden_size,
            'weight_min': self.weight_min,
            'weight_max': self.weight_max
        }
        base_config = super(AdaptiveFusion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# 辅助函数
def compute_uncertainty_metrics(mc_samples):
    """
    从 MC Dropout 采样中计算不确定性指标
    
    参数:
        mc_samples: MC 采样结果，shape 为 (N_samples, N_instances, 3)
    
    返回:
        mean_pred: 预测均值
        epistemic_var: 认知不确定性（模型不确定性）
        aleatoric_var: 数据不确定性（如果可用）
        total_var: 总不确定性
    """
    # 预测均值
    mean_pred = tf.reduce_mean(mc_samples, axis=0)
    
    # 认知不确定性（方差）
    epistemic_var = tf.reduce_var(mc_samples, axis=0)
    
    # 如果是贝叶斯输出，还可以计算 aleatoric
    # 这里假设输入只是 MC 采样
    
    total_var = epistemic_var
    
    return mean_pred, epistemic_var, total_var


def fusion_loss(y_true, y_pred_fused, y_pred_reg, y_pred_bayes, 
                var_bayes, alpha=0.5):
    """
    融合损失函数
    
    L = alpha * L_reg + (1-alpha) * L_bayes
    
    参数:
        y_true: 真实值
        y_pred_fused: 融合预测
        y_pred_reg: 回归分支预测
        y_pred_bayes: 贝叶斯分支预测（均值）
        var_bayes: 贝叶斯分支方差
        alpha: 平衡参数
    """
    # 回归分支损失（MSE）
    loss_reg = tf.reduce_mean(tf.square(y_true - y_pred_reg))
    
    # 贝叶斯分支损失（Gaussian NLL）
    epsilon = 1e-6
    nll = 0.5 * (tf.math.log(var_bayes + epsilon) + 
                 tf.square(y_true - y_pred_bayes) / (var_bayes + epsilon))
    loss_bayes = tf.reduce_mean(nll)
    
    # 融合损失
    loss_fused = alpha * loss_reg + (1 - alpha) * loss_bayes
    
    return loss_fused, loss_reg, loss_bayes
