"""
融合策略配置文件 - 扩展原始 config.py

使用方法:
    在 pose_estimator.py 中导入并应用
"""

from config import Config


class FusionConfig(Config):
    """
    融合策略配置类
    
    在原始 Config 基础上添加融合相关配置
    """
    
    # ==================== 融合策略开关 ====================
    FUSE_LOC_STRATEGIES = False  # 是否启用位置融合
    FUSION_MODE = 'uncertainty_weighted'  # 融合模式
    # 可选值:
    #   - 'uncertainty_weighted': 基于不确定性加权（推荐）
    #   - 'learned': 学习权重
    #   - 'simple_avg': 简单平均
    
    # ==================== 融合参数 ====================
    FUSION_WEIGHT_MIN = 0.1  # 最小融合权重
    FUSION_WEIGHT_MAX = 0.9  # 最大融合权重
    FUSION_EPSILON = 1e-6    # 数值稳定性参数
    
    # ==================== 自适应融合参数 ====================
    USE_ADAPTIVE_FUSION = False  # 是否使用自适应融合
    FUSION_HIDDEN_SIZE = 32    # 融合网络隐藏层大小
    
    # ==================== 损失函数参数 ====================
    FUSION_LOSS_ALPHA = 0.5    # 融合损失平衡参数
    
    # ==================== 实验配置 ====================
    EXPERIMENT_NAME = 'fusion_baseline'  # 实验名称
    
    def __init__(self):
        super().__init__()
        self.update()
    
    def update(self):
        """更新派生参数"""
        # 调用父类更新
        super().update()
        
        # 融合策略一致性检查
        if self.FUSE_LOC_STRATEGIES:
            assert self.BAYESIAN_LOC, "融合策略需要 BAYESIAN_LOC=True"
            assert self.REGRESS_LOC, "融合策略需要 REGRESS_LOC=True"
        
        if self.USE_ADAPTIVE_FUSION:
            assert self.FUSION_MODE == 'learned', \
                "自适应融合需要使用 'learned' 模式"
    
    def display(self):
        """显示配置信息"""
        super().display()
        print("\n=== 融合策略配置 ===")
        print(f"FUSE_LOC_STRATEGIES:  {self.FUSE_LOC_STRATEGIES}")
        print(f"FUSION_MODE:          {self.FUSION_MODE}")
        print(f"FUSION_WEIGHT_MIN:    {self.FUSION_WEIGHT_MIN}")
        print(f"FUSION_WEIGHT_MAX:    {self.FUSION_WEIGHT_MAX}")
        print(f"USE_ADAPTIVE_FUSION:  {self.USE_ADAPTIVE_FUSION}")
        print(f"EXPERIMENT_NAME:      {self.EXPERIMENT_NAME}")
        print()


# ==================== 预设配置 ====================

class FusionBaselineConfig(FusionConfig):
    """
    基础融合配置 - 使用不确定性加权
    """
    def __init__(self):
        super().__init__()
        self.FUSE_LOC_STRATEGIES = True
        self.FUSION_MODE = 'uncertainty_weighted'
        self.EXPERIMENT_NAME = 'fusion_baseline'


class FusionLearnedConfig(FusionConfig):
    """
    学习权重融合配置
    """
    def __init__(self):
        super().__init__()
        self.FUSE_LOC_STRATEGIES = True
        self.FUSION_MODE = 'learned'
        self.USE_ADAPTIVE_FUSION = True
        self.EXPERIMENT_NAME = 'fusion_learned'


class FusionSimpleConfig(FusionConfig):
    """
    简单平均融合配置
    """
    def __init__(self):
        super().__init__()
        self.FUSE_LOC_STRATEGIES = True
        self.FUSION_MODE = 'simple_avg'
        self.EXPERIMENT_NAME = 'fusion_simple'


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例 1: 使用基础融合配置
    config = FusionBaselineConfig()
    config.display()
    
    # 示例 2: 自定义配置
    config = FusionConfig()
    config.FUSE_LOC_STRATEGIES = True
    config.FUSION_MODE = 'uncertainty_weighted'
    config.FUSION_WEIGHT_MIN = 0.2
    config.FUSION_WEIGHT_MAX = 0.8
    config.EXPERIMENT_NAME = 'my_fusion_exp'
    config.update()
    config.display()
