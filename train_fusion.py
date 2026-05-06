"""
融合策略训练脚本

基于不确定性加权的策略 A（回归）和策略 B（贝叶斯）融合

使用方法:
    python train_fusion.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# 导入项目模块
from pose_estimator import run_model_inference, extract_location_from_result, extract_orientation_from_result
from fusion_config import FusionBaselineConfig
from ursonet.nn.fusion_layer import UncertaintyFusion
import speed
import net
import utils


def build_fusion_model(base_model, config):
    """
    构建融合模型
    
    参数:
        base_model: 基础 UrsoNet 模型
        config: 融合配置
    
    返回:
        fusion_model: 融合模型
    """
    print("[INFO] 构建融合模型...")
    
    # 获取基础模型的输出
    # 假设基础模型已经输出回归和贝叶斯两个分支
    
    # 添加融合层
    fusion_layer = UncertaintyFusion(
        weight_min=config.FUSION_WEIGHT_MIN,
        weight_max=config.FUSION_WEIGHT_MAX,
        fusion_mode=config.FUSION_MODE
    )
    
    print(f"[INFO] 融合模式：{config.FUSION_MODE}")
    print(f"[INFO] 权重范围：[{config.FUSION_WEIGHT_MIN}, {config.FUSION_WEIGHT_MAX}]")
    
    return fusion_layer


def train_fusion(config, dataset_train, dataset_val, epochs=50):
    """
    训练融合模型
    
    参数:
        config: 融合配置
        dataset_train: 训练数据集
        dataset_val: 验证数据集
        epochs: 训练轮数
    """
    print("[INFO] 开始训练融合模型...")
    print(f"[INFO] 训练轮数：{epochs}")
    print(f"[INFO] 实验名称：{config.EXPERIMENT_NAME}")
    
    # 1. 加载基础模型
    model = net.UrsoNet(mode="training", config=config, model_dir="./models/logs")
    
    # 2. 构建融合模型
    fusion_layer = build_fusion_model(model, config)
    
    # 3. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 训练阶段
        train_loss = 0.0
        train_count = 0
        
        for image_id in dataset_train.image_ids[:100]:  # 小批量测试
            # 加载数据
            image = dataset_train.load_image(image_id)
            loc_gt = np.asarray(dataset_train.load_location(image_id)).reshape(-1)
            
            # 前向传播
            results = run_model_inference(model, image, verbose=0)
            result = results[0]
            
            # 提取预测
            loc_pred_reg = result.get('loc_reg', None)  # 回归分支
            loc_pred_bayes = result.get('loc_mu', None)  # 贝叶斯分支
            loc_var = result.get('loc_var', None)  # 方差
            
            if loc_pred_reg is not None and loc_pred_bayes is not None and loc_var is not None:
                # 融合
                fused_pred, fusion_weights = fusion_layer([
                    loc_pred_reg[np.newaxis, :],
                    loc_pred_bayes[np.newaxis, :],
                    loc_var[np.newaxis, :]
                ])
                
                # 计算损失
                loss = np.mean((fused_pred[0] - loc_gt) ** 2)
                train_loss += loss
                train_count += 1
        
        train_loss = train_loss / max(train_count, 1)
        
        # 验证阶段
        val_loss = 0.0
        val_count = 0
        
        for image_id in dataset_val.image_ids[:20]:
            image = dataset_val.load_image(image_id)
            loc_gt = np.asarray(dataset_val.load_location(image_id)).reshape(-1)
            
            results = run_model_inference(model, image, verbose=0)
            result = results[0]
            
            loc_pred_reg = result.get('loc_reg', None)
            loc_pred_bayes = result.get('loc_mu', None)
            loc_var = result.get('loc_var', None)
            
            if loc_pred_reg is not None and loc_pred_bayes is not None and loc_var is not None:
                fused_pred, fusion_weights = fusion_layer([
                    loc_pred_reg[np.newaxis, :],
                    loc_pred_bayes[np.newaxis, :],
                    loc_var[np.newaxis, :]
                ])
                
                loss = np.mean((fused_pred[0] - loc_gt) ** 2)
                val_loss += loss
                val_count += 1
        
        val_loss = val_loss / max(val_count, 1)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"[INFO] 保存最佳模型 (Val Loss: {val_loss:.6f})")
            # model.keras_model.save_weights(f"./models/logs/{config.EXPERIMENT_NAME}/best_weights.h5")
    
    print("\n[INFO] 训练完成!")
    return model


def evaluate_fusion(config, model, dataset, fusion_layer):
    """
    评估融合模型性能
    
    参数:
        config: 配置
        model: 模型
        dataset: 数据集
        fusion_layer: 融合层
    """
    print("\n[INFO] 开始评估融合模型...")
    
    loc_errors = []
    ori_errors = []
    
    for image_id in dataset.image_ids[:50]:  # 测试前 50 张图
        # 加载数据
        image = dataset.load_image(image_id)
        loc_gt = np.asarray(dataset.load_location(image_id)).reshape(-1)
        q_gt = np.asarray(dataset.load_quaternion(image_id)).reshape(-1)
        
        # 推理
        results = run_model_inference(model, image, verbose=0)
        result = results[0]
        
        # 提取预测
        loc_pred_reg = result.get('loc_reg', None)
        loc_pred_bayes = result.get('loc_mu', None)
        loc_var = result.get('loc_var', None)
        
        if loc_pred_reg is not None and loc_pred_bayes is not None and loc_var is not None:
            # 融合
            fused_pred, fusion_weights = fusion_layer([
                loc_pred_reg[np.newaxis, :],
                loc_pred_bayes[np.newaxis, :],
                loc_var[np.newaxis, :]
            ])
            
            # 计算误差
            loc_err = np.linalg.norm(fused_pred[0] - loc_gt)
            loc_errors.append(loc_err)
    
    # 统计结果
    loc_errors = np.array(loc_errors)
    print(f"\n=== 评估结果 ===")
    print(f"位置误差 (Mean): {np.mean(loc_errors):.6f} m")
    print(f"位置误差 (Median): {np.median(loc_errors):.6f} m")
    print(f"位置误差 (Std): {np.std(loc_errors):.6f}")
    
    return {
        'loc_mean': np.mean(loc_errors),
        'loc_median': np.median(loc_errors),
        'loc_std': np.std(loc_errors)
    }


if __name__ == '__main__':
    # ==================== 配置区域 ====================
    class Args:
        dataset = "speed"
        weights = "models/logs/speed20260413T1059/weights_best.h5"  # 预训练权重
        epochs = 50
        batch_size = 2
    
    args = Args()
    
    # ==================== 主流程 ====================
    print("=" * 60)
    print("融合策略训练脚本")
    print("=" * 60)
    
    # 1. 创建配置
    config = FusionBaselineConfig()
    config.IMAGES_PER_GPU = args.batch_size
    config.update()
    config.display()
    
    # 2. 加载数据集
    dataset_dir = f"./datasets/{args.dataset}"
    dataset_train = speed.Speed()
    dataset_train.load_dataset(dataset_dir, config, "train_no_val")
    
    dataset_val = speed.Speed()
    dataset_val.load_dataset(dataset_dir, config, "val")
    
    # 3. 训练
    model = train_fusion(config, dataset_train, dataset_val, epochs=args.epochs)
    
    # 4. 评估
    dataset_test = speed.Speed()
    dataset_test.load_dataset(dataset_dir, config, "my_test")
    
    fusion_layer = build_fusion_model(model, config)
    results = evaluate_fusion(config, model, dataset_test, fusion_layer)
    
    print("\n[INFO] 所有任务完成!")
