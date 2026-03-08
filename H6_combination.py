"""
H6: 组合最优策略
基于F10 + H2时间集成 + H4异构集成

核心创新：
1. F10基础：元学习率控制器 + 自适应权重调整
2. H2策略：训练过程中保存多个checkpoint，集成预测
3. H4策略：集成Transformer + XGBoost + LightGBM
4. 双层集成：
   - 第一层：时间集成（多个Transformer checkpoint）
   - 第二层：异构集成（Transformer集成 + XGBoost + LightGBM）

目标：结合H2的+2.42%和H4的+0.67%提升，实现更大的性能增益
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import xgboost as xgb
from lightgbm import LGBMClassifier

# 从F10导入核心组件
from F10_adaptive_weights import (
    AdaptiveTrainer,
    search_thresholds_e8_style
)

# 从E8导入工具函数
from E8_fixed_issues import adjust_predictions_with_thresholds

# 从sklearn导入评估指标
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report

# 从enhanced_transformer_model导入模型
from enhanced_transformer_model import EnhancedTransformerModel


# ============================================================================
# 时间集成训练器（继承自F10的AdaptiveTrainer）
# ============================================================================

class TemporalEnsembleTrainer(AdaptiveTrainer):
    """时间集成训练器 - 保存多个checkpoint并集成预测"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, samples_per_class,
                 checkpoint_start=50, checkpoint_interval=5):
        super().__init__(model, train_loader, val_loader, test_loader, device, samples_per_class)
        
        self.checkpoint_start = checkpoint_start
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []  # 存储checkpoint的epoch和得分
        self.checkpoint_models = []  # 存储模型状态
    
    def train_adaptive(self, epochs=150, patience=30, model_name='H6_model', 
                      min_majority_f1=0.85, weight_update_interval=8):
        """训练并保存多个checkpoint"""
        print(f"\n开始时间集成训练...")
        print(f"  从Epoch {self.checkpoint_start}开始，每{self.checkpoint_interval}个epoch保存checkpoint")
        print(f"  使用元学习率控制器 + 自适应权重")
        
        best_score = 0
        patience_counter = 0
        weight_update_counter = 0
        
        for epoch in range(epochs):
            # 评估当前状态
            val_minority_recall, val_macro_f1, val_majority_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # 构建训练状态向量
            epoch_progress = (epoch + 1) / epochs
            training_state = [
                0.5,
                val_macro_f1,
                val_minority_recall,
                val_majority_f1,
                epoch_progress
            ]
            
            # 更新学习率
            current_lr = self.update_learning_rate(training_state)
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            
            # 重新评估
            val_minority_recall, val_macro_f1, val_majority_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # 综合得分
            combined_score = val_minority_recall * 0.4 + val_majority_f1 * 0.6
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Loss={train_loss:.4f}, LR={current_lr:.6f}")
                print(f"  Val F1 per class: [提前={val_f1_per_class[0]:.4f}, "
                      f"不流失={val_f1_per_class[1]:.4f}, 到期={val_f1_per_class[2]:.4f}]")
                print(f"  综合得分={combined_score:.4f}")
            
            # 动态调整权重
            weight_update_counter += 1
            if weight_update_counter >= weight_update_interval and epoch < epochs - 20:
                print(f"\n  [权重调整] Epoch {epoch+1}")
                self.criterion.update_weights(val_f1_per_class, target_f1=0.55)
                weight_update_counter = 0
            
            # 保存checkpoint（从指定epoch开始）
            if epoch + 1 >= self.checkpoint_start and (epoch + 1 - self.checkpoint_start) % self.checkpoint_interval == 0:
                if val_majority_f1 >= min_majority_f1:
                    checkpoint_state = {
                        'epoch': epoch + 1,
                        'model_state': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                        'score': combined_score
                    }
                    self.checkpoints.append((epoch + 1, combined_score))
                    self.checkpoint_models.append(checkpoint_state)
                    if (epoch + 1) % 10 == 0:
                        print(f"    ✓ 保存checkpoint (Epoch {epoch+1}, Score={combined_score:.4f})")
            
            # 保存最佳模型
            if val_majority_f1 >= min_majority_f1:
                if combined_score > best_score:
                    best_score = combined_score
                    patience_counter = 0
                    torch.save(self.model.state_dict(), f'{model_name}.pth')
                else:
                    patience_counter += 1
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n训练完成！")
        print(f"  保存了 {len(self.checkpoints)} 个checkpoint")
        print(f"  Checkpoint epochs: {[c[0] for c in self.checkpoints]}")
        
        return best_score
    
    def ensemble_predict(self, data_loader, n_models=10):
        """使用最后n_models个checkpoint进行集成预测"""
        if len(self.checkpoint_models) == 0:
            print("⚠️  没有保存的checkpoint，使用当前模型")
            _, _, _, _, probs, labels = self.evaluate(data_loader)
            return probs, labels
        
        # 选择最后n_models个checkpoint
        selected_checkpoints = self.checkpoint_models[-n_models:]
        n_selected = len(selected_checkpoints)
        
        print(f"\n使用 {n_selected} 个checkpoint进行集成预测")
        print(f"  Epochs: {[c['epoch'] for c in selected_checkpoints]}")
        
        # 指数加权（越晚的checkpoint权重越高）
        weights = np.arange(1, n_selected + 1, dtype=np.float32)
        weights = weights / weights.sum()
        print(f"  使用加权平均，权重: {weights}")
        
        # 收集所有预测
        all_probs = []
        labels = None
        
        for checkpoint in selected_checkpoints:
            # 加载checkpoint
            self.model.load_state_dict(checkpoint['model_state'])
            
            # 预测
            _, _, _, _, probs, lbls = self.evaluate(data_loader)
            all_probs.append(probs)
            
            if labels is None:
                labels = lbls
        
        # 加权平均
        ensemble_probs = np.zeros_like(all_probs[0])
        for i, probs in enumerate(all_probs):
            ensemble_probs += probs * weights[i]
        
        return ensemble_probs, labels


# ============================================================================
# 异构集成（Transformer时间集成 + XGBoost + LightGBM）
# ============================================================================

def train_heterogeneous_ensemble(X_train, y_train, X_val, y_val, X_threshold, y_threshold, 
                                 X_test, y_test, device):
    """训练异构集成模型"""
    
    print("\n" + "=" * 80)
    print("训练异构集成模型")
    print("=" * 80)
    print("集成策略:")
    print("  1. Transformer时间集成（F10 + H2）权重=0.5")
    print("  2. XGBoost（梯度提升树）权重=0.3")
    print("  3. LightGBM（轻量级梯度提升）权重=0.2")
    print("=" * 80)
    
    # 计算类别分布
    unique, counts = np.unique(y_train.numpy(), return_counts=True)
    samples_per_class = counts
    
    # 集成权重
    ensemble_weights = [0.5, 0.3, 0.2]
    print(f"\n集成权重: Transformer={ensemble_weights[0]}, XGBoost={ensemble_weights[1]}, LightGBM={ensemble_weights[2]}")
    
    # ========================================================================
    # [1/3] 训练Transformer时间集成
    # ========================================================================
    print("\n" + "=" * 80)
    print("[1/3] 训练Transformer时间集成（F10 + H2配置）")
    print("=" * 80)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    threshold_dataset = torch.utils.data.TensorDataset(X_threshold, y_threshold)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    threshold_loader = torch.utils.data.DataLoader(threshold_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 创建模型
    transformer_model = EnhancedTransformerModel(
        input_dim=X_train.shape[1],
        d_model=128,
        n_heads=8,
        n_layers=3,
        n_classes=3,
        dropout=0.1
    ).to(device)
    
    # 时间集成训练
    temporal_trainer = TemporalEnsembleTrainer(
        transformer_model, train_loader, val_loader, test_loader, device, samples_per_class,
        checkpoint_start=50, checkpoint_interval=5
    )
    
    temporal_trainer.train_adaptive(
        epochs=150,
        patience=30,
        model_name='H6_transformer',
        min_majority_f1=0.85,
        weight_update_interval=8
    )
    
    print("✓ Transformer时间集成训练完成")
    
    # ========================================================================
    # [2/3] 训练XGBoost
    # ========================================================================
    print("\n" + "=" * 80)
    print("[2/3] 训练XGBoost模型")
    print("=" * 80)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    
    xgb_model.fit(
        X_train.cpu().numpy(), y_train.cpu().numpy(),
        eval_set=[(X_val.cpu().numpy(), y_val.cpu().numpy())],
        verbose=50
    )
    
    print("✓ XGBoost训练完成")
    
    # ========================================================================
    # [3/3] 训练LightGBM
    # ========================================================================
    print("\n" + "=" * 80)
    print("[3/3] 训练LightGBM模型")
    print("=" * 80)
    
    lgb_model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=3,
        random_state=42,
        device='cpu',  # 改为CPU，避免OpenCL错误
        verbose=-1
    )
    
    lgb_model.fit(
        X_train.cpu().numpy(), y_train.cpu().numpy(),
        eval_set=[(X_val.cpu().numpy(), y_val.cpu().numpy())],
        eval_metric='multi_logloss'
    )
    
    print("✓ LightGBM训练完成")
    
    # ========================================================================
    # 阈值搜索（使用集成预测）
    # ========================================================================
    print("\n" + "=" * 80)
    print("阈值搜索（使用集成预测）")
    print("=" * 80)
    
    # Transformer时间集成预测
    transformer_probs, _ = temporal_trainer.ensemble_predict(threshold_loader, n_models=10)
    
    # XGBoost预测
    xgb_probs = xgb_model.predict_proba(X_threshold.cpu().numpy())
    
    # LightGBM预测
    lgb_probs = lgb_model.predict_proba(X_threshold.cpu().numpy())
    
    # 加权融合
    ensemble_probs = (
        ensemble_weights[0] * transformer_probs +
        ensemble_weights[1] * xgb_probs +
        ensemble_weights[2] * lgb_probs
    )
    
    # 搜索阈值
    best_thresholds = search_thresholds_e8_style(ensemble_probs, y_threshold.numpy())
    
    # ========================================================================
    # 测试集评估
    # ========================================================================
    print("\n" + "=" * 80)
    print("H6: 测试结果（双层集成）")
    print("=" * 80)
    
    # Transformer时间集成预测
    transformer_test_probs, _ = temporal_trainer.ensemble_predict(test_loader, n_models=10)
    
    # XGBoost预测
    xgb_test_probs = xgb_model.predict_proba(X_test.cpu().numpy())
    
    # LightGBM预测
    lgb_test_probs = lgb_model.predict_proba(X_test.cpu().numpy())
    
    # 加权融合
    ensemble_test_probs = (
        ensemble_weights[0] * transformer_test_probs +
        ensemble_weights[1] * xgb_test_probs +
        ensemble_weights[2] * lgb_test_probs
    )
    
    # 应用阈值
    test_preds = adjust_predictions_with_thresholds(ensemble_test_probs, best_thresholds)
    
    # 计算指标
    test_acc = accuracy_score(y_test.numpy(), test_preds)
    test_f1 = f1_score(y_test.numpy(), test_preds, average='macro')
    
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试Macro F1: {test_f1:.4f}")
    
    report = classification_report(y_test.numpy(), test_preds,
                                   target_names=['提前流失', '不流失', '到期流失'],
                                   digits=4, zero_division=0)
    print("\n分类报告:")
    print(report)
    
    # 详细指标
    recall_per_class = recall_score(y_test.numpy(), test_preds, average=None, zero_division=0)
    precision_per_class = precision_score(y_test.numpy(), test_preds, average=None, zero_division=0)
    f1_per_class = f1_score(y_test.numpy(), test_preds, average=None, zero_division=0)
    
    # 单模型性能对比
    print("\n" + "=" * 80)
    print("单模型 vs 集成性能对比")
    print("=" * 80)
    
    # Transformer单独
    transformer_preds = adjust_predictions_with_thresholds(transformer_test_probs, best_thresholds)
    transformer_f1 = f1_score(y_test.numpy(), transformer_preds, average='macro')
    print(f"  Transformer时间集成: {transformer_f1:.4f}")
    
    # XGBoost单独
    xgb_preds = adjust_predictions_with_thresholds(xgb_test_probs, best_thresholds)
    xgb_f1 = f1_score(y_test.numpy(), xgb_preds, average='macro')
    print(f"  XGBoost:             {xgb_f1:.4f}")
    
    # LightGBM单独
    lgb_preds = adjust_predictions_with_thresholds(lgb_test_probs, best_thresholds)
    lgb_f1 = f1_score(y_test.numpy(), lgb_preds, average='macro')
    print(f"  LightGBM:            {lgb_f1:.4f}")
    
    # 集成
    print(f"  双层集成 (H6):       {test_f1:.4f}")
    
    best_single = max(transformer_f1, xgb_f1, lgb_f1)
    improvement = (test_f1 - best_single) / best_single * 100
    print(f"  相比最佳单模型提升:  {improvement:+.2f}%")
    
    # 保存结果
    results = {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'recall_class_0': float(recall_per_class[0]),
        'recall_class_1': float(recall_per_class[1]),
        'recall_class_2': float(recall_per_class[2]),
        'precision_class_0': float(precision_per_class[0]),
        'precision_class_1': float(precision_per_class[1]),
        'precision_class_2': float(precision_per_class[2]),
        'f1_class_0': float(f1_per_class[0]),
        'f1_class_1': float(f1_per_class[1]),
        'f1_class_2': float(f1_per_class[2]),
        'minority_avg_recall': float((recall_per_class[0] + recall_per_class[2]) / 2),
        'majority_f1': float(f1_per_class[1]),
        'thresholds': best_thresholds,
        'ensemble_weights': ensemble_weights,
        'single_model_f1': {
            'transformer_temporal': float(transformer_f1),
            'xgboost': float(xgb_f1),
            'lightgbm': float(lgb_f1)
        },
        'optimization_strategy': 'F10 + H2_temporal_ensemble + H4_heterogeneous_ensemble',
        'n_temporal_checkpoints': len(temporal_trainer.checkpoints)
    }
    
    with open('H6_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ 结果已保存到 H6_results.json")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("实验H6: 组合最优策略（F10 + H2 + H4）")
    print("=" * 80)
    print("双层集成架构:")
    print("  第一层：时间集成（保存多个checkpoint，加权平均）")
    print("  第二层：异构集成（Transformer时间集成 + XGBoost + LightGBM）")
    print("  基础配置：F10的元学习率控制器 + 自适应权重调整")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载数据
    try:
        with open('../preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except:
        with open('/root/autodl-tmp/preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.LongTensor(data['y_test'])
    
    # E8的三分割策略
    val_size = len(X_val)
    split_idx = int(val_size * 0.5)
    
    X_val_new = X_val[:split_idx]
    y_val_new = y_val[:split_idx]
    X_threshold = X_val[split_idx:]
    y_threshold = y_val[split_idx:]
    
    print(f"\n数据分割:")
    print(f"  训练集: {len(X_train):,}")
    print(f"  验证集: {len(X_val_new):,}")
    print(f"  阈值搜索集: {len(X_threshold):,}")
    print(f"  测试集: {len(X_test):,}")
    
    # 训练异构集成
    results = train_heterogeneous_ensemble(
        X_train, y_train, X_val_new, y_val_new, 
        X_threshold, y_threshold, X_test, y_test, device
    )
    
    # 对比F10
    print("\n" + "=" * 80)
    print("H6 vs F10 对比")
    print("=" * 80)
    print(f"  F10: Macro F1 = 0.5305 (单Transformer)")
    print(f"  H2:  Macro F1 = 0.5433 (时间集成, +2.42%)")
    print(f"  H4:  Macro F1 = 0.5341 (异构集成, +0.67%)")
    print(f"  H6:  Macro F1 = {results['test_f1_macro']:.4f} (双层集成)")
    
    f10_baseline = 0.5305
    h6_improvement = (results['test_f1_macro'] - f10_baseline) / f10_baseline * 100
    print(f"  相比F10提升: {h6_improvement:+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

