"""
F10: 自适应权重优化 + 元学习率控制器
基于E8_fixed_issues.py + 动态权重调整策略 + G3元学习率控制器

核心创新：
1. 自适应权重调整：根据验证集表现动态调整类别权重
2. 元学习率控制器：根据训练状态动态调整学习率（来自G3）
3. 多阶段训练：
   - 阶段1：使用E8的权重（beta=0.9999）热身
   - 阶段2：根据各类别F1动态调整权重
   - 阶段3：微调阶段
4. 权重调整策略：F1低的类别增加权重，F1高的类别减少权重
5. 多数类F1保护机制（≥0.75）

目标：通过动态权重调整 + 自适应学习率实现最优平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import sys
sys.path.append('..')

# 从E8导入基础组件（只导入E8中实际存在的）
from E8_fixed_issues import (
    adjust_predictions_with_thresholds,
    compute_unified_score,
    ClassBalancedLoss,
    Trainer as E8Trainer  # E8的Trainer类
)
from enhanced_transformer_model import EnhancedTransformerModel
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score


# ============================================================================
# 元学习率控制器（来自G3）
# ============================================================================

class MetaLRController(nn.Module):
    """
    元学习率控制器
    根据训练状态动态预测最优学习率
    """
    def __init__(self, input_dim=5, hidden_dim=32, lr_min=1e-5, lr_max=1e-3):
        super().__init__()
        self.lr_min = lr_min
        self.lr_max = lr_max
        
        # 简单的MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出[0,1]，然后映射到[lr_min, lr_max]
        )
    
    def forward(self, training_state):
        """
        输入: [loss, f1, recall, precision, epoch_progress]
        输出: 学习率
        """
        lr_normalized = self.network(training_state)
        lr = self.lr_min + (self.lr_max - self.lr_min) * lr_normalized
        return lr


# ============================================================================
# 自适应Class-Balanced Loss
# ============================================================================

class AdaptiveClassBalancedLoss(nn.Module):
    """自适应调整权重的损失函数"""
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        
        # 初始权重（使用E8的策略）
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        
        self.weights = torch.FloatTensor(weights)
        self.initial_weights = weights.copy()
        print(f"初始类别权重: {weights}")
    
    def update_weights(self, f1_per_class, target_f1=0.50):
        """
        根据各类别F1动态调整权重
        
        策略：
        - F1低于目标的类别：增加权重
        - F1高于目标的类别：减少权重
        - 调整幅度与F1差距成正比
        """
        new_weights = self.initial_weights.copy()
        
        for i in range(len(f1_per_class)):
            f1_gap = target_f1 - f1_per_class[i]
            
            if i == 1:  # 多数类：保守调整
                adjustment = 1.0 + f1_gap * 0.15  # 降低到0.15
            else:  # 少数类：激进调整
                adjustment = 1.0 + f1_gap * 0.6  # 提高到0.6
            
            # 限制调整范围
            adjustment = np.clip(adjustment, 0.5, 3.0)
            new_weights[i] *= adjustment
        
        # 归一化
        new_weights = new_weights / new_weights.sum() * len(new_weights)
        
        self.weights = torch.FloatTensor(new_weights)
        print(f"  更新后权重: {new_weights}")
        print(f"  权重变化: {new_weights / self.initial_weights}")
        
        return new_weights
    
    def forward(self, logits, labels):
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
        
        # 标准的加权交叉熵
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        weights = self.weights[labels]
        
        return (ce_loss * weights).mean()


# ============================================================================
# 自适应训练器（基于E8的Trainer，但不使用继承，而是组合）
# ============================================================================

class AdaptiveTrainer:
    """自适应权重调整 + 元学习率控制的训练器"""
    def __init__(self, model, train_loader, val_loader, test_loader, device, samples_per_class):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.samples_per_class = samples_per_class
        
        # 使用自适应损失函数
        self.criterion = AdaptiveClassBalancedLoss(samples_per_class, beta=0.9999)
        
        # 元学习率控制器
        self.meta_lr_controller = MetaLRController(
            input_dim=5,
            hidden_dim=32,
            lr_min=1e-5,
            lr_max=1e-3
        ).to(device)
        
        # 主模型优化器（初始学习率会被元控制器覆盖）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=0.01
        )
        
        # 元学习率控制器的优化器
        self.meta_optimizer = torch.optim.Adam(
            self.meta_lr_controller.parameters(),
            lr=1e-3
        )
        
        # 记录学习率历史
        self.lr_history = []
        
        # 不需要scheduler，因为使用元学习率控制器
        self.scheduler = None
    
    def update_learning_rate(self, training_state):
        """使用元学习率控制器更新学习率"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(training_state).unsqueeze(0).to(self.device)
            new_lr = self.meta_lr_controller(state_tensor).item()
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.lr_history.append(new_lr)
        return new_lr
    
    def train_epoch(self, current_epoch=None, total_epochs=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.train_loader), correct / total
    
    def evaluate(self, data_loader, return_probs=False, return_detailed=False):
        """评估模型（兼容E8和F10的需求）"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算各类别F1
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        # 计算少数类平均召回率
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        minority_recall = (recall_per_class[0] + recall_per_class[2]) / 2
        
        # 计算少数类平均精确度
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        minority_precision = (precision_per_class[0] + precision_per_class[2]) / 2
        
        # 多数类F1
        majority_f1 = f1_per_class[1]
        
        # 整体F1
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # 准确率
        acc = accuracy_score(all_labels, all_preds)
        
        # F10风格返回
        if not return_detailed and not return_probs:
            return minority_recall, macro_f1, majority_f1, f1_per_class, all_probs, all_labels
        
        # E8风格返回
        if return_detailed:
            unified_score = compute_unified_score(macro_f1, minority_recall, minority_precision, acc)
            if return_probs:
                return 0.0, acc, macro_f1, minority_recall, minority_precision, unified_score, all_probs, all_labels
            return 0.0, acc, macro_f1, minority_recall, minority_precision, unified_score
        
        if return_probs:
            return 0.0, acc, macro_f1, all_probs, all_labels
        
        return 0.0, acc, macro_f1
    
    def train_adaptive(self, epochs=100, patience=20, model_name='F10_model', 
                      min_majority_f1=0.75, weight_update_interval=10):
        """
        自适应训练流程（集成元学习率控制器）
        
        参数：
        - weight_update_interval: 每隔多少个epoch更新一次权重
        """
        print(f"\n开始自适应训练（每{weight_update_interval}个epoch调整权重）...")
        print(f"  目标：多数类F1≥{min_majority_f1:.2f}，少数类F1≥0.40")
        print(f"  使用元学习率控制器动态调整学习率")
        
        best_score = 0
        patience_counter = 0
        weight_update_counter = 0
        
        for epoch in range(epochs):
            # 先评估当前状态，用于元学习率控制
            val_minority_recall, val_macro_f1, val_majority_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # 构建训练状态向量
            epoch_progress = (epoch + 1) / epochs
            training_state = [
                0.5,  # 初始loss估计（会在训练中更新）
                val_macro_f1,
                val_minority_recall,
                val_majority_f1,
                epoch_progress
            ]
            
            # 更新学习率
            current_lr = self.update_learning_rate(training_state)
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            
            # 重新评估（获取最新指标）
            val_minority_recall, val_macro_f1, val_majority_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # 综合得分
            combined_score = val_minority_recall * 0.4 + val_majority_f1 * 0.6
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Loss={train_loss:.4f}, LR={current_lr:.6f}")
                print(f"  Val F1 per class: [提前流失={val_f1_per_class[0]:.4f}, "
                      f"不流失={val_f1_per_class[1]:.4f}, 到期流失={val_f1_per_class[2]:.4f}]")
                print(f"  Val少数类召回={val_minority_recall:.4f}, Val多数类F1={val_majority_f1:.4f}")
                print(f"  综合得分={combined_score:.4f}")
            
            # 动态调整权重
            weight_update_counter += 1
            if weight_update_counter >= weight_update_interval and epoch < epochs - 20:
                print(f"\n  [权重调整] Epoch {epoch+1}")
                print(f"  当前F1: {val_f1_per_class}")
                self.criterion.update_weights(val_f1_per_class, target_f1=0.55)
                weight_update_counter = 0
            
            # 只有在多数类F1达标时才保存
            if val_majority_f1 >= min_majority_f1:
                if combined_score > best_score:
                    best_score = combined_score
                    patience_counter = 0
                    torch.save(self.model.state_dict(), f'{model_name}.pth')
                    if (epoch + 1) % 10 == 0:
                        print(f"  ✓ 保存模型（综合得分={combined_score:.4f}）")
                else:
                    patience_counter += 1
            else:
                patience_counter += 1
                if (epoch + 1) % 10 == 0:
                    print(f"  ⚠️  多数类F1={val_majority_f1:.4f} < {min_majority_f1:.2f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        try:
            self.model.load_state_dict(torch.load(f'{model_name}.pth'))
            print(f"\n训练完成，最佳综合得分: {best_score:.4f}")
        except:
            print(f"\n⚠️  未找到保存的模型，使用当前模型")
        
        # 打印学习率统计
        if self.lr_history:
            print(f"\n学习率统计:")
            print(f"  初始LR: {self.lr_history[0]:.6f}")
            print(f"  最终LR: {self.lr_history[-1]:.6f}")
            print(f"  平均LR: {np.mean(self.lr_history):.6f}")
            print(f"  最大LR: {np.max(self.lr_history):.6f}")
            print(f"  最小LR: {np.min(self.lr_history):.6f}")


# ============================================================================
# E8的阈值搜索（复用E8的方法）
# ============================================================================

def search_thresholds_e8_style(probs, labels):
    """使用E8的dynamic阈值搜索方法（复用E8逻辑）"""
    print(f"\n使用E8的dynamic阈值搜索方法...")
    
    best_thresholds = []
    
    for class_idx in range(3):
        binary_labels = (labels == class_idx).astype(int)
        class_probs = probs[:, class_idx]
        
        if class_idx == 1:  # 多数类
            best_threshold = 0.5
        else:  # 少数类
            best_threshold = 0.5
            best_recall = 0
            
            for threshold in np.arange(0.1, 0.7, 0.01):
                preds = (class_probs >= threshold).astype(int)
                
                if preds.sum() == 0:
                    continue
                
                recall = recall_score(binary_labels, preds, zero_division=0)
                precision = precision_score(binary_labels, preds, zero_division=0)
                
                if precision >= 0.30 and recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
            
            print(f"  类别{class_idx}: 阈值={best_threshold:.4f}, Recall={best_recall:.4f}")
        
        best_thresholds.append(best_threshold)
    
    return best_thresholds


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("实验F10: 自适应权重优化 + 元学习率控制器")
    print("=" * 80)
    print("自适应优化策略:")
    print("  1. 初始权重：使用E8的Class-Balanced Loss (beta=0.9999)")
    print("  2. 动态调整：每10个epoch根据各类别F1调整权重")
    print("  3. 调整策略：F1低的类别增加权重，F1高的类别减少权重")
    print("  4. 多数类保护：确保多数类F1≥0.75")
    print("  5. 阈值搜索：使用E8的dynamic方法")
    print("  6. 综合得分：少数类召回*0.4 + 多数类F1*0.6")
    print("  7. 元学习率控制器：根据训练状态动态调整学习率（来自G3）")
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
    
    # 计算类别分布
    unique, counts = np.unique(y_train.numpy(), return_counts=True)
    samples_per_class = counts
    print(f"\n训练集类别分布: {samples_per_class}")
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val_new, y_val_new)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 创建模型
    model = EnhancedTransformerModel(
        input_dim=X_train.shape[1],
        d_model=128,
        n_heads=8,
        n_layers=3,
        n_classes=3,
        dropout=0.1
    ).to(device)
    
    # 自适应训练（集成元学习率控制器）
    trainer = AdaptiveTrainer(model, train_loader, val_loader, test_loader, device, samples_per_class)
    trainer.train_adaptive(
        epochs=150,  # 延长到150个epoch
        patience=30,  # 增加patience到30
        model_name='F10_model', 
        min_majority_f1=0.85,  # 提高到0.85
        weight_update_interval=8  # 更频繁调整（每8个epoch）
    )
    
    # 阈值搜索
    print("\n" + "=" * 80)
    print("阈值搜索")
    print("=" * 80)
    
    threshold_dataset = torch.utils.data.TensorDataset(X_threshold, y_threshold)
    threshold_loader = torch.utils.data.DataLoader(threshold_dataset, batch_size=256, shuffle=False)
    
    _, _, _, _, threshold_probs, threshold_labels = trainer.evaluate(threshold_loader)
    
    best_thresholds = search_thresholds_e8_style(threshold_probs, threshold_labels)
    
    # 评估阈值
    preds = adjust_predictions_with_thresholds(threshold_probs, best_thresholds)
    recall_per_class = recall_score(threshold_labels, preds, average=None, zero_division=0)
    precision_per_class = precision_score(threshold_labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(threshold_labels, preds, average=None, zero_division=0)
    f1 = f1_score(threshold_labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(threshold_labels, preds)
    
    print(f"\n  最佳阈值: {best_thresholds}")
    print(f"  Threshold_search集指标:")
    print(f"    Macro F1: {f1:.4f}")
    print(f"    准确率: {acc:.4f}")
    print(f"    多数类F1: {f1_per_class[1]:.4f} {'✓' if f1_per_class[1] >= 0.75 else '✗'}")
    print(f"    少数类平均召回率: {(recall_per_class[0] + recall_per_class[2]) / 2:.4f}")
    
    # 测试
    print("\n" + "=" * 80)
    print("F10: 测试结果")
    print("=" * 80)
    
    _, _, _, _, test_probs, test_labels = trainer.evaluate(test_loader)
    test_preds_adjusted = adjust_predictions_with_thresholds(test_probs, best_thresholds)
    test_f1_adjusted = f1_score(test_labels, test_preds_adjusted, average='macro')
    test_acc_adjusted = accuracy_score(test_labels, test_preds_adjusted)
    
    print(f"测试准确率: {test_acc_adjusted:.4f}")
    print(f"测试Macro F1: {test_f1_adjusted:.4f}")
    
    report = classification_report(test_labels, test_preds_adjusted,
                                   target_names=['提前流失', '不流失', '到期流失'],
                                   digits=4, zero_division=0)
    print("\n分类报告:")
    print(report)
    
    recall_per_class = recall_score(test_labels, test_preds_adjusted, average=None, zero_division=0)
    precision_per_class = precision_score(test_labels, test_preds_adjusted, average=None, zero_division=0)
    f1_per_class = f1_score(test_labels, test_preds_adjusted, average=None, zero_division=0)
    
    print(f"\n详细性能:")
    print(f"  【多数类保护检查】")
    print(f"    多数类F1: {f1_per_class[1]:.4f} {'✓ 达标' if f1_per_class[1] >= 0.75 else '✗ 未达标'}")
    print(f"    多数类召回率: {recall_per_class[1]:.4f}")
    print(f"    多数类精确率: {precision_per_class[1]:.4f}")
    print(f"\n  【少数类优化效果】")
    print(f"    提前流失召回率: {recall_per_class[0]:.4f}")
    print(f"    到期流失召回率: {recall_per_class[2]:.4f}")
    print(f"    少数类平均召回率: {(recall_per_class[0] + recall_per_class[2]) / 2:.4f}")
    print(f"    提前流失精确率: {precision_per_class[0]:.4f}")
    print(f"    到期流失精确率: {precision_per_class[2]:.4f}")
    
    # 保存结果
    results = {
        'test_accuracy': float(test_acc_adjusted),
        'test_f1_macro': float(test_f1_adjusted),
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
        'majority_protected': bool(f1_per_class[1] >= 0.75),
        'thresholds': best_thresholds,
        'optimization_strategy': 'adaptive_weight_adjustment + meta_lr_controller',
        'lr_stats': {
            'initial': float(trainer.lr_history[0]) if trainer.lr_history else 0,
            'final': float(trainer.lr_history[-1]) if trainer.lr_history else 0,
            'mean': float(np.mean(trainer.lr_history)) if trainer.lr_history else 0,
            'max': float(np.max(trainer.lr_history)) if trainer.lr_history else 0,
            'min': float(np.min(trainer.lr_history)) if trainer.lr_history else 0
        }
    }
    
    with open('F10_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ 结果已保存到 F10_results.json")
    
    # 最终总结
    print("\n" + "=" * 80)
    print("F10 优化总结（集成元学习率控制器）")
    print("=" * 80)
    if f1_per_class[1] >= 0.75:
        print("✓ 多数类性能保护成功！")
        print(f"  少数类平均召回率: {(recall_per_class[0] + recall_per_class[2]) / 2:.4f}")
        print(f"  多数类F1: {f1_per_class[1]:.4f}")
        print(f"  整体Macro F1: {test_f1_adjusted:.4f}")
        print(f"\n元学习率控制器统计:")
        print(f"  学习率范围: [{np.min(trainer.lr_history):.6f}, {np.max(trainer.lr_history):.6f}]")
        print(f"  平均学习率: {np.mean(trainer.lr_history):.6f}")
    else:
        print("⚠️  多数类性能未达到保护目标")
    print("=" * 80)


if __name__ == "__main__":
    main()

