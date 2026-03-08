"""
E8>E7>E1
实验E8: 修复E7的4个关键错误
修复内容：
  错误1: 三分割数据（train/val/threshold_search/test）避免阈值过拟合
  错误2: 统一训练和阈值搜索的目标函数
  错误3: 修正泛化差距计算（使用相同阈值）
  错误4: 连续搜索最优策略
Beta = 0.9999 
root@autodl-container-e49a4eabc6-e4de2506:~/autodl-tmp/enhanced# python E8_fixed_issues.py

"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from scipy.optimize import minimize
import sys
sys.path.append('..')
from enhanced_transformer_model import EnhancedTransformerModel


def compute_class_balanced_weights(samples_per_class, beta=0.9999):
    """计算Class-Balanced权重"""
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)
    return weights


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss"""
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        weights = compute_class_balanced_weights(samples_per_class, beta)
        self.weights = torch.FloatTensor(weights)
        
    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)
        ce_loss = nn.CrossEntropyLoss(weight=weights)(logits, targets)
        return ce_loss


def adjust_predictions_with_thresholds(probs, thresholds):
    """使用不同阈值调整预测"""
    batch_size = probs.shape[0]
    predictions = np.zeros(batch_size, dtype=np.int64)
    thresholds_array = np.array(thresholds)
    
    for i in range(batch_size):
        adjusted_scores = probs[i] / thresholds_array
        predictions[i] = np.argmax(adjusted_scores)
    
    return predictions


def compute_unified_score(f1, minority_recall, minority_precision, acc):
    """
    统一评分函数（修复错误2 + 方案2优化）
    训练和阈值搜索使用相同的目标函数
    
    方案2: 平衡Precision和Recall
    - F1已经平衡了Precision和Recall，给予更高权重
    - Precision和Recall权重相等
    - 增加准确率权重，避免过度牺牲整体性能
    """
    return 0.4 * f1 + 0.2 * minority_recall + 0.2 * minority_precision + 0.2 * acc
# balance version

class BaseTrainer:
    """基础训练器 - E8的核心训练逻辑"""
    def __init__(self, model, train_loader, val_loader, test_loader, device, samples_per_class, beta=0.9999):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.samples_per_class = samples_per_class
        
        # Class-Balanced Loss (E1的配置)
        self.criterion = self._create_criterion(samples_per_class, beta)
        
        # 优化器和调度器（子类可以重写）
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
    
    def _create_criterion(self, samples_per_class, beta):
        """创建损失函数（子类可以重写）"""
        return ClassBalancedLoss(samples_per_class, beta=beta)
    
    def _create_optimizer(self):
        """创建优化器（子类可以重写）"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-4,
            weight_decay=0.01
        )
    
    def _create_scheduler(self):
        """创建学习率调度器（子类可以重写）"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
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
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        if return_detailed:
            recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
            precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
            minority_recall = (recall_per_class[0] + recall_per_class[2]) / 2
            minority_precision = (precision_per_class[0] + precision_per_class[2]) / 2
            unified_score = compute_unified_score(f1, minority_recall, minority_precision, acc)
            
            if return_probs:
                return total_loss / len(data_loader), acc, f1, minority_recall, minority_precision, unified_score, all_probs, all_labels
            return total_loss / len(data_loader), acc, f1, minority_recall, minority_precision, unified_score
        
        if return_probs:
            return total_loss / len(data_loader), acc, f1, all_probs, all_labels
        return total_loss / len(data_loader), acc, f1
    
    def train(self, epochs=100, patience=20, model_name='E8'):
        """训练主循环（子类可以重写）"""
        print("\n开始训练...")
        print(f"{'Epoch':<8s} {'Train Loss':<13s} {'Train Acc':<13s} {'Val Loss':<13s} "
              f"{'Val Acc':<13s} {'Val F1':<10s} {'Min Recall':<12s} {'Min Prec':<12s} {'Score':<10s}")
        print("-" * 120)
        
        best_score = 0
        best_val_f1 = 0
        best_minority_recall = 0
        best_minority_precision = 0
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1, minority_recall, minority_precision, unified_score = self.evaluate(
                self.val_loader, return_detailed=True
            )
            
            if epoch == 1 or epoch % 5 == 0:
                print(f"{epoch:<8d} {train_loss:<13.4f} {train_acc:<13.4f} {val_loss:<13.4f} "
                      f"{val_acc:<13.4f} {val_f1:<10.4f} {minority_recall:<12.4f} {minority_precision:<12.4f} {unified_score:<10.4f}")
            
            # 保存最佳模型（基于统一评分）
            if unified_score > best_score:
                best_score = unified_score
                best_val_f1 = val_f1
                best_minority_recall = minority_recall
                best_minority_precision = minority_precision
                best_epoch = epoch
                torch.save(self.model.state_dict(), f'{model_name}_best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            self.scheduler.step()
        
        print(f"\n最佳统一评分: {best_score:.4f} (Epoch {best_epoch})")
        print(f"  对应验证F1: {best_val_f1:.4f}")
        print(f"  对应少数类召回率: {best_minority_recall:.4f}")
        print(f"  对应少数类精确度: {best_minority_precision:.4f}")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'{model_name}_best_model.pth'))
        
        return best_val_f1, best_minority_recall, best_minority_precision


# 保持向后兼容
class Trainer(BaseTrainer):
    """E8的Trainer类（保持向后兼容）"""
    pass


def search_unified_thresholds(model, threshold_loader, device, method='continuous', 
                             min_precision=0.30, target_recall=0.50):
    """
    搜索统一阈值（修复错误1和错误2）+ 动态阈值 + Precision约束
    
    method: 'continuous' (连续优化) 或 'grid' (网格搜索) 或 'dynamic' (动态阈值)
    min_precision: 少数类最低精确度约束（默认30%）
    target_recall: 目标召回率（默认50%）
    """
    print(f"\n搜索统一阈值（使用独立threshold_search数据集，方法={method}）...")
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in threshold_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    if method == 'dynamic':
        # 动态阈值方法：为每个类别找到满足precision约束的最优recall
        print(f"  使用动态阈值（最低精确度={min_precision:.0%}，目标召回率={target_recall:.0%}）...")
        
        best_thresholds = []
        
        for class_idx in range(3):
            binary_labels = (all_labels == class_idx).astype(int)
            class_probs = all_probs[:, class_idx]
            
            if class_idx == 1:  # 多数类（不流失）
                # 使用较高阈值，保持高精确度
                best_threshold = 0.6
            else:  # 少数类
                # 搜索满足precision约束的最高recall
                best_threshold = 0.5
                best_recall = 0
                
                # 从高到低尝试阈值
                for threshold in np.arange(0.1, 0.7, 0.01):
                    preds = (class_probs >= threshold).astype(int)
                    
                    if preds.sum() == 0:
                        continue
                    
                    recall = recall_score(binary_labels, preds, zero_division=0)
                    precision = precision_score(binary_labels, preds, zero_division=0)
                    
                    # 如果precision满足约束，选择recall最高的
                    if precision >= min_precision and recall > best_recall:
                        best_recall = recall
                        best_threshold = threshold
                
                print(f"    类别{class_idx}: 阈值={best_threshold:.4f}, Recall={best_recall:.4f}")
            
            best_thresholds.append(best_threshold)
    
    elif method == 'continuous':
        # 连续优化方法 + Precision约束
        from scipy.optimize import minimize
        
        print(f"  使用连续优化算法（带Precision约束≥{min_precision:.0%}）...")
        
        # 定义目标函数（负的统一评分，因为minimize是最小化）
        def objective(thresholds):
            preds = adjust_predictions_with_thresholds(all_probs, thresholds)
            
            recall_per_class = recall_score(all_labels, preds, average=None, zero_division=0)
            precision_per_class = precision_score(all_labels, preds, average=None, zero_division=0)
            minority_recall = (recall_per_class[0] + recall_per_class[2]) / 2
            minority_precision = (precision_per_class[0] + precision_per_class[2]) / 2
            f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
            acc = accuracy_score(all_labels, preds)
            
            # 如果precision低于约束，施加惩罚
            precision_penalty = 0
            if minority_precision < min_precision:
                precision_penalty = 10 * (min_precision - minority_precision)
            
            # 方案2: 修改评分函数，更平衡地考虑Precision和Recall
            score = 0.4*f1 + 0.2*minority_recall + 0.2*minority_precision + 0.2*acc - precision_penalty
            return -score  # 负号因为要最小化
        
        # 初始阈值（基于经验）
        initial_thresholds = [0.35, 0.6, 0.35]
        
        # 约束：阈值在合理范围内
        bounds = [(0.2, 0.6), (0.5, 0.8), (0.2, 0.6)]
        
        # 使用优化算法
        result = minimize(
            objective,
            initial_thresholds,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        best_thresholds = result.x.tolist()
        best_thresholds = [round(t, 4) for t in best_thresholds]  # 保留4位小数
        
    else:
        # 网格搜索方法（原方法）
        print("  搜索范围:")
        print("    提前流失: [0.25, 0.30, 0.35]")
        print("    不流失:   [0.55, 0.60, 0.65]")
        print("    到期流失: [0.25, 0.30, 0.35]")
        
        best_score = 0
        best_thresholds = [0.5, 0.5, 0.5]
        
        for t0 in [0.25, 0.30, 0.35]:  # 提前流失
            for t1 in [0.55, 0.60, 0.65]:  # 不流失
                for t2 in [0.25, 0.30, 0.35]:  # 到期流失
                    thresholds = [t0, t1, t2]
                    preds = adjust_predictions_with_thresholds(all_probs, thresholds)
                    
                    recall_per_class = recall_score(all_labels, preds, average=None, zero_division=0)
                    precision_per_class = precision_score(all_labels, preds, average=None, zero_division=0)
                    minority_recall = (recall_per_class[0] + recall_per_class[2]) / 2
                    minority_precision = (precision_per_class[0] + precision_per_class[2]) / 2
                    
                    # 跳过不满足precision约束的组合
                    if minority_precision < min_precision:
                        continue
                    
                    f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
                    acc = accuracy_score(all_labels, preds)
                    
                    score = compute_unified_score(f1, minority_recall, minority_precision, acc)
                    
                    if score > best_score:
                        best_score = score
                        best_thresholds = thresholds
    
    # 评估最佳阈值
    preds = adjust_predictions_with_thresholds(all_probs, best_thresholds)
    recall_per_class = recall_score(all_labels, preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, preds, average=None, zero_division=0)
    minority_recall = (recall_per_class[0] + recall_per_class[2]) / 2
    minority_precision = (precision_per_class[0] + precision_per_class[2]) / 2
    f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, preds)
    
    best_metrics = {
        'f1': f1,
        'acc': acc,
        'minority_recall': minority_recall,
        'minority_precision': minority_precision,
        'recall_0': recall_per_class[0],
        'recall_2': recall_per_class[2],
        'precision_0': precision_per_class[0],
        'precision_2': precision_per_class[2]
    }
    
    print(f"\n  最佳阈值: {best_thresholds}")
    print(f"  Threshold_search集指标:")
    print(f"    Macro F1: {best_metrics['f1']:.4f}")
    print(f"    准确率: {best_metrics['acc']:.4f}")
    print(f"    少数类平均召回率: {best_metrics['minority_recall']:.4f}")
    print(f"    少数类平均精确度: {best_metrics['minority_precision']:.4f}")
    print(f"    提前流失: Recall={best_metrics['recall_0']:.4f}, Precision={best_metrics['precision_0']:.4f}")
    print(f"    到期流失: Recall={best_metrics['recall_2']:.4f}, Precision={best_metrics['precision_2']:.4f}")
    
    return best_thresholds, best_metrics


def main():
    print("=" * 80)
    print("实验E8: 修复E7的4个关键错误")
    print("=" * 80)
    print("修复内容:")
    print("  错误1: 三分割数据（train/val/threshold_search/test）避免阈值过拟合")
    print("  错误2: 统一训练和阈值搜索的目标函数")
    print("  错误3: 修正泛化差距计算（使用相同阈值）")
    print("  错误4: 使用固定Beta=0.9999（基于实验验证的最佳值）")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载数据
    print("加载数据: /root/autodl-tmp/preprocessed_data.pkl")
    with open('/root/autodl-tmp/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.LongTensor(data['y_test'])
    
    print(f"特征数: {X_train.shape[1]}")
    
    # 修复错误1: 三分割数据
    # 将原验证集分为 val (用于early stopping) 和 threshold_search (用于搜索阈值)
    val_size = len(X_val)
    split_idx = int(val_size * 0.5)
    
    X_val_new = X_val[:split_idx]
    y_val_new = y_val[:split_idx]
    X_threshold = X_val[split_idx:]
    y_threshold = y_val[split_idx:]
    
    print(f"\n数据分割（修复错误1）:")
    print(f"  训练集: {len(X_train):,}")
    print(f"  验证集（early stopping）: {len(X_val_new):,}")
    print(f"  阈值搜索集: {len(X_threshold):,}")
    print(f"  测试集: {len(X_test):,}")
    
    # 计算类别分布
    unique, counts = np.unique(y_train.numpy(), return_counts=True)
    samples_per_class = counts
    print(f"\n训练集类别分布: {samples_per_class}")
    for i, count in enumerate(samples_per_class):
        pct = count / samples_per_class.sum() * 100
        class_names = ['提前流失', '不流失', '到期流失']
        print(f"  类别{i} ({class_names[i]}): {count:,} ({pct:.2f}%)")
    
    # 固定Beta参数（基于实验结果）
    print("\n" + "=" * 80)
    print("使用固定Beta参数（基于实验验证的最佳值）")
    print("=" * 80)
    
    best_beta = 0.9999
    print(f"Beta参数: {best_beta}")
    print(f"选择理由:")
    print(f"  - 准确率高 (~82%)")
    print(f"  - 精确度高 (~53%)")
    print(f"  - 召回率适中 (~16%)")
    print(f"  - 误杀率低 (~10-12%)")
    print(f"  - 综合性能最均衡")
    print("=" * 80)
    
    # 使用固定Beta进行训练
    print("\n" + "=" * 80)
    print(f"使用Beta={best_beta}进行训练")
    print("=" * 80)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val_new, y_val_new)
    threshold_dataset = torch.utils.data.TensorDataset(X_threshold, y_threshold)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    threshold_loader = torch.utils.data.DataLoader(threshold_dataset, batch_size=256, shuffle=False)
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
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {num_params:,}")
    
    # 训练（使用最佳beta）
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, samples_per_class)
    trainer.criterion = ClassBalancedLoss(samples_per_class, beta=best_beta)
    best_val_f1, best_minority_recall, best_minority_precision = trainer.train(epochs=100, patience=20, model_name='E8_final')
    
    # 修复错误1: 在独立的threshold_search集上搜索阈值
    # 使用动态阈值 + Precision约束（最低30%精确度）
    print("\n" + "=" * 80)
    print("阈值搜索策略对比")
    print("=" * 80)
    
    # 方法1: 动态阈值（推荐）- 为每个类别独立优化
    print("\n【方法1: 动态阈值】")
    thresholds_dynamic, metrics_dynamic = search_unified_thresholds(
        model, threshold_loader, device, 
        method='dynamic',
        min_precision=0.30,
        target_recall=0.50
    )
    
    # 方法2: 连续优化 + Precision约束
    print("\n【方法2: 连续优化 + Precision约束】")
    thresholds_continuous, metrics_continuous = search_unified_thresholds(
        model, threshold_loader, device, 
        method='continuous',
        min_precision=0.30,
        target_recall=0.50
    )
    
    # 选择最佳方法（基于统一评分）
    score_dynamic = compute_unified_score(
        metrics_dynamic['f1'], 
        metrics_dynamic['minority_recall'],
        metrics_dynamic['minority_precision'],
        metrics_dynamic['acc']
    )
    score_continuous = compute_unified_score(
        metrics_continuous['f1'],
        metrics_continuous['minority_recall'],
        metrics_continuous['minority_precision'],
        metrics_continuous['acc']
    )
    
    print("\n" + "=" * 80)
    print("方法对比:")
    print(f"  动态阈值:   统一评分={score_dynamic:.4f}, 阈值={thresholds_dynamic}")
    print(f"  连续优化:   统一评分={score_continuous:.4f}, 阈值={thresholds_continuous}")
    
    if score_dynamic > score_continuous:
        print("\n✅ 选择: 动态阈值（表现更好）")
        best_thresholds = thresholds_dynamic
        threshold_metrics = metrics_dynamic
    else:
        print("\n✅ 选择: 连续优化（表现更好）")
        best_thresholds = thresholds_continuous
        threshold_metrics = metrics_continuous
    print("=" * 80)
    
    # 在验证集上评估（使用找到的阈值）- 用于计算泛化差距
    print("\n验证集（使用找到的阈值）:")
    val_loss, val_acc, val_f1, val_probs, val_labels = trainer.evaluate(val_loader, return_probs=True)
    val_preds_adjusted = adjust_predictions_with_thresholds(val_probs, best_thresholds)
    val_f1_adjusted = f1_score(val_labels, val_preds_adjusted, average='macro')
    val_acc_adjusted = accuracy_score(val_labels, val_preds_adjusted)
    print(f"  调整阈值后F1: {val_f1_adjusted:.4f}")
    print(f"  调整阈值后准确率: {val_acc_adjusted:.4f}")
    
    # 测试（标准阈值）
    print("\n" + "=" * 80)
    print("E8: 测试结果（标准阈值 [0.5, 0.5, 0.5]）")
    print("=" * 80)
    
    test_loss, test_acc, test_f1, test_probs, test_labels = trainer.evaluate(
        test_loader, return_probs=True
    )
    test_preds = np.argmax(test_probs, axis=1)
    
    print(f"测试Loss:        {test_loss:.4f}")
    print(f"测试准确率:      {test_acc:.4f}")
    print(f"测试Macro F1:    {test_f1:.4f}")
    
    # 测试（调整阈值）
    print("\n" + "=" * 80)
    print(f"E8: 测试结果（调整阈值 {best_thresholds}）")
    print("=" * 80)
    
    test_preds_adjusted = adjust_predictions_with_thresholds(test_probs, best_thresholds)
    test_f1_adjusted = f1_score(test_labels, test_preds_adjusted, average='macro')
    test_acc_adjusted = accuracy_score(test_labels, test_preds_adjusted)
    
    print(f"测试准确率:      {test_acc_adjusted:.4f}")
    print(f"测试Macro F1:    {test_f1_adjusted:.4f}")
    
    report_adjusted = classification_report(test_labels, test_preds_adjusted,
                                           target_names=['提前流失', '不流失', '到期流失'],
                                           digits=4, zero_division=0)
    print("\n分类报告:")
    print(report_adjusted)
    
    # 详细分析
    cm_adjusted = confusion_matrix(test_labels, test_preds_adjusted)
    print("\n混淆矩阵:")
    print(cm_adjusted)
    
    recall_per_class = recall_score(test_labels, test_preds_adjusted, average=None, zero_division=0)
    precision_per_class = precision_score(test_labels, test_preds_adjusted, average=None, zero_division=0)
    
    print(f"\n详细性能分析:")
    print(f"  提前流失: Recall={recall_per_class[0]:.4f}, Precision={precision_per_class[0]:.4f}")
    print(f"  不流失:   Recall={recall_per_class[1]:.4f}, Precision={precision_per_class[1]:.4f}")
    print(f"  到期流失: Recall={recall_per_class[2]:.4f}, Precision={precision_per_class[2]:.4f}")
    print(f"  少数类平均召回率: {(recall_per_class[0] + recall_per_class[2]) / 2:.4f}")
    print(f"  少数类平均精确度: {(precision_per_class[0] + precision_per_class[2]) / 2:.4f}")
    
    # 误杀与漏检分析
    false_positives_0 = cm_adjusted[:, 0].sum() - cm_adjusted[0, 0]
    false_positives_2 = cm_adjusted[:, 2].sum() - cm_adjusted[2, 2]
    false_negatives_0 = cm_adjusted[0].sum() - cm_adjusted[0, 0]
    false_negatives_2 = cm_adjusted[2].sum() - cm_adjusted[2, 2]
    total_minority = cm_adjusted[0].sum() + cm_adjusted[2].sum()
    
    print(f"\n误杀分析:")
    print(f"  预测为流失但实际不是: {false_positives_0 + false_positives_2:,}")
    print(f"  误杀率: {(false_positives_0 + false_positives_2) / len(test_labels) * 100:.2f}%")
    
    print(f"\n漏检分析:")
    print(f"  实际流失但未检出: {false_negatives_0 + false_negatives_2:,} / {total_minority:,}")
    print(f"  漏检率: {(false_negatives_0 + false_negatives_2) / total_minority * 100:.2f}%")
    
    # 修复错误3: 正确计算泛化差距（使用相同阈值）
    print("\n" + "=" * 80)
    print("修复错误3: 正确的泛化差距计算")
    print("=" * 80)
    print(f"验证集F1（调整阈值）: {val_f1_adjusted:.4f}")
    print(f"测试集F1（调整阈值）: {test_f1_adjusted:.4f}")
    generalization_gap_correct = abs(val_f1_adjusted - test_f1_adjusted)
    print(f"泛化差距（正确）: {generalization_gap_correct:.4f}")
    
    # 与E7对比
    print(f"\n与E7对比:")
    print(f"  E7: 提前流失召回=53.56%, 到期流失召回=22.33%, 准确率=78.23%, 泛化差距=0.0410(错误计算)")
    print(f"  E8: 提前流失召回={recall_per_class[0]*100:.2f}%, 到期流失召回={recall_per_class[2]*100:.2f}%, 准确率={test_acc_adjusted*100:.2f}%, 泛化差距={generalization_gap_correct:.4f}(正确)")
    
    print("=" * 80)
    
    # 保存结果
    results = {
        'best_beta': float(best_beta),
        'thresholds': best_thresholds,
        'test_accuracy': float(test_acc_adjusted),
        'test_f1_macro': float(test_f1_adjusted),
        'val_f1_adjusted': float(val_f1_adjusted),
        'generalization_gap_correct': float(generalization_gap_correct),
        'recall_class_0': float(recall_per_class[0]),
        'recall_class_1': float(recall_per_class[1]),
        'recall_class_2': float(recall_per_class[2]),
        'precision_class_0': float(precision_per_class[0]),
        'precision_class_1': float(precision_per_class[1]),
        'precision_class_2': float(precision_per_class[2]),
        'minority_avg_recall': float((recall_per_class[0] + recall_per_class[2]) / 2),
        'minority_avg_precision': float((precision_per_class[0] + precision_per_class[2]) / 2),
        'false_positives': int(false_positives_0 + false_positives_2),
        'false_negatives': int(false_negatives_0 + false_negatives_2),
        'miss_rate': float((false_negatives_0 + false_negatives_2) / total_minority)
    }
    
    with open('E8_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ 结果已保存到 E8_results.json")


if __name__ == "__main__":
    main()

