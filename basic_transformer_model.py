"""
基础深度学习模型：标准Transformer
用于客户流失预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# import time
import time

class InsuranceDataset(Dataset):
    """保险数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class BasicTransformerModel(nn.Module):
    """基础Transformer模型"""
    
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, 
                 n_classes=3, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Transformer编码器
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        
        # 投影到d_model维度
        x = self.input_projection(x)  # (batch_size, d_model)
        
        # 添加序列维度
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Transformer编码
        for block in self.transformer_blocks:
            x = block(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(x)  # (batch_size, n_classes)
        
        return logits


class TransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, model, device='cuda', lr=2e-4, weight_decay=0.01, class_weights=None):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 使用类别权重处理不平衡
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 不使用学习率调度器，保持固定学习率
        self.scheduler = None
        
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.patience = 15
        self.no_improve = 0
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 前向传播
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, acc
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, acc, f1
    
    def train(self, train_loader, val_loader, n_epochs=100):
        """训练模型"""
        print("\n开始训练...")
        print(f"{'Epoch':<6s} {'Train Loss':>12s} {'Train Acc':>12s} "
              f"{'Val Loss':>12s} {'Val Acc':>12s} {'Val F1':>12s}")
        print("-" * 75)
        
        for epoch in range(1, n_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)
            
            # 打印进度
            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:<6d} {train_loss:>12.4f} {train_acc:>12.4f} "
                      f"{val_loss:>12.4f} {val_acc:>12.4f} {val_f1:>12.4f}")
            
            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.no_improve = 0
                torch.save(self.model.state_dict(), 'best_model_basic_transformer.pth')
            else:
                self.no_improve += 1
            
            # Early stopping
            if self.no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\n最佳验证F1: {self.best_val_f1:.4f} (Epoch {self.best_epoch})")
    
    def test(self, test_loader, class_names=None):
        """测试模型"""
        if class_names is None:
            class_names = ['提前流失', '活跃客户', '到期流失']
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model_basic_transformer.pth'))
        
        print("\n" + "="*80)
        print("测试集评估")
        print("="*80)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
        
        # 计算指标
        test_loss = total_loss / len(test_loader)
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"\n测试结果:")
        print(f"  Loss:     {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Macro F1: {test_f1:.4f}")
        
        # 分类报告
        print(f"\n分类报告:")
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n混淆矩阵:")
        print(f"{'':20s}", end='')
        for name in class_names:
            print(f"预测:{name:10s}", end='')
        print()
        
        for i, name in enumerate(class_names):
            print(f"实际:{name:15s}", end='')
            for j in range(len(class_names)):
                print(f"{cm[i,j]:15d}", end='')
            print()
        
        return test_loss, test_acc, test_f1


def main():
    """主函数"""
    from data_preprocessing import InsuranceDataPreprocessor
    import pickle
    import os
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预处理数据
    if os.path.exists('../preprocessed_data.pkl'):
        print("加载预处理数据...")
        with open('../preprocessed_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    
    # 计算类别权重（处理不平衡）- 使用更温和的权重
    from collections import Counter
    class_counts = Counter(data_dict['y_train'])
    total = len(data_dict['y_train'])
    # 使用平方根缩放，避免权重过大
    class_weights = [np.sqrt(total / (len(class_counts) * class_counts[i])) for i in range(3)]
    print(f"\n类别权重: {[f'{w:.2f}' for w in class_weights]}")
    
    # 创建数据集
    train_dataset = InsuranceDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = InsuranceDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = InsuranceDataset(data_dict['X_test'], data_dict['y_test'])
    
    # 创建数据加载器 - 恢复原始batch size
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 创建模型 - 恢复到原始有效配置
    input_dim = data_dict['X_train'].shape[1]
    model = BasicTransformerModel(
        input_dim=input_dim,
        d_model=128,      # 恢复到128
        n_heads=4,        # 恢复到4
        n_layers=3,       # 恢复到3
        n_classes=3,
        dropout=0.1       # 恢复到0.1
    )
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型 - 使用原始配置
    trainer = TransformerTrainer(
        model, 
        device=device, 
        lr=2e-4,          # 恢复原始学习率
        weight_decay=0.01,
        class_weights=None  # 不使用类别权重
    )
    trainer.patience = 15  # 恢复原始patience
    trainer.train(train_loader, val_loader, n_epochs=100)
    
    # 测试模型
    class_names = ['提前流失', '活跃客户', '到期流失']
    trainer.test(test_loader, class_names)
    
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)


if __name__ == "__main__":
    main()

