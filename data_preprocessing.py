"""
数据预处理和特征工程模块
用于加载、清洗和预处理保险客户数据
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置 OpenMP 线程数（修复 libgomp 警告）
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'


class InsuranceDataPreprocessor:
    """保险数据预处理器"""
    
    def __init__(self, data_path):
        """
        初始化预处理器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # 定义特征类型
        # 移除会导致数据泄露的特征
        self.numerical_features = [
            'age', 'seniority_insured', 'seniority_policy',
            'premium', 'cost_claims_year', 'n_medical_services'
            # 移除: exposure_time (目标泄露), n_insured_* (聚合泄露), IICI* (聚合泄露)
        ]
        
        self.categorical_features = [
            'type_policy', 'type_product',  # 移除 type_policy_dg (冗余)
            'distribution_channel', 'gender', 'reimbursement', 'new_business', 
            'C_H', 'C_C', 'C_GI',  # 只保留通用人口统计，移除保险人口统计
            'C_GE_P', 'C_GE_S', 'C_GE_T'  # 只保留通用人口教育水平
            # 移除: C_II, C_IE_* (基于保险人口的聚合统计)
        ]
        
        # ⚠️ 地理-经济特征仅用于 GAT 模型，不在基础预处理中使用
        # 传统 ML 和基础 Transformer 不需要地理单元
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.df = pd.read_csv(self.data_path)
        
        # 删除无用列
        cols_to_drop = ['Unnamed: 42', '338']
        self.df = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns], errors='ignore')
        
        # ⚠️ 删除会导致目标泄露的列（这些是未来信息）
        leakage_cols = [
            'date_lapse_insured',      # 流失日期 - 目标泄露
            'date_lapse_policy',       # 保单流失日期 - 目标泄露
            'year_lapse_insured',      # 流失年份 - 目标泄露
            'year_lapse_policy',       # 保单流失年份 - 目标泄露
            'exposure_time',           # 风险暴露时间 - 包含流失信息
            'date_effect_insured',     # 生效日期 - 可能泄露
            'date_effect_policy',      # 保单生效日期 - 可能泄露
            'year_effect_insured',     # 生效年份 - 可能泄露
            'year_effect_policy',      # 保单生效年份 - 可能泄露
            'period',                  # 年份 - 时序泄露
            'ID_policy',               # ID列
            'ID_insured',              # ID列
            'type_policy_dg',          # 冗余特征（type_policy的细分）
            'n_insured_pc',            # 聚合统计 - 包含测试集信息
            'n_insured_mun',           # 聚合统计 - 包含测试集信息
            'n_insured_prov',          # 聚合统计 - 包含测试集信息
            'IICIMUN',                 # 聚合统计 - 包含测试集信息
            'IICIPROV',                # 聚合统计 - 包含测试集信息
            'C_II',                    # 基于保险人口的聚合统计
            'C_IE_P',                  # 基于保险人口的聚合统计
            'C_IE_S',                  # 基于保险人口的聚合统计
            'C_IE_T'                   # 基于保险人口的聚合统计
        ]
        
        print(f"\n  ⚠️  移除 {len([c for c in leakage_cols if c in self.df.columns])} 个会导致数据泄露的特征:")
        for col in leakage_cols:
            if col in self.df.columns:
                print(f"    - {col}")
        
        self.df = self.df.drop(columns=[col for col in leakage_cols if col in self.df.columns], errors='ignore')
        
        print(f"\n  总样本: {len(self.df):,}")
        print(f"  特征数: {self.df.shape[1]}")
        
        # 检查目标变量分布
        if 'lapse' in self.df.columns:
            lapse_dist = self.df['lapse'].value_counts().sort_index().to_dict()
            print(f"  Lapse分布: {lapse_dist}")
            print(f"    1=提前流失, 2=活跃客户, 3=到期流失")
        
        # 显示每列的缺失值统计
        missing_summary = self.df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
        if len(missing_summary) > 0:
            print(f"\n  缺失值统计（前10列）:")
            for col, count in missing_summary.head(10).items():
                pct = count / len(self.df) * 100
                print(f"    {col}: {count:,} ({pct:.2f}%)")
        
        return self.df
    
    def handle_missing_values(self):
        """
        处理缺失值：
        1. 如果某列全是空值，删除该列
        2. 删除含有缺失值的行
        """
        print("\n处理缺失值...")
        
        initial_rows = len(self.df)
        initial_cols = len(self.df.columns)
        
        # 步骤1：删除全是空值的列
        all_null_cols = self.df.columns[self.df.isnull().all()].tolist()
        if all_null_cols:
            print(f"\n  步骤1：删除 {len(all_null_cols)} 个全为空值的列:")
            for col in all_null_cols:
                print(f"    - {col}")
            self.df = self.df.drop(columns=all_null_cols)
        else:
            print(f"\n  步骤1：无全为空值的列")
        
        # 步骤2：统计每列的缺失值
        missing_counts = self.df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            print(f"\n  步骤2：发现 {len(missing_cols)} 列存在缺失值:")
            for col, count in missing_cols.items():
                print(f"    {col}: {count} 个缺失值 ({count/initial_rows*100:.2f}%)")
            
            # 删除任何含有缺失值的行
            self.df = self.df.dropna()
            
            removed_rows = initial_rows - len(self.df)
            print(f"\n  删除 {removed_rows:,} 行含缺失值的数据 ({removed_rows/initial_rows*100:.2f}%)")
            print(f"  剩余样本: {len(self.df):,}")
            print(f"  剩余列数: {len(self.df.columns)}")
        else:
            print(f"\n  步骤2：无缺失值")
            print(f"  剩余样本: {len(self.df):,}")
            print(f"  剩余列数: {len(self.df.columns)}")
    
    def encode_categorical_features(self, fit=True):
        """编码类别特征"""
        print("\n编码类别特征...")
        
        for col in self.categorical_features:
            if col in self.df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    self.df[col] = self.label_encoders[col].fit_transform(self.df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        self.df[col] = self.label_encoders[col].transform(self.df[col].astype(str))
                
                print(f"  {col}: {len(self.label_encoders[col].classes_)} 个类别")
    
    def normalize_numerical_features(self, fit=True):
        """标准化数值特征"""
        print("\n标准化数值特征...")
        
        available_features = [f for f in self.numerical_features if f in self.df.columns]
        
        if fit:
            self.df[available_features] = self.scaler.fit_transform(self.df[available_features])
        else:
            self.df[available_features] = self.scaler.transform(self.df[available_features])
        
        print(f"  标准化 {len(available_features)} 个数值特征")
    
    def stratified_random_split(self, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """
        分层随机划分（保持类别分布一致）
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_state: 随机种子
        """
        print("\n分层随机划分数据集...")
        
        # 先划分训练集和临时集（验证+测试）
        train_df, temp_df = train_test_split(
            self.df,
            test_size=(1 - train_ratio),
            stratify=self.df['lapse'],
            random_state=random_state
        )
        
        # 再从临时集中划分验证集和测试集
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df['lapse'],
            random_state=random_state
        )
        
        print(f"  训练集: {len(train_df):,} 样本")
        print(f"  验证集: {len(val_df):,} 样本")
        print(f"  测试集: {len(test_df):,} 样本")
        
        # 显示各集合的类别分布
        print("\n  类别分布检查:")
        for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
            dist = df['lapse'].value_counts(normalize=True).sort_index()
            print(f"    {name}: ", end='')
            for cls, pct in dist.items():
                print(f"类别{cls}={pct:.2%}  ", end='')
            print()
        
        return train_df, val_df, test_df
    
    def prepare_features_and_labels(self, df):
        """准备特征和标签"""
        # 排除目标变量和ID列以及日期列
        exclude_cols = [
            'lapse', 'ID', 'id', 'customer_id', 'ID_policy', 'ID_insured',
            'date_effect_insured', 'date_lapse_insured', 
            'date_effect_policy', 'date_lapse_policy',
            'year_effect_insured', 'year_lapse_insured',
            'year_effect_policy', 'year_lapse_policy',
            'period'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        if 'lapse' in df.columns:
            # 将lapse转换为0-based索引 (1,2,3 -> 0,1,2)
            y = df['lapse'].values - 1
        else:
            y = None
        
        return X, y, feature_cols
    
    def preprocess(self):
        """完整的预处理流程（修复数据泄露问题）"""
        print("="*80)
        print("开始数据预处理（防止数据泄露版本）")
        print("="*80)
        
        # 1. 加载数据（已移除泄露特征）
        self.load_data()
        
        # 2. 处理缺失值
        self.handle_missing_values()
        
        # 3. ⚠️ 先划分数据集（在任何编码/标准化之前）
        print("\n⚠️  关键步骤：先划分数据集，防止信息泄露")
        train_df, val_df, test_df = self.stratified_random_split()
        
        # 4. 在训练集上fit编码器和标准化器
        print("\n处理训练集...")
        self.df = train_df.copy()
        self.encode_categorical_features(fit=True)
        self.normalize_numerical_features(fit=True)
        
        # 5. 准备训练集特征
        X_train, y_train, feature_names = self.prepare_features_and_labels(self.df)
        train_df_processed = self.df.copy()
        
        # 6. 处理验证集（使用训练集的编码器）
        print("\n处理验证集...")
        self.df = val_df.copy()
        self.encode_categorical_features(fit=False)
        self.normalize_numerical_features(fit=False)
        X_val, y_val, _ = self.prepare_features_and_labels(self.df)
        val_df_processed = self.df.copy()
        
        # 7. 处理测试集（使用训练集的编码器）
        print("\n处理测试集...")
        self.df = test_df.copy()
        self.encode_categorical_features(fit=False)
        self.normalize_numerical_features(fit=False)
        X_test, y_test, _ = self.prepare_features_and_labels(self.df)
        test_df_processed = self.df.copy()
        
        print("\n" + "="*80)
        print("预处理完成（已防止数据泄露）")
        print("="*80)
        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test.shape}")
        print(f"\n✅ 数据泄露检查:")
        print(f"  - 已移除目标泄露特征（流失日期、暴露时间等）")
        print(f"  - 已移除聚合统计特征（包含测试集信息）")
        print(f"  - 编码器和标准化器仅在训练集上fit")
        
        # 保存预处理后的数据
        data_dict = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_names,
            'train_df': train_df_processed,
            'val_df': val_df_processed,
            'test_df': test_df_processed,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        # 保存到文件
        print("\n保存预处理数据...")
        with open('../preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        print("  ✓ 已保存到: preprocessed_data.pkl")
        
        return data_dict


def main():
    """测试预处理流程"""
    # 示例用法
    data_path = "insurance_data.csv"  
    
    preprocessor = InsuranceDataPreprocessor(data_path)
    data_dict = preprocessor.preprocess()
    
    print("\n特征名称:")
    for i, name in enumerate(data_dict['feature_names'][:10]):
        print(f"  {i}: {name}")
    print(f"  ... (共 {len(data_dict['feature_names'])} 个特征)")


if __name__ == "__main__":
    main()

