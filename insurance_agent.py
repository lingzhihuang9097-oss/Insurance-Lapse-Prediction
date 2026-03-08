"""
模块五：AI-Agent智能决策系统
基于H6模型 + Kimi AI大语言模型

功能：
1. 加载H6训练好的模型
2. 对客户进行风险预测
3. 提取注意力权重（关键特征）
4. 调用Kimi AI生成个性化决策建议
5. 人机协同机制（低置信度案例标记）
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Optional
import requests
from datetime import datetime

# 导入H6模型组件
from enhanced_transformer_model import EnhancedTransformerModel


# ============================================================================
# Kimi AI 接口封装
# ============================================================================

class KimiAIClient:
    """Kimi AI API客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.moonshot.cn/v1/chat/completions"
        self.model = "moonshot-v1-8k"  # 使用8k上下文模型
    
    def chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """
        调用Kimi AI进行对话
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度（0-1）
        
        Returns:
            AI生成的回复文本
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Kimi AI调用失败: {e}")
            return None


# ============================================================================
# 特征名称映射（用于可解释性）
# 基于 data_preprocessing.py 的实际特征
# ============================================================================

# 实际特征名称（根据数据预处理生成的特征）
# 数值特征 (6个)
NUMERICAL_FEATURES = [
    'age',                    # 年龄
    'seniority_insured',      # 被保险人资历（年限）
    'seniority_policy',       # 保单资历（年限）
    'premium',                # 保费
    'cost_claims_year',       # 年度理赔成本
    'n_medical_services'      # 医疗服务次数
]

# 类别特征 (15个)
CATEGORICAL_FEATURES = [
    'type_policy',            # 保单类型
    'type_product',           # 产品类型
    'distribution_channel',   # 分销渠道
    'gender',                 # 性别
    'reimbursement',          # 报销方式
    'new_business',           # 是否新业务
    'C_H',                    # 人口统计-家庭
    'C_C',                    # 人口统计-儿童
    'C_GI',                   # 人口统计-收入
    'C_GE_P',                 # 人口教育-小学
    'C_GE_S',                 # 人口教育-中学
    'C_GE_T'                  # 人口教育-高等
]

# 特征中文名称映射
FEATURE_NAMES_CN = {
    'age': '年龄',
    'seniority_insured': '被保险人资历（年）',
    'seniority_policy': '保单资历（年）',
    'premium': '保费金额',
    'cost_claims_year': '年度理赔成本',
    'n_medical_services': '医疗服务次数',
    'type_policy': '保单类型',
    'type_product': '产品类型',
    'distribution_channel': '分销渠道',
    'gender': '性别',
    'reimbursement': '报销方式',
    'new_business': '是否新业务',
    'C_H': '家庭人口比例',
    'C_C': '儿童人口比例',
    'C_GI': '收入水平指数',
    'C_GE_P': '小学教育比例',
    'C_GE_S': '中学教育比例',
    'C_GE_T': '高等教育比例'
}

# 特征详细描述
FEATURE_DESCRIPTIONS = {
    'age': '被保险人年龄',
    'seniority_insured': '被保险人在公司的年限',
    'seniority_policy': '保单持续年限',
    'premium': '年度保费金额',
    'cost_claims_year': '该年度的理赔成本',
    'n_medical_services': '使用医疗服务的次数',
    'type_policy': '保单类型分类',
    'type_product': '保险产品类型',
    'distribution_channel': '销售渠道',
    'gender': '被保险人性别',
    'reimbursement': '理赔报销方式',
    'new_business': '是否为新签约业务',
    'C_H': '所在地区家庭人口占比',
    'C_C': '所在地区儿童人口占比',
    'C_GI': '所在地区收入水平指数',
    'C_GE_P': '所在地区小学教育水平占比',
    'C_GE_S': '所在地区中学教育水平占比',
    'C_GE_T': '所在地区高等教育水平占比'
}

def get_feature_name(index: int, feature_names_list: list = None) -> str:
    """
    获取特征名称
    
    Args:
        index: 特征索引
        feature_names_list: 特征名称列表（从预处理数据中获取）
    
    Returns:
        特征的中文名称
    """
    if feature_names_list and index < len(feature_names_list):
        feature_key = feature_names_list[index]
        return FEATURE_NAMES_CN.get(feature_key, feature_key)
    return f'特征{index}'


FEATURE_NAMES = {
    # 基础信息
    'age': '年龄',
    'gender': '性别',
    'region': '地区',
    'education': '教育程度',
    'income': '收入水平',
    
    # 保单信息
    'policy_duration': '保单持续时间',
    'premium_amount': '保费金额',
    'coverage_amount': '保额',
    'policy_type': '保单类型',
    'payment_frequency': '缴费频率',
    
    # 行为特征
    'claim_count': '理赔次数',
    'claim_amount': '理赔金额',
    'service_calls': '客服咨询次数',
    'online_activity': '线上活跃度',
    'payment_delay': '缴费延迟次数',
    
    # 风险指标
    'risk_score': '风险评分',
    'credit_score': '信用评分',
    'health_score': '健康评分',
}


# ============================================================================
# 保险领域知识库
# ============================================================================

INSURANCE_KNOWLEDGE = """
# 保险客户流失管理专业知识

## 流失类型定义
1. **提前流失（Class 0）**：客户在保单到期前主动退保
   - 特征：通常发生在保单前1-2年
   - 原因：产品不匹配、价格敏感、服务不满意
   - 挽留难度：高

2. **不流失（Class 1）**：客户持续续保
   - 特征：稳定缴费、较少理赔、高满意度
   - 维护策略：定期关怀、增值服务

3. **到期流失（Class 2）**：客户在保单到期时不续保
   - 特征：保单即将到期、缴费意愿下降
   - 原因：产品过时、竞品吸引、需求变化
   - 挽留难度：中等

## 关键风险因素
- **高风险信号**：缴费延迟、客服投诉增加、线上活跃度下降
- **价格敏感性**：保费占收入比过高（>10%）
- **服务质量**：理赔处理时长、客服响应速度
- **产品匹配度**：保障范围与客户需求的契合度

## 干预策略库
1. **价格优惠**：保费折扣、分期付款、积分抵扣
2. **服务升级**：专属客服、快速理赔通道、健康管理服务
3. **产品调整**：保障升级、附加险推荐、保单组合优化
4. **情感关怀**：生日祝福、节日问候、定期回访

## 最佳实践
- 提前3个月启动到期客户挽留
- 对高价值客户（保费>5万）提供VIP服务
- 建立客户分层管理体系
- 定期进行客户满意度调查
"""


# ============================================================================
# AI-Agent决策系统
# ============================================================================

class InsuranceDecisionAgent:
    """保险智能决策Agent"""
    
    def __init__(self, model_path: str, api_key: str, device: str = 'cuda'):
        """
        初始化Agent
        
        Args:
            model_path: H6模型路径
            api_key: Kimi AI的API密钥
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"初始化AI-Agent决策系统...")
        print(f"   设备: {self.device}")
        
        # 加载数据（获取特征维度）
        self.load_data()
        
        # 加载H6模型
        self.model = self.load_model(model_path)
        
        # 初始化Kimi AI客户端
        self.kimi_client = KimiAIClient(api_key)
        
        # 置信度阈值（低于此值需要人工复核）
        self.confidence_threshold = 0.5
        
        print("AI-Agent初始化完成")
    
    def load_data(self):
        """加载数据以获取特征信息"""
        try:
            with open('../preprocessed_data.pkl', 'rb') as f:
                data = pickle.load(f)
        except:
            with open('/root/autodl-tmp/preprocessed_data.pkl', 'rb') as f:
                data = pickle.load(f)
        
        self.X_train = data['X_train']
        self.feature_dim = self.X_train.shape[1]
        
        # 从数据中获取实际的特征名称
        if 'feature_names' in data:
            self.feature_names_raw = data['feature_names']
            # 转换为中文名称
            self.feature_names = [
                FEATURE_NAMES_CN.get(name, name) 
                for name in self.feature_names_raw
            ]
        else:
            self.feature_names_raw = [f'feature_{i}' for i in range(self.feature_dim)]
            self.feature_names = [f'特征{i}' for i in range(self.feature_dim)]
        
        print(f"   特征维度: {self.feature_dim}")
        print(f"   特征示例: {', '.join(self.feature_names[:5])}")
    
    def load_model(self, model_path: str) -> nn.Module:
        """加载训练好的H6模型"""
        print(f"加载模型: {model_path}")
        
        model = EnhancedTransformerModel(
            input_dim=self.feature_dim,
            d_model=128,
            n_heads=8,
            n_layers=3,
            n_classes=3,
            dropout=0.1
        ).to(self.device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        return model
    
    def predict_with_attention(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测并提取注意力权重
        
        Args:
            features: 客户特征 (n_samples, n_features)
        
        Returns:
            predictions: 预测类别
            probabilities: 预测概率
            attention_weights: 注意力权重
        """
        # 先进行正常预测（不需要梯度）
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        # 单独计算特征重要性（需要梯度）
        attention_weights = self._compute_feature_importance(features)
        
        return predictions, probabilities, attention_weights
    
    def _compute_feature_importance(self, features: np.ndarray) -> np.ndarray:
        """
        计算特征重要性（简化版注意力权重）
        使用梯度方法
        
        Args:
            features: 客户特征 (n_samples, n_features)
        
        Returns:
            importance: 特征重要性 (n_samples, n_features)
        """
        self.model.train()  # 切换到训练模式以启用梯度
        
        X = torch.FloatTensor(features).to(self.device)
        X.requires_grad = True
        
        # 前向传播
        outputs = self.model(X)
        
        # 对每个样本计算梯度
        importance_list = []
        for i in range(len(X)):
            # 清零梯度
            if X.grad is not None:
                X.grad.zero_()
            
            # 对当前样本的预测求和并反向传播
            outputs[i].sum().backward(retain_graph=True)
            
            # 特征重要性 = |梯度| * |特征值|
            sample_importance = (X.grad[i].abs() * X[i].abs()).detach().cpu().numpy()
            importance_list.append(sample_importance)
        
        importance = np.array(importance_list)
        
        # 归一化
        importance = importance / (importance.sum(axis=1, keepdims=True) + 1e-8)
        
        self.model.eval()  # 切换回评估模式
        
        return importance
    
    def analyze_customer(self, customer_features: np.ndarray, customer_id: str = None) -> Dict:
        """
        分析单个客户
        
        Args:
            customer_features: 客户特征向量
            customer_id: 客户ID
        
        Returns:
            分析结果字典
        """
        # 确保是2D数组
        if customer_features.ndim == 1:
            customer_features = customer_features.reshape(1, -1)
        
        # 预测
        predictions, probabilities, attention_weights = self.predict_with_attention(customer_features)
        
        pred_class = predictions[0]
        pred_probs = probabilities[0]
        confidence = pred_probs.max()
        attention = attention_weights[0]
        
        # 提取关键特征（Top 5）
        top_indices = attention.argsort()[-5:][::-1]
        key_features = [
            {
                'name': self.feature_names[idx] if idx < len(self.feature_names) else f'特征{idx}',
                'index': int(idx),
                'value': float(customer_features[0, idx]),
                'importance': float(attention[idx]),
                'description': self._get_feature_description(idx, customer_features[0, idx])
            }
            for idx in top_indices
        ]
        
        # 风险等级
        risk_level = self._determine_risk_level(pred_class, confidence)
        
        # 流失类型
        churn_type = ['提前流失', '不流失', '到期流失'][pred_class]
        
        # 是否需要人工复核
        needs_review = confidence < self.confidence_threshold
        
        result = {
            'customer_id': customer_id or 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'class': int(pred_class),
                'type': churn_type,
                'confidence': float(confidence),
                'probabilities': {
                    '提前流失': float(pred_probs[0]),
                    '不流失': float(pred_probs[1]),
                    '到期流失': float(pred_probs[2])
                }
            },
            'risk_level': risk_level,
            'key_features': key_features,
            'needs_human_review': needs_review
        }
        
        return result
    
    def _determine_risk_level(self, pred_class: int, confidence: float) -> str:
        """确定风险等级"""
        if pred_class == 1:  # 不流失
            return '低风险'
        elif pred_class == 2 and confidence > 0.7:  # 到期流失，高置信度
            return '中风险'
        elif pred_class == 0 or (pred_class == 2 and confidence <= 0.7):
            return '高风险'
        else:
            return '中风险'
    
    def _get_feature_description(self, feature_idx: int, feature_value: float) -> str:
        """获取特征的详细描述"""
        # 获取原始特征名
        if hasattr(self, 'feature_names_raw') and feature_idx < len(self.feature_names_raw):
            feature_key = self.feature_names_raw[feature_idx]
        else:
            return f'{feature_value:.2f}'
        
        # 根据特征类型提供描述
        if feature_key == 'age':
            return f'{feature_value:.0f}岁'
        elif feature_key == 'seniority_insured':
            return f'{feature_value:.1f}年'
        elif feature_key == 'seniority_policy':
            return f'{feature_value:.1f}年'
        elif feature_key == 'premium':
            return f'{feature_value:.2f}元'
        elif feature_key == 'cost_claims_year':
            return f'{feature_value:.2f}元'
        elif feature_key == 'n_medical_services':
            return f'{feature_value:.0f}次'
        elif feature_key == 'gender':
            return '男' if feature_value > 0.5 else '女'
        elif feature_key == 'new_business':
            return '是' if feature_value > 0.5 else '否'
        elif feature_key in ['C_H', 'C_C', 'C_GI', 'C_GE_P', 'C_GE_S', 'C_GE_T']:
            return f'{feature_value:.2%}'
        elif feature_key in CATEGORICAL_FEATURES:
            return f'类别{int(feature_value)}'
        else:
            return f'{feature_value:.2f}'
    
    def generate_decision(self, analysis_result: Dict) -> Dict:
        """
        生成AI决策建议
        
        Args:
            analysis_result: 客户分析结果
        
        Returns:
            决策建议字典
        """
        print(f"\n为客户 {analysis_result['customer_id']} 生成决策建议...")
        
        # 构建提示词
        prompt = self._build_prompt(analysis_result)
        
        # 调用Kimi AI
        messages = [
            {
                "role": "system",
                "content": f"你是一位资深的保险客户关系管理专家，精通客户流失预防和挽留策略。\n\n{INSURANCE_KNOWLEDGE}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        ai_response = self.kimi_client.chat(messages, temperature=0.7)
        
        if ai_response is None:
            # 如果AI调用失败，使用规则生成基础建议
            ai_response = self._generate_fallback_decision(analysis_result)
        
        # 解析AI响应
        decision = {
            'customer_id': analysis_result['customer_id'],
            'timestamp': datetime.now().isoformat(),
            'ai_analysis': ai_response,
            'needs_human_review': analysis_result['needs_human_review'],
            'confidence': analysis_result['prediction']['confidence']
        }
        
        return decision
    
    def _build_prompt(self, analysis_result: Dict) -> str:
        """构建结构化提示词"""
        pred = analysis_result['prediction']
        
        # 关键特征描述
        key_features_desc = "\n".join([
            f"  - {f['name']} (特征{f['index']}): {f['description']} (重要性: {f['importance']:.2%})"
            for f in analysis_result['key_features']
        ])
        
        prompt = f"""
请基于以下客户信息，提供专业的流失风险分析和挽留建议：

## 客户基本信息
- 客户ID: {analysis_result['customer_id']}
- 分析时间: {analysis_result['timestamp']}

## 模型预测结果
- 预测类型: {pred['type']}
- 风险等级: {analysis_result['risk_level']}
- 预测置信度: {pred['confidence']:.2%}
- 各类别概率:
  * 提前流失: {pred['probabilities']['提前流失']:.2%}
  * 不流失: {pred['probabilities']['不流失']:.2%}
  * 到期流失: {pred['probabilities']['到期流失']:.2%}

## 关键影响因素（按重要性排序）
{key_features_desc}

## 请提供以下分析：

### 1. 流失原因诊断
请根据预测结果和关键特征，分析客户可能流失的主要原因（2-3点）。

### 2. 干预策略推荐
请提供3个具体的挽留策略，包括：
- 策略名称
- 具体措施
- 预期效果
- 实施优先级（高/中/低）

### 3. 预期效果评估
评估实施这些策略后，客户留存概率的提升幅度。

### 4. 备选方案
如果主要策略效果不佳，提供1-2个备选方案。

### 5. 注意事项
提醒业务人员在执行过程中需要注意的关键点。

请以专业、简洁的方式回答，每个部分用明确的标题分隔。
"""
        return prompt
    
    def _generate_fallback_decision(self, analysis_result: Dict) -> str:
        """生成备用决策（当AI调用失败时）"""
        pred_class = analysis_result['prediction']['class']
        risk_level = analysis_result['risk_level']
        
        if pred_class == 0:  # 提前流失
            return """
### 1. 流失原因诊断
- 客户对当前产品或服务不满意
- 可能存在价格敏感性问题
- 竞品吸引或需求变化

### 2. 干预策略推荐
**策略1：紧急挽留计划**
- 措施：48小时内安排客户经理一对一沟通
- 预期效果：了解真实流失原因
- 优先级：高

**策略2：个性化优惠方案**
- 措施：提供保费折扣或增值服务
- 预期效果：降低价格敏感性
- 优先级：高

**策略3：产品重新匹配**
- 措施：根据客户需求调整保障方案
- 预期效果：提升产品满意度
- 优先级：中

### 3. 预期效果评估
实施上述策略后，预计客户留存概率可提升30-50%。

### 4. 备选方案
如主要策略无效，考虑：
- 转介绍奖励：鼓励客户推荐新客户
- 暂停保单：提供保单暂停选项而非退保

### 5. 注意事项
- 快速响应，避免客户情绪恶化
- 真诚沟通，了解真实需求
- 记录反馈，优化后续服务
"""
        elif pred_class == 2:  # 到期流失
            return """
### 1. 流失原因诊断
- 保单即将到期，续保意愿不强
- 可能对产品价值认知不足
- 竞品可能提供更优方案

### 2. 干预策略推荐
**策略1：提前续保激励**
- 措施：提前3个月启动续保沟通，提供早鸟优惠
- 预期效果：提升续保率
- 优先级：高

**策略2：产品升级推荐**
- 措施：推荐更适合的新产品或附加险
- 预期效果：增加客户价值感知
- 优先级：中

**策略3：服务体验提升**
- 措施：提供VIP服务、健康管理等增值服务
- 预期效果：增强客户粘性
- 优先级：中

### 3. 预期效果评估
实施上述策略后，预计续保率可提升20-40%。

### 4. 备选方案
- 分期付款：降低一次性缴费压力
- 保障组合：提供家庭保障套餐

### 5. 注意事项
- 提前启动，避免临期被动
- 强调产品价值和保障意义
- 关注客户生命周期变化
"""
        else:  # 不流失
            return """
### 1. 客户状态分析
- 客户当前状态良好，流失风险低
- 建议继续维护良好关系

### 2. 维护策略推荐
**策略1：定期关怀**
- 措施：节日问候、生日祝福、定期回访
- 预期效果：保持客户满意度
- 优先级：中

**策略2：增值服务**
- 措施：提供健康咨询、理财规划等服务
- 预期效果：提升客户价值
- 优先级：中

**策略3：交叉销售**
- 措施：适时推荐其他保险产品
- 预期效果：增加客户终身价值
- 优先级：低

### 3. 预期效果评估
持续维护可保持95%以上的留存率。

### 4. 备选方案
- 客户转介绍计划
- 忠诚客户奖励计划

### 5. 注意事项
- 避免过度营销
- 关注客户需求变化
- 保持服务质量
"""
    
    def batch_analyze(self, features: np.ndarray, customer_ids: List[str] = None) -> List[Dict]:
        """
        批量分析客户
        
        Args:
            features: 客户特征矩阵 (n_samples, n_features)
            customer_ids: 客户ID列表
        
        Returns:
            分析结果列表
        """
        if customer_ids is None:
            customer_ids = [f"Customer_{i}" for i in range(len(features))]
        
        results = []
        for i, (feat, cid) in enumerate(zip(features, customer_ids)):
            print(f"\n分析客户 {i+1}/{len(features)}: {cid}")
            analysis = self.analyze_customer(feat, cid)
            decision = self.generate_decision(analysis)
            
            results.append({
                'analysis': analysis,
                'decision': decision
            })
        
        return results
    
    def export_report(self, results: List[Dict], output_file: str = 'decision_report.json'):
        """导出决策报告"""
        # 转换所有numpy/torch类型为Python原生类型
        def convert_to_serializable(obj):
            """递归转换对象为JSON可序列化格式"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 转换结果
        serializable_results = convert_to_serializable(results)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_customers': len(results),
            'high_risk_count': sum(1 for r in results if r['analysis']['risk_level'] == '高风险'),
            'needs_review_count': sum(1 for r in results if r['analysis']['needs_human_review']),
            'results': serializable_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 决策报告已导出: {output_file}")
        print(f"   总客户数: {report['total_customers']}")
        print(f"   高风险客户: {report['high_risk_count']}")
        print(f"   需人工复核: {report['needs_review_count']}")


# ============================================================================
# 主函数 - 演示
# ============================================================================

def main():
    print("=" * 80)
    print("AI-Agent智能决策系统 - 演示")
    print("=" * 80)
    
    # 初始化Agent
    agent = InsuranceDecisionAgent(
        model_path='H6_transformer.pth',
        api_key='sk-mPZ6yp8J52QeOJqqIjjeV9dauH4GOB1B3XYoo82OtiZGPRIx',
        device='cuda'
    )
    
    # 加载测试数据
    try:
        with open('../preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except:
        with open('/root/autodl-tmp/preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 选择几个代表性样本进行演示
    print("\n" + "=" * 80)
    print("演示：分析代表性客户")
    print("=" * 80)
    
    # 选择不同类别的样本
    sample_indices = []
    for class_id in [0, 1, 2]:
        class_indices = np.where(y_test == class_id)[0]
        if len(class_indices) > 0:
            sample_indices.append(class_indices[0])
    
    sample_features = X_test[sample_indices]
    sample_ids = [f"DEMO_Customer_{i+1}" for i in range(len(sample_indices))]
    
    # 批量分析
    results = agent.batch_analyze(sample_features, sample_ids)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("决策结果汇总")
    print("=" * 80)
    
    for i, result in enumerate(results):
        analysis = result['analysis']
        decision = result['decision']
        
        print(f"\n{'='*80}")
        print(f"客户 {i+1}: {analysis['customer_id']}")
        print(f"{'='*80}")
        print(f"预测类型: {analysis['prediction']['type']}")
        print(f"风险等级: {analysis['risk_level']}")
        print(f"置信度: {analysis['prediction']['confidence']:.2%}")
        print(f"需要人工复核: {'是' if analysis['needs_human_review'] else '否'}")
        
        print(f"\n关键特征:")
        for feat in analysis['key_features'][:3]:
            print(f"  - {feat['name']}: {feat['value']:.2f} (重要性: {feat['importance']:.2%})")
        
        print(f"\nAI决策建议:")
        print(decision['ai_analysis'][:500] + "..." if len(decision['ai_analysis']) > 500 else decision['ai_analysis'])
    
    # 导出完整报告
    agent.export_report(results, 'demo_decision_report.json')
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

