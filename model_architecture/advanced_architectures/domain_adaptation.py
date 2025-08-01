#!/usr/bin/env python3
"""
域适应实现

为PINNs提供域适应功能，包括：
- 域对抗训练
- 多域学习
- 迁移学习
- 域泛化
- 自适应特征提取

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

class GradientReversalLayer(nn.Module):
    """
    梯度反转层
    
    用于域对抗训练的梯度反转
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        初始化梯度反转层
        
        Args:
            alpha: 梯度反转强度
        """
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量（前向不变，反向梯度反转）
        """
        return GradientReversalFunction.apply(x, self.alpha)

class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转函数
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None

class DomainClassifier(nn.Module):
    """
    域分类器
    
    用于区分不同域的特征
    """
    
    def __init__(self, feature_dim: int, num_domains: int, 
                 hidden_dims: List[int] = [256, 128]):
        """
        初始化域分类器
        
        Args:
            feature_dim: 特征维度
            num_domains: 域数量
            hidden_dims: 隐藏层维度列表
        """
        super(DomainClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        
        # 构建分类器网络
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_domains))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征张量 [batch_size, feature_dim]
            
        Returns:
            torch.Tensor: 域分类logits [batch_size, num_domains]
        """
        return self.classifier(features)

class DomainAdversarialNetwork(nn.Module):
    """
    域对抗网络
    
    实现域对抗训练的完整网络
    """
    
    def __init__(self, feature_extractor: nn.Module, task_predictor: nn.Module,
                 domain_classifier: nn.Module, lambda_domain: float = 1.0):
        """
        初始化域对抗网络
        
        Args:
            feature_extractor: 特征提取器
            task_predictor: 任务预测器
            domain_classifier: 域分类器
            lambda_domain: 域损失权重
        """
        super(DomainAdversarialNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.task_predictor = task_predictor
        self.domain_classifier = domain_classifier
        self.lambda_domain = lambda_domain
        
        self.gradient_reversal = GradientReversalLayer(alpha=lambda_domain)
    
    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            alpha: 梯度反转强度（可选）
            
        Returns:
            Dict: 包含任务预测和域预测的字典
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 任务预测
        task_output = self.task_predictor(features)
        
        # 域预测（带梯度反转）
        if alpha is not None:
            self.gradient_reversal.alpha = alpha
        
        reversed_features = self.gradient_reversal(features)
        domain_output = self.domain_classifier(reversed_features)
        
        return {
            'task_output': task_output,
            'domain_output': domain_output,
            'features': features
        }

class MultiDomainPINN(nn.Module):
    """
    多域PINN
    
    处理多个域的PINN模型
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_domains: int,
                 hidden_dims: List[int] = [256, 256, 256],
                 domain_specific_layers: int = 2):
        """
        初始化多域PINN
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_domains: 域数量
            hidden_dims: 隐藏层维度
            domain_specific_layers: 域特定层数量
        """
        super(MultiDomainPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_domains = num_domains
        
        # 共享特征提取器
        shared_layers = []
        input_size = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-domain_specific_layers]):
            shared_layers.extend([
                nn.Linear(input_size, hidden_dim),
                nn.Tanh()
            ])
            input_size = hidden_dim
        
        self.shared_encoder = nn.Sequential(*shared_layers)
        
        # 域特定网络
        self.domain_specific_networks = nn.ModuleList()
        for domain_id in range(num_domains):
            domain_layers = []
            current_dim = input_size
            
            for hidden_dim in hidden_dims[-domain_specific_layers:]:
                domain_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.Tanh()
                ])
                current_dim = hidden_dim
            
            domain_layers.append(nn.Linear(current_dim, output_dim))
            self.domain_specific_networks.append(nn.Sequential(*domain_layers))
        
        # 域分类器
        self.domain_classifier = DomainClassifier(
            feature_dim=input_size,
            num_domains=num_domains
        )
        
        # 梯度反转层
        self.gradient_reversal = GradientReversalLayer()
    
    def forward(self, x: torch.Tensor, domain_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            domain_id: 域ID（可选）
            
        Returns:
            Dict: 包含预测结果和域分类的字典
        """
        # 共享特征提取
        shared_features = self.shared_encoder(x)
        
        # 域分类
        reversed_features = self.gradient_reversal(shared_features)
        domain_logits = self.domain_classifier(reversed_features)
        
        if domain_id is not None:
            # 使用指定域
            task_output = self.domain_specific_networks[domain_id](shared_features)
            return {
                'task_output': task_output,
                'domain_logits': domain_logits,
                'shared_features': shared_features
            }
        else:
            # 所有域的输出
            domain_outputs = []
            for network in self.domain_specific_networks:
                domain_outputs.append(network(shared_features))
            
            return {
                'domain_outputs': torch.stack(domain_outputs, dim=1),  # [batch_size, num_domains, output_dim]
                'domain_logits': domain_logits,
                'shared_features': shared_features
            }

class TransferLearningPINN(nn.Module):
    """
    迁移学习PINN
    
    从源域迁移到目标域的PINN
    """
    
    def __init__(self, source_model: nn.Module, target_layers: List[int],
                 freeze_source: bool = True):
        """
        初始化迁移学习PINN
        
        Args:
            source_model: 源域模型
            target_layers: 目标域特定层索引
            freeze_source: 是否冻结源域参数
        """
        super(TransferLearningPINN, self).__init__()
        
        self.source_model = source_model
        self.target_layers = target_layers
        
        # 冻结源域参数
        if freeze_source:
            for param in self.source_model.parameters():
                param.requires_grad = False
        
        # 获取源模型结构信息
        self._extract_model_info()
        
        # 创建目标域特定层
        self._create_target_layers()
    
    def _extract_model_info(self):
        """
        提取源模型信息
        """
        # TODO: 实现模型信息提取
        # 分析源模型的层结构和维度
        pass
    
    def _create_target_layers(self):
        """
        创建目标域特定层
        """
        # TODO: 实现目标层创建
        # 根据target_layers创建新的层
        self.target_specific_layers = nn.ModuleList()
        pass
    
    def forward(self, x: torch.Tensor, use_target: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            use_target: 是否使用目标域层
            
        Returns:
            torch.Tensor: 输出张量
        """
        if use_target:
            # TODO: 实现目标域前向传播
            # 使用源模型的部分层 + 目标域特定层
            pass
        else:
            # 仅使用源模型
            return self.source_model(x)

class DomainGeneralizationPINN(nn.Module):
    """
    域泛化PINN
    
    学习域不变特征的PINN
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_domains: int,
                 hidden_dims: List[int] = [256, 256, 256],
                 invariant_penalty: float = 1.0):
        """
        初始化域泛化PINN
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_domains: 域数量
            hidden_dims: 隐藏层维度
            invariant_penalty: 不变性惩罚权重
        """
        super(DomainGeneralizationPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_domains = num_domains
        self.invariant_penalty = invariant_penalty
        
        # 特征提取器
        feature_layers = []
        input_size = input_dim
        
        for hidden_dim in hidden_dims:
            feature_layers.extend([
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_size = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(input_size, output_dim)
        )
        
        # 域判别器（用于计算域不变性损失）
        self.domain_discriminator = DomainClassifier(
            feature_dim=input_size,
            num_domains=num_domains
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Dict: 包含预测和特征的字典
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 任务预测
        predictions = self.predictor(features)
        
        # 域判别（用于计算不变性损失）
        domain_logits = self.domain_discriminator(features.detach())
        
        return {
            'predictions': predictions,
            'features': features,
            'domain_logits': domain_logits
        }
    
    def compute_invariance_loss(self, features: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """
        计算域不变性损失
        
        Args:
            features: 特征张量
            domain_labels: 域标签
            
        Returns:
            torch.Tensor: 不变性损失
        """
        # 计算域间特征分布差异
        unique_domains = torch.unique(domain_labels)
        
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算域间均值差异
        domain_means = []
        for domain in unique_domains:
            domain_mask = (domain_labels == domain)
            if domain_mask.sum() > 0:
                domain_mean = features[domain_mask].mean(dim=0)
                domain_means.append(domain_mean)
        
        if len(domain_means) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算均值间的距离
        invariance_loss = 0.0
        for i in range(len(domain_means)):
            for j in range(i + 1, len(domain_means)):
                invariance_loss += F.mse_loss(domain_means[i], domain_means[j])
        
        return invariance_loss * self.invariant_penalty

class AdaptiveFeatureExtractor(nn.Module):
    """
    自适应特征提取器
    
    根据输入自适应调整特征提取策略
    """
    
    def __init__(self, input_dim: int, feature_dim: int, num_experts: int = 4,
                 hidden_dim: int = 256):
        """
        初始化自适应特征提取器
        
        Args:
            input_dim: 输入维度
            feature_dim: 特征维度
            num_experts: 专家网络数量
            hidden_dim: 隐藏层维度
        """
        super(AdaptiveFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        
        # 门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 专家网络
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
            self.experts.append(expert)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            Tuple: (特征张量, 门控权重)
        """
        # 计算门控权重
        gating_weights = self.gating_network(x)  # [batch_size, num_experts]
        
        # 计算专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch_size, feature_dim]
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, feature_dim, num_experts]
        
        # 加权组合
        gating_weights = gating_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
        features = torch.bmm(expert_outputs, gating_weights.transpose(1, 2)).squeeze(-1)
        
        return features, gating_weights.squeeze(1)

class GlacierDomainAdapter(nn.Module):
    """
    冰川域适应器
    
    专门用于冰川数据的域适应
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 source_domains: List[str], target_domain: str,
                 adaptation_method: str = 'adversarial'):
        """
        初始化冰川域适应器
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            source_domains: 源域列表
            target_domain: 目标域
            adaptation_method: 适应方法
        """
        super(GlacierDomainAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.source_domains = source_domains
        self.target_domain = target_domain
        self.adaptation_method = adaptation_method
        
        num_domains = len(source_domains) + 1  # +1 for target domain
        
        if adaptation_method == 'adversarial':
            # 域对抗适应
            self.adapter = MultiDomainPINN(
                input_dim=input_dim,
                output_dim=output_dim,
                num_domains=num_domains
            )
        elif adaptation_method == 'generalization':
            # 域泛化适应
            self.adapter = DomainGeneralizationPINN(
                input_dim=input_dim,
                output_dim=output_dim,
                num_domains=num_domains
            )
        else:
            raise ValueError(f"未支持的适应方法: {adaptation_method}")
        
        # 冰川物理约束
        self.physics_constraints = self._create_physics_constraints()
    
    def _create_physics_constraints(self) -> nn.Module:
        """
        创建冰川物理约束
        
        Returns:
            nn.Module: 物理约束模块
        """
        # TODO: 实现冰川物理约束
        # 包括质量守恒、动量守恒等
        return nn.Identity()
    
    def forward(self, x: torch.Tensor, domain_info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            domain_info: 域信息字典
            
        Returns:
            Dict: 适应结果字典
        """
        # 基础适应
        adapter_output = self.adapter(x)
        
        # 应用物理约束
        if 'physics_quantities' in domain_info:
            physics_output = self.physics_constraints(adapter_output['task_output'])
            adapter_output['physics_constrained'] = physics_output
        
        # 添加域信息
        adapter_output['domain_info'] = domain_info
        
        return adapter_output
    
    def compute_adaptation_loss(self, outputs: Dict[str, torch.Tensor], 
                              targets: torch.Tensor, domain_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算适应损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
            domain_labels: 域标签
            
        Returns:
            Dict: 损失字典
        """
        losses = {}
        
        # 任务损失
        if 'task_output' in outputs:
            losses['task_loss'] = F.mse_loss(outputs['task_output'], targets)
        
        # 域对抗损失
        if 'domain_logits' in outputs:
            domain_loss = F.cross_entropy(outputs['domain_logits'], domain_labels)
            losses['domain_loss'] = domain_loss
        
        # 物理约束损失
        if 'physics_constrained' in outputs:
            physics_loss = F.mse_loss(outputs['physics_constrained'], targets)
            losses['physics_loss'] = physics_loss
        
        # 不变性损失（如果使用域泛化）
        if hasattr(self.adapter, 'compute_invariance_loss') and 'features' in outputs:
            invariance_loss = self.adapter.compute_invariance_loss(
                outputs['features'], domain_labels
            )
            losses['invariance_loss'] = invariance_loss
        
        return losses

if __name__ == "__main__":
    # 测试域适应组件
    input_dim, output_dim = 10, 3
    batch_size = 32
    num_domains = 3
    
    # 测试多域PINN
    multi_domain_pinn = MultiDomainPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_domains=num_domains
    )
    
    x = torch.randn(batch_size, input_dim)
    output = multi_domain_pinn(x, domain_id=0)
    print(f"多域PINN输出形状: {output['task_output'].shape}")
    
    # 测试域泛化PINN
    domain_gen_pinn = DomainGeneralizationPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_domains=num_domains
    )
    
    output = domain_gen_pinn(x)
    print(f"域泛化PINN输出形状: {output['predictions'].shape}")
    
    # 测试冰川域适应器
    glacier_adapter = GlacierDomainAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        source_domains=['himalaya', 'karakoram'],
        target_domain='tibetan_plateau'
    )
    
    domain_info = {'region': 'tibetan_plateau'}
    output = glacier_adapter(x, domain_info)
    print(f"冰川域适应器输出键: {list(output.keys())}")