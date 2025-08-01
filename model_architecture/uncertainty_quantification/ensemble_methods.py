#!/usr/bin/env python3
"""
集成方法用于不确定性量化

实现各种集成方法来量化模型不确定性，包括：
- 深度集成
- Bootstrap集成
- 多种子集成
- 集成蒸馏
- 不确定性分解

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
import copy
import random
from collections import defaultdict

class DeepEnsemble:
    """
    深度集成
    
    实现深度集成方法用于不确定性量化
    """
    
    def __init__(self, model_factory: Callable, num_models: int = 5,
                 diversity_regularization: bool = True):
        """
        初始化深度集成
        
        Args:
            model_factory: 模型工厂函数
            num_models: 集成模型数量
            diversity_regularization: 是否使用多样性正则化
        """
        self.model_factory = model_factory
        self.num_models = num_models
        self.diversity_regularization = diversity_regularization
        self.models = []
        self.optimizers = []
        
        # 创建集成模型
        for i in range(num_models):
            model = model_factory()
            self.models.append(model)
    
    def initialize_optimizers(self, optimizer_class, **optimizer_kwargs):
        """
        初始化优化器
        
        Args:
            optimizer_class: 优化器类
            **optimizer_kwargs: 优化器参数
        """
        self.optimizers = []
        for model in self.models:
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            self.optimizers.append(optimizer)
    
    def diversity_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        多样性损失
        
        Args:
            predictions: 预测列表
            
        Returns:
            torch.Tensor: 多样性损失
        """
        if len(predictions) < 2:
            return torch.tensor(0.0)
        
        # 计算预测之间的相关性
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i = predictions[i].flatten()
                pred_j = predictions[j].flatten()
                
                # 皮尔逊相关系数
                mean_i = torch.mean(pred_i)
                mean_j = torch.mean(pred_j)
                
                numerator = torch.sum((pred_i - mean_i) * (pred_j - mean_j))
                denominator = torch.sqrt(
                    torch.sum((pred_i - mean_i)**2) * torch.sum((pred_j - mean_j)**2)
                )
                
                correlation = numerator / (denominator + 1e-8)
                correlations.append(correlation)
        
        # 多样性损失：鼓励低相关性
        diversity_loss = torch.mean(torch.stack(correlations)**2)
        
        return diversity_loss
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                  loss_fn: Callable, physics_loss_fn: Callable = None,
                  x_physics: torch.Tensor = None,
                  diversity_weight: float = 0.1) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            x: 输入数据
            y: 目标数据
            loss_fn: 损失函数
            physics_loss_fn: 物理损失函数
            x_physics: 物理点
            diversity_weight: 多样性权重
            
        Returns:
            Dict: 损失字典
        """
        total_loss = 0.0
        data_losses = []
        physics_losses = []
        predictions = []
        
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            optimizer.zero_grad()
            
            # 数据损失
            pred = model(x)
            data_loss = loss_fn(pred, y)
            total_model_loss = data_loss
            
            predictions.append(pred.detach())
            data_losses.append(data_loss.item())
            
            # 物理损失
            if physics_loss_fn is not None and x_physics is not None:
                physics_pred = model(x_physics)
                physics_loss = physics_loss_fn(x_physics, physics_pred)
                total_model_loss += physics_loss
                physics_losses.append(physics_loss.item())
            
            # 多样性正则化
            if self.diversity_regularization and len(predictions) > 1:
                diversity_loss = self.diversity_loss(predictions)
                total_model_loss += diversity_weight * diversity_loss
            
            total_model_loss.backward()
            optimizer.step()
            
            total_loss += total_model_loss.item()
        
        return {
            'total_loss': total_loss / self.num_models,
            'data_loss': np.mean(data_losses),
            'physics_loss': np.mean(physics_losses) if physics_losses else 0.0,
            'diversity_loss': self.diversity_loss(predictions).item() if len(predictions) > 1 else 0.0
        }
    
    def predict(self, x: torch.Tensor, return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        集成预测
        
        Args:
            x: 输入数据
            return_individual: 是否返回个体预测
            
        Returns:
            Dict: 预测结果
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # 统计量
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        var = torch.var(predictions, dim=0)
        
        # 分位数
        quantiles = torch.quantile(predictions, torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975]), dim=0)
        
        results = {
            'mean': mean,
            'std': std,
            'var': var,
            'q025': quantiles[0],
            'q25': quantiles[1],
            'median': quantiles[2],
            'q75': quantiles[3],
            'q975': quantiles[4]
        }
        
        if return_individual:
            results['individual_predictions'] = predictions
        
        return results
    
    def save_ensemble(self, filepath: str):
        """
        保存集成模型
        
        Args:
            filepath: 文件路径
        """
        ensemble_state = {
            'num_models': self.num_models,
            'model_states': [model.state_dict() for model in self.models]
        }
        torch.save(ensemble_state, filepath)
    
    def load_ensemble(self, filepath: str):
        """
        加载集成模型
        
        Args:
            filepath: 文件路径
        """
        ensemble_state = torch.load(filepath)
        
        for i, state_dict in enumerate(ensemble_state['model_states']):
            if i < len(self.models):
                self.models[i].load_state_dict(state_dict)

class BootstrapEnsemble:
    """
    Bootstrap集成
    
    使用Bootstrap采样创建集成
    """
    
    def __init__(self, model_factory: Callable, num_models: int = 5,
                 bootstrap_ratio: float = 1.0):
        """
        初始化Bootstrap集成
        
        Args:
            model_factory: 模型工厂函数
            num_models: 集成模型数量
            bootstrap_ratio: Bootstrap采样比例
        """
        self.model_factory = model_factory
        self.num_models = num_models
        self.bootstrap_ratio = bootstrap_ratio
        self.models = []
        self.optimizers = []
        
        # 创建集成模型
        for i in range(num_models):
            model = model_factory()
            self.models.append(model)
    
    def bootstrap_sample(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bootstrap采样
        
        Args:
            x: 输入数据
            y: 目标数据
            
        Returns:
            Tuple: Bootstrap样本
        """
        n_samples = x.shape[0]
        sample_size = int(n_samples * self.bootstrap_ratio)
        
        # 有放回采样
        indices = torch.randint(0, n_samples, (sample_size,))
        
        return x[indices], y[indices]
    
    def train_ensemble(self, x: torch.Tensor, y: torch.Tensor,
                      loss_fn: Callable, optimizer_class,
                      num_epochs: int = 100, **optimizer_kwargs) -> List[Dict[str, float]]:
        """
        训练Bootstrap集成
        
        Args:
            x: 输入数据
            y: 目标数据
            loss_fn: 损失函数
            optimizer_class: 优化器类
            num_epochs: 训练轮数
            **optimizer_kwargs: 优化器参数
            
        Returns:
            List: 训练历史
        """
        training_histories = []
        
        for i, model in enumerate(self.models):
            print(f"训练模型 {i+1}/{self.num_models}")
            
            # Bootstrap采样
            x_boot, y_boot = self.bootstrap_sample(x, y)
            
            # 优化器
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            
            # 训练历史
            history = []
            
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                
                pred = model(x_boot)
                loss = loss_fn(pred, y_boot)
                
                loss.backward()
                optimizer.step()
                
                history.append(loss.item())
            
            training_histories.append(history)
        
        return training_histories
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Bootstrap集成预测
        
        Args:
            x: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'predictions': predictions
        }

class MultiSeedEnsemble:
    """
    多种子集成
    
    使用不同随机种子训练相同架构的模型
    """
    
    def __init__(self, model_factory: Callable, num_models: int = 5,
                 seeds: List[int] = None):
        """
        初始化多种子集成
        
        Args:
            model_factory: 模型工厂函数
            num_models: 集成模型数量
            seeds: 随机种子列表
        """
        self.model_factory = model_factory
        self.num_models = num_models
        
        if seeds is None:
            self.seeds = list(range(num_models))
        else:
            self.seeds = seeds[:num_models]
        
        self.models = []
        self.optimizers = []
    
    def initialize_models(self):
        """
        初始化模型（使用不同种子）
        """
        self.models = []
        
        for seed in self.seeds:
            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # 创建模型
            model = self.model_factory()
            self.models.append(model)
    
    def train_ensemble(self, x: torch.Tensor, y: torch.Tensor,
                      loss_fn: Callable, optimizer_class,
                      num_epochs: int = 100, **optimizer_kwargs) -> List[Dict[str, float]]:
        """
        训练多种子集成
        
        Args:
            x: 输入数据
            y: 目标数据
            loss_fn: 损失函数
            optimizer_class: 优化器类
            num_epochs: 训练轮数
            **optimizer_kwargs: 优化器参数
            
        Returns:
            List: 训练历史
        """
        if not self.models:
            self.initialize_models()
        
        training_histories = []
        
        for i, (model, seed) in enumerate(zip(self.models, self.seeds)):
            print(f"训练模型 {i+1}/{self.num_models} (种子: {seed})")
            
            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # 优化器
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            
            # 训练历史
            history = []
            
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                
                pred = model(x)
                loss = loss_fn(pred, y)
                
                loss.backward()
                optimizer.step()
                
                history.append(loss.item())
            
            training_histories.append(history)
        
        return training_histories
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        多种子集成预测
        
        Args:
            x: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'predictions': predictions
        }

class EnsembleDistillation:
    """
    集成蒸馏
    
    将集成知识蒸馏到单个模型中
    """
    
    def __init__(self, teacher_ensemble: Union[DeepEnsemble, BootstrapEnsemble, MultiSeedEnsemble],
                 student_model: nn.Module, temperature: float = 3.0):
        """
        初始化集成蒸馏
        
        Args:
            teacher_ensemble: 教师集成
            student_model: 学生模型
            temperature: 蒸馏温度
        """
        self.teacher_ensemble = teacher_ensemble
        self.student_model = student_model
        self.temperature = temperature
    
    def distillation_loss(self, student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor,
                         true_labels: torch.Tensor = None,
                         alpha: float = 0.7) -> torch.Tensor:
        """
        蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师集成输出
            true_labels: 真实标签
            alpha: 蒸馏权重
            
        Returns:
            torch.Tensor: 蒸馏损失
        """
        # 软目标损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)
        
        # 硬目标损失
        if true_labels is not None:
            hard_loss = F.mse_loss(student_logits, true_labels)
            total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        else:
            total_loss = soft_loss
        
        return total_loss
    
    def train_student(self, x: torch.Tensor, y: torch.Tensor = None,
                     optimizer_class = torch.optim.Adam,
                     num_epochs: int = 100, alpha: float = 0.7,
                     **optimizer_kwargs) -> List[float]:
        """
        训练学生模型
        
        Args:
            x: 输入数据
            y: 真实标签（可选）
            optimizer_class: 优化器类
            num_epochs: 训练轮数
            alpha: 蒸馏权重
            **optimizer_kwargs: 优化器参数
            
        Returns:
            List: 训练历史
        """
        optimizer = optimizer_class(self.student_model.parameters(), **optimizer_kwargs)
        
        # 获取教师集成预测
        teacher_predictions = self.teacher_ensemble.predict(x)
        teacher_logits = teacher_predictions['mean']
        
        history = []
        
        for epoch in range(num_epochs):
            self.student_model.train()
            optimizer.zero_grad()
            
            student_logits = self.student_model(x)
            loss = self.distillation_loss(student_logits, teacher_logits, y, alpha)
            
            loss.backward()
            optimizer.step()
            
            history.append(loss.item())
        
        return history
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        学生模型预测
        
        Args:
            x: 输入数据
            
        Returns:
            torch.Tensor: 预测结果
        """
        self.student_model.eval()
        with torch.no_grad():
            return self.student_model(x)

class UncertaintyDecomposition:
    """
    不确定性分解
    
    分解总不确定性为认知不确定性和偶然不确定性
    """
    
    def __init__(self):
        """
        初始化不确定性分解
        """
        pass
    
    def decompose_uncertainty(self, ensemble_predictions: torch.Tensor,
                            individual_uncertainties: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        分解不确定性
        
        Args:
            ensemble_predictions: 集成预测 [num_models, batch_size, ...]
            individual_uncertainties: 个体模型不确定性（可选）
            
        Returns:
            Dict: 不确定性分解结果
        """
        # 集成统计量
        ensemble_mean = torch.mean(ensemble_predictions, dim=0)
        
        # 总不确定性（集成预测的方差）
        total_uncertainty = torch.var(ensemble_predictions, dim=0)
        
        # 认知不确定性（模型间的不一致性）
        model_means = ensemble_predictions
        epistemic_uncertainty = torch.var(model_means, dim=0)
        
        # 偶然不确定性（数据固有噪声）
        if individual_uncertainties is not None:
            # 如果有个体模型的不确定性估计
            aleatoric_uncertainty = torch.mean(individual_uncertainties, dim=0)
        else:
            # 简化估计：总不确定性减去认知不确定性
            aleatoric_uncertainty = torch.clamp(
                total_uncertainty - epistemic_uncertainty, min=0
            )
        
        return {
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'ensemble_mean': ensemble_mean
        }
    
    def mutual_information(self, ensemble_predictions: torch.Tensor) -> torch.Tensor:
        """
        计算互信息（认知不确定性的另一种度量）
        
        Args:
            ensemble_predictions: 集成预测
            
        Returns:
            torch.Tensor: 互信息
        """
        # 集成预测的熵
        ensemble_mean = torch.mean(ensemble_predictions, dim=0)
        ensemble_entropy = -torch.sum(ensemble_mean * torch.log(ensemble_mean + 1e-8), dim=-1)
        
        # 个体预测熵的期望
        individual_entropies = []
        for i in range(ensemble_predictions.shape[0]):
            pred = ensemble_predictions[i]
            entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=-1)
            individual_entropies.append(entropy)
        
        expected_entropy = torch.mean(torch.stack(individual_entropies), dim=0)
        
        # 互信息 = 集成熵 - 期望熵
        mutual_info = ensemble_entropy - expected_entropy
        
        return mutual_info

class EnsembleManager:
    """
    集成管理器
    
    统一管理不同类型的集成方法
    """
    
    def __init__(self):
        """
        初始化集成管理器
        """
        self.ensembles = {}
        self.uncertainty_decomposer = UncertaintyDecomposition()
    
    def add_ensemble(self, name: str, ensemble):
        """
        添加集成
        
        Args:
            name: 集成名称
            ensemble: 集成对象
        """
        self.ensembles[name] = ensemble
    
    def compare_ensembles(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        比较不同集成方法
        
        Args:
            x: 输入数据
            
        Returns:
            Dict: 比较结果
        """
        results = {}
        
        for name, ensemble in self.ensembles.items():
            predictions = ensemble.predict(x)
            
            # 不确定性分解
            if 'predictions' in predictions:
                uncertainty_decomp = self.uncertainty_decomposer.decompose_uncertainty(
                    predictions['predictions']
                )
                predictions.update(uncertainty_decomp)
            
            results[name] = predictions
        
        return results
    
    def ensemble_of_ensembles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        集成的集成（元集成）
        
        Args:
            x: 输入数据
            
        Returns:
            Dict: 元集成结果
        """
        all_predictions = []
        
        for ensemble in self.ensembles.values():
            pred_dict = ensemble.predict(x)
            if 'mean' in pred_dict:
                all_predictions.append(pred_dict['mean'])
            elif 'predictions' in pred_dict:
                # 如果返回的是个体预测，取平均
                mean_pred = torch.mean(pred_dict['predictions'], dim=0)
                all_predictions.append(mean_pred)
        
        if all_predictions:
            all_predictions = torch.stack(all_predictions, dim=0)
            
            meta_mean = torch.mean(all_predictions, dim=0)
            meta_std = torch.std(all_predictions, dim=0)
            
            return {
                'meta_mean': meta_mean,
                'meta_std': meta_std,
                'meta_predictions': all_predictions
            }
        else:
            return {}

if __name__ == "__main__":
    # 测试集成方法
    torch.manual_seed(42)
    
    # 简单模型工厂
    def model_factory():
        return nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    # 测试数据
    x = torch.randn(100, 2)
    y = torch.randn(100, 1)
    x_test = torch.randn(20, 2)
    
    # 损失函数
    loss_fn = nn.MSELoss()
    
    # 测试深度集成
    print("测试深度集成...")
    deep_ensemble = DeepEnsemble(model_factory, num_models=3)
    deep_ensemble.initialize_optimizers(torch.optim.Adam, lr=0.01)
    
    # 训练几步
    for _ in range(5):
        loss_dict = deep_ensemble.train_step(x, y, loss_fn)
    
    deep_predictions = deep_ensemble.predict(x_test)
    print(f"深度集成预测形状: {deep_predictions['mean'].shape}")
    print(f"深度集成不确定性形状: {deep_predictions['std'].shape}")
    
    # 测试Bootstrap集成
    print("\n测试Bootstrap集成...")
    bootstrap_ensemble = BootstrapEnsemble(model_factory, num_models=3)
    bootstrap_ensemble.train_ensemble(x, y, loss_fn, torch.optim.Adam, num_epochs=10, lr=0.01)
    
    bootstrap_predictions = bootstrap_ensemble.predict(x_test)
    print(f"Bootstrap集成预测形状: {bootstrap_predictions['mean'].shape}")
    
    # 测试多种子集成
    print("\n测试多种子集成...")
    multiseed_ensemble = MultiSeedEnsemble(model_factory, num_models=3, seeds=[1, 2, 3])
    multiseed_ensemble.train_ensemble(x, y, loss_fn, torch.optim.Adam, num_epochs=10, lr=0.01)
    
    multiseed_predictions = multiseed_ensemble.predict(x_test)
    print(f"多种子集成预测形状: {multiseed_predictions['mean'].shape}")
    
    # 测试不确定性分解
    print("\n测试不确定性分解...")
    uncertainty_decomposer = UncertaintyDecomposition()
    decomp_results = uncertainty_decomposer.decompose_uncertainty(
        deep_predictions['individual_predictions'] if 'individual_predictions' in deep_predictions 
        else bootstrap_predictions['predictions']
    )
    print(f"不确定性分解键: {list(decomp_results.keys())}")
    
    # 测试集成管理器
    print("\n测试集成管理器...")
    manager = EnsembleManager()
    manager.add_ensemble('deep', deep_ensemble)
    manager.add_ensemble('bootstrap', bootstrap_ensemble)
    manager.add_ensemble('multiseed', multiseed_ensemble)
    
    comparison = manager.compare_ensembles(x_test)
    print(f"集成比较结果键: {list(comparison.keys())}")
    
    meta_results = manager.ensemble_of_ensembles(x_test)
    if meta_results:
        print(f"元集成结果键: {list(meta_results.keys())}")
        print(f"元集成预测形状: {meta_results['meta_mean'].shape}")