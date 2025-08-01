#!/usr/bin/env python3
"""
共形预测用于不确定性量化

实现共形预测方法来提供预测区间的统计保证，包括：
- 标准共形预测
- 交叉共形预测
- 分位数共形预测
- 自适应共形预测
- 条件共形预测

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
import warnings

class BaseConformalPredictor(ABC):
    """
    共形预测基类
    
    定义共形预测的基本接口
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        """
        初始化共形预测器
        
        Args:
            model: 预测模型
            alpha: 显著性水平 (1-alpha为置信水平)
        """
        self.model = model
        self.alpha = alpha
        self.quantile = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """
        在校准集上拟合共形预测器
        
        Args:
            X_cal: 校准集输入
            y_cal: 校准集标签
        """
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        进行共形预测
        
        Args:
            X: 输入数据
            
        Returns:
            Dict: 预测结果包含预测区间
        """
        pass
    
    def _compute_conformity_scores(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算一致性分数
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            torch.Tensor: 一致性分数
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        # 默认使用绝对误差作为一致性分数
        scores = torch.abs(predictions - y)
        return scores
    
    def _compute_quantile(self, scores: torch.Tensor) -> float:
        """
        计算分位数
        
        Args:
            scores: 一致性分数
            
        Returns:
            float: 分位数值
        """
        n = len(scores)
        # 计算 (1-alpha)(1+1/n) 分位数
        q_level = (1 - self.alpha) * (1 + 1/n)
        q_level = min(q_level, 1.0)  # 确保不超过1
        
        quantile = torch.quantile(scores, q_level)
        return quantile.item()

class SplitConformalPredictor(BaseConformalPredictor):
    """
    分割共形预测
    
    使用独立的校准集进行共形预测
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        """
        初始化分割共形预测器
        
        Args:
            model: 预测模型
            alpha: 显著性水平
        """
        super().__init__(model, alpha)
    
    def fit(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """
        在校准集上拟合
        
        Args:
            X_cal: 校准集输入
            y_cal: 校准集标签
        """
        # 计算校准集上的一致性分数
        scores = self._compute_conformity_scores(X_cal, y_cal)
        
        # 计算分位数
        self.quantile = self._compute_quantile(scores)
        self.is_fitted = True
    
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        进行共形预测
        
        Args:
            X: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("共形预测器尚未拟合")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        # 构建预测区间
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return {
            'predictions': predictions,
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower
        }

class CrossConformalPredictor(BaseConformalPredictor):
    """
    交叉共形预测
    
    使用交叉验证进行共形预测，无需独立校准集
    """
    
    def __init__(self, model_factory: Callable, alpha: float = 0.1, n_folds: int = 5):
        """
        初始化交叉共形预测器
        
        Args:
            model_factory: 模型工厂函数
            alpha: 显著性水平
            n_folds: 交叉验证折数
        """
        super().__init__(None, alpha)
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.fold_models = []
        self.fold_quantiles = []
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
           optimizer_class=torch.optim.Adam, num_epochs: int = 100, **optimizer_kwargs):
        """
        使用交叉验证拟合
        
        Args:
            X: 输入数据
            y: 标签数据
            optimizer_class: 优化器类
            num_epochs: 训练轮数
            **optimizer_kwargs: 优化器参数
        """
        # 转换为numpy进行交叉验证分割
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        self.fold_models = []
        self.fold_quantiles = []
        all_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_np)):
            print(f"训练折 {fold + 1}/{self.n_folds}")
            
            # 分割数据
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # 创建并训练模型
            model = self.model_factory()
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                
                pred = model(X_train)
                loss = F.mse_loss(pred, y_train)
                
                loss.backward()
                optimizer.step()
            
            # 计算验证集上的一致性分数
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                scores = torch.abs(val_pred - y_val)
                all_scores.append(scores)
            
            self.fold_models.append(model)
        
        # 合并所有分数并计算全局分位数
        all_scores = torch.cat(all_scores, dim=0)
        self.quantile = self._compute_quantile(all_scores)
        self.is_fitted = True
    
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        进行交叉共形预测
        
        Args:
            X: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("交叉共形预测器尚未拟合")
        
        # 使用所有折的模型进行预测
        fold_predictions = []
        
        for model in self.fold_models:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                fold_predictions.append(pred)
        
        # 平均预测
        predictions = torch.mean(torch.stack(fold_predictions), dim=0)
        
        # 构建预测区间
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return {
            'predictions': predictions,
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'fold_predictions': torch.stack(fold_predictions)
        }

class QuantileConformalPredictor(BaseConformalPredictor):
    """
    分位数共形预测
    
    使用分位数回归进行共形预测
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        """
        初始化分位数共形预测器
        
        Args:
            model: 分位数回归模型（输出3个值：下分位数、中位数、上分位数）
            alpha: 显著性水平
        """
        super().__init__(model, alpha)
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - alpha / 2
    
    def _compute_conformity_scores(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算分位数一致性分数
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            torch.Tensor: 一致性分数
        """
        self.model.eval()
        with torch.no_grad():
            quantile_preds = self.model(X)  # [batch_size, 3] (lower, median, upper)
        
        lower_pred = quantile_preds[:, 0:1]
        upper_pred = quantile_preds[:, 2:3]
        
        # 计算一致性分数：如果真实值在预测区间内，分数为0；否则为距离区间的最小距离
        scores = torch.maximum(
            lower_pred - y,  # 如果y < lower_pred，则为正值
            y - upper_pred   # 如果y > upper_pred，则为正值
        )
        scores = torch.maximum(scores, torch.zeros_like(scores))
        
        return scores
    
    def fit(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """
        在校准集上拟合
        
        Args:
            X_cal: 校准集输入
            y_cal: 校准集标签
        """
        scores = self._compute_conformity_scores(X_cal, y_cal)
        self.quantile = self._compute_quantile(scores)
        self.is_fitted = True
    
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        进行分位数共形预测
        
        Args:
            X: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("分位数共形预测器尚未拟合")
        
        self.model.eval()
        with torch.no_grad():
            quantile_preds = self.model(X)
        
        lower_pred = quantile_preds[:, 0:1]
        median_pred = quantile_preds[:, 1:2]
        upper_pred = quantile_preds[:, 2:3]
        
        # 调整预测区间
        adjusted_lower = lower_pred - self.quantile
        adjusted_upper = upper_pred + self.quantile
        
        return {
            'predictions': median_pred,
            'lower': adjusted_lower,
            'upper': adjusted_upper,
            'interval_width': adjusted_upper - adjusted_lower,
            'raw_quantiles': quantile_preds
        }

class AdaptiveConformalPredictor(BaseConformalPredictor):
    """
    自适应共形预测
    
    根据预测难度自适应调整预测区间
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.1, 
                 adaptation_rate: float = 0.1):
        """
        初始化自适应共形预测器
        
        Args:
            model: 预测模型
            alpha: 显著性水平
            adaptation_rate: 自适应率
        """
        super().__init__(model, alpha)
        self.adaptation_rate = adaptation_rate
        self.difficulty_estimator = None
    
    def _estimate_difficulty(self, X: torch.Tensor) -> torch.Tensor:
        """
        估计预测难度
        
        Args:
            X: 输入数据
            
        Returns:
            torch.Tensor: 难度分数
        """
        if self.difficulty_estimator is None:
            # 简单的难度估计：使用输入特征的方差
            difficulty = torch.var(X, dim=1, keepdim=True)
        else:
            # 使用训练好的难度估计器
            difficulty = self.difficulty_estimator(X)
        
        return difficulty
    
    def fit(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """
        拟合自适应共形预测器
        
        Args:
            X_cal: 校准集输入
            y_cal: 校准集标签
        """
        # 计算基础一致性分数
        base_scores = self._compute_conformity_scores(X_cal, y_cal)
        
        # 估计难度
        difficulties = self._estimate_difficulty(X_cal)
        
        # 根据难度调整分数
        adjusted_scores = base_scores * (1 + self.adaptation_rate * difficulties)
        
        # 计算调整后的分位数
        self.quantile = self._compute_quantile(adjusted_scores)
        
        # 保存校准数据用于自适应
        self.cal_difficulties = difficulties
        self.cal_scores = base_scores
        
        self.is_fitted = True
    
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        进行自适应共形预测
        
        Args:
            X: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("自适应共形预测器尚未拟合")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        # 估计测试数据的难度
        test_difficulties = self._estimate_difficulty(X)
        
        # 根据难度自适应调整区间宽度
        adaptive_quantiles = self.quantile * (1 + self.adaptation_rate * test_difficulties)
        
        # 构建预测区间
        lower = predictions - adaptive_quantiles
        upper = predictions + adaptive_quantiles
        
        return {
            'predictions': predictions,
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'difficulties': test_difficulties,
            'adaptive_quantiles': adaptive_quantiles
        }

class ConditionalConformalPredictor(BaseConformalPredictor):
    """
    条件共形预测
    
    根据输入特征条件调整预测区间
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.1, 
                 num_bins: int = 10):
        """
        初始化条件共形预测器
        
        Args:
            model: 预测模型
            alpha: 显著性水平
            num_bins: 条件分箱数量
        """
        super().__init__(model, alpha)
        self.num_bins = num_bins
        self.bin_quantiles = {}
        self.bin_boundaries = None
    
    def _assign_bins(self, X: torch.Tensor) -> torch.Tensor:
        """
        将输入分配到条件分箱
        
        Args:
            X: 输入数据
            
        Returns:
            torch.Tensor: 分箱索引
        """
        # 使用第一个特征进行分箱（可以扩展为更复杂的分箱策略）
        feature = X[:, 0]
        
        if self.bin_boundaries is None:
            # 基于分位数创建分箱边界
            quantiles = torch.linspace(0, 1, self.num_bins + 1)
            self.bin_boundaries = torch.quantile(feature, quantiles)
        
        # 分配分箱
        bins = torch.searchsorted(self.bin_boundaries[1:-1], feature)
        bins = torch.clamp(bins, 0, self.num_bins - 1)
        
        return bins
    
    def fit(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        """
        拟合条件共形预测器
        
        Args:
            X_cal: 校准集输入
            y_cal: 校准集标签
        """
        # 计算一致性分数
        scores = self._compute_conformity_scores(X_cal, y_cal)
        
        # 分配分箱
        bins = self._assign_bins(X_cal)
        
        # 为每个分箱计算分位数
        self.bin_quantiles = {}
        
        for bin_idx in range(self.num_bins):
            bin_mask = (bins == bin_idx)
            
            if bin_mask.sum() > 0:
                bin_scores = scores[bin_mask]
                self.bin_quantiles[bin_idx] = self._compute_quantile(bin_scores)
            else:
                # 如果分箱为空，使用全局分位数
                self.bin_quantiles[bin_idx] = self._compute_quantile(scores)
        
        self.is_fitted = True
    
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        进行条件共形预测
        
        Args:
            X: 输入数据
            
        Returns:
            Dict: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("条件共形预测器尚未拟合")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        # 分配分箱
        bins = self._assign_bins(X)
        
        # 为每个样本获取对应的分位数
        quantiles = torch.zeros(X.shape[0], 1)
        
        for bin_idx in range(self.num_bins):
            bin_mask = (bins == bin_idx)
            if bin_mask.sum() > 0:
                quantiles[bin_mask] = self.bin_quantiles[bin_idx]
        
        # 构建预测区间
        lower = predictions - quantiles
        upper = predictions + quantiles
        
        return {
            'predictions': predictions,
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'bins': bins,
            'quantiles': quantiles
        }

class ConformalPredictionEvaluator:
    """
    共形预测评估器
    
    评估共形预测的性能
    """
    
    def __init__(self):
        """
        初始化评估器
        """
        pass
    
    def coverage_rate(self, y_true: torch.Tensor, lower: torch.Tensor, 
                     upper: torch.Tensor) -> float:
        """
        计算覆盖率
        
        Args:
            y_true: 真实值
            lower: 预测区间下界
            upper: 预测区间上界
            
        Returns:
            float: 覆盖率
        """
        covered = (y_true >= lower) & (y_true <= upper)
        return covered.float().mean().item()
    
    def average_width(self, lower: torch.Tensor, upper: torch.Tensor) -> float:
        """
        计算平均区间宽度
        
        Args:
            lower: 预测区间下界
            upper: 预测区间上界
            
        Returns:
            float: 平均宽度
        """
        widths = upper - lower
        return widths.mean().item()
    
    def conditional_coverage(self, y_true: torch.Tensor, lower: torch.Tensor,
                           upper: torch.Tensor, conditions: torch.Tensor,
                           num_bins: int = 10) -> Dict[str, float]:
        """
        计算条件覆盖率
        
        Args:
            y_true: 真实值
            lower: 预测区间下界
            upper: 预测区间上界
            conditions: 条件变量
            num_bins: 分箱数量
            
        Returns:
            Dict: 条件覆盖率
        """
        # 基于条件变量分箱
        quantiles = torch.linspace(0, 1, num_bins + 1)
        bin_boundaries = torch.quantile(conditions, quantiles)
        bins = torch.searchsorted(bin_boundaries[1:-1], conditions)
        bins = torch.clamp(bins, 0, num_bins - 1)
        
        conditional_coverages = {}
        
        for bin_idx in range(num_bins):
            bin_mask = (bins == bin_idx)
            
            if bin_mask.sum() > 0:
                bin_coverage = self.coverage_rate(
                    y_true[bin_mask], lower[bin_mask], upper[bin_mask]
                )
                conditional_coverages[f'bin_{bin_idx}'] = bin_coverage
        
        return conditional_coverages
    
    def efficiency_metrics(self, y_true: torch.Tensor, lower: torch.Tensor,
                          upper: torch.Tensor) -> Dict[str, float]:
        """
        计算效率指标
        
        Args:
            y_true: 真实值
            lower: 预测区间下界
            upper: 预测区间上界
            
        Returns:
            Dict: 效率指标
        """
        coverage = self.coverage_rate(y_true, lower, upper)
        avg_width = self.average_width(lower, upper)
        
        # 标准化宽度（相对于真实值的标准差）
        y_std = torch.std(y_true).item()
        normalized_width = avg_width / y_std if y_std > 0 else float('inf')
        
        # 效率分数：覆盖率除以标准化宽度
        efficiency = coverage / normalized_width if normalized_width > 0 else 0
        
        return {
            'coverage_rate': coverage,
            'average_width': avg_width,
            'normalized_width': normalized_width,
            'efficiency_score': efficiency
        }

class ConformalPredictionManager:
    """
    共形预测管理器
    
    统一管理不同类型的共形预测方法
    """
    
    def __init__(self):
        """
        初始化管理器
        """
        self.predictors = {}
        self.evaluator = ConformalPredictionEvaluator()
    
    def add_predictor(self, name: str, predictor: BaseConformalPredictor):
        """
        添加共形预测器
        
        Args:
            name: 预测器名称
            predictor: 预测器对象
        """
        self.predictors[name] = predictor
    
    def compare_predictors(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        比较不同共形预测器
        
        Args:
            X_test: 测试输入
            y_test: 测试标签
            
        Returns:
            Dict: 比较结果
        """
        results = {}
        
        for name, predictor in self.predictors.items():
            if not predictor.is_fitted:
                warnings.warn(f"预测器 {name} 尚未拟合，跳过评估")
                continue
            
            # 进行预测
            pred_results = predictor.predict(X_test)
            
            # 评估性能
            metrics = self.evaluator.efficiency_metrics(
                y_test, pred_results['lower'], pred_results['upper']
            )
            
            results[name] = metrics
        
        return results
    
    def ensemble_prediction(self, X: torch.Tensor, method: str = 'average') -> Dict[str, torch.Tensor]:
        """
        集成多个共形预测器
        
        Args:
            X: 输入数据
            method: 集成方法 ('average', 'intersection', 'union')
            
        Returns:
            Dict: 集成预测结果
        """
        if not self.predictors:
            raise ValueError("没有可用的预测器")
        
        all_predictions = []
        all_lowers = []
        all_uppers = []
        
        for predictor in self.predictors.values():
            if predictor.is_fitted:
                pred_results = predictor.predict(X)
                all_predictions.append(pred_results['predictions'])
                all_lowers.append(pred_results['lower'])
                all_uppers.append(pred_results['upper'])
        
        if not all_predictions:
            raise ValueError("没有已拟合的预测器")
        
        all_predictions = torch.stack(all_predictions)
        all_lowers = torch.stack(all_lowers)
        all_uppers = torch.stack(all_uppers)
        
        if method == 'average':
            # 平均集成
            ensemble_pred = torch.mean(all_predictions, dim=0)
            ensemble_lower = torch.mean(all_lowers, dim=0)
            ensemble_upper = torch.mean(all_uppers, dim=0)
        
        elif method == 'intersection':
            # 交集：最保守的区间
            ensemble_pred = torch.mean(all_predictions, dim=0)
            ensemble_lower = torch.max(all_lowers, dim=0)[0]
            ensemble_upper = torch.min(all_uppers, dim=0)[0]
        
        elif method == 'union':
            # 并集：最宽松的区间
            ensemble_pred = torch.mean(all_predictions, dim=0)
            ensemble_lower = torch.min(all_lowers, dim=0)[0]
            ensemble_upper = torch.max(all_uppers, dim=0)[0]
        
        else:
            raise ValueError(f"未知的集成方法: {method}")
        
        return {
            'predictions': ensemble_pred,
            'lower': ensemble_lower,
            'upper': ensemble_upper,
            'interval_width': ensemble_upper - ensemble_lower
        }

if __name__ == "__main__":
    # 测试共形预测方法
    torch.manual_seed(42)
    
    # 简单模型
    model = nn.Sequential(
        nn.Linear(2, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # 分位数模型（输出3个分位数）
    quantile_model = nn.Sequential(
        nn.Linear(2, 50),
        nn.ReLU(),
        nn.Linear(50, 3)
    )
    
    # 测试数据
    X_train = torch.randn(200, 2)
    y_train = torch.randn(200, 1)
    X_cal = torch.randn(100, 2)
    y_cal = torch.randn(100, 1)
    X_test = torch.randn(50, 2)
    y_test = torch.randn(50, 1)
    
    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
    
    # 测试分割共形预测
    print("测试分割共形预测...")
    split_cp = SplitConformalPredictor(model, alpha=0.1)
    split_cp.fit(X_cal, y_cal)
    split_results = split_cp.predict(X_test)
    print(f"分割共形预测区间宽度: {split_results['interval_width'].mean().item():.4f}")
    
    # 测试交叉共形预测
    print("\n测试交叉共形预测...")
    def model_factory():
        return nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
    
    cross_cp = CrossConformalPredictor(model_factory, alpha=0.1, n_folds=3)
    cross_cp.fit(X_train, y_train, num_epochs=50, lr=0.01)
    cross_results = cross_cp.predict(X_test)
    print(f"交叉共形预测区间宽度: {cross_results['interval_width'].mean().item():.4f}")
    
    # 测试自适应共形预测
    print("\n测试自适应共形预测...")
    adaptive_cp = AdaptiveConformalPredictor(model, alpha=0.1, adaptation_rate=0.2)
    adaptive_cp.fit(X_cal, y_cal)
    adaptive_results = adaptive_cp.predict(X_test)
    print(f"自适应共形预测区间宽度: {adaptive_results['interval_width'].mean().item():.4f}")
    
    # 测试条件共形预测
    print("\n测试条件共形预测...")
    conditional_cp = ConditionalConformalPredictor(model, alpha=0.1, num_bins=5)
    conditional_cp.fit(X_cal, y_cal)
    conditional_results = conditional_cp.predict(X_test)
    print(f"条件共形预测区间宽度: {conditional_results['interval_width'].mean().item():.4f}")
    
    # 评估性能
    print("\n评估共形预测性能...")
    evaluator = ConformalPredictionEvaluator()
    
    # 分割共形预测评估
    split_metrics = evaluator.efficiency_metrics(y_test, split_results['lower'], split_results['upper'])
    print(f"分割共形预测覆盖率: {split_metrics['coverage_rate']:.4f}")
    print(f"分割共形预测效率分数: {split_metrics['efficiency_score']:.4f}")
    
    # 交叉共形预测评估
    cross_metrics = evaluator.efficiency_metrics(y_test, cross_results['lower'], cross_results['upper'])
    print(f"交叉共形预测覆盖率: {cross_metrics['coverage_rate']:.4f}")
    print(f"交叉共形预测效率分数: {cross_metrics['efficiency_score']:.4f}")
    
    # 测试管理器
    print("\n测试共形预测管理器...")
    manager = ConformalPredictionManager()
    manager.add_predictor('split', split_cp)
    manager.add_predictor('cross', cross_cp)
    manager.add_predictor('adaptive', adaptive_cp)
    manager.add_predictor('conditional', conditional_cp)
    
    comparison = manager.compare_predictors(X_test, y_test)
    print("预测器比较结果:")
    for name, metrics in comparison.items():
        print(f"  {name}: 覆盖率={metrics['coverage_rate']:.4f}, 效率={metrics['efficiency_score']:.4f}")
    
    # 集成预测
    ensemble_results = manager.ensemble_prediction(X_test, method='average')
    ensemble_metrics = evaluator.efficiency_metrics(y_test, ensemble_results['lower'], ensemble_results['upper'])
    print(f"\n集成预测覆盖率: {ensemble_metrics['coverage_rate']:.4f}")
    print(f"集成预测效率分数: {ensemble_metrics['efficiency_score']:.4f}")