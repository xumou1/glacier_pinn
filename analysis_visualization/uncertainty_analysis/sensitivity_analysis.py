#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感性分析模块

本模块提供冰川PINNs模型敏感性分析功能，包括：
- 局部敏感性分析
- 全局敏感性分析（Sobol指数）
- Morris筛选法
- 参数重要性排序
- 交互效应分析
- 敏感性可视化

作者: 冰川PINNs项目组
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import itertools
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    敏感性分析器
    
    提供多种敏感性分析方法，用于评估模型参数和输入变量的重要性。
    """
    
    def __init__(self, model: Optional[Callable] = None):
        """
        初始化敏感性分析器
        
        参数:
            model: 待分析的模型（可调用对象）
        """
        self.model = model
        self.scaler = StandardScaler()
        
    def local_sensitivity_analysis(self,
                                 X_base: np.ndarray,
                                 parameter_names: List[str],
                                 perturbation_size: float = 0.01,
                                 method: str = 'finite_difference') -> Dict:
        """
        局部敏感性分析
        
        参数:
            X_base: 基准参数值
            parameter_names: 参数名称列表
            perturbation_size: 扰动大小
            method: 计算方法 ('finite_difference', 'gradient')
            
        返回:
            局部敏感性分析结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行敏感性分析")
            
            n_params = len(X_base)
            sensitivities = np.zeros(n_params)
            
            # 计算基准输出
            if hasattr(self.model, 'predict'):
                y_base = self.model.predict(X_base.reshape(1, -1))[0]
            else:
                y_base = self.model(X_base)
            
            logger.info(f"基准输出: {y_base}")
            
            for i in range(n_params):
                if method == 'finite_difference':
                    # 有限差分法
                    X_perturbed = X_base.copy()
                    X_perturbed[i] += perturbation_size
                    
                    if hasattr(self.model, 'predict'):
                        y_perturbed = self.model.predict(X_perturbed.reshape(1, -1))[0]
                    else:
                        y_perturbed = self.model(X_perturbed)
                    
                    # 计算敏感性
                    sensitivity = (y_perturbed - y_base) / perturbation_size
                    sensitivities[i] = sensitivity
                    
                elif method == 'gradient':
                    # 梯度法（需要模型支持梯度计算）
                    if hasattr(self.model, 'gradient'):
                        gradient = self.model.gradient(X_base)
                        sensitivities[i] = gradient[i]
                    else:
                        # 回退到有限差分
                        X_perturbed = X_base.copy()
                        X_perturbed[i] += perturbation_size
                        
                        if hasattr(self.model, 'predict'):
                            y_perturbed = self.model.predict(X_perturbed.reshape(1, -1))[0]
                        else:
                            y_perturbed = self.model(X_perturbed)
                        
                        sensitivity = (y_perturbed - y_base) / perturbation_size
                        sensitivities[i] = sensitivity
            
            # 计算归一化敏感性
            normalized_sensitivities = np.abs(sensitivities) / np.sum(np.abs(sensitivities))
            
            # 排序
            importance_order = np.argsort(np.abs(sensitivities))[::-1]
            
            return {
                'method': 'local_sensitivity',
                'base_point': X_base,
                'base_output': y_base,
                'sensitivities': sensitivities,
                'normalized_sensitivities': normalized_sensitivities,
                'parameter_names': parameter_names,
                'importance_ranking': {
                    'indices': importance_order,
                    'names': [parameter_names[i] for i in importance_order],
                    'values': sensitivities[importance_order]
                },
                'perturbation_size': perturbation_size
            }
            
        except Exception as e:
            logger.error(f"局部敏感性分析失败: {e}")
            raise
    
    def sobol_sensitivity_analysis(self,
                                 parameter_bounds: List[Tuple[float, float]],
                                 parameter_names: List[str],
                                 n_samples: int = 1000,
                                 calc_second_order: bool = True) -> Dict:
        """
        Sobol全局敏感性分析
        
        参数:
            parameter_bounds: 参数边界列表 [(min, max), ...]
            parameter_names: 参数名称列表
            n_samples: 采样数量
            calc_second_order: 是否计算二阶效应
            
        返回:
            Sobol敏感性分析结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行敏感性分析")
            
            n_params = len(parameter_bounds)
            
            # 生成Sobol序列
            A, B, AB_matrices = self._generate_sobol_matrices(
                parameter_bounds, n_samples, n_params
            )
            
            # 计算模型输出
            logger.info("计算模型输出...")
            
            y_A = self._evaluate_model_batch(A)
            y_B = self._evaluate_model_batch(B)
            
            y_AB = {}
            for i in range(n_params):
                y_AB[i] = self._evaluate_model_batch(AB_matrices[i])
            
            # 计算Sobol指数
            sobol_indices = self._calculate_sobol_indices(
                y_A, y_B, y_AB, calc_second_order
            )
            
            # 添加参数名称
            first_order = {}
            total_order = {}
            
            for i, name in enumerate(parameter_names):
                first_order[name] = sobol_indices['first_order'][i]
                total_order[name] = sobol_indices['total_order'][i]
            
            result = {
                'method': 'sobol',
                'n_samples': n_samples,
                'parameter_names': parameter_names,
                'parameter_bounds': parameter_bounds,
                'first_order_indices': first_order,
                'total_order_indices': total_order,
                'first_order_array': sobol_indices['first_order'],
                'total_order_array': sobol_indices['total_order']
            }
            
            if calc_second_order and 'second_order' in sobol_indices:
                result['second_order_indices'] = sobol_indices['second_order']
            
            # 参数重要性排序
            total_importance = np.array(list(total_order.values()))
            importance_order = np.argsort(total_importance)[::-1]
            
            result['importance_ranking'] = {
                'indices': importance_order,
                'names': [parameter_names[i] for i in importance_order],
                'total_indices': total_importance[importance_order]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sobol敏感性分析失败: {e}")
            raise
    
    def morris_screening(self,
                       parameter_bounds: List[Tuple[float, float]],
                       parameter_names: List[str],
                       n_trajectories: int = 100,
                       n_levels: int = 10) -> Dict:
        """
        Morris筛选法
        
        参数:
            parameter_bounds: 参数边界列表
            parameter_names: 参数名称列表
            n_trajectories: 轨迹数量
            n_levels: 离散化水平数
            
        返回:
            Morris筛选结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行敏感性分析")
            
            n_params = len(parameter_bounds)
            
            # 生成Morris采样
            trajectories, delta = self._generate_morris_trajectories(
                parameter_bounds, n_trajectories, n_levels
            )
            
            # 计算基本效应
            elementary_effects = []
            
            logger.info(f"计算{n_trajectories}条轨迹的基本效应...")
            
            for traj_idx, trajectory in enumerate(trajectories):
                traj_effects = []
                
                for step in range(n_params):
                    # 当前点和下一点
                    x1 = trajectory[step]
                    x2 = trajectory[step + 1]
                    
                    # 计算模型输出
                    y1 = self._evaluate_model_single(x1)
                    y2 = self._evaluate_model_single(x2)
                    
                    # 找到变化的参数
                    changed_param = np.where(x1 != x2)[0][0]
                    
                    # 计算基本效应
                    effect = (y2 - y1) / delta
                    traj_effects.append((changed_param, effect))
                
                elementary_effects.append(traj_effects)
                
                if (traj_idx + 1) % 20 == 0:
                    logger.info(f"完成轨迹: {traj_idx + 1}/{n_trajectories}")
            
            # 计算Morris指标
            morris_indices = self._calculate_morris_indices(
                elementary_effects, n_params
            )
            
            # 添加参数名称
            mu_star = {}
            sigma = {}
            mu = {}
            
            for i, name in enumerate(parameter_names):
                mu_star[name] = morris_indices['mu_star'][i]
                sigma[name] = morris_indices['sigma'][i]
                mu[name] = morris_indices['mu'][i]
            
            # 参数分类
            classification = self._classify_morris_parameters(
                morris_indices['mu_star'], morris_indices['sigma'], parameter_names
            )
            
            return {
                'method': 'morris',
                'n_trajectories': n_trajectories,
                'n_levels': n_levels,
                'parameter_names': parameter_names,
                'parameter_bounds': parameter_bounds,
                'mu_star': mu_star,
                'sigma': sigma,
                'mu': mu,
                'mu_star_array': morris_indices['mu_star'],
                'sigma_array': morris_indices['sigma'],
                'mu_array': morris_indices['mu'],
                'classification': classification,
                'elementary_effects': elementary_effects
            }
            
        except Exception as e:
            logger.error(f"Morris筛选失败: {e}")
            raise
    
    def feature_importance_analysis(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  feature_names: List[str],
                                  method: str = 'permutation') -> Dict:
        """
        特征重要性分析
        
        参数:
            X: 输入特征
            y: 目标变量
            feature_names: 特征名称列表
            method: 分析方法 ('permutation', 'random_forest', 'correlation')
            
        返回:
            特征重要性分析结果
        """
        try:
            if method == 'permutation':
                return self._permutation_importance(X, y, feature_names)
            elif method == 'random_forest':
                return self._random_forest_importance(X, y, feature_names)
            elif method == 'correlation':
                return self._correlation_importance(X, y, feature_names)
            else:
                raise ValueError(f"不支持的方法: {method}")
                
        except Exception as e:
            logger.error(f"特征重要性分析失败: {e}")
            raise
    
    def interaction_analysis(self,
                           parameter_bounds: List[Tuple[float, float]],
                           parameter_names: List[str],
                           n_samples: int = 500) -> Dict:
        """
        交互效应分析
        
        参数:
            parameter_bounds: 参数边界列表
            parameter_names: 参数名称列表
            n_samples: 采样数量
            
        返回:
            交互效应分析结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行交互效应分析")
            
            n_params = len(parameter_bounds)
            
            # 生成采样点
            samples = self._latin_hypercube_sampling(parameter_bounds, n_samples)
            
            # 计算模型输出
            outputs = self._evaluate_model_batch(samples)
            
            # 计算两两交互效应
            interaction_matrix = np.zeros((n_params, n_params))
            
            logger.info("计算参数交互效应...")
            
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    # 计算交互效应强度
                    interaction_strength = self._calculate_interaction_strength(
                        samples[:, i], samples[:, j], outputs
                    )
                    
                    interaction_matrix[i, j] = interaction_strength
                    interaction_matrix[j, i] = interaction_strength
            
            # 找到最强的交互效应
            strong_interactions = []
            threshold = np.percentile(interaction_matrix[np.triu_indices(n_params, k=1)], 75)
            
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    if interaction_matrix[i, j] > threshold:
                        strong_interactions.append({
                            'param1': parameter_names[i],
                            'param2': parameter_names[j],
                            'strength': interaction_matrix[i, j]
                        })
            
            # 按强度排序
            strong_interactions.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'method': 'interaction_analysis',
                'parameter_names': parameter_names,
                'interaction_matrix': interaction_matrix,
                'strong_interactions': strong_interactions,
                'threshold': threshold,
                'samples': samples,
                'outputs': outputs
            }
            
        except Exception as e:
            logger.error(f"交互效应分析失败: {e}")
            raise
    
    def comprehensive_sensitivity_analysis(self,
                                         parameter_bounds: List[Tuple[float, float]],
                                         parameter_names: List[str],
                                         X_base: Optional[np.ndarray] = None,
                                         methods: List[str] = ['local', 'sobol', 'morris']) -> Dict:
        """
        综合敏感性分析
        
        参数:
            parameter_bounds: 参数边界列表
            parameter_names: 参数名称列表
            X_base: 局部分析的基准点
            methods: 分析方法列表
            
        返回:
            综合敏感性分析结果
        """
        try:
            results = {
                'parameter_names': parameter_names,
                'parameter_bounds': parameter_bounds,
                'methods_used': methods
            }
            
            # 局部敏感性分析
            if 'local' in methods:
                if X_base is None:
                    # 使用参数边界的中点作为基准
                    X_base = np.array([(b[0] + b[1]) / 2 for b in parameter_bounds])
                
                logger.info("执行局部敏感性分析...")
                local_results = self.local_sensitivity_analysis(
                    X_base, parameter_names
                )
                results['local'] = local_results
            
            # Sobol全局敏感性分析
            if 'sobol' in methods:
                logger.info("执行Sobol全局敏感性分析...")
                sobol_results = self.sobol_sensitivity_analysis(
                    parameter_bounds, parameter_names, n_samples=1000
                )
                results['sobol'] = sobol_results
            
            # Morris筛选
            if 'morris' in methods:
                logger.info("执行Morris筛选...")
                morris_results = self.morris_screening(
                    parameter_bounds, parameter_names, n_trajectories=50
                )
                results['morris'] = morris_results
            
            # 交互效应分析
            if 'interaction' in methods:
                logger.info("执行交互效应分析...")
                interaction_results = self.interaction_analysis(
                    parameter_bounds, parameter_names
                )
                results['interaction'] = interaction_results
            
            # 综合排序
            results['comprehensive_ranking'] = self._create_comprehensive_ranking(
                results, parameter_names
            )
            
            return results
            
        except Exception as e:
            logger.error(f"综合敏感性分析失败: {e}")
            raise
    
    def _generate_sobol_matrices(self,
                               parameter_bounds: List[Tuple[float, float]],
                               n_samples: int,
                               n_params: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        生成Sobol采样矩阵
        """
        # 简化的Sobol序列生成（实际应用中建议使用专门的库如SALib）
        np.random.seed(42)
        
        # 生成两个独立的随机矩阵
        A = np.random.uniform(0, 1, (n_samples, n_params))
        B = np.random.uniform(0, 1, (n_samples, n_params))
        
        # 缩放到参数边界
        for i, (low, high) in enumerate(parameter_bounds):
            A[:, i] = A[:, i] * (high - low) + low
            B[:, i] = B[:, i] * (high - low) + low
        
        # 生成AB矩阵（用于计算一阶效应）
        AB_matrices = {}
        for i in range(n_params):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            AB_matrices[i] = AB_i
        
        return A, B, AB_matrices
    
    def _evaluate_model_batch(self, X: np.ndarray) -> np.ndarray:
        """
        批量评估模型
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            return np.array([self.model(x) for x in X])
    
    def _evaluate_model_single(self, x: np.ndarray) -> float:
        """
        单点评估模型
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(x.reshape(1, -1))[0]
        else:
            return self.model(x)
    
    def _calculate_sobol_indices(self,
                               y_A: np.ndarray,
                               y_B: np.ndarray,
                               y_AB: Dict,
                               calc_second_order: bool) -> Dict:
        """
        计算Sobol指数
        """
        # 计算方差
        var_y = np.var(np.concatenate([y_A, y_B]))
        
        n_params = len(y_AB)
        first_order = np.zeros(n_params)
        total_order = np.zeros(n_params)
        
        for i in range(n_params):
            # 一阶效应
            first_order[i] = (np.mean(y_B * (y_AB[i] - y_A))) / var_y
            
            # 总效应
            total_order[i] = 1 - (np.mean(y_A * (y_AB[i] - y_B))) / var_y
        
        result = {
            'first_order': first_order,
            'total_order': total_order
        }
        
        # 二阶效应（简化计算）
        if calc_second_order:
            second_order = {}
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    # 这里是简化的二阶效应计算
                    second_order[(i, j)] = max(0, total_order[i] + total_order[j] - 
                                              first_order[i] - first_order[j])
            result['second_order'] = second_order
        
        return result
    
    def _generate_morris_trajectories(self,
                                    parameter_bounds: List[Tuple[float, float]],
                                    n_trajectories: int,
                                    n_levels: int) -> Tuple[List, float]:
        """
        生成Morris轨迹
        """
        n_params = len(parameter_bounds)
        delta = 1 / (2 * (n_levels - 1))
        
        trajectories = []
        
        for _ in range(n_trajectories):
            # 生成一条轨迹
            trajectory = []
            
            # 起始点
            start_point = np.random.uniform(0, 1, n_params)
            
            # 离散化
            for i in range(n_params):
                level = int(start_point[i] * n_levels)
                start_point[i] = level / (n_levels - 1)
            
            # 缩放到参数边界
            for i, (low, high) in enumerate(parameter_bounds):
                start_point[i] = start_point[i] * (high - low) + low
            
            trajectory.append(start_point.copy())
            
            # 生成轨迹的其余点
            current_point = start_point.copy()
            param_order = np.random.permutation(n_params)
            
            for param_idx in param_order:
                # 在当前参数上移动delta
                low, high = parameter_bounds[param_idx]
                
                # 归一化当前值
                normalized_val = (current_point[param_idx] - low) / (high - low)
                
                # 移动delta
                if normalized_val + delta <= 1:
                    normalized_val += delta
                else:
                    normalized_val -= delta
                
                # 缩放回原始范围
                current_point[param_idx] = normalized_val * (high - low) + low
                trajectory.append(current_point.copy())
            
            trajectories.append(trajectory)
        
        return trajectories, delta
    
    def _calculate_morris_indices(self,
                                elementary_effects: List,
                                n_params: int) -> Dict:
        """
        计算Morris指标
        """
        # 收集每个参数的基本效应
        effects_by_param = [[] for _ in range(n_params)]
        
        for trajectory_effects in elementary_effects:
            for param_idx, effect in trajectory_effects:
                effects_by_param[param_idx].append(effect)
        
        # 计算Morris指标
        mu = np.zeros(n_params)
        mu_star = np.zeros(n_params)
        sigma = np.zeros(n_params)
        
        for i in range(n_params):
            if effects_by_param[i]:
                effects = np.array(effects_by_param[i])
                mu[i] = np.mean(effects)
                mu_star[i] = np.mean(np.abs(effects))
                sigma[i] = np.std(effects)
        
        return {
            'mu': mu,
            'mu_star': mu_star,
            'sigma': sigma
        }
    
    def _classify_morris_parameters(self,
                                  mu_star: np.ndarray,
                                  sigma: np.ndarray,
                                  parameter_names: List[str]) -> Dict:
        """
        Morris参数分类
        """
        # 计算阈值
        mu_star_threshold = np.percentile(mu_star, 75)
        sigma_threshold = np.percentile(sigma, 75)
        
        classification = {
            'important_linear': [],
            'important_nonlinear': [],
            'unimportant': []
        }
        
        for i, name in enumerate(parameter_names):
            if mu_star[i] > mu_star_threshold:
                if sigma[i] > sigma_threshold:
                    classification['important_nonlinear'].append(name)
                else:
                    classification['important_linear'].append(name)
            else:
                classification['unimportant'].append(name)
        
        return classification
    
    def _permutation_importance(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              feature_names: List[str]) -> Dict:
        """
        置换重要性分析
        """
        if self.model is None:
            # 使用随机森林作为默认模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        else:
            model = self.model
            if hasattr(model, 'fit'):
                model.fit(X, y)
        
        # 计算置换重要性
        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=42
        )
        
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = {
                'mean': perm_importance.importances_mean[i],
                'std': perm_importance.importances_std[i]
            }
        
        # 排序
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1]['mean'], reverse=True)
        
        return {
            'method': 'permutation',
            'importances': importance_dict,
            'ranking': [item[0] for item in sorted_features],
            'importance_array': perm_importance.importances_mean
        }
    
    def _random_forest_importance(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                feature_names: List[str]) -> Dict:
        """
        随机森林重要性分析
        """
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = importances[i]
        
        # 排序
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            'method': 'random_forest',
            'importances': importance_dict,
            'ranking': [item[0] for item in sorted_features],
            'importance_array': importances
        }
    
    def _correlation_importance(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              feature_names: List[str]) -> Dict:
        """
        相关性重要性分析
        """
        correlations = np.abs([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
        
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = correlations[i]
        
        # 排序
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            'method': 'correlation',
            'importances': importance_dict,
            'ranking': [item[0] for item in sorted_features],
            'importance_array': correlations
        }
    
    def _latin_hypercube_sampling(self,
                                parameter_bounds: List[Tuple[float, float]],
                                n_samples: int) -> np.ndarray:
        """
        拉丁超立方采样
        """
        n_params = len(parameter_bounds)
        samples = np.zeros((n_samples, n_params))
        
        for i in range(n_params):
            # 生成均匀分布的样本
            uniform_samples = np.random.uniform(0, 1, n_samples)
            
            # 拉丁超立方采样
            intervals = np.linspace(0, 1, n_samples + 1)
            for j in range(n_samples):
                uniform_samples[j] = np.random.uniform(intervals[j], intervals[j + 1])
            
            # 随机排列
            np.random.shuffle(uniform_samples)
            
            # 缩放到参数边界
            low, high = parameter_bounds[i]
            samples[:, i] = uniform_samples * (high - low) + low
        
        return samples
    
    def _calculate_interaction_strength(self,
                                      param1_values: np.ndarray,
                                      param2_values: np.ndarray,
                                      outputs: np.ndarray) -> float:
        """
        计算两个参数的交互效应强度
        """
        # 使用相关系数的乘积作为交互强度的简单度量
        corr1 = abs(stats.pearsonr(param1_values, outputs)[0])
        corr2 = abs(stats.pearsonr(param2_values, outputs)[0])
        
        # 计算联合效应
        combined_effect = abs(stats.pearsonr(
            param1_values * param2_values, outputs
        )[0])
        
        # 交互强度 = 联合效应 - 独立效应之和
        interaction_strength = max(0, combined_effect - corr1 - corr2)
        
        return interaction_strength
    
    def _create_comprehensive_ranking(self,
                                    results: Dict,
                                    parameter_names: List[str]) -> Dict:
        """
        创建综合参数重要性排序
        """
        n_params = len(parameter_names)
        scores = np.zeros(n_params)
        
        # 收集各方法的排序
        rankings = []
        
        if 'local' in results:
            local_ranking = results['local']['importance_ranking']['indices']
            rankings.append(local_ranking)
        
        if 'sobol' in results:
            sobol_total = results['sobol']['total_order_array']
            sobol_ranking = np.argsort(sobol_total)[::-1]
            rankings.append(sobol_ranking)
        
        if 'morris' in results:
            morris_mu_star = results['morris']['mu_star_array']
            morris_ranking = np.argsort(morris_mu_star)[::-1]
            rankings.append(morris_ranking)
        
        # 计算平均排名
        for ranking in rankings:
            for rank, param_idx in enumerate(ranking):
                scores[param_idx] += (n_params - rank)
        
        # 归一化
        if len(rankings) > 0:
            scores = scores / len(rankings)
        
        # 最终排序
        final_ranking = np.argsort(scores)[::-1]
        
        return {
            'ranking_indices': final_ranking,
            'ranking_names': [parameter_names[i] for i in final_ranking],
            'scores': scores[final_ranking],
            'methods_combined': len(rankings)
        }

class SensitivityVisualizer:
    """
    敏感性分析可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_local_sensitivity(self,
                             results: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制局部敏感性分析结果
        
        参数:
            results: 局部敏感性分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 敏感性条形图
        ax1 = axes[0]
        param_names = results['parameter_names']
        sensitivities = results['sensitivities']
        
        # 按绝对值排序
        sorted_indices = np.argsort(np.abs(sensitivities))[::-1]
        sorted_names = [param_names[i] for i in sorted_indices]
        sorted_values = sensitivities[sorted_indices]
        
        colors = ['red' if x < 0 else 'blue' for x in sorted_values]
        bars = ax1.barh(range(len(sorted_names)), sorted_values, color=colors)
        
        ax1.set_yticks(range(len(sorted_names)))
        ax1.set_yticklabels(sorted_names)
        ax1.set_xlabel('敏感性')
        ax1.set_title('局部敏感性分析')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            ax1.text(value + 0.01 * max(np.abs(sorted_values)), i,
                    f'{value:.3f}', va='center')
        
        # 归一化敏感性饼图
        ax2 = axes[1]
        normalized_sens = results['normalized_sensitivities']
        sorted_norm_sens = normalized_sens[sorted_indices]
        
        # 只显示前5个最重要的参数
        top_n = min(5, len(param_names))
        top_names = sorted_names[:top_n]
        top_values = sorted_norm_sens[:top_n]
        
        if len(top_names) < len(param_names):
            top_names.append('其他')
            top_values = np.append(top_values, np.sum(sorted_norm_sens[top_n:]))
        
        ax2.pie(top_values, labels=top_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('归一化敏感性分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sobol_indices(self,
                         results: Dict,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制Sobol指数
        
        参数:
            results: Sobol分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sobol敏感性分析', fontsize=16, fontweight='bold')
        
        param_names = results['parameter_names']
        first_order = results['first_order_array']
        total_order = results['total_order_array']
        
        # 一阶和总效应对比
        ax1 = axes[0, 0]
        x = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, first_order, width, label='一阶效应', alpha=0.8)
        bars2 = ax1.bar(x + width/2, total_order, width, label='总效应', alpha=0.8)
        
        ax1.set_xlabel('参数')
        ax1.set_ylabel('Sobol指数')
        ax1.set_title('一阶效应 vs 总效应')
        ax1.set_xticks(x)
        ax1.set_xticklabels(param_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 总效应排序
        ax2 = axes[0, 1]
        sorted_indices = np.argsort(total_order)[::-1]
        sorted_names = [param_names[i] for i in sorted_indices]
        sorted_total = total_order[sorted_indices]
        
        bars = ax2.barh(range(len(sorted_names)), sorted_total, color='orange')
        ax2.set_yticks(range(len(sorted_names)))
        ax2.set_yticklabels(sorted_names)
        ax2.set_xlabel('总效应指数')
        ax2.set_title('参数重要性排序')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, sorted_total)):
            ax2.text(value + 0.01, i, f'{value:.3f}', va='center')
        
        # 交互效应（总效应 - 一阶效应）
        ax3 = axes[1, 0]
        interaction_effects = total_order - first_order
        
        bars = ax3.bar(param_names, interaction_effects, color='green', alpha=0.7)
        ax3.set_xlabel('参数')
        ax3.set_ylabel('交互效应')
        ax3.set_title('交互效应强度')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 累积效应
        ax4 = axes[1, 1]
        cumulative_first = np.cumsum(np.sort(first_order)[::-1])
        cumulative_total = np.cumsum(np.sort(total_order)[::-1])
        
        ax4.plot(range(1, len(cumulative_first) + 1), cumulative_first, 
                'o-', label='累积一阶效应')
        ax4.plot(range(1, len(cumulative_total) + 1), cumulative_total, 
                's-', label='累积总效应')
        
        ax4.set_xlabel('参数数量')
        ax4.set_ylabel('累积效应')
        ax4.set_title('累积效应分析')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_morris_screening(self,
                            results: Dict,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制Morris筛选结果
        
        参数:
            results: Morris筛选结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Morris筛选分析', fontsize=16, fontweight='bold')
        
        param_names = results['parameter_names']
        mu_star = results['mu_star_array']
        sigma = results['sigma_array']
        mu = results['mu_array']
        
        # Morris散点图 (μ* vs σ)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(mu_star, sigma, s=100, alpha=0.7, c=range(len(param_names)), 
                            cmap='viridis')
        
        # 添加参数标签
        for i, name in enumerate(param_names):
            ax1.annotate(name, (mu_star[i], sigma[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('μ* (平均绝对效应)')
        ax1.set_ylabel('σ (标准差)')
        ax1.set_title('Morris散点图')
        ax1.grid(True, alpha=0.3)
        
        # 添加分类线
        mu_star_threshold = np.percentile(mu_star, 75)
        sigma_threshold = np.percentile(sigma, 75)
        ax1.axvline(x=mu_star_threshold, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(y=sigma_threshold, color='red', linestyle='--', alpha=0.7)
        
        # μ*排序
        ax2 = axes[0, 1]
        sorted_indices = np.argsort(mu_star)[::-1]
        sorted_names = [param_names[i] for i in sorted_indices]
        sorted_mu_star = mu_star[sorted_indices]
        
        bars = ax2.barh(range(len(sorted_names)), sorted_mu_star, color='skyblue')
        ax2.set_yticks(range(len(sorted_names)))
        ax2.set_yticklabels(sorted_names)
        ax2.set_xlabel('μ*')
        ax2.set_title('参数重要性 (μ*)')
        ax2.grid(True, alpha=0.3)
        
        # 参数分类
        ax3 = axes[1, 0]
        classification = results['classification']
        
        categories = list(classification.keys())
        counts = [len(classification[cat]) for cat in categories]
        colors = ['green', 'orange', 'gray']
        
        bars = ax3.bar(categories, counts, color=colors)
        ax3.set_ylabel('参数数量')
        ax3.set_title('参数分类')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(count), ha='center', va='bottom')
        
        # μ vs μ*
        ax4 = axes[1, 1]
        ax4.scatter(mu, mu_star, s=100, alpha=0.7)
        
        # 添加参数标签
        for i, name in enumerate(param_names):
            ax4.annotate(name, (mu[i], mu_star[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('μ (平均效应)')
        ax4.set_ylabel('μ* (平均绝对效应)')
        ax4.set_title('μ vs μ*')
        ax4.grid(True, alpha=0.3)
        
        # 添加对角线
        min_val = min(np.min(mu), np.min(mu_star))
        max_val = max(np.max(mu), np.max(mu_star))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comprehensive_comparison(self,
                                    results: Dict,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制综合敏感性分析比较
        
        参数:
            results: 综合分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        param_names = results['parameter_names']
        methods = results['methods_used']
        
        # 计算子图数量
        n_methods = len([m for m in methods if m in results])
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('综合敏感性分析比较', fontsize=16, fontweight='bold')
        
        # 方法比较雷达图
        ax1 = axes[0, 0]
        
        # 收集各方法的排序
        rankings_data = {}
        
        if 'local' in results:
            local_sens = np.abs(results['local']['sensitivities'])
            local_norm = local_sens / np.sum(local_sens)
            rankings_data['局部敏感性'] = local_norm
        
        if 'sobol' in results:
            sobol_total = results['sobol']['total_order_array']
            rankings_data['Sobol总效应'] = sobol_total
        
        if 'morris' in results:
            morris_mu_star = results['morris']['mu_star_array']
            morris_norm = morris_mu_star / np.sum(morris_mu_star)
            rankings_data['Morris μ*'] = morris_norm
        
        # 绘制比较条形图
        x = np.arange(len(param_names))
        width = 0.8 / len(rankings_data)
        
        for i, (method, values) in enumerate(rankings_data.items()):
            ax1.bar(x + i * width, values, width, label=method, alpha=0.8)
        
        ax1.set_xlabel('参数')
        ax1.set_ylabel('归一化重要性')
        ax1.set_title('方法比较')
        ax1.set_xticks(x + width * (len(rankings_data) - 1) / 2)
        ax1.set_xticklabels(param_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 综合排序
        ax2 = axes[0, 1]
        if 'comprehensive_ranking' in results:
            comp_ranking = results['comprehensive_ranking']
            ranking_names = comp_ranking['ranking_names']
            scores = comp_ranking['scores']
            
            bars = ax2.barh(range(len(ranking_names)), scores, color='purple', alpha=0.7)
            ax2.set_yticks(range(len(ranking_names)))
            ax2.set_yticklabels(ranking_names)
            ax2.set_xlabel('综合得分')
            ax2.set_title('综合重要性排序')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax2.text(score + 0.01 * max(scores), i, f'{score:.2f}', va='center')
        
        # 相关性矩阵
        ax3 = axes[1, 0]
        if len(rankings_data) > 1:
            # 计算方法间的相关性
            methods_list = list(rankings_data.keys())
            n_methods = len(methods_list)
            correlation_matrix = np.zeros((n_methods, n_methods))
            
            for i in range(n_methods):
                for j in range(n_methods):
                    corr, _ = stats.pearsonr(rankings_data[methods_list[i]], 
                                           rankings_data[methods_list[j]])
                    correlation_matrix[i, j] = corr
            
            im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax3.set_xticks(range(n_methods))
            ax3.set_yticks(range(n_methods))
            ax3.set_xticklabels(methods_list, rotation=45)
            ax3.set_yticklabels(methods_list)
            ax3.set_title('方法相关性')
            
            # 添加数值标签
            for i in range(n_methods):
                for j in range(n_methods):
                    text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax3)
        
        # 参数重要性分布
        ax4 = axes[1, 1]
        if rankings_data:
            # 计算每个参数在不同方法中的平均排名
            avg_importance = np.zeros(len(param_names))
            
            for values in rankings_data.values():
                avg_importance += values / len(rankings_data)
            
            # 饼图显示重要性分布
            top_n = min(5, len(param_names))
            sorted_indices = np.argsort(avg_importance)[::-1]
            
            top_names = [param_names[i] for i in sorted_indices[:top_n]]
            top_values = avg_importance[sorted_indices[:top_n]]
            
            if len(param_names) > top_n:
                top_names.append('其他')
                top_values = np.append(top_values, np.sum(avg_importance[sorted_indices[top_n:]]))
            
            ax4.pie(top_values, labels=top_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title('平均重要性分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """
    主函数 - 演示敏感性分析功能
    """
    # 创建示例模型
    def test_model(x):
        """
        测试模型：非线性函数
        y = x1^2 + 2*x2 + 0.5*x3*x4 + 0.1*x5
        """
        return x[0]**2 + 2*x[1] + 0.5*x[2]*x[3] + 0.1*x[4]
    
    # 参数设置
    parameter_names = ['温度', '降水', '海拔', '坡度', '辐射']
    parameter_bounds = [(-10, 10), (0, 2000), (0, 5000), (0, 45), (100, 400)]
    
    # 初始化分析器
    analyzer = SensitivityAnalyzer(model=test_model)
    visualizer = SensitivityVisualizer()
    
    print("开始敏感性分析演示...")
    
    # 1. 局部敏感性分析
    print("\n1. 局部敏感性分析")
    X_base = np.array([0, 1000, 2500, 20, 250])  # 基准点
    
    local_results = analyzer.local_sensitivity_analysis(
        X_base, parameter_names, perturbation_size=0.01
    )
    
    print("局部敏感性排序:")
    for i, (name, sens) in enumerate(zip(
        local_results['importance_ranking']['names'],
        local_results['importance_ranking']['values']
    )):
        print(f"  {i+1}. {name}: {sens:.4f}")
    
    # 2. Sobol全局敏感性分析
    print("\n2. Sobol全局敏感性分析")
    sobol_results = analyzer.sobol_sensitivity_analysis(
        parameter_bounds, parameter_names, n_samples=500
    )
    
    print("Sobol总效应排序:")
    for i, (name, total_idx) in enumerate(zip(
        sobol_results['importance_ranking']['names'],
        sobol_results['importance_ranking']['total_indices']
    )):
        print(f"  {i+1}. {name}: {total_idx:.4f}")
    
    # 3. Morris筛选
    print("\n3. Morris筛选")
    morris_results = analyzer.morris_screening(
        parameter_bounds, parameter_names, n_trajectories=50
    )
    
    print("Morris参数分类:")
    classification = morris_results['classification']
    for category, params in classification.items():
        print(f"  {category}: {params}")
    
    # 4. 综合分析
    print("\n4. 综合敏感性分析")
    comprehensive_results = analyzer.comprehensive_sensitivity_analysis(
        parameter_bounds, parameter_names, X_base, 
        methods=['local', 'sobol', 'morris']
    )
    
    print("综合重要性排序:")
    comp_ranking = comprehensive_results['comprehensive_ranking']
    for i, (name, score) in enumerate(zip(
        comp_ranking['ranking_names'],
        comp_ranking['scores']
    )):
        print(f"  {i+1}. {name}: {score:.2f}")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 局部敏感性
    fig1 = visualizer.plot_local_sensitivity(local_results)
    
    # Sobol指数
    fig2 = visualizer.plot_sobol_indices(sobol_results)
    
    # Morris筛选
    fig3 = visualizer.plot_morris_screening(morris_results)
    
    # 综合比较
    fig4 = visualizer.plot_comprehensive_comparison(comprehensive_results)
    
    plt.show()
    
    print("\n敏感性分析完成！")
    
    # 输出总结
    print("\n=== 分析总结 ===")
    print(f"最重要的参数: {comp_ranking['ranking_names'][0]}")
    print(f"最不重要的参数: {comp_ranking['ranking_names'][-1]}")
    print(f"分析方法数量: {comp_ranking['methods_combined']}")

if __name__ == "__main__":
    main()