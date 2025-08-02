#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础实验配置模块

该模块定义了冰川PINNs实验的基础配置类和常用配置模板。

作者: 冰川研究团队
日期: 2024
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """模型配置"""
    # 网络架构
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    activation: str = "tanh"
    initialization: str = "xavier_normal"
    dropout_rate: float = 0.0
    batch_normalization: bool = False
    
    # 输入输出维度
    input_dim: int = 4  # [x, y, z, t]
    output_dim: int = 3  # [velocity_x, velocity_y, thickness]
    
    # 物理约束
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    data_weight: float = 1.0
    
    # 不确定性量化
    enable_uncertainty: bool = False
    uncertainty_method: str = "ensemble"  # "ensemble", "bayesian", "monte_carlo"
    n_ensemble: int = 5


@dataclass
class TrainingConfig:
    """训练配置"""
    # 优化器设置
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine", "step", "exponential", "plateau"
    lr_decay_steps: int = 1000
    lr_decay_rate: float = 0.95
    min_lr: float = 1e-6
    
    # 训练参数
    max_epochs: int = 10000
    batch_size: int = 1024
    validation_split: float = 0.2
    early_stopping_patience: int = 500
    
    # 损失函数权重调度
    adaptive_weights: bool = True
    weight_update_frequency: int = 100
    
    # 梯度处理
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # 检查点
    save_frequency: int = 1000
    keep_best_only: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    # 数据源
    data_sources: List[str] = field(default_factory=lambda: [
        "hugonnet", "farinotti", "millan", "rgi"
    ])
    
    # 空间范围
    spatial_bounds: Dict[str, float] = field(default_factory=lambda: {
        "lon_min": 75.0, "lon_max": 105.0,
        "lat_min": 25.0, "lat_max": 40.0
    })
    
    # 时间范围
    temporal_bounds: Dict[str, str] = field(default_factory=lambda: {
        "start_date": "2000-01-01",
        "end_date": "2020-12-31"
    })
    
    # 数据预处理
    normalization: str = "standard"  # "standard", "minmax", "robust"
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    
    # 采样策略
    sampling_strategy: str = "adaptive"  # "uniform", "adaptive", "physics_guided"
    n_collocation_points: int = 10000
    n_boundary_points: int = 1000
    n_initial_points: int = 1000
    
    # 数据增强
    data_augmentation: bool = False
    noise_level: float = 0.01


@dataclass
class ValidationConfig:
    """验证配置"""
    # 验证策略
    validation_methods: List[str] = field(default_factory=lambda: [
        "temporal_holdout", "spatial_holdout", "glacier_wise"
    ])
    
    # 交叉验证
    k_fold: int = 5
    holdout_ratio: float = 0.2
    
    # 独立验证数据
    independent_datasets: List[str] = field(default_factory=lambda: [
        "grace", "icesat", "field_data"
    ])
    
    # 物理一致性检查
    physics_validation: bool = True
    conservation_tolerance: float = 1e-3
    
    # 性能指标
    metrics: List[str] = field(default_factory=lambda: [
        "rmse", "mae", "r2", "nse", "kge"
    ])


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 实验元信息
    name: str = "glacier_pinns_experiment"
    description: str = "青藏高原冰川PINNs建模实验"
    version: str = "1.0.0"
    author: str = "冰川研究团队"
    
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # 计算资源
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # 输出设置
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    save_predictions: bool = True
    save_visualizations: bool = True
    
    # 随机种子
    random_seed: int = 42
    
    def save(self, filepath: Union[str, Path]) -> None:
        """保存配置到文件"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """从文件加载配置"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建配置"""
        # 递归创建嵌套的dataclass
        if 'model' in data:
            data['model'] = ModelConfig(**data['model'])
        if 'training' in data:
            data['training'] = TrainingConfig(**data['training'])
        if 'data' in data:
            data['data'] = DataConfig(**data['data'])
        if 'validation' in data:
            data['validation'] = ValidationConfig(**data['validation'])
        
        return cls(**data)
    
    def update(self, updates: Dict[str, Any]) -> 'ExperimentConfig':
        """更新配置"""
        config_dict = asdict(self)
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, updates)
        return self.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        # 检查模型配置
        if len(self.model.hidden_layers) == 0:
            errors.append("模型必须至少有一个隐藏层")
        
        if self.model.input_dim <= 0 or self.model.output_dim <= 0:
            errors.append("输入输出维度必须大于0")
        
        # 检查训练配置
        if self.training.learning_rate <= 0:
            errors.append("学习率必须大于0")
        
        if self.training.max_epochs <= 0:
            errors.append("最大训练轮数必须大于0")
        
        if self.training.batch_size <= 0:
            errors.append("批次大小必须大于0")
        
        # 检查数据配置
        if not self.data.data_sources:
            errors.append("必须指定至少一个数据源")
        
        if self.data.n_collocation_points <= 0:
            errors.append("配点数量必须大于0")
        
        # 检查验证配置
        if not self.validation.validation_methods:
            errors.append("必须指定至少一种验证方法")
        
        if self.validation.k_fold <= 1:
            errors.append("交叉验证折数必须大于1")
        
        return errors


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Union[str, Path] = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def create_default_configs(self) -> Dict[str, ExperimentConfig]:
        """创建默认配置模板"""
        configs = {}
        
        # 基础配置
        configs['base'] = ExperimentConfig(
            name="base_experiment",
            description="基础冰川PINNs实验配置"
        )
        
        # 高精度配置
        configs['high_precision'] = ExperimentConfig(
            name="high_precision_experiment",
            description="高精度冰川建模实验",
            model=ModelConfig(
                hidden_layers=[256, 256, 256, 256, 256],
                dropout_rate=0.1,
                batch_normalization=True
            ),
            training=TrainingConfig(
                max_epochs=20000,
                batch_size=2048,
                learning_rate=5e-4,
                early_stopping_patience=1000
            ),
            data=DataConfig(
                n_collocation_points=50000,
                n_boundary_points=5000,
                sampling_strategy="physics_guided"
            )
        )
        
        # 快速原型配置
        configs['fast_prototype'] = ExperimentConfig(
            name="fast_prototype_experiment",
            description="快速原型验证实验",
            model=ModelConfig(
                hidden_layers=[64, 64, 64],
                physics_weight=0.5
            ),
            training=TrainingConfig(
                max_epochs=1000,
                batch_size=512,
                learning_rate=1e-2,
                early_stopping_patience=100
            ),
            data=DataConfig(
                n_collocation_points=1000,
                n_boundary_points=100,
                data_sources=["hugonnet"]
            )
        )
        
        # 不确定性量化配置
        configs['uncertainty'] = ExperimentConfig(
            name="uncertainty_quantification_experiment",
            description="不确定性量化实验",
            model=ModelConfig(
                enable_uncertainty=True,
                uncertainty_method="ensemble",
                n_ensemble=10
            ),
            training=TrainingConfig(
                max_epochs=15000,
                adaptive_weights=True
            )
        )
        
        # 多尺度配置
        configs['multiscale'] = ExperimentConfig(
            name="multiscale_experiment",
            description="多尺度建模实验",
            data=DataConfig(
                sampling_strategy="multiscale",
                n_collocation_points=20000,
                data_augmentation=True
            ),
            training=TrainingConfig(
                lr_scheduler="plateau",
                adaptive_weights=True,
                weight_update_frequency=50
            )
        )
        
        return configs
    
    def save_default_configs(self) -> None:
        """保存默认配置到文件"""
        configs = self.create_default_configs()
        
        for name, config in configs.items():
            config.save(self.config_dir / f"{name}_config.yaml")
        
        print(f"已保存 {len(configs)} 个默认配置到 {self.config_dir}")
    
    def list_configs(self) -> List[str]:
        """列出所有可用配置"""
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def load_config(self, name: str) -> ExperimentConfig:
        """加载指定配置"""
        yaml_path = self.config_dir / f"{name}.yaml"
        json_path = self.config_dir / f"{name}.json"
        
        if yaml_path.exists():
            return ExperimentConfig.load(yaml_path)
        elif json_path.exists():
            return ExperimentConfig.load(json_path)
        else:
            raise FileNotFoundError(f"配置文件不存在: {name}")
    
    def create_experiment_config(self, 
                               base_config: str = "base",
                               modifications: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        """基于基础配置创建新的实验配置"""
        config = self.load_config(base_config)
        
        if modifications:
            config = config.update(modifications)
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        return config


def create_config_manager(config_dir: str = "./configs") -> ConfigManager:
    """创建配置管理器的工厂函数"""
    return ConfigManager(config_dir)


if __name__ == "__main__":
    # 示例用法
    
    # 创建配置管理器
    manager = create_config_manager("./experiment_configs")
    
    # 生成默认配置
    print("创建默认配置...")
    manager.save_default_configs()
    
    # 列出所有配置
    print(f"\n可用配置: {manager.list_configs()}")
    
    # 加载并修改配置
    print("\n加载基础配置并进行修改...")
    modifications = {
        "name": "custom_experiment",
        "model": {
            "hidden_layers": [128, 256, 256, 128],
            "activation": "relu"
        },
        "training": {
            "learning_rate": 5e-4,
            "max_epochs": 5000
        }
    }
    
    custom_config = manager.create_experiment_config(
        base_config="base_config",
        modifications=modifications
    )
    
    # 保存自定义配置
    custom_config.save("./experiment_configs/custom_config.yaml")
    print("自定义配置已保存")
    
    # 验证配置
    errors = custom_config.validate()
    if errors:
        print(f"配置验证错误: {errors}")
    else:
        print("配置验证通过")
    
    # 显示配置摘要
    print(f"\n实验配置摘要:")
    print(f"名称: {custom_config.name}")
    print(f"模型层数: {len(custom_config.model.hidden_layers)}")
    print(f"隐藏单元: {custom_config.model.hidden_layers}")
    print(f"学习率: {custom_config.training.learning_rate}")
    print(f"最大轮数: {custom_config.training.max_epochs}")
    print(f"数据源: {custom_config.data.data_sources}")
    print(f"验证方法: {custom_config.validation.validation_methods}")