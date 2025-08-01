#!/usr/bin/env python3
"""
验证测试模块

提供完整的模型验证和测试功能，包括：
- 交叉验证和独立验证
- 性能指标评估
- 物理一致性验证
- 不确定性评估
- 模型比较和基准测试

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

# 交叉验证
from .cross_validation import (
    # 基础交叉验证
    KFoldValidator,
    StratifiedKFoldValidator,
    TimeSeriesSplitValidator,
    SpatialKFoldValidator,
    
    # 高级交叉验证
    NestedCrossValidator,
    GroupKFoldValidator,
    LeaveOneOutValidator,
    BlockCrossValidator,
    
    # 交叉验证管理
    CrossValidationManager,
    ValidationResultAnalyzer
)

# 独立验证
from .independent_validation import (
    # 独立数据验证
    IndependentValidator,
    HoldoutValidator,
    TemporalValidator,
    SpatialValidator,
    
    # 外部数据验证
    ExternalDataValidator,
    FieldDataValidator,
    SatelliteDataValidator,
    ModelComparisonValidator,
    
    # 验证管理
    IndependentValidationManager,
    ValidationReportGenerator
)

# 性能指标
from .performance_metrics import (
    # 基础指标
    RegressionMetrics,
    ClassificationMetrics,
    TimeSeriesMetrics,
    SpatialMetrics,
    
    # 高级指标
    PhysicsAwareMetrics,
    UncertaintyMetrics,
    RobustnessMetrics,
    EfficiencyMetrics,
    
    # 指标管理
    MetricsCalculator,
    MetricsComparator,
    BenchmarkManager
)

# 物理验证
from .physics_validation import (
    # 物理一致性检查
    PhysicsConsistencyChecker,
    ConservationLawValidator,
    BoundaryConditionValidator,
    SymmetryValidator,
    
    # 物理约束验证
    ConstraintValidator,
    PDEResidualAnalyzer,
    EnergyConservationChecker,
    MassConservationChecker,
    
    # 冰川物理验证
    GlacierPhysicsValidator,
    FlowLawValidator,
    MassBalanceValidator,
    ThermodynamicsValidator,
    
    # 验证管理
    PhysicsValidationManager,
    PhysicsReportGenerator
)

__all__ = [
    # 交叉验证
    'KFoldValidator',
    'StratifiedKFoldValidator',
    'TimeSeriesSplitValidator',
    'SpatialKFoldValidator',
    'NestedCrossValidator',
    'GroupKFoldValidator',
    'LeaveOneOutValidator',
    'BlockCrossValidator',
    'CrossValidationManager',
    'ValidationResultAnalyzer',
    
    # 独立验证
    'IndependentValidator',
    'HoldoutValidator',
    'TemporalValidator',
    'SpatialValidator',
    'ExternalDataValidator',
    'FieldDataValidator',
    'SatelliteDataValidator',
    'ModelComparisonValidator',
    'IndependentValidationManager',
    'ValidationReportGenerator',
    
    # 性能指标
    'RegressionMetrics',
    'ClassificationMetrics',
    'TimeSeriesMetrics',
    'SpatialMetrics',
    'PhysicsAwareMetrics',
    'UncertaintyMetrics',
    'RobustnessMetrics',
    'EfficiencyMetrics',
    'MetricsCalculator',
    'MetricsComparator',
    'BenchmarkManager',
    
    # 物理验证
    'PhysicsConsistencyChecker',
    'ConservationLawValidator',
    'BoundaryConditionValidator',
    'SymmetryValidator',
    'ConstraintValidator',
    'PDEResidualAnalyzer',
    'EnergyConservationChecker',
    'MassConservationChecker',
    'GlacierPhysicsValidator',
    'FlowLawValidator',
    'MassBalanceValidator',
    'ThermodynamicsValidator',
    'PhysicsValidationManager',
    'PhysicsReportGenerator',
    
    # 便捷函数
    'create_validation_suite',
    'run_comprehensive_validation',
    'compare_models',
    'generate_validation_report'
]

# 版本信息
__version__ = '1.0.0'

# 模块文档
__doc__ = """
验证测试模块

本模块提供了完整的模型验证和测试功能，专门针对PINNs模型的特点，
包括物理一致性验证、不确定性评估和多维度性能评估。

主要组件：

1. 交叉验证 (cross_validation):
   - KFoldValidator: K折交叉验证
   - StratifiedKFoldValidator: 分层K折验证
   - TimeSeriesSplitValidator: 时间序列分割验证
   - SpatialKFoldValidator: 空间K折验证
   - NestedCrossValidator: 嵌套交叉验证
   - GroupKFoldValidator: 分组K折验证
   - LeaveOneOutValidator: 留一法验证
   - BlockCrossValidator: 块交叉验证

2. 独立验证 (independent_validation):
   - IndependentValidator: 独立数据集验证
   - HoldoutValidator: 留出法验证
   - TemporalValidator: 时间外推验证
   - SpatialValidator: 空间外推验证
   - ExternalDataValidator: 外部数据验证
   - FieldDataValidator: 野外观测数据验证
   - SatelliteDataValidator: 卫星数据验证
   - ModelComparisonValidator: 模型对比验证

3. 性能指标 (performance_metrics):
   - RegressionMetrics: 回归指标（MSE, MAE, R²等）
   - ClassificationMetrics: 分类指标（准确率、精确率等）
   - TimeSeriesMetrics: 时间序列指标（趋势、季节性等）
   - SpatialMetrics: 空间指标（空间相关性、梯度等）
   - PhysicsAwareMetrics: 物理感知指标
   - UncertaintyMetrics: 不确定性指标
   - RobustnessMetrics: 鲁棒性指标
   - EfficiencyMetrics: 效率指标

4. 物理验证 (physics_validation):
   - PhysicsConsistencyChecker: 物理一致性检查
   - ConservationLawValidator: 守恒定律验证
   - BoundaryConditionValidator: 边界条件验证
   - SymmetryValidator: 对称性验证
   - ConstraintValidator: 约束条件验证
   - PDEResidualAnalyzer: PDE残差分析
   - GlacierPhysicsValidator: 冰川物理验证
   - FlowLawValidator: 流动定律验证
   - MassBalanceValidator: 质量平衡验证
   - ThermodynamicsValidator: 热力学验证

验证策略：

1. 多层次验证:
   - 数据层面：数据质量和一致性
   - 模型层面：架构和参数合理性
   - 物理层面：物理定律遵循程度
   - 应用层面：实际问题解决能力

2. 多维度评估:
   - 准确性：预测精度和误差分析
   - 鲁棒性：对噪声和扰动的敏感性
   - 泛化性：跨域和跨时间的适应性
   - 效率性：计算成本和时间复杂度
   - 可解释性：模型行为的可理解性

3. 多尺度验证:
   - 点尺度：单点预测精度
   - 局部尺度：小区域空间模式
   - 区域尺度：大范围空间分布
   - 全局尺度：整体系统行为

使用示例：

```python
import torch
import numpy as np
from validation_testing import (
    create_validation_suite,
    run_comprehensive_validation,
    compare_models
)

# 1. 创建验证套件
validation_suite = create_validation_suite(
    validation_types=['cross_validation', 'independent', 'physics'],
    cv_folds=5,
    test_ratio=0.2,
    physics_checks=['conservation', 'boundary', 'symmetry']
)

# 2. 准备数据和模型
from model_architecture import create_glacier_pinn

model = create_glacier_pinn(
    input_dim=3,
    output_dim=4,
    hidden_dims=[100, 100, 100]
)

# 训练数据
X_train = torch.randn(1000, 3)
y_train = torch.randn(1000, 4)

# 测试数据
X_test = torch.randn(200, 3)
y_test = torch.randn(200, 4)

# 物理域数据
X_physics = torch.randn(500, 3)

# 3. 运行综合验证
validation_results = run_comprehensive_validation(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_physics=X_physics,
    validation_suite=validation_suite
)

# 4. 查看验证结果
print("交叉验证结果:")
print(f"平均MSE: {validation_results['cross_validation']['mse_mean']:.6f}")
print(f"MSE标准差: {validation_results['cross_validation']['mse_std']:.6f}")

print("\n独立验证结果:")
print(f"测试MSE: {validation_results['independent']['test_mse']:.6f}")
print(f"R²分数: {validation_results['independent']['r2_score']:.4f}")

print("\n物理验证结果:")
print(f"质量守恒误差: {validation_results['physics']['mass_conservation_error']:.6f}")
print(f"边界条件满足度: {validation_results['physics']['boundary_satisfaction']:.4f}")

# 5. 模型比较
from model_architecture import create_advanced_glacier_pinn

# 创建另一个模型进行比较
advanced_model = create_advanced_glacier_pinn(
    input_dim=3,
    output_dim=4,
    use_multiscale=True,
    use_attention=True
)

# 比较两个模型
comparison_results = compare_models(
    models={'basic': model, 'advanced': advanced_model},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    validation_suite=validation_suite
)

print("\n模型比较结果:")
for model_name, results in comparison_results.items():
    print(f"{model_name}模型:")
    print(f"  测试MSE: {results['test_mse']:.6f}")
    print(f"  物理一致性: {results['physics_consistency']:.4f}")
    print(f"  计算效率: {results['efficiency_score']:.4f}")
```

高级验证示例：

```python
from validation_testing.cross_validation import (
    SpatialKFoldValidator,
    TimeSeriesSplitValidator
)
from validation_testing.physics_validation import (
    GlacierPhysicsValidator,
    MassBalanceValidator
)
from validation_testing.performance_metrics import (
    PhysicsAwareMetrics,
    UncertaintyMetrics
)

# 1. 空间交叉验证
spatial_cv = SpatialKFoldValidator(
    n_splits=5,
    spatial_column=['x', 'y'],
    buffer_distance=1000  # 1km缓冲区
)

spatial_scores = spatial_cv.validate(
    model=model,
    X=X_train,
    y=y_train,
    scoring=['mse', 'mae', 'r2']
)

print("空间交叉验证结果:")
for metric, scores in spatial_scores.items():
    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# 2. 时间序列验证
time_cv = TimeSeriesSplitValidator(
    n_splits=5,
    test_size=0.2,
    gap=30  # 30天间隔
)

# 假设有时间信息
time_indices = np.arange(len(X_train))
time_scores = time_cv.validate(
    model=model,
    X=X_train,
    y=y_train,
    time_indices=time_indices
)

print("\n时间序列验证结果:")
for metric, scores in time_scores.items():
    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# 3. 冰川物理验证
glacier_validator = GlacierPhysicsValidator(
    physics_laws=['mass_balance', 'momentum_balance', 'glen_flow'],
    tolerance=1e-3
)

physics_results = glacier_validator.validate(
    model=model,
    X_physics=X_physics,
    domain_bounds={'x': [0, 10000], 'y': [0, 5000], 't': [0, 365]}
)

print("\n冰川物理验证结果:")
for law, result in physics_results.items():
    print(f"{law}: 满足度 = {result['satisfaction']:.4f}, 误差 = {result['error']:.6f}")

# 4. 质量平衡验证
mass_balance_validator = MassBalanceValidator(
    accumulation_rate=2.0,  # m/year
    ablation_rate=3.0,      # m/year
    tolerance=0.1
)

mass_balance_result = mass_balance_validator.validate(
    model=model,
    X_surface=X_physics[:100],  # 表面点
    time_span=365  # 一年
)

print(f"\n质量平衡验证: 误差 = {mass_balance_result['error']:.4f} m/year")

# 5. 物理感知指标
physics_metrics = PhysicsAwareMetrics(
    physics_weight=0.3,
    conservation_weight=0.4,
    boundary_weight=0.3
)

y_pred = model(X_test)
physics_score = physics_metrics.compute(
    y_true=y_test,
    y_pred=y_pred,
    X=X_test,
    model=model
)

print(f"\n物理感知综合评分: {physics_score:.4f}")

# 6. 不确定性指标
from model_architecture.uncertainty_quantification import MCDropoutPINN

mc_model = MCDropoutPINN(base_model=model, n_samples=100)
uncertainty_metrics = UncertaintyMetrics()

y_pred_samples = mc_model.predict_samples(X_test)
uncertainty_scores = uncertainty_metrics.compute(
    y_true=y_test,
    y_pred_samples=y_pred_samples
)

print("\n不确定性指标:")
for metric, score in uncertainty_scores.items():
    print(f"{metric}: {score:.4f}")
```

验证报告生成：

```python
from validation_testing import generate_validation_report

# 生成综合验证报告
report = generate_validation_report(
    validation_results=validation_results,
    model_info={'name': 'Glacier PINN', 'version': '1.0'},
    data_info={'source': 'Dussaillant 2025', 'region': 'Tibetan Plateau'},
    output_format='html',
    include_plots=True,
    save_path='validation_report.html'
)

print(f"验证报告已保存到: {report['save_path']}")
print(f"报告包含 {len(report['sections'])} 个部分")
```

设计原则：

1. 全面性: 覆盖模型的各个方面和应用场景
2. 层次性: 从基础到高级，从局部到全局
3. 专业性: 针对PINNs和冰川物理的特殊需求
4. 可扩展性: 易于添加新的验证方法和指标
5. 自动化: 减少人工干预，提高验证效率

验证最佳实践：

1. 验证策略:
   - 多种验证方法结合使用
   - 注重物理一致性验证
   - 考虑不确定性和鲁棒性
   - 进行长期稳定性测试

2. 数据准备:
   - 确保验证数据的独立性
   - 保持数据分布的代表性
   - 考虑时空相关性
   - 包含边界情况和极端情况

3. 结果解释:
   - 综合考虑多个指标
   - 分析误差的空间和时间分布
   - 识别模型的优势和局限性
   - 提供改进建议

4. 持续改进:
   - 建立验证基准
   - 跟踪性能变化
   - 更新验证标准
   - 积累验证经验

注意事项：

1. 避免数据泄露和过拟合
2. 考虑计算资源和时间成本
3. 注意验证结果的统计显著性
4. 重视物理合理性而非仅仅数值精度
5. 建立合理的验证阈值和标准
"""

# 便捷函数
def create_validation_suite(
    validation_types: list = ['cross_validation', 'independent', 'physics'],
    cv_folds: int = 5,
    test_ratio: float = 0.2,
    physics_checks: list = ['conservation', 'boundary'],
    **kwargs
):
    """
    创建验证套件
    
    Args:
        validation_types: 验证类型列表
        cv_folds: 交叉验证折数
        test_ratio: 测试集比例
        physics_checks: 物理检查列表
        **kwargs: 其他参数
        
    Returns:
        ValidationSuite: 验证套件
    """
    class ValidationSuite:
        def __init__(self):
            self.validators = {}
            self.metrics = {}
            self.config = kwargs
        
        def add_validator(self, name, validator):
            self.validators[name] = validator
        
        def add_metrics(self, name, metrics):
            self.metrics[name] = metrics
    
    suite = ValidationSuite()
    
    # 添加交叉验证
    if 'cross_validation' in validation_types:
        suite.add_validator(
            'cross_validation',
            CrossValidationManager(
                cv_folds=cv_folds,
                **kwargs.get('cv_params', {})
            )
        )
    
    # 添加独立验证
    if 'independent' in validation_types:
        suite.add_validator(
            'independent',
            IndependentValidationManager(
                test_ratio=test_ratio,
                **kwargs.get('independent_params', {})
            )
        )
    
    # 添加物理验证
    if 'physics' in validation_types:
        suite.add_validator(
            'physics',
            PhysicsValidationManager(
                physics_checks=physics_checks,
                **kwargs.get('physics_params', {})
            )
        )
    
    # 添加性能指标
    suite.add_metrics('regression', RegressionMetrics())
    suite.add_metrics('physics', PhysicsAwareMetrics())
    suite.add_metrics('uncertainty', UncertaintyMetrics())
    
    return suite

def run_comprehensive_validation(
    model,
    X_train, y_train,
    X_test=None, y_test=None,
    X_physics=None,
    validation_suite=None,
    **kwargs
):
    """
    运行综合验证
    
    Args:
        model: 待验证的模型
        X_train: 训练输入数据
        y_train: 训练输出数据
        X_test: 测试输入数据
        y_test: 测试输出数据
        X_physics: 物理域数据
        validation_suite: 验证套件
        **kwargs: 其他参数
        
    Returns:
        dict: 验证结果
    """
    if validation_suite is None:
        validation_suite = create_validation_suite(**kwargs)
    
    results = {}
    
    # 运行各种验证
    for name, validator in validation_suite.validators.items():
        if name == 'cross_validation':
            results[name] = validator.validate(
                model=model,
                X=X_train,
                y=y_train
            )
        elif name == 'independent' and X_test is not None:
            results[name] = validator.validate(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
        elif name == 'physics' and X_physics is not None:
            results[name] = validator.validate(
                model=model,
                X_physics=X_physics
            )
    
    return results

def compare_models(
    models: dict,
    X_train, y_train,
    X_test=None, y_test=None,
    validation_suite=None,
    **kwargs
):
    """
    比较多个模型
    
    Args:
        models: 模型字典 {name: model}
        X_train: 训练输入数据
        y_train: 训练输出数据
        X_test: 测试输入数据
        y_test: 测试输出数据
        validation_suite: 验证套件
        **kwargs: 其他参数
        
    Returns:
        dict: 比较结果
    """
    comparison_results = {}
    
    for name, model in models.items():
        print(f"验证模型: {name}")
        results = run_comprehensive_validation(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            validation_suite=validation_suite,
            **kwargs
        )
        comparison_results[name] = results
    
    return comparison_results

def generate_validation_report(
    validation_results: dict,
    model_info: dict = None,
    data_info: dict = None,
    output_format: str = 'html',
    include_plots: bool = True,
    save_path: str = None,
    **kwargs
):
    """
    生成验证报告
    
    Args:
        validation_results: 验证结果
        model_info: 模型信息
        data_info: 数据信息
        output_format: 输出格式 ('html', 'pdf', 'markdown')
        include_plots: 是否包含图表
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        dict: 报告信息
    """
    report_generator = ValidationReportGenerator(
        output_format=output_format,
        include_plots=include_plots,
        **kwargs
    )
    
    report = report_generator.generate(
        validation_results=validation_results,
        model_info=model_info,
        data_info=data_info,
        save_path=save_path
    )
    
    return report

# 默认配置
DEFAULT_VALIDATION_CONFIG = {
    'cv_folds': 5,
    'test_ratio': 0.2,
    'physics_tolerance': 1e-3,
    'uncertainty_samples': 100,
    'spatial_buffer': 1000,
    'temporal_gap': 30,
    'metrics': ['mse', 'mae', 'r2', 'physics_consistency']
}

# 验证阈值
VALIDATION_THRESHOLDS = {
    'mse_threshold': 1e-3,
    'r2_threshold': 0.8,
    'physics_consistency_threshold': 0.9,
    'mass_conservation_threshold': 0.1,
    'boundary_satisfaction_threshold': 0.95,
    'uncertainty_coverage_threshold': 0.9
}