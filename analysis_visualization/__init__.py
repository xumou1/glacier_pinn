#!/usr/bin/env python3
"""
分析可视化模块

提供完整的分析和可视化功能，包括：
- 空间分析和可视化
- 时间序列分析
- 不确定性分析
- 交互式可视化
- 结果解释和报告生成

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

# 空间分析
from .spatial_analysis import (
    # 空间统计分析
    SpatialStatistics,
    SpatialCorrelation,
    SpatialClustering,
    SpatialInterpolation,
    
    # 空间模式分析
    SpatialPatternAnalyzer,
    HotspotAnalyzer,
    SpatialTrendAnalyzer,
    SpatialAnomalyDetector,
    
    # 空间可视化
    SpatialVisualizer,
    MapVisualizer,
    ContourPlotter,
    VectorFieldPlotter,
    
    # 冰川空间分析
    GlacierSpatialAnalyzer,
    FlowFieldAnalyzer,
    ElevationAnalyzer,
    ThicknessAnalyzer
)

# 时间分析
from .temporal_analysis import (
    # 时间序列分析
    TimeSeriesAnalyzer,
    TrendAnalyzer,
    SeasonalityAnalyzer,
    ChangePointDetector,
    
    # 时间模式分析
    TemporalPatternAnalyzer,
    CyclicPatternDetector,
    TemporalAnomalyDetector,
    ForecastAnalyzer,
    
    # 时间可视化
    TimeSeriesVisualizer,
    TrendPlotter,
    SeasonalPlotter,
    HeatmapPlotter,
    
    # 冰川时间分析
    GlacierTemporalAnalyzer,
    MassBalanceAnalyzer,
    VelocityTrendAnalyzer,
    ClimateResponseAnalyzer
)

# 不确定性分析
from .uncertainty_analysis import (
    # 不确定性量化
    UncertaintyQuantifier,
    SensitivityAnalyzer,
    VarianceDecomposer,
    UncertaintyPropagator,
    
    # 不确定性可视化
    UncertaintyVisualizer,
    ConfidenceIntervalPlotter,
    UncertaintyMapPlotter,
    SensitivityPlotter,
    
    # 可靠性分析
    ReliabilityAnalyzer,
    ConfidenceAssessment,
    PredictionIntervalAnalyzer,
    RiskAssessment,
    
    # 冰川不确定性分析
    GlacierUncertaintyAnalyzer,
    ProjectionUncertaintyAnalyzer,
    ParameterUncertaintyAnalyzer,
    ModelUncertaintyAnalyzer
)

# 交互式可视化
from .interactive_visualization import (
    # 交互式组件
    InteractivePlotter,
    DashboardCreator,
    WidgetManager,
    AnimationCreator,
    
    # 3D可视化
    ThreeDVisualizer,
    VolumeRenderer,
    SurfaceRenderer,
    ParticleRenderer,
    
    # Web可视化
    WebVisualizer,
    MapboxVisualizer,
    PlotlyVisualizer,
    BokehVisualizer,
    
    # 冰川交互式可视化
    GlacierDashboard,
    GlacierAnimator,
    Glacier3DVisualizer,
    GlacierWebApp
)

__all__ = [
    # 空间分析
    'SpatialStatistics',
    'SpatialCorrelation',
    'SpatialClustering',
    'SpatialInterpolation',
    'SpatialPatternAnalyzer',
    'HotspotAnalyzer',
    'SpatialTrendAnalyzer',
    'SpatialAnomalyDetector',
    'SpatialVisualizer',
    'MapVisualizer',
    'ContourPlotter',
    'VectorFieldPlotter',
    'GlacierSpatialAnalyzer',
    'FlowFieldAnalyzer',
    'ElevationAnalyzer',
    'ThicknessAnalyzer',
    
    # 时间分析
    'TimeSeriesAnalyzer',
    'TrendAnalyzer',
    'SeasonalityAnalyzer',
    'ChangePointDetector',
    'TemporalPatternAnalyzer',
    'CyclicPatternDetector',
    'TemporalAnomalyDetector',
    'ForecastAnalyzer',
    'TimeSeriesVisualizer',
    'TrendPlotter',
    'SeasonalPlotter',
    'HeatmapPlotter',
    'GlacierTemporalAnalyzer',
    'MassBalanceAnalyzer',
    'VelocityTrendAnalyzer',
    'ClimateResponseAnalyzer',
    
    # 不确定性分析
    'UncertaintyQuantifier',
    'SensitivityAnalyzer',
    'VarianceDecomposer',
    'UncertaintyPropagator',
    'UncertaintyVisualizer',
    'ConfidenceIntervalPlotter',
    'UncertaintyMapPlotter',
    'SensitivityPlotter',
    'ReliabilityAnalyzer',
    'ConfidenceAssessment',
    'PredictionIntervalAnalyzer',
    'RiskAssessment',
    'GlacierUncertaintyAnalyzer',
    'ProjectionUncertaintyAnalyzer',
    'ParameterUncertaintyAnalyzer',
    'ModelUncertaintyAnalyzer',
    
    # 交互式可视化
    'InteractivePlotter',
    'DashboardCreator',
    'WidgetManager',
    'AnimationCreator',
    'ThreeDVisualizer',
    'VolumeRenderer',
    'SurfaceRenderer',
    'ParticleRenderer',
    'WebVisualizer',
    'MapboxVisualizer',
    'PlotlyVisualizer',
    'BokehVisualizer',
    'GlacierDashboard',
    'GlacierAnimator',
    'Glacier3DVisualizer',
    'GlacierWebApp',
    
    # 便捷函数
    'create_analysis_suite',
    'generate_comprehensive_report',
    'create_interactive_dashboard',
    'analyze_glacier_dynamics'
]

# 版本信息
__version__ = '1.0.0'

# 模块文档
__doc__ = """
分析可视化模块

本模块提供了完整的分析和可视化功能，专门针对冰川PINNs模型的结果分析，
包括空间模式、时间演化、不确定性评估和交互式探索。

主要组件：

1. 空间分析 (spatial_analysis):
   - SpatialStatistics: 空间统计分析（Moran's I、Geary's C等）
   - SpatialCorrelation: 空间相关性分析
   - SpatialClustering: 空间聚类分析（K-means、DBSCAN等）
   - SpatialInterpolation: 空间插值（克里金、IDW等）
   - SpatialPatternAnalyzer: 空间模式识别
   - HotspotAnalyzer: 热点分析（Getis-Ord Gi*）
   - SpatialTrendAnalyzer: 空间趋势分析
   - SpatialAnomalyDetector: 空间异常检测
   - GlacierSpatialAnalyzer: 冰川空间分析集成

2. 时间分析 (temporal_analysis):
   - TimeSeriesAnalyzer: 时间序列分析（ARIMA、STL等）
   - TrendAnalyzer: 趋势分析（Mann-Kendall、Sen's slope）
   - SeasonalityAnalyzer: 季节性分析（FFT、小波分析）
   - ChangePointDetector: 变点检测（CUSUM、PELT）
   - TemporalPatternAnalyzer: 时间模式识别
   - CyclicPatternDetector: 周期模式检测
   - TemporalAnomalyDetector: 时间异常检测
   - ForecastAnalyzer: 预测分析
   - GlacierTemporalAnalyzer: 冰川时间分析集成

3. 不确定性分析 (uncertainty_analysis):
   - UncertaintyQuantifier: 不确定性量化（贝叶斯、频率学）
   - SensitivityAnalyzer: 敏感性分析（Sobol指数、Morris方法）
   - VarianceDecomposer: 方差分解
   - UncertaintyPropagator: 不确定性传播
   - ReliabilityAnalyzer: 可靠性分析
   - ConfidenceAssessment: 置信度评估
   - PredictionIntervalAnalyzer: 预测区间分析
   - RiskAssessment: 风险评估
   - GlacierUncertaintyAnalyzer: 冰川不确定性分析集成

4. 交互式可视化 (interactive_visualization):
   - InteractivePlotter: 交互式绘图（matplotlib、plotly）
   - DashboardCreator: 仪表板创建（Dash、Streamlit）
   - WidgetManager: 交互组件管理
   - AnimationCreator: 动画创建
   - ThreeDVisualizer: 3D可视化（mayavi、plotly）
   - VolumeRenderer: 体积渲染
   - SurfaceRenderer: 表面渲染
   - WebVisualizer: Web可视化
   - GlacierDashboard: 冰川专用仪表板

分析功能：

1. 空间分析:
   - 冰川几何特征分析
   - 流场模式识别
   - 厚度分布分析
   - 高程变化模式
   - 空间相关性评估
   - 热点和冷点识别

2. 时间分析:
   - 质量平衡趋势
   - 速度变化分析
   - 季节性模式识别
   - 气候响应分析
   - 长期演化预测
   - 极端事件检测

3. 不确定性分析:
   - 参数不确定性评估
   - 模型结构不确定性
   - 预测不确定性量化
   - 敏感性排序
   - 风险概率评估
   - 决策支持分析

4. 交互式探索:
   - 多维数据探索
   - 实时参数调整
   - 情景分析
   - 比较分析
   - 动态可视化
   - 协作分析平台

使用示例：

```python
import numpy as np
import torch
from analysis_visualization import (
    create_analysis_suite,
    generate_comprehensive_report,
    create_interactive_dashboard
)

# 1. 创建分析套件
analysis_suite = create_analysis_suite(
    analysis_types=['spatial', 'temporal', 'uncertainty'],
    spatial_methods=['correlation', 'clustering', 'hotspot'],
    temporal_methods=['trend', 'seasonality', 'changepoint'],
    uncertainty_methods=['sensitivity', 'propagation', 'reliability']
)

# 2. 准备数据
# 模型预测结果
y_pred = torch.randn(1000, 4)  # u, v, h, p
X_coords = torch.randn(1000, 3)  # x, y, t

# 不确定性信息
y_uncertainty = torch.randn(1000, 4)

# 时间序列数据
time_series_data = {
    'time': np.arange('2000-01', '2021-01', dtype='datetime64[M]'),
    'mass_balance': np.random.randn(252),
    'velocity': np.random.randn(252),
    'thickness': np.random.randn(252)
}

# 3. 运行空间分析
from analysis_visualization.spatial_analysis import GlacierSpatialAnalyzer

spatial_analyzer = GlacierSpatialAnalyzer()
spatial_results = spatial_analyzer.analyze(
    predictions=y_pred,
    coordinates=X_coords[:, :2],  # x, y
    variables=['velocity_u', 'velocity_v', 'thickness', 'pressure']
)

print("空间分析结果:")
print(f"空间自相关性: {spatial_results['spatial_autocorr']:.4f}")
print(f"聚类数量: {spatial_results['n_clusters']}")
print(f"热点区域: {len(spatial_results['hotspots'])} 个")

# 4. 运行时间分析
from analysis_visualization.temporal_analysis import GlacierTemporalAnalyzer

temporal_analyzer = GlacierTemporalAnalyzer()
temporal_results = temporal_analyzer.analyze(
    time_series=time_series_data,
    variables=['mass_balance', 'velocity', 'thickness']
)

print("\n时间分析结果:")
for var, result in temporal_results.items():
    print(f"{var}:")
    print(f"  趋势: {result['trend']['slope']:.4f} /年")
    print(f"  季节性强度: {result['seasonality']['strength']:.4f}")
    print(f"  变点数量: {len(result['changepoints'])}")

# 5. 运行不确定性分析
from analysis_visualization.uncertainty_analysis import GlacierUncertaintyAnalyzer

uncertainty_analyzer = GlacierUncertaintyAnalyzer()
uncertainty_results = uncertainty_analyzer.analyze(
    predictions=y_pred,
    uncertainties=y_uncertainty,
    coordinates=X_coords
)

print("\n不确定性分析结果:")
print(f"平均不确定性: {uncertainty_results['mean_uncertainty']:.4f}")
print(f"不确定性覆盖率: {uncertainty_results['coverage_probability']:.4f}")
print(f"可靠性评分: {uncertainty_results['reliability_score']:.4f}")

# 6. 创建交互式仪表板
dashboard = create_interactive_dashboard(
    data={
        'predictions': y_pred,
        'coordinates': X_coords,
        'uncertainties': y_uncertainty,
        'time_series': time_series_data
    },
    dashboard_type='glacier',
    include_3d=True,
    include_animation=True
)

print(f"\n仪表板已创建: {dashboard.url}")

# 7. 生成综合报告
report = generate_comprehensive_report(
    spatial_results=spatial_results,
    temporal_results=temporal_results,
    uncertainty_results=uncertainty_results,
    output_format='html',
    include_interactive=True,
    save_path='analysis_report.html'
)

print(f"\n分析报告已保存: {report['save_path']}")
```

高级分析示例：

```python
# 1. 详细空间分析
from analysis_visualization.spatial_analysis import (
    SpatialCorrelation,
    HotspotAnalyzer,
    FlowFieldAnalyzer
)

# 空间相关性分析
spatial_corr = SpatialCorrelation(method='moran')
corr_results = spatial_corr.analyze(
    data=y_pred[:, 2],  # 厚度数据
    coordinates=X_coords[:, :2],
    weights_type='queen'
)

print(f"Moran's I: {corr_results['morans_i']:.4f}")
print(f"p值: {corr_results['p_value']:.4f}")

# 热点分析
hotspot_analyzer = HotspotAnalyzer()
hotspots = hotspot_analyzer.detect(
    data=y_pred[:, :2],  # 速度数据
    coordinates=X_coords[:, :2],
    method='getis_ord'
)

print(f"热点数量: {len(hotspots['hot_spots'])}")
print(f"冷点数量: {len(hotspots['cold_spots'])}")

# 流场分析
flow_analyzer = FlowFieldAnalyzer()
flow_results = flow_analyzer.analyze(
    velocity_u=y_pred[:, 0],
    velocity_v=y_pred[:, 1],
    coordinates=X_coords[:, :2]
)

print(f"平均流速: {flow_results['mean_velocity']:.4f} m/年")
print(f"流向一致性: {flow_results['flow_consistency']:.4f}")
print(f"涡度: {flow_results['vorticity_mean']:.6f}")

# 2. 详细时间分析
from analysis_visualization.temporal_analysis import (
    TrendAnalyzer,
    SeasonalityAnalyzer,
    ChangePointDetector
)

# 趋势分析
trend_analyzer = TrendAnalyzer(method='mann_kendall')
trend_results = trend_analyzer.analyze(
    data=time_series_data['mass_balance'],
    time=time_series_data['time']
)

print(f"\n趋势显著性: {trend_results['significant']}")
print(f"趋势斜率: {trend_results['slope']:.4f} m/年")
print(f"置信区间: [{trend_results['slope_ci'][0]:.4f}, {trend_results['slope_ci'][1]:.4f}]")

# 季节性分析
seasonality_analyzer = SeasonalityAnalyzer(method='stl')
seasonality_results = seasonality_analyzer.analyze(
    data=time_series_data['velocity'],
    time=time_series_data['time'],
    period=12  # 月度数据
)

print(f"季节性强度: {seasonality_results['seasonal_strength']:.4f}")
print(f"趋势强度: {seasonality_results['trend_strength']:.4f}")
print(f"残差方差: {seasonality_results['residual_variance']:.4f}")

# 变点检测
changepoint_detector = ChangePointDetector(method='pelt')
changepoints = changepoint_detector.detect(
    data=time_series_data['thickness'],
    time=time_series_data['time']
)

print(f"检测到 {len(changepoints)} 个变点:")
for i, cp in enumerate(changepoints):
    print(f"  变点 {i+1}: {cp['time']} (置信度: {cp['confidence']:.4f})")

# 3. 详细不确定性分析
from analysis_visualization.uncertainty_analysis import (
    SensitivityAnalyzer,
    UncertaintyPropagator,
    ReliabilityAnalyzer
)

# 敏感性分析
sensitivity_analyzer = SensitivityAnalyzer(method='sobol')
sensitivity_results = sensitivity_analyzer.analyze(
    model_function=lambda x: model(x),  # 假设有模型函数
    parameter_ranges={
        'param1': [0.5, 1.5],
        'param2': [0.1, 0.9],
        'param3': [10, 100]
    },
    n_samples=1000
)

print("\n敏感性分析结果:")
for param, indices in sensitivity_results['first_order'].items():
    print(f"{param}: S1 = {indices:.4f}")

# 不确定性传播
uncertainty_propagator = UncertaintyPropagator()
propagation_results = uncertainty_propagator.propagate(
    input_uncertainties={
        'thickness': 0.1,  # 10%不确定性
        'velocity': 0.05,  # 5%不确定性
        'temperature': 1.0  # 1K不确定性
    },
    model_function=lambda x: model(x),
    n_samples=1000
)

print(f"输出不确定性: {propagation_results['output_std']:.4f}")
print(f"主要贡献源: {propagation_results['main_contributor']}")

# 可靠性分析
reliability_analyzer = ReliabilityAnalyzer()
reliability_results = reliability_analyzer.assess(
    predictions=y_pred,
    uncertainties=y_uncertainty,
    observations=y_test,  # 假设有观测数据
    confidence_level=0.9
)

print(f"可靠性指标: {reliability_results['reliability_index']:.4f}")
print(f"校准误差: {reliability_results['calibration_error']:.4f}")
print(f"锐度: {reliability_results['sharpness']:.4f}")
```

可视化示例：

```python
# 1. 空间可视化
from analysis_visualization.spatial_analysis import SpatialVisualizer

spatial_viz = SpatialVisualizer()

# 创建空间分布图
spatial_viz.plot_spatial_distribution(
    data=y_pred[:, 2],  # 厚度
    coordinates=X_coords[:, :2],
    title='冰川厚度分布',
    colormap='viridis',
    save_path='thickness_distribution.png'
)

# 创建流场图
spatial_viz.plot_vector_field(
    u=y_pred[:, 0],
    v=y_pred[:, 1],
    coordinates=X_coords[:, :2],
    title='冰川流场',
    save_path='flow_field.png'
)

# 2. 时间可视化
from analysis_visualization.temporal_analysis import TimeSeriesVisualizer

temporal_viz = TimeSeriesVisualizer()

# 创建时间序列图
temporal_viz.plot_time_series(
    data=time_series_data,
    variables=['mass_balance', 'velocity'],
    title='冰川时间序列',
    save_path='time_series.png'
)

# 创建趋势分解图
temporal_viz.plot_decomposition(
    data=time_series_data['thickness'],
    time=time_series_data['time'],
    title='厚度变化分解',
    save_path='decomposition.png'
)

# 3. 不确定性可视化
from analysis_visualization.uncertainty_analysis import UncertaintyVisualizer

uncertainty_viz = UncertaintyVisualizer()

# 创建不确定性地图
uncertainty_viz.plot_uncertainty_map(
    predictions=y_pred[:, 2],
    uncertainties=y_uncertainty[:, 2],
    coordinates=X_coords[:, :2],
    title='厚度预测不确定性',
    save_path='uncertainty_map.png'
)

# 创建置信区间图
uncertainty_viz.plot_confidence_intervals(
    predictions=y_pred,
    uncertainties=y_uncertainty,
    time=X_coords[:, 2],
    title='预测置信区间',
    save_path='confidence_intervals.png'
)

# 4. 交互式可视化
from analysis_visualization.interactive_visualization import GlacierDashboard

# 创建冰川仪表板
dashboard = GlacierDashboard(
    title='青藏高原冰川分析仪表板'
)

# 添加组件
dashboard.add_spatial_map(
    data=y_pred,
    coordinates=X_coords,
    variables=['velocity_u', 'velocity_v', 'thickness', 'pressure']
)

dashboard.add_time_series(
    data=time_series_data,
    variables=['mass_balance', 'velocity', 'thickness']
)

dashboard.add_uncertainty_panel(
    predictions=y_pred,
    uncertainties=y_uncertainty
)

dashboard.add_3d_visualization(
    surface_data=y_pred[:, 2],  # 厚度作为表面
    coordinates=X_coords[:, :2]
)

# 启动仪表板
dashboard.run(port=8050, debug=True)
print("仪表板运行在: http://localhost:8050")
```

设计原则：

1. 模块化: 各分析组件可独立使用或组合使用
2. 可扩展性: 易于添加新的分析方法和可视化技术
3. 交互性: 支持交互式探索和实时分析
4. 专业性: 针对冰川物理问题的专门设计
5. 可重现性: 确保分析结果的可重现性

分析最佳实践：

1. 多维度分析: 结合空间、时间和不确定性分析
2. 层次化探索: 从全局到局部，从概览到细节
3. 交互式验证: 通过交互式工具验证分析结果
4. 可视化驱动: 用可视化引导分析发现
5. 报告生成: 自动生成专业的分析报告

使用建议：

1. 根据研究目标选择合适的分析方法
2. 注意数据质量对分析结果的影响
3. 结合领域知识解释分析结果
4. 使用交互式工具进行深入探索
5. 定期更新分析方法和可视化技术

注意事项：

1. 大数据集的内存和计算限制
2. 统计分析的假设条件检验
3. 可视化的色彩和符号选择
4. 交互式组件的响应性能
5. 分析结果的统计显著性
"""

# 便捷函数
def create_analysis_suite(
    analysis_types: list = ['spatial', 'temporal', 'uncertainty'],
    spatial_methods: list = ['correlation', 'clustering'],
    temporal_methods: list = ['trend', 'seasonality'],
    uncertainty_methods: list = ['sensitivity', 'propagation'],
    **kwargs
):
    """
    创建分析套件
    
    Args:
        analysis_types: 分析类型列表
        spatial_methods: 空间分析方法
        temporal_methods: 时间分析方法
        uncertainty_methods: 不确定性分析方法
        **kwargs: 其他参数
        
    Returns:
        AnalysisSuite: 分析套件
    """
    class AnalysisSuite:
        def __init__(self):
            self.analyzers = {}
            self.visualizers = {}
            self.config = kwargs
        
        def add_analyzer(self, name, analyzer):
            self.analyzers[name] = analyzer
        
        def add_visualizer(self, name, visualizer):
            self.visualizers[name] = visualizer
    
    suite = AnalysisSuite()
    
    # 添加空间分析
    if 'spatial' in analysis_types:
        suite.add_analyzer('spatial', GlacierSpatialAnalyzer(
            methods=spatial_methods,
            **kwargs.get('spatial_params', {})
        ))
        suite.add_visualizer('spatial', SpatialVisualizer())
    
    # 添加时间分析
    if 'temporal' in analysis_types:
        suite.add_analyzer('temporal', GlacierTemporalAnalyzer(
            methods=temporal_methods,
            **kwargs.get('temporal_params', {})
        ))
        suite.add_visualizer('temporal', TimeSeriesVisualizer())
    
    # 添加不确定性分析
    if 'uncertainty' in analysis_types:
        suite.add_analyzer('uncertainty', GlacierUncertaintyAnalyzer(
            methods=uncertainty_methods,
            **kwargs.get('uncertainty_params', {})
        ))
        suite.add_visualizer('uncertainty', UncertaintyVisualizer())
    
    return suite

def generate_comprehensive_report(
    spatial_results: dict = None,
    temporal_results: dict = None,
    uncertainty_results: dict = None,
    output_format: str = 'html',
    include_interactive: bool = True,
    save_path: str = None,
    **kwargs
):
    """
    生成综合分析报告
    
    Args:
        spatial_results: 空间分析结果
        temporal_results: 时间分析结果
        uncertainty_results: 不确定性分析结果
        output_format: 输出格式
        include_interactive: 是否包含交互式组件
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        dict: 报告信息
    """
    from .report_generator import ComprehensiveReportGenerator
    
    generator = ComprehensiveReportGenerator(
        output_format=output_format,
        include_interactive=include_interactive,
        **kwargs
    )
    
    report = generator.generate(
        spatial_results=spatial_results,
        temporal_results=temporal_results,
        uncertainty_results=uncertainty_results,
        save_path=save_path
    )
    
    return report

def create_interactive_dashboard(
    data: dict,
    dashboard_type: str = 'glacier',
    include_3d: bool = True,
    include_animation: bool = True,
    **kwargs
):
    """
    创建交互式仪表板
    
    Args:
        data: 数据字典
        dashboard_type: 仪表板类型
        include_3d: 是否包含3D可视化
        include_animation: 是否包含动画
        **kwargs: 其他参数
        
    Returns:
        Dashboard: 交互式仪表板
    """
    if dashboard_type == 'glacier':
        dashboard = GlacierDashboard(
            include_3d=include_3d,
            include_animation=include_animation,
            **kwargs
        )
    else:
        dashboard = DashboardCreator(
            dashboard_type=dashboard_type,
            **kwargs
        )
    
    # 添加数据
    dashboard.load_data(data)
    
    return dashboard

def analyze_glacier_dynamics(
    predictions: dict,
    coordinates: dict,
    time_series: dict = None,
    uncertainties: dict = None,
    analysis_config: dict = None,
    **kwargs
):
    """
    分析冰川动力学
    
    Args:
        predictions: 预测结果
        coordinates: 坐标信息
        time_series: 时间序列数据
        uncertainties: 不确定性信息
        analysis_config: 分析配置
        **kwargs: 其他参数
        
    Returns:
        dict: 分析结果
    """
    # 创建分析套件
    suite = create_analysis_suite(
        **(analysis_config or {})
    )
    
    results = {}
    
    # 空间分析
    if 'spatial' in suite.analyzers:
        results['spatial'] = suite.analyzers['spatial'].analyze(
            predictions=predictions,
            coordinates=coordinates,
            **kwargs.get('spatial_params', {})
        )
    
    # 时间分析
    if 'temporal' in suite.analyzers and time_series is not None:
        results['temporal'] = suite.analyzers['temporal'].analyze(
            time_series=time_series,
            **kwargs.get('temporal_params', {})
        )
    
    # 不确定性分析
    if 'uncertainty' in suite.analyzers and uncertainties is not None:
        results['uncertainty'] = suite.analyzers['uncertainty'].analyze(
            predictions=predictions,
            uncertainties=uncertainties,
            coordinates=coordinates,
            **kwargs.get('uncertainty_params', {})
        )
    
    return results

# 默认配置
DEFAULT_ANALYSIS_CONFIG = {
    'spatial_resolution': 100,  # 米
    'temporal_resolution': 'monthly',
    'confidence_level': 0.95,
    'n_bootstrap': 1000,
    'n_permutations': 999,
    'significance_level': 0.05
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colormap': 'viridis',
    'font_size': 12,
    'line_width': 2,
    'marker_size': 50,
    'alpha': 0.7
}

# 交互式配置
INTERACTIVE_CONFIG = {
    'port': 8050,
    'host': 'localhost',
    'debug': False,
    'auto_open_browser': True,
    'update_interval': 1000,  # 毫秒
    'cache_timeout': 3600     # 秒
}