#!/usr/bin/env python3
"""
未来预测模块

本模块提供基于气候变化情景的冰川未来预测功能，包括：
- 冰川体积和面积变化预测
- 海平面上升贡献估算
- 水资源可用性预测
- 极端事件频率变化
- 不确定性量化

作者: 青藏高原冰川PINNs项目组
日期: 2025-01-28
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from scipy import stats
import warnings


class ProjectionScenario(Enum):
    """预测情景类型"""
    SSP126 = "SSP1-2.6"  # 低排放情景
    SSP245 = "SSP2-4.5"  # 中等排放情景
    SSP370 = "SSP3-7.0"  # 高排放情景
    SSP585 = "SSP5-8.5"  # 极高排放情景


class ProjectionVariable(Enum):
    """预测变量类型"""
    VOLUME = "volume"
    AREA = "area"
    THICKNESS = "thickness"
    RUNOFF = "runoff"
    SEA_LEVEL = "sea_level"
    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"


@dataclass
class ProjectionConfig:
    """未来预测配置"""
    start_year: int = 2025
    end_year: int = 2100
    time_step: int = 1  # 年
    scenarios: List[ProjectionScenario] = None
    variables: List[ProjectionVariable] = None
    uncertainty_quantiles: List[float] = None
    
    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = list(ProjectionScenario)
        if self.variables is None:
            self.variables = list(ProjectionVariable)
        if self.uncertainty_quantiles is None:
            self.uncertainty_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]


class GlacierProjectionModel:
    """冰川预测模型"""
    
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.projections = {}
        
    def project_volume_change(self, 
                            scenario: ProjectionScenario,
                            initial_volume: float,
                            temperature_change: np.ndarray,
                            precipitation_change: np.ndarray) -> np.ndarray:
        """预测冰川体积变化"""
        years = np.arange(self.config.start_year, self.config.end_year + 1)
        volume_change = np.zeros_like(years, dtype=float)
        
        # 体积变化率模型 (简化的度日模型)
        sensitivity = -0.8  # m³/°C/year per m³
        precipitation_factor = 0.3  # 降水补给因子
        
        current_volume = initial_volume
        for i, year in enumerate(years):
            if i < len(temperature_change) and i < len(precipitation_change):
                # 温度驱动的消融
                melt_loss = sensitivity * current_volume * temperature_change[i]
                # 降水补给
                precip_gain = precipitation_factor * current_volume * precipitation_change[i]
                
                volume_change[i] = melt_loss + precip_gain
                current_volume = max(0, current_volume + volume_change[i])
            else:
                volume_change[i] = 0
                
        return volume_change
    
    def project_area_change(self, 
                          volume_change: np.ndarray,
                          initial_area: float) -> np.ndarray:
        """基于体积变化预测面积变化"""
        # 使用体积-面积关系 V = c * A^γ
        gamma = 1.375  # 典型的体积-面积指数
        
        area_change = np.zeros_like(volume_change)
        cumulative_volume_change = np.cumsum(volume_change)
        
        for i in range(len(volume_change)):
            if cumulative_volume_change[i] != 0:
                relative_volume_change = cumulative_volume_change[i] / initial_area
                area_change[i] = initial_area * (relative_volume_change / gamma)
            
        return area_change


class SeaLevelContribution:
    """海平面上升贡献计算"""
    
    def __init__(self):
        self.ice_density = 917  # kg/m³
        self.water_density = 1000  # kg/m³
        self.ocean_area = 3.61e14  # m²
        
    def calculate_contribution(self, volume_loss: np.ndarray) -> np.ndarray:
        """计算海平面上升贡献"""
        # 考虑密度差异
        water_equivalent = volume_loss * (self.ice_density / self.water_density)
        # 转换为海平面上升高度 (mm)
        sea_level_rise = (water_equivalent / self.ocean_area) * 1000
        return sea_level_rise


class UncertaintyQuantification:
    """不确定性量化"""
    
    def __init__(self, config: ProjectionConfig):
        self.config = config
        
    def monte_carlo_projection(self,
                             projection_func,
                             parameters: Dict,
                             n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """蒙特卡洛不确定性分析"""
        results = []
        
        for _ in range(n_samples):
            # 添加参数不确定性
            perturbed_params = self._perturb_parameters(parameters)
            result = projection_func(**perturbed_params)
            results.append(result)
            
        results = np.array(results)
        
        # 计算分位数
        quantiles = {}
        for q in self.config.uncertainty_quantiles:
            quantiles[f"q{int(q*100)}"] = np.percentile(results, q*100, axis=0)
            
        return quantiles
    
    def _perturb_parameters(self, parameters: Dict) -> Dict:
        """添加参数扰动"""
        perturbed = parameters.copy()
        
        # 为不同参数添加不同的不确定性
        uncertainty_factors = {
            'temperature_change': 0.2,  # 20%不确定性
            'precipitation_change': 0.3,  # 30%不确定性
            'initial_volume': 0.1,  # 10%不确定性
            'initial_area': 0.05  # 5%不确定性
        }
        
        for param, factor in uncertainty_factors.items():
            if param in perturbed:
                if isinstance(perturbed[param], np.ndarray):
                    noise = np.random.normal(1, factor, perturbed[param].shape)
                    perturbed[param] = perturbed[param] * noise
                else:
                    noise = np.random.normal(1, factor)
                    perturbed[param] = perturbed[param] * noise
                    
        return perturbed


class FutureProjectionAnalyzer:
    """未来预测分析器"""
    
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.glacier_model = GlacierProjectionModel(config)
        self.sea_level = SeaLevelContribution()
        self.uncertainty = UncertaintyQuantification(config)
        self.results = {}
        
    def run_projections(self,
                       glacier_data: Dict,
                       climate_projections: Dict) -> Dict:
        """运行完整的未来预测分析"""
        results = {}
        
        for scenario in self.config.scenarios:
            scenario_results = {}
            
            # 获取气候数据
            temp_change = climate_projections[scenario.value]['temperature']
            precip_change = climate_projections[scenario.value]['precipitation']
            
            # 冰川体积预测
            volume_change = self.glacier_model.project_volume_change(
                scenario,
                glacier_data['initial_volume'],
                temp_change,
                precip_change
            )
            
            # 冰川面积预测
            area_change = self.glacier_model.project_area_change(
                volume_change,
                glacier_data['initial_area']
            )
            
            # 海平面贡献
            sea_level_contrib = self.sea_level.calculate_contribution(
                np.abs(volume_change[volume_change < 0])  # 只考虑体积损失
            )
            
            scenario_results = {
                'volume_change': volume_change,
                'area_change': area_change,
                'sea_level_contribution': sea_level_contrib,
                'years': np.arange(self.config.start_year, self.config.end_year + 1)
            }
            
            results[scenario.value] = scenario_results
            
        self.results = results
        return results
    
    def analyze_trends(self) -> Dict:
        """分析预测趋势"""
        trend_analysis = {}
        
        for scenario, data in self.results.items():
            years = data['years']
            
            # 线性趋势分析
            volume_trend = stats.linregress(years, np.cumsum(data['volume_change']))
            area_trend = stats.linregress(years, np.cumsum(data['area_change']))
            
            trend_analysis[scenario] = {
                'volume_trend_slope': volume_trend.slope,
                'volume_trend_pvalue': volume_trend.pvalue,
                'area_trend_slope': area_trend.slope,
                'area_trend_pvalue': area_trend.pvalue,
                'total_volume_loss': np.sum(data['volume_change'][data['volume_change'] < 0]),
                'total_area_loss': np.sum(data['area_change'][data['area_change'] < 0]),
                'total_sea_level_contribution': np.sum(data['sea_level_contribution'])
            }
            
        return trend_analysis
    
    def generate_summary_report(self) -> str:
        """生成预测摘要报告"""
        if not self.results:
            return "未运行预测分析"
            
        trends = self.analyze_trends()
        
        report = "\n=== 冰川未来预测分析报告 ===\n\n"
        report += f"预测时间范围: {self.config.start_year}-{self.config.end_year}\n\n"
        
        for scenario, trend_data in trends.items():
            report += f"--- {scenario} 情景 ---\n"
            report += f"总体积损失: {trend_data['total_volume_loss']:.2e} m³\n"
            report += f"总面积损失: {trend_data['total_area_loss']:.2e} m²\n"
            report += f"海平面贡献: {trend_data['total_sea_level_contribution']:.3f} mm\n"
            report += f"体积变化趋势: {trend_data['volume_trend_slope']:.2e} m³/年\n"
            report += f"面积变化趋势: {trend_data['area_trend_slope']:.2e} m²/年\n\n"
            
        return report
    
    def plot_projections(self, save_path: Optional[str] = None):
        """绘制预测结果"""
        if not self.results:
            print("未运行预测分析")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('冰川未来预测结果', fontsize=16)
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (scenario, data) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            years = data['years']
            
            # 累积体积变化
            axes[0, 0].plot(years, np.cumsum(data['volume_change']), 
                          label=scenario, color=color, linewidth=2)
            
            # 累积面积变化
            axes[0, 1].plot(years, np.cumsum(data['area_change']), 
                          label=scenario, color=color, linewidth=2)
            
            # 海平面贡献
            axes[1, 0].plot(years[:len(data['sea_level_contribution'])], 
                          np.cumsum(data['sea_level_contribution']), 
                          label=scenario, color=color, linewidth=2)
            
            # 年变化率
            axes[1, 1].plot(years, data['volume_change'], 
                          label=scenario, color=color, linewidth=2)
        
        # 设置子图
        axes[0, 0].set_title('累积体积变化')
        axes[0, 0].set_ylabel('体积变化 (m³)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('累积面积变化')
        axes[0, 1].set_ylabel('面积变化 (m²)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('累积海平面贡献')
        axes[1, 0].set_ylabel('海平面上升 (mm)')
        axes[1, 0].set_xlabel('年份')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('年体积变化率')
        axes[1, 1].set_ylabel('体积变化率 (m³/年)')
        axes[1, 1].set_xlabel('年份')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_future_projection_analyzer(config: Optional[ProjectionConfig] = None) -> FutureProjectionAnalyzer:
    """创建未来预测分析器"""
    if config is None:
        config = ProjectionConfig()
    return FutureProjectionAnalyzer(config)


# 示例使用
if __name__ == "__main__":
    # 创建配置
    config = ProjectionConfig(
        start_year=2025,
        end_year=2100,
        scenarios=[ProjectionScenario.SSP245, ProjectionScenario.SSP585]
    )
    
    # 创建分析器
    analyzer = create_future_projection_analyzer(config)
    
    # 模拟数据
    glacier_data = {
        'initial_volume': 1e9,  # m³
        'initial_area': 1e6     # m²
    }
    
    # 模拟气候预测数据
    years = np.arange(2025, 2101)
    climate_projections = {
        'SSP2-4.5': {
            'temperature': np.linspace(0, 2.5, len(years)),  # 温度变化
            'precipitation': np.random.normal(0, 0.1, len(years))  # 降水变化
        },
        'SSP5-8.5': {
            'temperature': np.linspace(0, 4.5, len(years)),
            'precipitation': np.random.normal(0, 0.15, len(years))
        }
    }
    
    # 运行预测
    results = analyzer.run_projections(glacier_data, climate_projections)
    
    # 生成报告
    report = analyzer.generate_summary_report()
    print(report)
    
    # 绘制结果
    analyzer.plot_projections()