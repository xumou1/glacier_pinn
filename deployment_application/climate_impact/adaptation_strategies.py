#!/usr/bin/env python3
"""
适应策略模块

本模块提供基于冰川变化预测的气候适应策略制定功能，包括：
- 水资源管理适应策略
- 灾害风险减缓措施
- 基础设施适应性规划
- 生态系统保护策略
- 社会经济适应方案

作者: 青藏高原冰川PINNs项目组
日期: 2025-01-28
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import json


class AdaptationSector(Enum):
    """适应部门类型"""
    WATER_RESOURCES = "water_resources"
    AGRICULTURE = "agriculture"
    INFRASTRUCTURE = "infrastructure"
    ECOSYSTEM = "ecosystem"
    DISASTER_RISK = "disaster_risk"
    ENERGY = "energy"
    TOURISM = "tourism"
    URBAN_PLANNING = "urban_planning"


class AdaptationTimeframe(Enum):
    """适应时间框架"""
    SHORT_TERM = "short_term"  # 1-5年
    MEDIUM_TERM = "medium_term"  # 5-20年
    LONG_TERM = "long_term"  # 20-50年
    VERY_LONG_TERM = "very_long_term"  # 50年以上


class AdaptationType(Enum):
    """适应类型"""
    STRUCTURAL = "structural"  # 结构性措施
    NON_STRUCTURAL = "non_structural"  # 非结构性措施
    ECOSYSTEM_BASED = "ecosystem_based"  # 基于生态系统
    INSTITUTIONAL = "institutional"  # 制度性措施
    TECHNOLOGICAL = "technological"  # 技术性措施


class ImplementationComplexity(Enum):
    """实施复杂度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AdaptationMeasure:
    """适应措施"""
    id: str
    name: str
    description: str
    sector: AdaptationSector
    adaptation_type: AdaptationType
    timeframe: AdaptationTimeframe
    complexity: ImplementationComplexity
    cost_estimate: float  # 成本估算（万元）
    effectiveness_score: float  # 有效性评分（0-1）
    co_benefits: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)


@dataclass
class AdaptationContext:
    """适应背景信息"""
    region_name: str
    population: int
    economic_level: str  # "low", "medium", "high"
    climate_vulnerability: float  # 0-1
    institutional_capacity: float  # 0-1
    financial_capacity: float  # 0-1
    technical_capacity: float  # 0-1
    glacier_dependence: float  # 对冰川的依赖程度 0-1


class AdaptationStrategy(ABC):
    """适应策略基类"""
    
    def __init__(self, context: AdaptationContext):
        self.context = context
        self.measures = []
        
    @abstractmethod
    def generate_measures(self, climate_projections: Dict) -> List[AdaptationMeasure]:
        """生成适应措施"""
        pass
    
    @abstractmethod
    def prioritize_measures(self, measures: List[AdaptationMeasure]) -> List[AdaptationMeasure]:
        """优先级排序"""
        pass


class WaterResourcesAdaptation(AdaptationStrategy):
    """水资源适应策略"""
    
    def generate_measures(self, climate_projections: Dict) -> List[AdaptationMeasure]:
        """生成水资源适应措施"""
        measures = []
        
        # 基于预测的水资源变化生成措施
        water_stress = climate_projections.get('water_stress_increase', 0)
        
        if water_stress > 0.3:  # 高水资源压力
            measures.extend([
                AdaptationMeasure(
                    id="WR001",
                    name="建设高山水库",
                    description="在冰川下游建设调节水库，储存融水用于干旱期供水",
                    sector=AdaptationSector.WATER_RESOURCES,
                    adaptation_type=AdaptationType.STRUCTURAL,
                    timeframe=AdaptationTimeframe.MEDIUM_TERM,
                    complexity=ImplementationComplexity.HIGH,
                    cost_estimate=50000,
                    effectiveness_score=0.8,
                    co_benefits=["洪水控制", "水力发电", "生态保护"],
                    prerequisites=["环境影响评估", "资金筹措", "技术设计"],
                    risks=["生态影响", "地质风险", "成本超支"],
                    indicators=["储水量", "供水保证率", "生态流量"]
                ),
                AdaptationMeasure(
                    id="WR002",
                    name="节水灌溉系统",
                    description="推广滴灌、喷灌等高效节水灌溉技术",
                    sector=AdaptationSector.WATER_RESOURCES,
                    adaptation_type=AdaptationType.TECHNOLOGICAL,
                    timeframe=AdaptationTimeframe.SHORT_TERM,
                    complexity=ImplementationComplexity.MEDIUM,
                    cost_estimate=5000,
                    effectiveness_score=0.7,
                    co_benefits=["提高农业产量", "减少土壤盐碱化"],
                    prerequisites=["农民培训", "技术推广"],
                    risks=["初期投资高", "技术维护"],
                    indicators=["灌溉效率", "用水量减少比例"]
                )
            ])
            
        if water_stress > 0.5:  # 极高水资源压力
            measures.append(
                AdaptationMeasure(
                    id="WR003",
                    name="跨流域调水",
                    description="从水资源丰富地区向缺水地区调水",
                    sector=AdaptationSector.WATER_RESOURCES,
                    adaptation_type=AdaptationType.STRUCTURAL,
                    timeframe=AdaptationTimeframe.LONG_TERM,
                    complexity=ImplementationComplexity.VERY_HIGH,
                    cost_estimate=200000,
                    effectiveness_score=0.9,
                    co_benefits=["区域发展平衡"],
                    prerequisites=["跨区域协调", "巨额投资", "复杂工程"],
                    risks=["生态破坏", "社会冲突", "技术风险"],
                    indicators=["调水量", "受益人口", "生态影响"]
                )
            )
            
        return measures
    
    def prioritize_measures(self, measures: List[AdaptationMeasure]) -> List[AdaptationMeasure]:
        """水资源措施优先级排序"""
        def priority_score(measure):
            # 综合考虑有效性、成本、实施复杂度
            effectiveness = measure.effectiveness_score
            cost_factor = 1 / (1 + measure.cost_estimate / 10000)  # 成本越低越好
            complexity_factor = {
                ImplementationComplexity.LOW: 1.0,
                ImplementationComplexity.MEDIUM: 0.8,
                ImplementationComplexity.HIGH: 0.6,
                ImplementationComplexity.VERY_HIGH: 0.4
            }[measure.complexity]
            
            return effectiveness * cost_factor * complexity_factor
            
        return sorted(measures, key=priority_score, reverse=True)


class DisasterRiskAdaptation(AdaptationStrategy):
    """灾害风险适应策略"""
    
    def generate_measures(self, climate_projections: Dict) -> List[AdaptationMeasure]:
        """生成灾害风险适应措施"""
        measures = []
        
        glof_risk = climate_projections.get('glof_risk_increase', 0)
        avalanche_risk = climate_projections.get('avalanche_risk_increase', 0)
        
        if glof_risk > 0.3:
            measures.extend([
                AdaptationMeasure(
                    id="DR001",
                    name="冰湖监测预警系统",
                    description="建立实时冰湖监测和早期预警系统",
                    sector=AdaptationSector.DISASTER_RISK,
                    adaptation_type=AdaptationType.TECHNOLOGICAL,
                    timeframe=AdaptationTimeframe.SHORT_TERM,
                    complexity=ImplementationComplexity.MEDIUM,
                    cost_estimate=3000,
                    effectiveness_score=0.8,
                    co_benefits=["科学研究", "旅游安全"],
                    prerequisites=["技术设备", "专业人员", "通信网络"],
                    risks=["设备故障", "维护成本"],
                    indicators=["监测覆盖率", "预警准确率", "响应时间"]
                ),
                AdaptationMeasure(
                    id="DR002",
                    name="冰湖治理工程",
                    description="对高危冰湖实施排水、加固等治理措施",
                    sector=AdaptationSector.DISASTER_RISK,
                    adaptation_type=AdaptationType.STRUCTURAL,
                    timeframe=AdaptationTimeframe.MEDIUM_TERM,
                    complexity=ImplementationComplexity.HIGH,
                    cost_estimate=20000,
                    effectiveness_score=0.9,
                    co_benefits=["生态保护", "景观改善"],
                    prerequisites=["详细勘察", "工程设计", "环保审批"],
                    risks=["工程风险", "生态影响", "成本高"],
                    indicators=["治理湖泊数量", "风险降低程度"]
                )
            ])
            
        if avalanche_risk > 0.3:
            measures.append(
                AdaptationMeasure(
                    id="DR003",
                    name="雪崩防护工程",
                    description="在易发雪崩区域建设防护设施",
                    sector=AdaptationSector.DISASTER_RISK,
                    adaptation_type=AdaptationType.STRUCTURAL,
                    timeframe=AdaptationTimeframe.MEDIUM_TERM,
                    complexity=ImplementationComplexity.HIGH,
                    cost_estimate=15000,
                    effectiveness_score=0.7,
                    co_benefits=["交通安全", "旅游发展"],
                    prerequisites=["地质勘察", "工程设计"],
                    risks=["维护成本", "景观影响"],
                    indicators=["防护覆盖率", "事故减少率"]
                )
            )
            
        return measures
    
    def prioritize_measures(self, measures: List[AdaptationMeasure]) -> List[AdaptationMeasure]:
        """灾害风险措施优先级排序"""
        def priority_score(measure):
            # 灾害风险措施优先考虑有效性和紧急性
            effectiveness = measure.effectiveness_score
            urgency_factor = {
                AdaptationTimeframe.SHORT_TERM: 1.0,
                AdaptationTimeframe.MEDIUM_TERM: 0.8,
                AdaptationTimeframe.LONG_TERM: 0.6,
                AdaptationTimeframe.VERY_LONG_TERM: 0.4
            }[measure.timeframe]
            
            return effectiveness * urgency_factor
            
        return sorted(measures, key=priority_score, reverse=True)


class EcosystemAdaptation(AdaptationStrategy):
    """生态系统适应策略"""
    
    def generate_measures(self, climate_projections: Dict) -> List[AdaptationMeasure]:
        """生成生态系统适应措施"""
        measures = [
            AdaptationMeasure(
                id="EC001",
                name="生态廊道建设",
                description="建设连接不同生境的生态廊道，促进物种迁移",
                sector=AdaptationSector.ECOSYSTEM,
                adaptation_type=AdaptationType.ECOSYSTEM_BASED,
                timeframe=AdaptationTimeframe.MEDIUM_TERM,
                complexity=ImplementationComplexity.MEDIUM,
                cost_estimate=8000,
                effectiveness_score=0.7,
                co_benefits=["生物多样性保护", "碳汇功能", "景观美化"],
                prerequisites=["生态调查", "规划设计", "土地协调"],
                risks=["维护成本", "效果不确定"],
                indicators=["廊道连通性", "物种多样性", "生态功能"]
            ),
            AdaptationMeasure(
                id="EC002",
                name="湿地恢复工程",
                description="恢复退化湿地，增强生态系统韧性",
                sector=AdaptationSector.ECOSYSTEM,
                adaptation_type=AdaptationType.ECOSYSTEM_BASED,
                timeframe=AdaptationTimeframe.LONG_TERM,
                complexity=ImplementationComplexity.MEDIUM,
                cost_estimate=12000,
                effectiveness_score=0.8,
                co_benefits=["水质净化", "洪水调节", "碳储存"],
                prerequisites=["湿地调查", "恢复技术", "长期监测"],
                risks=["恢复失败", "外来物种入侵"],
                indicators=["湿地面积", "水质指标", "鸟类多样性"]
            )
        ]
        
        return measures
    
    def prioritize_measures(self, measures: List[AdaptationMeasure]) -> List[AdaptationMeasure]:
        """生态系统措施优先级排序"""
        def priority_score(measure):
            # 生态措施重视长期效益和协同效益
            effectiveness = measure.effectiveness_score
            co_benefits_factor = len(measure.co_benefits) / 5  # 协同效益越多越好
            
            return effectiveness * (1 + co_benefits_factor)
            
        return sorted(measures, key=priority_score, reverse=True)


class AdaptationPlanningFramework:
    """适应规划框架"""
    
    def __init__(self, context: AdaptationContext):
        self.context = context
        self.strategies = {
            AdaptationSector.WATER_RESOURCES: WaterResourcesAdaptation(context),
            AdaptationSector.DISASTER_RISK: DisasterRiskAdaptation(context),
            AdaptationSector.ECOSYSTEM: EcosystemAdaptation(context)
        }
        self.adaptation_plan = []
        
    def develop_comprehensive_plan(self, climate_projections: Dict) -> List[AdaptationMeasure]:
        """制定综合适应计划"""
        all_measures = []
        
        # 从各部门策略生成措施
        for sector, strategy in self.strategies.items():
            sector_measures = strategy.generate_measures(climate_projections)
            prioritized_measures = strategy.prioritize_measures(sector_measures)
            all_measures.extend(prioritized_measures)
            
        # 综合优先级排序
        final_plan = self._comprehensive_prioritization(all_measures)
        
        # 考虑实施约束
        feasible_plan = self._apply_implementation_constraints(final_plan)
        
        self.adaptation_plan = feasible_plan
        return feasible_plan
    
    def _comprehensive_prioritization(self, measures: List[AdaptationMeasure]) -> List[AdaptationMeasure]:
        """综合优先级排序"""
        def comprehensive_score(measure):
            # 多准则决策分析
            effectiveness = measure.effectiveness_score
            
            # 成本效益比
            cost_effectiveness = effectiveness / (measure.cost_estimate / 1000 + 1)
            
            # 实施可行性
            feasibility = {
                ImplementationComplexity.LOW: 1.0,
                ImplementationComplexity.MEDIUM: 0.8,
                ImplementationComplexity.HIGH: 0.6,
                ImplementationComplexity.VERY_HIGH: 0.4
            }[measure.complexity]
            
            # 紧急性
            urgency = {
                AdaptationTimeframe.SHORT_TERM: 1.0,
                AdaptationTimeframe.MEDIUM_TERM: 0.8,
                AdaptationTimeframe.LONG_TERM: 0.6,
                AdaptationTimeframe.VERY_LONG_TERM: 0.4
            }[measure.timeframe]
            
            # 协同效益
            co_benefits_score = len(measure.co_benefits) / 5
            
            # 综合评分
            return (effectiveness * 0.3 + 
                   cost_effectiveness * 0.25 + 
                   feasibility * 0.2 + 
                   urgency * 0.15 + 
                   co_benefits_score * 0.1)
            
        return sorted(measures, key=comprehensive_score, reverse=True)
    
    def _apply_implementation_constraints(self, measures: List[AdaptationMeasure]) -> List[AdaptationMeasure]:
        """应用实施约束"""
        feasible_measures = []
        total_cost = 0
        
        # 假设总预算约束
        budget_limit = self.context.financial_capacity * 100000  # 基于财政能力
        
        for measure in measures:
            if total_cost + measure.cost_estimate <= budget_limit:
                # 检查技术能力约束
                if self._check_technical_feasibility(measure):
                    feasible_measures.append(measure)
                    total_cost += measure.cost_estimate
                    
        return feasible_measures
    
    def _check_technical_feasibility(self, measure: AdaptationMeasure) -> bool:
        """检查技术可行性"""
        required_capacity = {
            ImplementationComplexity.LOW: 0.3,
            ImplementationComplexity.MEDIUM: 0.5,
            ImplementationComplexity.HIGH: 0.7,
            ImplementationComplexity.VERY_HIGH: 0.9
        }[measure.complexity]
        
        return self.context.technical_capacity >= required_capacity
    
    def generate_implementation_timeline(self) -> Dict:
        """生成实施时间表"""
        timeline = {
            AdaptationTimeframe.SHORT_TERM: [],
            AdaptationTimeframe.MEDIUM_TERM: [],
            AdaptationTimeframe.LONG_TERM: [],
            AdaptationTimeframe.VERY_LONG_TERM: []
        }
        
        for measure in self.adaptation_plan:
            timeline[measure.timeframe].append(measure)
            
        return timeline
    
    def calculate_total_investment(self) -> Dict:
        """计算总投资需求"""
        total_cost = sum(measure.cost_estimate for measure in self.adaptation_plan)
        
        cost_by_timeframe = {}
        for timeframe in AdaptationTimeframe:
            timeframe_cost = sum(
                measure.cost_estimate for measure in self.adaptation_plan 
                if measure.timeframe == timeframe
            )
            cost_by_timeframe[timeframe.value] = timeframe_cost
            
        cost_by_sector = {}
        for sector in AdaptationSector:
            sector_cost = sum(
                measure.cost_estimate for measure in self.adaptation_plan 
                if measure.sector == sector
            )
            cost_by_sector[sector.value] = sector_cost
            
        return {
            'total_cost': total_cost,
            'cost_by_timeframe': cost_by_timeframe,
            'cost_by_sector': cost_by_sector
        }
    
    def export_plan_to_json(self, filepath: str):
        """导出适应计划到JSON文件"""
        plan_data = {
            'context': {
                'region_name': self.context.region_name,
                'population': self.context.population,
                'economic_level': self.context.economic_level,
                'climate_vulnerability': self.context.climate_vulnerability,
                'institutional_capacity': self.context.institutional_capacity,
                'financial_capacity': self.context.financial_capacity,
                'technical_capacity': self.context.technical_capacity,
                'glacier_dependence': self.context.glacier_dependence
            },
            'measures': [
                {
                    'id': measure.id,
                    'name': measure.name,
                    'description': measure.description,
                    'sector': measure.sector.value,
                    'adaptation_type': measure.adaptation_type.value,
                    'timeframe': measure.timeframe.value,
                    'complexity': measure.complexity.value,
                    'cost_estimate': measure.cost_estimate,
                    'effectiveness_score': measure.effectiveness_score,
                    'co_benefits': measure.co_benefits,
                    'prerequisites': measure.prerequisites,
                    'risks': measure.risks,
                    'indicators': measure.indicators
                }
                for measure in self.adaptation_plan
            ],
            'investment_summary': self.calculate_total_investment(),
            'implementation_timeline': {
                timeframe.value: [measure.id for measure in measures]
                for timeframe, measures in self.generate_implementation_timeline().items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)


def create_adaptation_planner(context: AdaptationContext) -> AdaptationPlanningFramework:
    """创建适应规划器"""
    return AdaptationPlanningFramework(context)


# 示例使用
if __name__ == "__main__":
    # 创建适应背景
    context = AdaptationContext(
        region_name="青藏高原东南部",
        population=500000,
        economic_level="medium",
        climate_vulnerability=0.7,
        institutional_capacity=0.6,
        financial_capacity=0.5,
        technical_capacity=0.6,
        glacier_dependence=0.8
    )
    
    # 创建规划框架
    planner = create_adaptation_planner(context)
    
    # 模拟气候预测
    climate_projections = {
        'water_stress_increase': 0.4,
        'glof_risk_increase': 0.3,
        'avalanche_risk_increase': 0.2,
        'temperature_increase': 2.5,
        'precipitation_change': -0.1
    }
    
    # 制定适应计划
    adaptation_plan = planner.develop_comprehensive_plan(climate_projections)
    
    # 输出结果
    print(f"\n=== {context.region_name} 气候适应计划 ===\n")
    print(f"共制定 {len(adaptation_plan)} 项适应措施:\n")
    
    for i, measure in enumerate(adaptation_plan, 1):
        print(f"{i}. {measure.name}")
        print(f"   部门: {measure.sector.value}")
        print(f"   时间框架: {measure.timeframe.value}")
        print(f"   成本估算: {measure.cost_estimate:.0f} 万元")
        print(f"   有效性评分: {measure.effectiveness_score:.2f}")
        print(f"   描述: {measure.description}")
        print()
    
    # 投资摘要
    investment = planner.calculate_total_investment()
    print(f"总投资需求: {investment['total_cost']:.0f} 万元")
    
    # 导出计划
    planner.export_plan_to_json('adaptation_plan.json')
    print("\n适应计划已导出到 adaptation_plan.json")