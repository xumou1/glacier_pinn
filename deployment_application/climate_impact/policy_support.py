#!/usr/bin/env python3
"""
政策支持模块

本模块提供基于冰川变化科学证据的政策制定支持功能，包括：
- 政策影响评估
- 科学证据整合
- 政策建议生成
- 利益相关者分析
- 政策实施监测

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
from datetime import datetime, timedelta


class PolicyDomain(Enum):
    """政策领域"""
    CLIMATE_CHANGE = "climate_change"
    WATER_MANAGEMENT = "water_management"
    DISASTER_RISK_REDUCTION = "disaster_risk_reduction"
    ENVIRONMENTAL_PROTECTION = "environmental_protection"
    SUSTAINABLE_DEVELOPMENT = "sustainable_development"
    INTERNATIONAL_COOPERATION = "international_cooperation"
    ECONOMIC_DEVELOPMENT = "economic_development"
    SOCIAL_WELFARE = "social_welfare"


class PolicyLevel(Enum):
    """政策层级"""
    INTERNATIONAL = "international"
    NATIONAL = "national"
    REGIONAL = "regional"
    LOCAL = "local"


class PolicyType(Enum):
    """政策类型"""
    REGULATORY = "regulatory"  # 监管政策
    ECONOMIC = "economic"  # 经济政策
    INFORMATIONAL = "informational"  # 信息政策
    VOLUNTARY = "voluntary"  # 自愿政策
    INSTITUTIONAL = "institutional"  # 制度政策


class EvidenceStrength(Enum):
    """证据强度"""
    VERY_HIGH = "very_high"  # 非常高
    HIGH = "high"  # 高
    MEDIUM = "medium"  # 中等
    LOW = "low"  # 低
    VERY_LOW = "very_low"  # 非常低


@dataclass
class ScientificEvidence:
    """科学证据"""
    id: str
    title: str
    description: str
    data_source: str
    methodology: str
    key_findings: List[str]
    uncertainty_level: float  # 0-1
    confidence_level: float  # 0-1
    spatial_scale: str
    temporal_scale: str
    policy_relevance: float  # 0-1
    last_updated: datetime


@dataclass
class Stakeholder:
    """利益相关者"""
    id: str
    name: str
    type: str  # "government", "ngo", "private", "academic", "community"
    influence_level: float  # 0-1
    interest_level: float  # 0-1
    position: str  # "supportive", "neutral", "opposed"
    concerns: List[str]
    resources: List[str]
    contact_info: Dict[str, str]


@dataclass
class PolicyRecommendation:
    """政策建议"""
    id: str
    title: str
    description: str
    domain: PolicyDomain
    policy_type: PolicyType
    level: PolicyLevel
    priority: str  # "high", "medium", "low"
    evidence_base: List[str]  # 支撑证据ID列表
    target_stakeholders: List[str]  # 目标利益相关者ID列表
    expected_outcomes: List[str]
    implementation_steps: List[str]
    resource_requirements: Dict[str, float]
    timeline: str
    success_indicators: List[str]
    potential_barriers: List[str]
    mitigation_strategies: List[str]


class EvidenceIntegrator:
    """证据整合器"""
    
    def __init__(self):
        self.evidence_database = {}
        
    def add_evidence(self, evidence: ScientificEvidence):
        """添加科学证据"""
        self.evidence_database[evidence.id] = evidence
        
    def integrate_glacier_data(self, glacier_projections: Dict) -> List[ScientificEvidence]:
        """整合冰川数据为政策证据"""
        evidence_list = []
        
        # 冰川体积变化证据
        if 'volume_change' in glacier_projections:
            volume_evidence = ScientificEvidence(
                id="GLAC_VOL_001",
                title="青藏高原冰川体积变化预测",
                description="基于PINNs模型的冰川体积变化预测结果",
                data_source="PINNs模型预测",
                methodology="物理信息神经网络",
                key_findings=[
                    f"预计到2100年冰川体积将减少{abs(glacier_projections['volume_change']):.1%}",
                    "体积损失主要集中在低海拔区域",
                    "不同排放情景下差异显著"
                ],
                uncertainty_level=0.2,
                confidence_level=0.8,
                spatial_scale="青藏高原",
                temporal_scale="2025-2100",
                policy_relevance=0.9,
                last_updated=datetime.now()
            )
            evidence_list.append(volume_evidence)
            
        # 水资源影响证据
        if 'runoff_change' in glacier_projections:
            water_evidence = ScientificEvidence(
                id="WATER_RUN_001",
                title="冰川融水径流变化预测",
                description="冰川变化对下游径流的影响评估",
                data_source="水文模型计算",
                methodology="冰川-水文耦合模型",
                key_findings=[
                    f"年径流量预计变化{glacier_projections['runoff_change']:.1%}",
                    "季节性径流分布将发生显著变化",
                    "极端干旱和洪水风险增加"
                ],
                uncertainty_level=0.3,
                confidence_level=0.7,
                spatial_scale="主要河流流域",
                temporal_scale="2025-2100",
                policy_relevance=0.95,
                last_updated=datetime.now()
            )
            evidence_list.append(water_evidence)
            
        # 灾害风险证据
        if 'disaster_risk' in glacier_projections:
            disaster_evidence = ScientificEvidence(
                id="DISASTER_001",
                title="冰川相关灾害风险评估",
                description="GLOF、冰崩等灾害风险变化评估",
                data_source="灾害风险模型",
                methodology="多灾种风险评估",
                key_findings=[
                    f"GLOF风险增加{glacier_projections['disaster_risk']:.1%}",
                    "高风险区域主要分布在冰湖密集区",
                    "需要加强监测和预警"
                ],
                uncertainty_level=0.4,
                confidence_level=0.6,
                spatial_scale="高风险区域",
                temporal_scale="近期-中期",
                policy_relevance=0.85,
                last_updated=datetime.now()
            )
            evidence_list.append(disaster_evidence)
            
        return evidence_list
    
    def assess_evidence_quality(self, evidence: ScientificEvidence) -> EvidenceStrength:
        """评估证据质量"""
        # 综合考虑不确定性、置信度和政策相关性
        quality_score = (
            (1 - evidence.uncertainty_level) * 0.4 +
            evidence.confidence_level * 0.4 +
            evidence.policy_relevance * 0.2
        )
        
        if quality_score >= 0.8:
            return EvidenceStrength.VERY_HIGH
        elif quality_score >= 0.6:
            return EvidenceStrength.HIGH
        elif quality_score >= 0.4:
            return EvidenceStrength.MEDIUM
        elif quality_score >= 0.2:
            return EvidenceStrength.LOW
        else:
            return EvidenceStrength.VERY_LOW
    
    def synthesize_evidence(self, domain: PolicyDomain) -> Dict:
        """综合特定领域的证据"""
        relevant_evidence = [
            evidence for evidence in self.evidence_database.values()
            if evidence.policy_relevance > 0.5
        ]
        
        synthesis = {
            'total_evidence_count': len(relevant_evidence),
            'high_quality_evidence': [
                evidence for evidence in relevant_evidence
                if self.assess_evidence_quality(evidence) in [EvidenceStrength.HIGH, EvidenceStrength.VERY_HIGH]
            ],
            'key_findings': [],
            'uncertainty_assessment': {},
            'policy_implications': []
        }
        
        # 提取关键发现
        for evidence in relevant_evidence:
            synthesis['key_findings'].extend(evidence.key_findings)
            
        # 不确定性评估
        uncertainties = [evidence.uncertainty_level for evidence in relevant_evidence]
        synthesis['uncertainty_assessment'] = {
            'mean_uncertainty': np.mean(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'min_uncertainty': np.min(uncertainties)
        }
        
        return synthesis


class StakeholderAnalyzer:
    """利益相关者分析器"""
    
    def __init__(self):
        self.stakeholders = {}
        
    def add_stakeholder(self, stakeholder: Stakeholder):
        """添加利益相关者"""
        self.stakeholders[stakeholder.id] = stakeholder
        
    def analyze_stakeholder_landscape(self) -> Dict:
        """分析利益相关者格局"""
        analysis = {
            'total_stakeholders': len(self.stakeholders),
            'by_type': {},
            'by_position': {},
            'influence_distribution': {},
            'key_stakeholders': []
        }
        
        # 按类型分析
        for stakeholder in self.stakeholders.values():
            stakeholder_type = stakeholder.type
            if stakeholder_type not in analysis['by_type']:
                analysis['by_type'][stakeholder_type] = 0
            analysis['by_type'][stakeholder_type] += 1
            
            # 按立场分析
            position = stakeholder.position
            if position not in analysis['by_position']:
                analysis['by_position'][position] = 0
            analysis['by_position'][position] += 1
            
            # 识别关键利益相关者（高影响力和高兴趣）
            if stakeholder.influence_level > 0.7 and stakeholder.interest_level > 0.7:
                analysis['key_stakeholders'].append(stakeholder)
                
        return analysis
    
    def identify_coalition_opportunities(self) -> List[Dict]:
        """识别联盟机会"""
        coalitions = []
        
        supportive_stakeholders = [
            s for s in self.stakeholders.values() 
            if s.position == "supportive" and s.influence_level > 0.5
        ]
        
        if len(supportive_stakeholders) >= 2:
            coalitions.append({
                'type': 'supportive_coalition',
                'members': supportive_stakeholders,
                'potential_impact': sum(s.influence_level for s in supportive_stakeholders),
                'common_interests': self._find_common_interests(supportive_stakeholders)
            })
            
        return coalitions
    
    def _find_common_interests(self, stakeholders: List[Stakeholder]) -> List[str]:
        """找到共同利益"""
        if not stakeholders:
            return []
            
        # 简化实现：找到所有利益相关者都关心的问题
        all_concerns = [set(s.concerns) for s in stakeholders]
        common_concerns = set.intersection(*all_concerns) if all_concerns else set()
        
        return list(common_concerns)


class PolicyRecommendationEngine:
    """政策建议引擎"""
    
    def __init__(self, evidence_integrator: EvidenceIntegrator, 
                 stakeholder_analyzer: StakeholderAnalyzer):
        self.evidence_integrator = evidence_integrator
        self.stakeholder_analyzer = stakeholder_analyzer
        self.recommendations = {}
        
    def generate_water_policy_recommendations(self, evidence_synthesis: Dict) -> List[PolicyRecommendation]:
        """生成水资源政策建议"""
        recommendations = []
        
        # 基于证据生成建议
        if evidence_synthesis['uncertainty_assessment']['mean_uncertainty'] < 0.3:
            # 高确定性证据支持的政策
            recommendations.append(
                PolicyRecommendation(
                    id="WATER_POL_001",
                    title="建立冰川融水监测预警系统",
                    description="建立覆盖主要冰川区域的融水监测和预警系统，为水资源管理提供科学依据",
                    domain=PolicyDomain.WATER_MANAGEMENT,
                    policy_type=PolicyType.REGULATORY,
                    level=PolicyLevel.NATIONAL,
                    priority="high",
                    evidence_base=["WATER_RUN_001", "GLAC_VOL_001"],
                    target_stakeholders=["water_ministry", "meteorological_bureau"],
                    expected_outcomes=[
                        "提高水资源预测精度",
                        "减少极端事件损失",
                        "优化水资源配置"
                    ],
                    implementation_steps=[
                        "制定监测网络规划",
                        "采购安装监测设备",
                        "建立数据共享平台",
                        "培训技术人员"
                    ],
                    resource_requirements={
                        "funding": 50000000,  # 5000万元
                        "personnel": 100,
                        "equipment": 200
                    },
                    timeline="3年",
                    success_indicators=[
                        "监测站点覆盖率>80%",
                        "预警准确率>85%",
                        "数据共享率>90%"
                    ],
                    potential_barriers=[
                        "资金不足",
                        "技术人员缺乏",
                        "部门协调困难"
                    ],
                    mitigation_strategies=[
                        "多渠道筹措资金",
                        "加强人才培养",
                        "建立协调机制"
                    ]
                )
            )
            
        # 跨流域水资源配置政策
        recommendations.append(
            PolicyRecommendation(
                id="WATER_POL_002",
                title="制定跨流域水资源适应性配置政策",
                description="基于冰川变化预测，制定跨流域水资源配置和调度政策",
                domain=PolicyDomain.WATER_MANAGEMENT,
                policy_type=PolicyType.INSTITUTIONAL,
                level=PolicyLevel.REGIONAL,
                priority="high",
                evidence_base=["WATER_RUN_001"],
                target_stakeholders=["regional_governments", "water_users"],
                expected_outcomes=[
                    "提高水资源利用效率",
                    "减少区域水资源冲突",
                    "增强系统韧性"
                ],
                implementation_steps=[
                    "开展流域水资源评估",
                    "制定配置方案",
                    "建立协调机制",
                    "实施试点项目"
                ],
                resource_requirements={
                    "funding": 20000000,
                    "personnel": 50,
                    "infrastructure": 10
                },
                timeline="5年",
                success_indicators=[
                    "水资源配置效率提升20%",
                    "用水冲突减少50%",
                    "供水保证率>95%"
                ],
                potential_barriers=[
                    "利益协调困难",
                    "技术标准不统一",
                    "法律法规滞后"
                ],
                mitigation_strategies=[
                    "建立利益补偿机制",
                    "统一技术标准",
                    "完善法律框架"
                ]
            )
        )
        
        return recommendations
    
    def generate_disaster_risk_recommendations(self) -> List[PolicyRecommendation]:
        """生成灾害风险政策建议"""
        return [
            PolicyRecommendation(
                id="DISASTER_POL_001",
                title="建立冰川灾害风险管理体系",
                description="建立覆盖监测、预警、应急响应的综合灾害风险管理体系",
                domain=PolicyDomain.DISASTER_RISK_REDUCTION,
                policy_type=PolicyType.INSTITUTIONAL,
                level=PolicyLevel.NATIONAL,
                priority="high",
                evidence_base=["DISASTER_001"],
                target_stakeholders=["emergency_management", "local_governments"],
                expected_outcomes=[
                    "减少灾害损失",
                    "提高应急响应能力",
                    "保护人民生命财产安全"
                ],
                implementation_steps=[
                    "制定管理体系框架",
                    "建立监测预警网络",
                    "完善应急预案",
                    "开展演练培训"
                ],
                resource_requirements={
                    "funding": 30000000,
                    "personnel": 200,
                    "equipment": 150
                },
                timeline="3年",
                success_indicators=[
                    "灾害损失减少30%",
                    "预警时间提前2小时",
                    "应急响应时间缩短50%"
                ],
                potential_barriers=[
                    "部门协调困难",
                    "资金投入不足",
                    "技术能力有限"
                ],
                mitigation_strategies=[
                    "建立统一指挥体系",
                    "多元化资金来源",
                    "加强技术合作"
                ]
            )
        ]
    
    def prioritize_recommendations(self, recommendations: List[PolicyRecommendation]) -> List[PolicyRecommendation]:
        """优先级排序"""
        def priority_score(rec):
            # 基于优先级、证据强度、利益相关者支持度等
            priority_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}[rec.priority]
            evidence_count = len(rec.evidence_base)
            stakeholder_count = len(rec.target_stakeholders)
            
            return priority_weight * (1 + evidence_count * 0.1 + stakeholder_count * 0.05)
            
        return sorted(recommendations, key=priority_score, reverse=True)


class PolicyImpactAssessment:
    """政策影响评估"""
    
    def __init__(self):
        self.assessment_results = {}
        
    def assess_economic_impact(self, recommendation: PolicyRecommendation) -> Dict:
        """评估经济影响"""
        # 简化的经济影响评估
        total_cost = sum(recommendation.resource_requirements.values())
        
        # 估算效益（基于预期结果）
        benefit_multiplier = {
            PolicyDomain.WATER_MANAGEMENT: 3.0,
            PolicyDomain.DISASTER_RISK_REDUCTION: 4.0,
            PolicyDomain.ENVIRONMENTAL_PROTECTION: 2.0
        }.get(recommendation.domain, 2.5)
        
        estimated_benefit = total_cost * benefit_multiplier
        
        return {
            'total_cost': total_cost,
            'estimated_benefit': estimated_benefit,
            'benefit_cost_ratio': estimated_benefit / total_cost if total_cost > 0 else 0,
            'payback_period': total_cost / (estimated_benefit / 10) if estimated_benefit > 0 else float('inf')
        }
    
    def assess_social_impact(self, recommendation: PolicyRecommendation) -> Dict:
        """评估社会影响"""
        return {
            'affected_population': 1000000,  # 简化估算
            'vulnerable_groups_benefit': 0.8,
            'public_acceptance': 0.7,
            'employment_creation': len(recommendation.implementation_steps) * 50
        }
    
    def assess_environmental_impact(self, recommendation: PolicyRecommendation) -> Dict:
        """评估环境影响"""
        return {
            'ecosystem_protection': 0.8,
            'carbon_footprint': -0.1,  # 负值表示减排
            'biodiversity_impact': 0.6,
            'resource_efficiency': 0.7
        }


class PolicySupportSystem:
    """政策支持系统"""
    
    def __init__(self):
        self.evidence_integrator = EvidenceIntegrator()
        self.stakeholder_analyzer = StakeholderAnalyzer()
        self.recommendation_engine = PolicyRecommendationEngine(
            self.evidence_integrator, self.stakeholder_analyzer
        )
        self.impact_assessment = PolicyImpactAssessment()
        
    def comprehensive_policy_analysis(self, glacier_projections: Dict) -> Dict:
        """综合政策分析"""
        # 1. 整合科学证据
        evidence_list = self.evidence_integrator.integrate_glacier_data(glacier_projections)
        for evidence in evidence_list:
            self.evidence_integrator.add_evidence(evidence)
            
        # 2. 综合证据
        water_evidence = self.evidence_integrator.synthesize_evidence(PolicyDomain.WATER_MANAGEMENT)
        disaster_evidence = self.evidence_integrator.synthesize_evidence(PolicyDomain.DISASTER_RISK_REDUCTION)
        
        # 3. 生成政策建议
        water_recommendations = self.recommendation_engine.generate_water_policy_recommendations(water_evidence)
        disaster_recommendations = self.recommendation_engine.generate_disaster_risk_recommendations()
        
        all_recommendations = water_recommendations + disaster_recommendations
        prioritized_recommendations = self.recommendation_engine.prioritize_recommendations(all_recommendations)
        
        # 4. 影响评估
        impact_assessments = {}
        for rec in prioritized_recommendations:
            impact_assessments[rec.id] = {
                'economic': self.impact_assessment.assess_economic_impact(rec),
                'social': self.impact_assessment.assess_social_impact(rec),
                'environmental': self.impact_assessment.assess_environmental_impact(rec)
            }
            
        return {
            'evidence_summary': {
                'water_management': water_evidence,
                'disaster_risk_reduction': disaster_evidence
            },
            'policy_recommendations': prioritized_recommendations,
            'impact_assessments': impact_assessments,
            'implementation_roadmap': self._create_implementation_roadmap(prioritized_recommendations)
        }
    
    def _create_implementation_roadmap(self, recommendations: List[PolicyRecommendation]) -> Dict:
        """创建实施路线图"""
        roadmap = {
            'phase_1': [],  # 0-2年
            'phase_2': [],  # 2-5年
            'phase_3': []   # 5年以上
        }
        
        for rec in recommendations:
            if "1年" in rec.timeline or "2年" in rec.timeline:
                roadmap['phase_1'].append(rec)
            elif "3年" in rec.timeline or "5年" in rec.timeline:
                roadmap['phase_2'].append(rec)
            else:
                roadmap['phase_3'].append(rec)
                
        return roadmap
    
    def generate_policy_brief(self, analysis_results: Dict) -> str:
        """生成政策简报"""
        brief = "\n=== 青藏高原冰川变化政策建议简报 ===\n\n"
        
        brief += "## 执行摘要\n"
        brief += "基于最新的冰川变化科学预测，本简报提出了水资源管理和灾害风险减缓的政策建议。\n\n"
        
        brief += "## 关键发现\n"
        for domain, evidence in analysis_results['evidence_summary'].items():
            brief += f"### {domain}\n"
            for finding in evidence['key_findings'][:3]:  # 前3个关键发现
                brief += f"- {finding}\n"
            brief += "\n"
            
        brief += "## 政策建议\n"
        for i, rec in enumerate(analysis_results['policy_recommendations'][:5], 1):  # 前5个建议
            brief += f"### {i}. {rec.title}\n"
            brief += f"**优先级**: {rec.priority}\n"
            brief += f"**实施时间**: {rec.timeline}\n"
            brief += f"**预期成果**: {', '.join(rec.expected_outcomes[:2])}\n\n"
            
        brief += "## 实施建议\n"
        brief += "建议按照三个阶段实施：\n"
        roadmap = analysis_results['implementation_roadmap']
        brief += f"- 第一阶段(0-2年): {len(roadmap['phase_1'])}项措施\n"
        brief += f"- 第二阶段(2-5年): {len(roadmap['phase_2'])}项措施\n"
        brief += f"- 第三阶段(5年以上): {len(roadmap['phase_3'])}项措施\n\n"
        
        return brief


def create_policy_support_system() -> PolicySupportSystem:
    """创建政策支持系统"""
    return PolicySupportSystem()


# 示例使用
if __name__ == "__main__":
    # 创建政策支持系统
    policy_system = create_policy_support_system()
    
    # 模拟冰川预测数据
    glacier_projections = {
        'volume_change': -0.3,  # 体积减少30%
        'runoff_change': -0.15,  # 径流减少15%
        'disaster_risk': 0.4,   # 灾害风险增加40%
        'temperature_increase': 2.5,
        'precipitation_change': -0.1
    }
    
    # 添加一些利益相关者
    stakeholders = [
        Stakeholder(
            id="water_ministry",
            name="水利部",
            type="government",
            influence_level=0.9,
            interest_level=0.8,
            position="supportive",
            concerns=["水资源安全", "供水保障"],
            resources=["政策制定权", "资金投入"],
            contact_info={"email": "contact@mwr.gov.cn"}
        ),
        Stakeholder(
            id="local_governments",
            name="地方政府",
            type="government",
            influence_level=0.7,
            interest_level=0.9,
            position="supportive",
            concerns=["经济发展", "民生保障"],
            resources=["执行能力", "地方资金"],
            contact_info={"email": "local@gov.cn"}
        )
    ]
    
    for stakeholder in stakeholders:
        policy_system.stakeholder_analyzer.add_stakeholder(stakeholder)
    
    # 进行综合政策分析
    analysis_results = policy_system.comprehensive_policy_analysis(glacier_projections)
    
    # 生成政策简报
    policy_brief = policy_system.generate_policy_brief(analysis_results)
    print(policy_brief)
    
    # 输出详细建议
    print("\n=== 详细政策建议 ===\n")
    for i, rec in enumerate(analysis_results['policy_recommendations'], 1):
        print(f"{i}. {rec.title}")
        print(f"   领域: {rec.domain.value}")
        print(f"   类型: {rec.policy_type.value}")
        print(f"   层级: {rec.level.value}")
        print(f"   优先级: {rec.priority}")
        print(f"   描述: {rec.description}")
        
        # 显示影响评估
        if rec.id in analysis_results['impact_assessments']:
            impact = analysis_results['impact_assessments'][rec.id]
            print(f"   经济影响: 成本效益比 {impact['economic']['benefit_cost_ratio']:.2f}")
            print(f"   社会影响: 受益人口 {impact['social']['affected_population']:,}")
        print()