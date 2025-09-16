"""
Analysis engine that orchestrates all analyzers and provides comprehensive model analysis.
"""

import time
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .base_analyzer import BaseAnalyzer, AnalysisResult, ProblemType, AnalysisType
from .classification import ClassificationAnalyzer
from .regression import RegressionAnalyzer
from .bias_fairness import BiasAnalyzer
from .data_quality import DataQualityAnalyzer
from .explainability import ExplainabilityAnalyzer

if TYPE_CHECKING:
    from utils.export_manager import ExportManager


@dataclass
class AnalysisReport:
    """
    Comprehensive analysis report containing results from all analyzers.
    """
    problem_type: ProblemType
    sample_size: int
    feature_count: int

    # Individual analyzer results
    performance_analysis: Optional[AnalysisResult] = None
    bias_analysis: Optional[AnalysisResult] = None
    data_quality_analysis: Optional[AnalysisResult] = None
    explainability_analysis: Optional[AnalysisResult] = None

    # Summary metrics
    overall_score: float = 0.0
    key_insights: List[str] = None
    priority_recommendations: List[str] = None

    # Metadata
    analysis_timestamp: str = ""
    total_execution_time: float = 0.0

    def __post_init__(self):
        if self.key_insights is None:
            self.key_insights = []
        if self.priority_recommendations is None:
            self.priority_recommendations = []


class AnalysisEngine:
    """
    Central orchestrator for all ML model analysis.

    Coordinates multiple analyzers, auto-detects analysis types,
    and provides comprehensive reporting capabilities.
    """

    def __init__(self):
        # Initialize all analyzers
        self.analyzers = {
            AnalysisType.PERFORMANCE: {
                ProblemType.BINARY_CLASSIFICATION: ClassificationAnalyzer(),
                ProblemType.MULTICLASS_CLASSIFICATION: ClassificationAnalyzer(),
                ProblemType.REGRESSION: RegressionAnalyzer()
            },
            AnalysisType.BIAS_FAIRNESS: BiasAnalyzer(),
            AnalysisType.DATA_QUALITY: DataQualityAnalyzer(),
            AnalysisType.EXPLAINABILITY: ExplainabilityAnalyzer()
        }

    def analyze_model(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        analysis_types: Optional[List[AnalysisType]] = None,
        protected_attributes: Optional[List[str]] = None,
        probability_scores: Optional[Union[pd.Series, np.ndarray]] = None,
        model=None,
        **kwargs
    ) -> AnalysisReport:
        """
        Perform comprehensive model analysis.

        Args:
            data: Input features/data
            predictions: Model predictions
            true_values: Ground truth values (optional)
            problem_type: Type of ML problem (auto-detected if None)
            analysis_types: Types of analysis to perform (all if None)
            protected_attributes: Protected attributes for bias analysis
            probability_scores: Prediction probabilities (for classification)
            model: Original model object (for better explainability)
            **kwargs: Additional parameters

        Returns:
            Comprehensive AnalysisReport
        """
        start_time = time.time()

        # Auto-detect problem type if not provided
        if problem_type is None:
            problem_type = self._auto_detect_problem_type(predictions, true_values)

        # Default to all analysis types if not specified
        if analysis_types is None:
            analysis_types = [
                AnalysisType.PERFORMANCE,
                AnalysisType.BIAS_FAIRNESS,
                AnalysisType.DATA_QUALITY,
                AnalysisType.EXPLAINABILITY
            ]

        # Create analysis report
        report = AnalysisReport(
            problem_type=problem_type,
            sample_size=len(data),
            feature_count=len(data.columns),
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Perform each type of analysis
        for analysis_type in analysis_types:
            try:
                result = self._run_analysis(
                    analysis_type=analysis_type,
                    data=data,
                    predictions=predictions,
                    true_values=true_values,
                    problem_type=problem_type,
                    protected_attributes=protected_attributes,
                    probability_scores=probability_scores,
                    model=model,
                    **kwargs
                )

                # Store results in report
                if analysis_type == AnalysisType.PERFORMANCE:
                    report.performance_analysis = result
                elif analysis_type == AnalysisType.BIAS_FAIRNESS:
                    report.bias_analysis = result
                elif analysis_type == AnalysisType.DATA_QUALITY:
                    report.data_quality_analysis = result
                elif analysis_type == AnalysisType.EXPLAINABILITY:
                    report.explainability_analysis = result

            except Exception as e:
                # Log error but continue with other analyses
                print(f"Warning: {analysis_type.value} analysis failed: {str(e)}")
                continue

        # Generate summary insights and recommendations
        report.key_insights = self._generate_summary_insights(report)
        report.priority_recommendations = self._generate_priority_recommendations(report)
        report.overall_score = self._calculate_overall_score(report)

        # Update execution time
        report.total_execution_time = time.time() - start_time

        return report

    def _auto_detect_problem_type(
        self,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> ProblemType:
        """Auto-detect the ML problem type from predictions and true values."""
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(true_values, pd.Series):
            true_values = true_values.values

        # Check unique values in predictions
        unique_predictions = np.unique(predictions)

        # Binary classification: exactly 2 unique values
        if len(unique_predictions) == 2:
            return ProblemType.BINARY_CLASSIFICATION

        # Check if predictions are integers and small number of classes
        if np.all(predictions == predictions.astype(int)) and len(unique_predictions) <= 20:
            return ProblemType.MULTICLASS_CLASSIFICATION

        # Check true values if available for additional context
        if true_values is not None:
            unique_true = np.unique(true_values)
            if len(unique_true) <= 20 and np.all(true_values == true_values.astype(int)):
                if len(unique_true) == 2:
                    return ProblemType.BINARY_CLASSIFICATION
                else:
                    return ProblemType.MULTICLASS_CLASSIFICATION

        # Default to regression for continuous values
        return ProblemType.REGRESSION

    def _run_analysis(
        self,
        analysis_type: AnalysisType,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        problem_type: ProblemType,
        **kwargs
    ) -> AnalysisResult:
        """Run a specific type of analysis."""

        if analysis_type == AnalysisType.PERFORMANCE:
            # Get the appropriate performance analyzer
            analyzer = self.analyzers[analysis_type].get(problem_type)
            if analyzer is None:
                raise ValueError(f"No performance analyzer available for {problem_type}")

            # Pass probability scores if available
            probability_scores = kwargs.get('probability_scores')
            return analyzer.analyze(
                data=data,
                predictions=predictions,
                true_values=true_values,
                problem_type=problem_type,
                probability_scores=probability_scores
            )

        elif analysis_type == AnalysisType.BIAS_FAIRNESS:
            analyzer = self.analyzers[analysis_type]
            protected_attributes = kwargs.get('protected_attributes')

            return analyzer.analyze(
                data=data,
                predictions=predictions,
                true_values=true_values,
                problem_type=problem_type,
                protected_attributes=protected_attributes
            )

        elif analysis_type == AnalysisType.DATA_QUALITY:
            analyzer = self.analyzers[analysis_type]
            return analyzer.analyze(
                data=data,
                predictions=predictions,
                true_values=true_values,
                problem_type=problem_type
            )

        elif analysis_type == AnalysisType.EXPLAINABILITY:
            analyzer = self.analyzers[analysis_type]
            model = kwargs.get('model')

            return analyzer.analyze(
                data=data,
                predictions=predictions,
                true_values=true_values,
                problem_type=problem_type,
                model=model
            )

        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    def _generate_summary_insights(self, report: AnalysisReport) -> List[str]:
        """Generate high-level summary insights from all analyses."""
        insights = []

        # Performance insights
        if report.performance_analysis:
            perf_insights = report.performance_analysis.insights[:2]  # Top 2
            insights.extend(perf_insights)

        # Bias insights (high priority)
        if report.bias_analysis:
            bias_insights = [insight for insight in report.bias_analysis.insights if "🚨" in insight or "⚠️" in insight]
            insights.extend(bias_insights[:2])  # Top 2 critical bias issues

        # Data quality insights (critical issues only)
        if report.data_quality_analysis:
            quality_insights = [insight for insight in report.data_quality_analysis.insights if "🚨" in insight]
            insights.extend(quality_insights[:1])  # Top critical issue

        # Explainability insights
        if report.explainability_analysis:
            exp_insights = report.explainability_analysis.insights[:1]  # Top insight
            insights.extend(exp_insights)

        # Add overall assessment
        overall_score = report.overall_score
        if overall_score > 0.8:
            insights.insert(0, "✅ Overall model performance and fairness appear good")
        elif overall_score > 0.6:
            insights.insert(0, "⚠️ Model has moderate performance with some areas for improvement")
        else:
            insights.insert(0, "🚨 Model has significant issues requiring attention")

        return insights[:6]  # Limit to top 6 insights

    def _generate_priority_recommendations(self, report: AnalysisReport) -> List[str]:
        """Generate prioritized recommendations from all analyses."""
        recommendations = []

        # Critical bias issues first
        if report.bias_analysis:
            bias_recs = [rec for rec in report.bias_analysis.recommendations if "🚨" in rec or "⚖️" in rec]
            recommendations.extend(bias_recs[:2])

        # Critical data quality issues
        if report.data_quality_analysis:
            quality_recs = [rec for rec in report.data_quality_analysis.recommendations if "🔧" in rec]
            recommendations.extend(quality_recs[:2])

        # Performance improvements
        if report.performance_analysis:
            perf_recs = report.performance_analysis.recommendations[:2]
            recommendations.extend(perf_recs)

        # Explainability recommendations
        if report.explainability_analysis:
            exp_recs = report.explainability_analysis.recommendations[:1]
            recommendations.extend(exp_recs)

        return recommendations[:5]  # Limit to top 5 recommendations

    def _calculate_overall_score(self, report: AnalysisReport) -> float:
        """Calculate an overall quality score for the model."""
        scores = []
        weights = []

        # Performance score (weight: 0.4)
        if report.performance_analysis:
            perf_score = self._extract_performance_score(report.performance_analysis, report.problem_type)
            scores.append(perf_score)
            weights.append(0.4)

        # Bias score (weight: 0.3) - inverted because lower bias is better
        if report.bias_analysis:
            bias_score = 1.0 - report.bias_analysis.metrics.get('overall_bias_score', 0.5)
            scores.append(bias_score)
            weights.append(0.3)

        # Data quality score (weight: 0.2)
        if report.data_quality_analysis:
            quality_score = report.data_quality_analysis.confidence_score
            scores.append(quality_score)
            weights.append(0.2)

        # Explainability score (weight: 0.1)
        if report.explainability_analysis:
            exp_score = self._extract_explainability_score(report.explainability_analysis)
            scores.append(exp_score)
            weights.append(0.1)

        if not scores:
            return 0.5  # Default neutral score

        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _extract_performance_score(self, analysis: AnalysisResult, problem_type: ProblemType) -> float:
        """Extract a normalized performance score from analysis results."""
        metrics = analysis.metrics

        if problem_type == ProblemType.REGRESSION:
            # For regression, use R² (already 0-1, higher is better)
            r2 = metrics.get('r2_score', 0)
            return max(0, min(1, r2))  # Clamp to 0-1

        else:  # Classification
            # For classification, use accuracy
            accuracy = metrics.get('accuracy', 0.5)
            return accuracy

    def _extract_explainability_score(self, analysis: AnalysisResult) -> float:
        """Extract a normalized explainability score."""
        metrics = analysis.metrics

        # Use feature diversity as a proxy for explainability quality
        diversity = metrics.get('feature_diversity', 0.5)
        return diversity

    def get_quick_summary(self, report: AnalysisReport) -> Dict[str, Any]:
        """Get a quick summary of the analysis report."""
        summary = {
            'problem_type': report.problem_type.value,
            'overall_score': round(report.overall_score, 3),
            'sample_size': report.sample_size,
            'feature_count': report.feature_count,
            'key_insights': report.key_insights[:3],  # Top 3
            'priority_recommendations': report.priority_recommendations[:3],  # Top 3
            'execution_time': round(report.total_execution_time, 2)
        }

        # Add key metrics from each analysis
        if report.performance_analysis:
            summary['performance'] = self._extract_key_performance_metrics(
                report.performance_analysis, report.problem_type
            )

        if report.bias_analysis:
            summary['bias_score'] = round(report.bias_analysis.metrics.get('overall_bias_score', 0), 3)

        if report.data_quality_analysis:
            summary['data_quality'] = {
                'completeness': round(report.data_quality_analysis.metrics.get('completeness_rate', 0), 3),
                'confidence_score': round(report.data_quality_analysis.confidence_score, 3)
            }

        return summary

    def _extract_key_performance_metrics(self, analysis: AnalysisResult, problem_type: ProblemType) -> Dict[str, float]:
        """Extract key performance metrics for summary."""
        metrics = analysis.metrics

        if problem_type == ProblemType.REGRESSION:
            return {
                'r2_score': round(metrics.get('r2_score', 0), 3),
                'rmse': round(metrics.get('root_mean_squared_error', 0), 3),
                'mae': round(metrics.get('mean_absolute_error', 0), 3)
            }
        else:  # Classification
            return {
                'accuracy': round(metrics.get('accuracy', 0), 3),
                'precision': round(metrics.get('precision', 0), 3),
                'recall': round(metrics.get('recall', 0), 3),
                'f1_score': round(metrics.get('f1_score', 0), 3)
            }

    def get_available_analyzers(self) -> Dict[str, List[str]]:
        """Get information about available analyzers."""
        return {
            'performance_analyzers': [
                'Classification Analyzer (Binary & Multiclass)',
                'Regression Analyzer'
            ],
            'specialized_analyzers': [
                'Bias & Fairness Analyzer',
                'Data Quality Analyzer',
                'Explainability Analyzer'
            ],
            'supported_problem_types': [pt.value for pt in ProblemType if pt != ProblemType.AUTO_DETECT]
        }