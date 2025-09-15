"""
Bias and fairness analyzer for comprehensive bias detection across protected attributes.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .base_analyzer import BaseAnalyzer, AnalysisResult, ProblemType, AnalysisType


class BiasAnalyzer(BaseAnalyzer):
    """
    Comprehensive analyzer for bias and fairness detection.

    Supports demographic parity, equalized odds, intersectional analysis,
    and comprehensive bias metrics across multiple protected attributes.
    """

    def __init__(self):
        super().__init__(
            name="Bias & Fairness Analyzer",
            supported_problem_types=[
                ProblemType.BINARY_CLASSIFICATION,
                ProblemType.MULTICLASS_CLASSIFICATION,
                ProblemType.REGRESSION
            ]
        )
        self.analysis_type = AnalysisType.BIAS_FAIRNESS

    def analyze(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        protected_attributes: Optional[List[str]] = None,
        **kwargs
    ) -> AnalysisResult:
        """
        Perform comprehensive bias and fairness analysis.
        """
        start_time = time.time()

        # Input validation
        validation_errors = self.validate_inputs(data, predictions, true_values, problem_type)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")

        # Auto-detect problem type if not provided
        if problem_type is None:
            problem_type = self.detect_problem_type(predictions, true_values)

        # Auto-detect protected attributes if not provided
        if protected_attributes is None:
            protected_attributes = self._detect_protected_attributes(data)

        if not protected_attributes:
            raise ValueError("No protected attributes found or specified for bias analysis")

        # Convert to numpy arrays
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(true_values, pd.Series):
            true_values = true_values.values

        # Create base result
        result = self._create_base_result(
            problem_type=problem_type,
            execution_time=0,
            sample_size=len(data),
            feature_count=len(data.columns)
        )

        # Perform bias analysis
        result.metrics = self._calculate_bias_metrics(
            data, predictions, true_values, protected_attributes, problem_type
        )

        # Generate insights and recommendations
        result.insights = self._generate_insights(result.metrics, protected_attributes)
        result.recommendations = self._generate_recommendations(result.metrics, protected_attributes)

        # Create visualizations
        result.plots = self._create_plots(
            data, predictions, true_values, protected_attributes, problem_type
        )

        # Detailed analysis
        result.detailed_results = self._detailed_analysis(
            data, predictions, true_values, protected_attributes, problem_type
        )

        # Update execution time
        result.execution_time = time.time() - start_time

        return result

    def _detect_protected_attributes(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect potential protected attributes."""
        protected_keywords = [
            'gender', 'race', 'ethnicity', 'age', 'religion', 'disability',
            'sexual_orientation', 'nationality', 'protected', 'sensitive',
            'demographic', 'group', 'category', 'sex'
        ]

        detected = []
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in protected_keywords):
                # Check if it's categorical with reasonable number of unique values
                if data[col].dtype == 'object' or data[col].nunique() <= 10:
                    detected.append(col)

        return detected

    def _calculate_bias_metrics(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray],
        protected_attributes: List[str],
        problem_type: ProblemType
    ) -> Dict[str, float]:
        """Calculate comprehensive bias metrics."""
        metrics = {}

        for attr in protected_attributes:
            if attr not in data.columns:
                continue

            attr_data = data[attr]
            unique_groups = attr_data.unique()

            # Skip if only one group
            if len(unique_groups) <= 1:
                continue

            # Demographic parity (statistical parity)
            positive_rates = {}
            for group in unique_groups:
                mask = attr_data == group
                if problem_type == ProblemType.REGRESSION:
                    # For regression, use above-median as "positive"
                    positive_rates[group] = np.mean(predictions[mask] > np.median(predictions))
                else:
                    # For classification
                    positive_rates[group] = np.mean(predictions[mask] == 1) if len(np.unique(predictions)) == 2 else np.mean(predictions[mask])

            # Calculate disparate impact ratio
            if len(positive_rates) >= 2:
                rates = list(positive_rates.values())
                max_rate = max(rates)
                min_rate = min(rates)
                metrics[f'{attr}_disparate_impact_ratio'] = min_rate / max_rate if max_rate > 0 else 0

                # Demographic parity difference
                metrics[f'{attr}_demographic_parity_diff'] = max_rate - min_rate

            # If true values available, calculate additional fairness metrics
            if true_values is not None:
                metrics.update(self._calculate_equalized_odds_metrics(
                    attr_data, predictions, true_values, attr, problem_type
                ))

        # Overall bias score
        metrics['overall_bias_score'] = self._calculate_overall_bias_score(metrics)

        return metrics

    def _calculate_equalized_odds_metrics(
        self,
        attr_data: pd.Series,
        predictions: np.ndarray,
        true_values: np.ndarray,
        attr_name: str,
        problem_type: ProblemType
    ) -> Dict[str, float]:
        """Calculate equalized odds and related metrics."""
        metrics = {}
        unique_groups = attr_data.unique()

        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            # True positive rates and false positive rates by group
            tpr_by_group = {}
            fpr_by_group = {}
            accuracy_by_group = {}

            for group in unique_groups:
                mask = attr_data == group
                group_true = true_values[mask]
                group_pred = predictions[mask]

                if len(group_true) == 0:
                    continue

                # Calculate TPR and FPR for binary case
                if problem_type == ProblemType.BINARY_CLASSIFICATION:
                    tp = np.sum((group_true == 1) & (group_pred == 1))
                    fp = np.sum((group_true == 0) & (group_pred == 1))
                    tn = np.sum((group_true == 0) & (group_pred == 0))
                    fn = np.sum((group_true == 1) & (group_pred == 0))

                    tpr_by_group[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr_by_group[group] = fp / (fp + tn) if (fp + tn) > 0 else 0

                # Accuracy for all classification types
                accuracy_by_group[group] = np.mean(group_true == group_pred)

            # Equalized odds (TPR difference)
            if len(tpr_by_group) >= 2:
                tpr_values = list(tpr_by_group.values())
                metrics[f'{attr_name}_tpr_difference'] = max(tpr_values) - min(tpr_values)

                fpr_values = list(fpr_by_group.values())
                metrics[f'{attr_name}_fpr_difference'] = max(fpr_values) - min(fpr_values)

            # Accuracy parity
            if len(accuracy_by_group) >= 2:
                acc_values = list(accuracy_by_group.values())
                metrics[f'{attr_name}_accuracy_difference'] = max(acc_values) - min(acc_values)

        elif problem_type == ProblemType.REGRESSION:
            # For regression: calculate performance differences
            mae_by_group = {}
            r2_by_group = {}

            for group in unique_groups:
                mask = attr_data == group
                group_true = true_values[mask]
                group_pred = predictions[mask]

                if len(group_true) == 0:
                    continue

                mae_by_group[group] = np.mean(np.abs(group_true - group_pred))

                # R² calculation
                ss_res = np.sum((group_true - group_pred) ** 2)
                ss_tot = np.sum((group_true - np.mean(group_true)) ** 2)
                r2_by_group[group] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Performance parity
            if len(mae_by_group) >= 2:
                mae_values = list(mae_by_group.values())
                metrics[f'{attr_name}_mae_difference'] = max(mae_values) - min(mae_values)

                r2_values = list(r2_by_group.values())
                metrics[f'{attr_name}_r2_difference'] = max(r2_values) - min(r2_values)

        return metrics

    def _calculate_overall_bias_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall bias score (0 = no bias, 1 = maximum bias)."""
        bias_indicators = []

        # Collect disparate impact ratios
        for key, value in metrics.items():
            if 'disparate_impact_ratio' in key:
                # Convert to bias indicator (further from 1.0 = more bias)
                bias_indicators.append(abs(1.0 - value))
            elif any(suffix in key for suffix in ['_difference', '_diff']):
                # Direct bias measures
                bias_indicators.append(value)

        if not bias_indicators:
            return 0.0

        # Return average bias score, capped at 1.0
        return min(1.0, np.mean(bias_indicators))

    def _generate_insights(self, metrics: Dict[str, float], protected_attributes: List[str]) -> List[str]:
        """Generate actionable insights from bias metrics."""
        insights = []

        # Overall bias assessment
        overall_bias = metrics.get('overall_bias_score', 0)
        if overall_bias < 0.1:
            insights.append("✅ Low overall bias detected - model appears relatively fair")
        elif overall_bias < 0.3:
            insights.append("⚠️ Moderate bias detected - some disparities present")
        else:
            insights.append("🚨 High bias detected - significant fairness concerns")

        # Attribute-specific insights
        for attr in protected_attributes:
            disparate_impact = metrics.get(f'{attr}_disparate_impact_ratio', 1.0)

            if disparate_impact < 0.8:
                insights.append(f"🚨 {attr}: Disparate impact detected (ratio: {disparate_impact:.2f})")
            elif disparate_impact < 0.9:
                insights.append(f"⚠️ {attr}: Potential bias concern (ratio: {disparate_impact:.2f})")

            # Demographic parity
            demo_parity = metrics.get(f'{attr}_demographic_parity_diff', 0)
            if demo_parity > 0.2:
                insights.append(f"📊 {attr}: Large difference in positive prediction rates ({demo_parity:.2f})")

            # Equalized odds
            tpr_diff = metrics.get(f'{attr}_tpr_difference', 0)
            if tpr_diff > 0.1:
                insights.append(f"⚖️ {attr}: Unequal true positive rates across groups")

        return insights

    def _generate_recommendations(self, metrics: Dict[str, float], protected_attributes: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        overall_bias = metrics.get('overall_bias_score', 0)

        if overall_bias > 0.3:
            recommendations.append("🔧 Consider bias mitigation techniques (pre-processing, in-processing, post-processing)")
            recommendations.append("📊 Collect more balanced training data across protected groups")

        # Specific recommendations by metric type
        for attr in protected_attributes:
            disparate_impact = metrics.get(f'{attr}_disparate_impact_ratio', 1.0)

            if disparate_impact < 0.8:
                recommendations.append(f"⚖️ {attr}: Apply threshold optimization to improve fairness")
                recommendations.append(f"🎯 {attr}: Consider demographic parity constraints during training")

            tpr_diff = metrics.get(f'{attr}_tpr_difference', 0)
            if tpr_diff > 0.15:
                recommendations.append(f"📊 {attr}: Implement equalized odds post-processing")

            demo_parity = metrics.get(f'{attr}_demographic_parity_diff', 0)
            if demo_parity > 0.25:
                recommendations.append(f"🔄 {attr}: Apply re-sampling or re-weighting techniques")

        if overall_bias > 0.2:
            recommendations.append("📋 Conduct regular bias audits and monitoring")
            recommendations.append("🎓 Consider fairness-aware machine learning algorithms")

        if len(recommendations) == 0:
            recommendations.append("✅ Model shows good fairness properties - maintain monitoring")

        return recommendations

    def _create_plots(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray],
        protected_attributes: List[str],
        problem_type: ProblemType
    ) -> Dict[str, Any]:
        """Create comprehensive bias visualization plots."""
        plots = {}

        # Prediction distribution by protected attribute
        plots['prediction_distribution'] = self._plot_prediction_distribution(
            data, predictions, protected_attributes
        )

        # Bias metrics heatmap
        plots['bias_heatmap'] = self._plot_bias_heatmap(data, predictions, true_values, protected_attributes, problem_type)

        # Performance by group
        if true_values is not None:
            plots['performance_by_group'] = self._plot_performance_by_group(
                data, predictions, true_values, protected_attributes, problem_type
            )

        # Intersectional analysis
        if len(protected_attributes) >= 2:
            plots['intersectional_analysis'] = self._plot_intersectional_analysis(
                data, predictions, protected_attributes[:2]  # Limit to first 2 for visualization
            )

        return plots

    def _plot_prediction_distribution(self, data: pd.DataFrame, predictions: np.ndarray, protected_attributes: List[str]):
        """Plot prediction distributions by protected attributes."""
        if not protected_attributes:
            return None

        attr = protected_attributes[0]  # Use first attribute
        if attr not in data.columns:
            return None

        df = pd.DataFrame({
            'predictions': predictions,
            attr: data[attr].values
        })

        fig = px.box(
            df, x=attr, y='predictions',
            title=f'Prediction Distribution by {attr}',
            points='outliers'
        )

        fig.update_layout(width=700, height=400)
        return fig

    def _plot_bias_heatmap(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray],
        protected_attributes: List[str],
        problem_type: ProblemType
    ):
        """Create bias metrics heatmap."""
        bias_data = []

        for attr in protected_attributes:
            if attr not in data.columns:
                continue

            attr_data = data[attr]
            unique_groups = attr_data.unique()

            for group in unique_groups:
                mask = attr_data == group
                group_predictions = predictions[mask]

                if problem_type == ProblemType.REGRESSION:
                    avg_prediction = np.mean(group_predictions)
                    metric_value = avg_prediction
                else:
                    positive_rate = np.mean(group_predictions == 1) if len(np.unique(predictions)) == 2 else np.mean(group_predictions)
                    metric_value = positive_rate

                bias_data.append({
                    'Attribute': attr,
                    'Group': str(group),
                    'Metric': metric_value,
                    'Sample_Size': len(group_predictions)
                })

        if not bias_data:
            return None

        df = pd.DataFrame(bias_data)

        fig = px.scatter(
            df, x='Attribute', y='Group', size='Sample_Size', color='Metric',
            title='Bias Metrics by Protected Attribute and Group',
            color_continuous_scale='RdYlBu_r'
        )

        fig.update_layout(width=800, height=500)
        return fig

    def _plot_performance_by_group(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: np.ndarray,
        protected_attributes: List[str],
        problem_type: ProblemType
    ):
        """Plot performance metrics by protected groups."""
        if not protected_attributes:
            return None

        attr = protected_attributes[0]
        if attr not in data.columns:
            return None

        performance_data = []
        attr_data = data[attr]
        unique_groups = attr_data.unique()

        for group in unique_groups:
            mask = attr_data == group
            group_true = true_values[mask]
            group_pred = predictions[mask]

            if len(group_true) == 0:
                continue

            if problem_type == ProblemType.REGRESSION:
                mae = np.mean(np.abs(group_true - group_pred))
                r2 = 1 - np.sum((group_true - group_pred)**2) / np.sum((group_true - np.mean(group_true))**2) if np.var(group_true) > 0 else 0
                performance_data.append({'Group': str(group), 'MAE': mae, 'R²': r2})
            else:
                accuracy = np.mean(group_true == group_pred)
                performance_data.append({'Group': str(group), 'Accuracy': accuracy})

        if not performance_data:
            return None

        df = pd.DataFrame(performance_data)

        if problem_type == ProblemType.REGRESSION:
            fig = make_subplots(rows=1, cols=2, subplot_titles=['MAE by Group', 'R² by Group'])
            fig.add_trace(go.Bar(x=df['Group'], y=df['MAE'], name='MAE'), row=1, col=1)
            fig.add_trace(go.Bar(x=df['Group'], y=df['R²'], name='R²'), row=1, col=2)
        else:
            fig = px.bar(df, x='Group', y='Accuracy', title=f'Accuracy by {attr}')

        fig.update_layout(width=800, height=400)
        return fig

    def _plot_intersectional_analysis(self, data: pd.DataFrame, predictions: np.ndarray, attributes: List[str]):
        """Create intersectional bias analysis plot."""
        if len(attributes) < 2:
            return None

        attr1, attr2 = attributes[0], attributes[1]
        if attr1 not in data.columns or attr2 not in data.columns:
            return None

        # Create intersectional groups
        intersectional_data = []
        for val1 in data[attr1].unique():
            for val2 in data[attr2].unique():
                mask = (data[attr1] == val1) & (data[attr2] == val2)
                group_predictions = predictions[mask]

                if len(group_predictions) > 0:
                    avg_prediction = np.mean(group_predictions)
                    intersectional_data.append({
                        attr1: str(val1),
                        attr2: str(val2),
                        'Average_Prediction': avg_prediction,
                        'Sample_Size': len(group_predictions)
                    })

        if not intersectional_data:
            return None

        df = pd.DataFrame(intersectional_data)

        fig = px.scatter(
            df, x=attr1, y=attr2, size='Sample_Size', color='Average_Prediction',
            title=f'Intersectional Analysis: {attr1} × {attr2}',
            color_continuous_scale='RdYlBu_r'
        )

        fig.update_layout(width=700, height=500)
        return fig

    def _detailed_analysis(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray],
        protected_attributes: List[str],
        problem_type: ProblemType
    ) -> Dict[str, Any]:
        """Perform detailed bias analysis."""
        detailed = {}

        # Group statistics
        detailed['group_statistics'] = {}
        for attr in protected_attributes:
            if attr not in data.columns:
                continue

            attr_data = data[attr]
            group_stats = {}

            for group in attr_data.unique():
                mask = attr_data == group
                group_predictions = predictions[mask]

                stats = {
                    'sample_size': int(len(group_predictions)),
                    'percentage_of_total': float(len(group_predictions) / len(predictions) * 100)
                }

                if problem_type == ProblemType.REGRESSION:
                    stats.update({
                        'mean_prediction': float(np.mean(group_predictions)),
                        'std_prediction': float(np.std(group_predictions)),
                        'median_prediction': float(np.median(group_predictions))
                    })
                else:
                    if len(np.unique(predictions)) == 2:
                        stats['positive_rate'] = float(np.mean(group_predictions == 1))
                    else:
                        stats['prediction_distribution'] = {
                            str(val): float(np.mean(group_predictions == val))
                            for val in np.unique(predictions)
                        }

                if true_values is not None:
                    group_true = true_values[mask]
                    if problem_type == ProblemType.REGRESSION:
                        stats['mae'] = float(np.mean(np.abs(group_true - group_predictions)))
                        stats['rmse'] = float(np.sqrt(np.mean((group_true - group_predictions)**2)))
                    else:
                        stats['accuracy'] = float(np.mean(group_true == group_predictions))

                group_stats[str(group)] = stats

            detailed['group_statistics'][attr] = group_stats

        # Bias summary
        detailed['bias_summary'] = {
            'total_protected_attributes': len(protected_attributes),
            'attributes_with_bias': sum(1 for attr in protected_attributes
                                      if any(f'{attr}_' in key for key in detailed.get('bias_metrics', {}).keys())),
            'bias_severity': 'high' if detailed.get('overall_bias_score', 0) > 0.3 else
                           'moderate' if detailed.get('overall_bias_score', 0) > 0.1 else 'low'
        }

        return detailed