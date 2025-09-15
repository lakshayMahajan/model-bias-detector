"""
Regression analyzer for comprehensive regression model performance analysis.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

from .base_analyzer import BaseAnalyzer, AnalysisResult, ProblemType, AnalysisType


class RegressionAnalyzer(BaseAnalyzer):
    """
    Comprehensive analyzer for regression problems.

    Provides R², MAE, RMSE, residual analysis, and advanced regression diagnostics.
    """

    def __init__(self):
        super().__init__(
            name="Regression Analyzer",
            supported_problem_types=[ProblemType.REGRESSION]
        )
        self.analysis_type = AnalysisType.PERFORMANCE

    def analyze(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        **kwargs
    ) -> AnalysisResult:
        """
        Perform comprehensive regression analysis.
        """
        start_time = time.time()

        # Input validation
        validation_errors = self.validate_inputs(data, predictions, true_values, problem_type)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")

        # Convert to numpy arrays
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(true_values, pd.Series):
            true_values = true_values.values

        # Create base result
        result = self._create_base_result(
            problem_type=ProblemType.REGRESSION,
            execution_time=0,
            sample_size=len(data),
            feature_count=len(data.columns)
        )

        if true_values is not None:
            # Calculate core metrics
            result.metrics = self._calculate_metrics(true_values, predictions)

            # Generate insights and recommendations
            result.insights = self._generate_insights(result.metrics)
            result.recommendations = self._generate_recommendations(result.metrics)

            # Create visualizations
            result.plots = self._create_plots(true_values, predictions)

            # Detailed analysis
            result.detailed_results = self._detailed_analysis(true_values, predictions)
        else:
            # No true values - prediction distribution analysis only
            result.metrics = self._prediction_distribution_metrics(predictions)
            result.insights = ["Analysis performed on predictions only (no true values provided)"]
            result.recommendations = ["Obtain true values for comprehensive regression evaluation"]
            result.plots = self._prediction_only_plots(predictions)

        # Update execution time
        result.execution_time = time.time() - start_time

        return result

    def _calculate_metrics(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        metrics = {}

        # Handle edge cases
        if len(true_values) == 0 or len(predictions) == 0:
            return {'error': 'Empty arrays provided'}

        # Basic metrics
        metrics['r2_score'] = r2_score(true_values, predictions)
        metrics['mean_squared_error'] = mean_squared_error(true_values, predictions)
        metrics['root_mean_squared_error'] = np.sqrt(metrics['mean_squared_error'])
        metrics['mean_absolute_error'] = mean_absolute_error(true_values, predictions)

        # Advanced metrics
        try:
            metrics['mean_absolute_percentage_error'] = mean_absolute_percentage_error(true_values, predictions)
        except:
            metrics['mean_absolute_percentage_error'] = float('inf')

        metrics['explained_variance_score'] = explained_variance_score(true_values, predictions)

        # Residual analysis
        residuals = true_values - predictions
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = float(stats.skew(residuals))
        metrics['residual_kurtosis'] = float(stats.kurtosis(residuals))

        # Relative metrics
        if np.mean(np.abs(true_values)) > 0:
            metrics['relative_mae'] = metrics['mean_absolute_error'] / np.mean(np.abs(true_values))
            metrics['relative_rmse'] = metrics['root_mean_squared_error'] / np.mean(np.abs(true_values))

        # Maximum error
        metrics['max_error'] = np.max(np.abs(residuals))

        # Correlation
        if np.std(true_values) > 0 and np.std(predictions) > 0:
            metrics['pearson_correlation'] = float(np.corrcoef(true_values, predictions)[0, 1])

        return metrics

    def _prediction_distribution_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for prediction distribution analysis."""
        return {
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions)),
            'prediction_min': float(np.min(predictions)),
            'prediction_max': float(np.max(predictions)),
            'prediction_range': float(np.max(predictions) - np.min(predictions)),
            'prediction_median': float(np.median(predictions)),
            'prediction_skewness': float(stats.skew(predictions)),
            'prediction_kurtosis': float(stats.kurtosis(predictions))
        }

    def _generate_insights(self, metrics: Dict[str, float]) -> List[str]:
        """Generate actionable insights from metrics."""
        insights = []

        # R² insights
        r2 = metrics.get('r2_score', 0)
        if r2 < 0:
            insights.append("⚠️ Negative R² - model performs worse than predicting the mean")
        elif r2 < 0.3:
            insights.append("📉 Low R² (< 0.3) - weak explanatory power")
        elif r2 < 0.7:
            insights.append("📊 Moderate R² - reasonable but improvable performance")
        elif r2 > 0.9:
            insights.append("🎯 Excellent R² (> 0.9) - strong model performance")

        # Residual analysis insights
        residual_skew = metrics.get('residual_skewness', 0)
        if abs(residual_skew) > 1:
            insights.append("📊 Highly skewed residuals - model may have systematic bias")
        elif abs(residual_skew) > 0.5:
            insights.append("📈 Moderately skewed residuals - check for outliers")

        residual_kurtosis = metrics.get('residual_kurtosis', 0)
        if abs(residual_kurtosis) > 2:
            insights.append("📏 Heavy-tailed residuals - potential outliers affecting model")

        # Error magnitude insights
        mape = metrics.get('mean_absolute_percentage_error', float('inf'))
        if mape != float('inf'):
            if mape < 0.1:
                insights.append("✅ Excellent prediction accuracy (MAPE < 10%)")
            elif mape < 0.2:
                insights.append("👍 Good prediction accuracy (MAPE < 20%)")
            elif mape > 0.5:
                insights.append("⚠️ Poor prediction accuracy (MAPE > 50%)")

        # Correlation insights
        corr = metrics.get('pearson_correlation', 0)
        if corr < 0.7:
            insights.append("📉 Low correlation between predictions and actual values")
        elif corr > 0.95:
            insights.append("🎯 Very high correlation - excellent linear relationship")

        return insights

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        r2 = metrics.get('r2_score', 0)
        mape = metrics.get('mean_absolute_percentage_error', float('inf'))
        residual_skew = metrics.get('residual_skewness', 0)

        if r2 < 0.5:
            recommendations.append("🔧 Consider feature engineering or polynomial features")
            recommendations.append("📊 Try different algorithms (Random Forest, XGBoost, Neural Networks)")

        if mape > 0.3:
            recommendations.append("📈 High prediction error - check for data quality issues")
            recommendations.append("🎯 Consider removing outliers or using robust regression methods")

        if abs(residual_skew) > 1:
            recommendations.append("📏 Skewed residuals suggest trying log transformation of target variable")
            recommendations.append("🔍 Investigate systematic patterns in residuals")

        residual_mean = abs(metrics.get('residual_mean', 0))
        if residual_mean > 0.1 * metrics.get('residual_std', 1):
            recommendations.append("⚖️ Non-zero mean residuals indicate systematic bias")

        if len(recommendations) == 0:
            recommendations.append("✅ Model performance is good - consider ensemble methods for further improvement")

        return recommendations

    def _create_plots(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Create comprehensive visualization plots."""
        plots = {}

        # Actual vs Predicted scatter plot
        plots['actual_vs_predicted'] = self._plot_actual_vs_predicted(true_values, predictions)

        # Residual plots
        plots['residuals'] = self._plot_residuals(true_values, predictions)

        # Distribution comparison
        plots['distribution_comparison'] = self._plot_distribution_comparison(true_values, predictions)

        # QQ plot for residuals
        plots['qq_plot'] = self._plot_qq_residuals(true_values, predictions)

        return plots

    def _prediction_only_plots(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Create plots for prediction-only analysis."""
        plots = {}
        plots['prediction_distribution'] = self._plot_prediction_distribution(predictions)
        return plots

    def _plot_actual_vs_predicted(self, true_values: np.ndarray, predictions: np.ndarray):
        """Create actual vs predicted scatter plot."""
        # Calculate R²
        r2 = r2_score(true_values, predictions)

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=true_values,
            y=predictions,
            mode='markers',
            name='Predictions',
            marker=dict(size=6, opacity=0.6),
            text=[f'Actual: {t:.2f}<br>Predicted: {p:.2f}<br>Error: {abs(t-p):.2f}'
                  for t, p in zip(true_values, predictions)],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red', width=2)
        ))

        fig.update_layout(
            title=f'Actual vs Predicted Values (R² = {r2:.3f})',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            width=600,
            height=500,
            showlegend=True
        )

        return fig

    def _plot_residuals(self, true_values: np.ndarray, predictions: np.ndarray):
        """Create residual analysis plots."""
        residuals = true_values - predictions

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Residuals vs Predicted',
                'Residuals vs Actual',
                'Residual Distribution',
                'Residuals vs Index'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers', name='Residuals vs Predicted',
                      marker=dict(size=4, opacity=0.6)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # Residuals vs Actual
        fig.add_trace(
            go.Scatter(x=true_values, y=residuals, mode='markers', name='Residuals vs Actual',
                      marker=dict(size=4, opacity=0.6)),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        # Residual distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Residual Distribution', nbinsx=30),
            row=2, col=1
        )

        # Residuals vs Index (order)
        fig.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, mode='markers',
                      name='Residuals vs Index', marker=dict(size=4, opacity=0.6)),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

        fig.update_layout(
            title='Residual Analysis',
            height=800,
            width=1000,
            showlegend=False
        )

        return fig

    def _plot_distribution_comparison(self, true_values: np.ndarray, predictions: np.ndarray):
        """Compare distributions of actual vs predicted values."""
        fig = go.Figure()

        # Actual values distribution
        fig.add_trace(go.Histogram(
            x=true_values,
            name='Actual Values',
            opacity=0.7,
            nbinsx=30
        ))

        # Predicted values distribution
        fig.add_trace(go.Histogram(
            x=predictions,
            name='Predicted Values',
            opacity=0.7,
            nbinsx=30
        ))

        fig.update_layout(
            title='Distribution Comparison: Actual vs Predicted',
            xaxis_title='Value',
            yaxis_title='Frequency',
            barmode='overlay',
            width=700,
            height=400
        )

        return fig

    def _plot_qq_residuals(self, true_values: np.ndarray, predictions: np.ndarray):
        """Create Q-Q plot for residuals normality check."""
        residuals = true_values - predictions

        # Calculate quantiles
        osm, osr = stats.probplot(residuals, dist="norm", plot=None)

        fig = go.Figure()

        # Q-Q plot
        fig.add_trace(go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            name='Residuals',
            marker=dict(size=5, opacity=0.7)
        ))

        # Reference line
        fig.add_trace(go.Scatter(
            x=[np.min(osm), np.max(osm)],
            y=[np.min(osm), np.max(osm)],
            mode='lines',
            name='Normal Distribution',
            line=dict(dash='dash', color='red')
        ))

        fig.update_layout(
            title='Q-Q Plot: Residuals vs Normal Distribution',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            width=500,
            height=400
        )

        return fig

    def _plot_prediction_distribution(self, predictions: np.ndarray):
        """Create prediction distribution plot."""
        fig = px.histogram(
            x=predictions,
            title='Prediction Distribution',
            nbins=30,
            marginal='box'
        )

        fig.update_layout(width=600, height=400)
        return fig

    def _detailed_analysis(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Perform detailed regression analysis."""
        detailed = {}
        residuals = true_values - predictions

        # Statistical tests
        detailed['normality_test'] = {
            'shapiro_wilk': stats.shapiro(residuals[:min(5000, len(residuals))]),  # Limit for large datasets
            'jarque_bera': stats.jarque_bera(residuals)
        }

        # Outlier detection
        q75, q25 = np.percentile(residuals, [75, 25])
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers = residuals[(residuals < q25 - outlier_threshold) | (residuals > q75 + outlier_threshold)]

        detailed['outlier_analysis'] = {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(residuals) * 100,
            'outlier_threshold': float(outlier_threshold)
        }

        # Error distribution by magnitude
        abs_errors = np.abs(residuals)
        detailed['error_distribution'] = {
            'errors_within_1_std': np.sum(abs_errors <= np.std(residuals)) / len(residuals),
            'errors_within_2_std': np.sum(abs_errors <= 2 * np.std(residuals)) / len(residuals),
            'errors_within_3_std': np.sum(abs_errors <= 3 * np.std(residuals)) / len(residuals)
        }

        # Performance by prediction magnitude
        true_ranges = np.percentile(true_values, [25, 50, 75])
        detailed['performance_by_range'] = {}

        for i, (low, high) in enumerate([(np.min(true_values), true_ranges[0]),
                                        (true_ranges[0], true_ranges[1]),
                                        (true_ranges[1], true_ranges[2]),
                                        (true_ranges[2], np.max(true_values))]):
            mask = (true_values >= low) & (true_values <= high)
            if np.sum(mask) > 0:
                range_r2 = r2_score(true_values[mask], predictions[mask])
                range_mae = mean_absolute_error(true_values[mask], predictions[mask])
                detailed['performance_by_range'][f'quartile_{i+1}'] = {
                    'r2': float(range_r2),
                    'mae': float(range_mae),
                    'sample_count': int(np.sum(mask))
                }

        return detailed