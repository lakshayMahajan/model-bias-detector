"""
Data quality analyzer for comprehensive data validation and quality assessment.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

from .base_analyzer import BaseAnalyzer, AnalysisResult, ProblemType, AnalysisType


class DataQualityAnalyzer(BaseAnalyzer):
    """
    Comprehensive analyzer for data quality assessment.

    Validates data integrity, detects outliers, missing values,
    and provides comprehensive quality metrics.
    """

    def __init__(self):
        super().__init__(
            name="Data Quality Analyzer",
            supported_problem_types=[
                ProblemType.BINARY_CLASSIFICATION,
                ProblemType.MULTICLASS_CLASSIFICATION,
                ProblemType.REGRESSION,
                ProblemType.AUTO_DETECT
            ]
        )
        self.analysis_type = AnalysisType.DATA_QUALITY

    def analyze(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        **kwargs
    ) -> AnalysisResult:
        """
        Perform comprehensive data quality analysis.
        """
        start_time = time.time()

        # Input validation
        validation_errors = self.validate_inputs(data, predictions, true_values, problem_type)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")

        # Auto-detect problem type if not provided
        if problem_type is None:
            problem_type = self.detect_problem_type(predictions, true_values)

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

        # Calculate quality metrics
        result.metrics = self._calculate_quality_metrics(data, predictions, true_values)

        # Generate insights and recommendations
        result.insights = self._generate_insights(result.metrics, data)
        result.recommendations = self._generate_recommendations(result.metrics, data)

        # Create visualizations
        result.plots = self._create_plots(data, predictions, true_values)

        # Detailed analysis
        result.detailed_results = self._detailed_analysis(data, predictions, true_values)

        # Update execution time
        result.execution_time = time.time() - start_time

        # Set confidence score based on data quality
        result.confidence_score = self._calculate_confidence_score(result.metrics)

        return result

    def _calculate_quality_metrics(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate comprehensive data quality metrics."""
        metrics = {}

        # Basic completeness metrics
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        metrics['completeness_rate'] = (total_cells - missing_cells) / total_cells
        metrics['missing_percentage'] = (missing_cells / total_cells) * 100

        # Column-wise missing data
        metrics['columns_with_missing'] = (data.isnull().any()).sum()
        metrics['percentage_columns_with_missing'] = (metrics['columns_with_missing'] / len(data.columns)) * 100

        # Row-wise missing data
        rows_with_missing = data.isnull().any(axis=1).sum()
        metrics['rows_with_missing'] = rows_with_missing
        metrics['percentage_rows_with_missing'] = (rows_with_missing / len(data)) * 100

        # Duplicate detection
        duplicates = data.duplicated().sum()
        metrics['duplicate_rows'] = duplicates
        metrics['duplicate_percentage'] = (duplicates / len(data)) * 100

        # Data type consistency
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        metrics['numeric_columns'] = len(numeric_cols)
        metrics['categorical_columns'] = len(categorical_cols)
        metrics['total_columns'] = len(data.columns)

        # Outlier detection for numeric columns
        outlier_info = self._detect_outliers(data[numeric_cols])
        metrics['outlier_percentage'] = outlier_info['percentage']
        metrics['columns_with_outliers'] = outlier_info['columns_affected']

        # Predictions quality
        if len(predictions) > 0:
            metrics['predictions_missing'] = np.isnan(predictions).sum() if predictions.dtype.kind in 'fc' else 0
            metrics['predictions_infinite'] = np.isinf(predictions).sum() if predictions.dtype.kind in 'fc' else 0

            unique_predictions = len(np.unique(predictions))
            metrics['prediction_diversity'] = unique_predictions / len(predictions)

        # True values quality (if available)
        if true_values is not None:
            metrics['true_values_missing'] = np.isnan(true_values).sum() if true_values.dtype.kind in 'fc' else 0
            metrics['true_values_infinite'] = np.isinf(true_values).sum() if true_values.dtype.kind in 'fc' else 0

            # Class balance (for classification)
            unique_true = np.unique(true_values)
            if len(unique_true) <= 20:  # Likely classification
                class_counts = [np.sum(true_values == cls) for cls in unique_true]
                metrics['class_balance_ratio'] = min(class_counts) / max(class_counts) if max(class_counts) > 0 else 0

        # Feature correlation issues
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            # Find highly correlated pairs (exclude diagonal)
            np.fill_diagonal(corr_matrix.values, 0)
            high_corr_pairs = (corr_matrix > 0.95).sum().sum() / 2  # Divide by 2 to avoid double counting
            metrics['highly_correlated_features'] = high_corr_pairs

        # Data range and distribution issues
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                # Check for zero variance
                if col_data.var() == 0:
                    metrics[f'{col}_zero_variance'] = 1

                # Check for extreme skewness
                try:
                    skewness = abs(stats.skew(col_data))
                    if skewness > 3:
                        metrics.setdefault('highly_skewed_features', 0)
                        metrics['highly_skewed_features'] += 1
                except:
                    pass

        return metrics

    def _detect_outliers(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns using IQR method."""
        outlier_info = {'percentage': 0, 'columns_affected': 0, 'details': {}}

        total_outliers = 0
        total_values = 0
        columns_with_outliers = 0

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) == 0:
                continue

            total_values += len(col_data)

            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            if outliers > 0:
                columns_with_outliers += 1
                total_outliers += outliers
                outlier_info['details'][col] = {
                    'count': outliers,
                    'percentage': (outliers / len(col_data)) * 100
                }

        outlier_info['percentage'] = (total_outliers / total_values * 100) if total_values > 0 else 0
        outlier_info['columns_affected'] = columns_with_outliers

        return outlier_info

    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on data quality metrics."""
        score = 1.0

        # Penalize for missing data
        missing_penalty = metrics.get('missing_percentage', 0) / 100 * 0.3
        score -= missing_penalty

        # Penalize for duplicates
        duplicate_penalty = min(metrics.get('duplicate_percentage', 0) / 100 * 0.2, 0.2)
        score -= duplicate_penalty

        # Penalize for outliers
        outlier_penalty = min(metrics.get('outlier_percentage', 0) / 100 * 0.2, 0.2)
        score -= outlier_penalty

        # Penalize for infinite/missing predictions
        pred_issues = (metrics.get('predictions_missing', 0) + metrics.get('predictions_infinite', 0))
        if pred_issues > 0:
            score -= 0.3

        return max(0.0, score)

    def _generate_insights(self, metrics: Dict[str, float], data: pd.DataFrame) -> List[str]:
        """Generate actionable insights from quality metrics."""
        insights = []

        # Missing data insights
        missing_pct = metrics.get('missing_percentage', 0)
        if missing_pct > 20:
            insights.append(f"🚨 High missing data rate ({missing_pct:.1f}%) - significant data quality concern")
        elif missing_pct > 5:
            insights.append(f"⚠️ Moderate missing data ({missing_pct:.1f}%) - may impact model performance")
        elif missing_pct == 0:
            insights.append("✅ No missing data detected - excellent completeness")

        # Duplicate insights
        dup_pct = metrics.get('duplicate_percentage', 0)
        if dup_pct > 10:
            insights.append(f"🔄 High duplicate rate ({dup_pct:.1f}%) - may indicate data collection issues")
        elif dup_pct > 0:
            insights.append(f"📄 {dup_pct:.1f}% duplicate rows detected")

        # Outlier insights
        outlier_pct = metrics.get('outlier_percentage', 0)
        if outlier_pct > 15:
            insights.append(f"📊 High outlier rate ({outlier_pct:.1f}%) - check for data collection errors")
        elif outlier_pct > 5:
            insights.append(f"🎯 Moderate outliers ({outlier_pct:.1f}%) detected - may need treatment")

        # Prediction quality insights
        pred_missing = metrics.get('predictions_missing', 0)
        pred_infinite = metrics.get('predictions_infinite', 0)
        if pred_missing > 0 or pred_infinite > 0:
            insights.append(f"⚠️ Prediction quality issues: {pred_missing} missing, {pred_infinite} infinite values")

        # Class balance insights
        class_balance = metrics.get('class_balance_ratio', 1)
        if class_balance < 0.1:
            insights.append("⚖️ Severe class imbalance detected - consider resampling techniques")
        elif class_balance < 0.3:
            insights.append("📊 Moderate class imbalance - may affect model performance")

        # Feature correlation insights
        high_corr = metrics.get('highly_correlated_features', 0)
        if high_corr > 0:
            insights.append(f"🔗 {high_corr} highly correlated feature pairs - consider dimensionality reduction")

        # Skewness insights
        skewed_features = metrics.get('highly_skewed_features', 0)
        if skewed_features > 0:
            insights.append(f"📈 {skewed_features} highly skewed features - consider transformations")

        return insights

    def _generate_recommendations(self, metrics: Dict[str, float], data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Missing data recommendations
        missing_pct = metrics.get('missing_percentage', 0)
        if missing_pct > 10:
            recommendations.append("🔧 Apply missing data imputation or consider removing incomplete samples")
            recommendations.append("📊 Analyze missingness patterns - consider MCAR, MAR, or MNAR mechanisms")

        # Duplicate recommendations
        dup_pct = metrics.get('duplicate_percentage', 0)
        if dup_pct > 1:
            recommendations.append("🔄 Remove duplicate rows to prevent data leakage")

        # Outlier recommendations
        outlier_pct = metrics.get('outlier_percentage', 0)
        if outlier_pct > 10:
            recommendations.append("🎯 Investigate outliers: validate, transform, or remove if appropriate")
            recommendations.append("📊 Consider robust modeling techniques less sensitive to outliers")

        # Prediction quality recommendations
        pred_issues = metrics.get('predictions_missing', 0) + metrics.get('predictions_infinite', 0)
        if pred_issues > 0:
            recommendations.append("⚠️ Fix prediction quality issues before analysis")

        # Class balance recommendations
        class_balance = metrics.get('class_balance_ratio', 1)
        if class_balance < 0.2:
            recommendations.append("⚖️ Address class imbalance with SMOTE, undersampling, or class weights")

        # Feature correlation recommendations
        high_corr = metrics.get('highly_correlated_features', 0)
        if high_corr > 3:
            recommendations.append("🔗 Apply PCA or feature selection to reduce multicollinearity")

        # General recommendations
        if missing_pct < 5 and dup_pct < 1 and outlier_pct < 5:
            recommendations.append("✅ Data quality is good - proceed with confidence")
        else:
            recommendations.append("📋 Implement data quality monitoring and validation pipelines")

        return recommendations

    def _create_plots(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Create comprehensive data quality visualization plots."""
        plots = {}

        # Missing data heatmap
        plots['missing_data_heatmap'] = self._plot_missing_data_heatmap(data)

        # Data completeness overview
        plots['completeness_overview'] = self._plot_completeness_overview(data)

        # Outlier detection plot
        plots['outlier_analysis'] = self._plot_outlier_analysis(data)

        # Feature distribution overview
        plots['feature_distributions'] = self._plot_feature_distributions(data)

        # Correlation heatmap
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plots['correlation_heatmap'] = self._plot_correlation_heatmap(data[numeric_cols])

        return plots

    def _plot_missing_data_heatmap(self, data: pd.DataFrame):
        """Create missing data heatmap."""
        if data.isnull().sum().sum() == 0:
            # No missing data
            fig = go.Figure()
            fig.add_annotation(
                text="No Missing Data Detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=20
            )
            fig.update_layout(title="Missing Data Analysis", width=600, height=400)
            return fig

        # Calculate missing data percentage per column
        missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)

        fig = go.Figure(data=go.Bar(
            x=missing_pct.index,
            y=missing_pct.values,
            marker_color='red',
            opacity=0.7
        ))

        fig.update_layout(
            title='Missing Data by Column',
            xaxis_title='Columns',
            yaxis_title='Missing Percentage (%)',
            width=800,
            height=400,
            xaxis_tickangle=45
        )

        return fig

    def _plot_completeness_overview(self, data: pd.DataFrame):
        """Create data completeness overview."""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        complete_cells = total_cells - missing_cells

        fig = go.Figure(data=[go.Pie(
            labels=['Complete', 'Missing'],
            values=[complete_cells, missing_cells],
            hole=0.4,
            marker_colors=['green', 'red']
        )])

        fig.update_layout(
            title=f'Data Completeness Overview<br>({complete_cells:,} complete, {missing_cells:,} missing)',
            width=500,
            height=400
        )

        return fig

    def _plot_outlier_analysis(self, data: pd.DataFrame):
        """Create outlier analysis plot."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:6]  # Limit to first 6 for readability

        if len(numeric_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No Numeric Columns for Outlier Analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Outlier Analysis", width=600, height=400)
            return fig

        fig = go.Figure()

        for col in numeric_cols:
            fig.add_trace(go.Box(
                y=data[col].dropna(),
                name=col,
                boxpoints='outliers'
            ))

        fig.update_layout(
            title='Outlier Detection (Box Plots)',
            yaxis_title='Values',
            width=800,
            height=500,
            showlegend=False
        )

        return fig

    def _plot_feature_distributions(self, data: pd.DataFrame):
        """Create feature distribution overview."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]  # Limit for performance

        if len(numeric_cols) == 0:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{col} Distribution' for col in numeric_cols[:4]]
        )

        for i, col in enumerate(numeric_cols[:4]):
            row = (i // 2) + 1
            col_num = (i % 2) + 1

            fig.add_trace(
                go.Histogram(x=data[col].dropna(), name=col, showlegend=False),
                row=row, col=col_num
            )

        fig.update_layout(
            title='Feature Distributions',
            height=600,
            width=800
        )

        return fig

    def _plot_correlation_heatmap(self, numeric_data: pd.DataFrame):
        """Create correlation heatmap for numeric features."""
        if len(numeric_data.columns) < 2:
            return None

        corr_matrix = numeric_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Feature Correlation Matrix',
            width=700,
            height=600,
            xaxis_tickangle=45
        )

        return fig

    def _detailed_analysis(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform detailed data quality analysis."""
        detailed = {}

        # Column-wise analysis
        detailed['column_analysis'] = {}
        for col in data.columns:
            col_info = {
                'dtype': str(data[col].dtype),
                'missing_count': int(data[col].isnull().sum()),
                'missing_percentage': float(data[col].isnull().sum() / len(data) * 100),
                'unique_values': int(data[col].nunique()),
                'most_frequent': str(data[col].mode().iloc[0]) if len(data[col].mode()) > 0 else None
            }

            if data[col].dtype in ['int64', 'float64']:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    col_info.update({
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'skewness': float(stats.skew(col_data)),
                        'kurtosis': float(stats.kurtosis(col_data))
                    })

            detailed['column_analysis'][col] = col_info

        # Data profiling summary
        detailed['profiling_summary'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime']).columns),
            'memory_usage_mb': float(data.memory_usage(deep=True).sum() / 1024 / 1024)
        }

        # Quality score breakdown
        detailed['quality_scores'] = {
            'completeness': 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])),
            'uniqueness': 1 - (data.duplicated().sum() / len(data)),
            'consistency': self._calculate_consistency_score(data),
            'validity': self._calculate_validity_score(data, predictions, true_values)
        }

        return detailed

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        # Simple consistency check based on data types and value ranges
        score = 1.0

        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for mixed case inconsistencies
                if data[col].nunique() < len(data) / 2:  # Likely categorical
                    unique_values = data[col].dropna().unique()
                    if len(unique_values) > 1:
                        # Simple check for case inconsistencies
                        lower_count = sum(1 for val in unique_values if isinstance(val, str) and val.islower())
                        if 0 < lower_count < len(unique_values):
                            score -= 0.1

        return max(0.0, score)

    def _calculate_validity_score(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray]
    ) -> float:
        """Calculate data validity score."""
        score = 1.0

        # Check for invalid values in predictions
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            score -= 0.3

        # Check for invalid values in true values
        if true_values is not None and (np.any(np.isnan(true_values)) or np.any(np.isinf(true_values))):
            score -= 0.3

        # Check for extreme outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                # Z-score based outlier detection
                z_scores = np.abs(stats.zscore(col_data))
                extreme_outliers = np.sum(z_scores > 4)  # Very extreme outliers
                if extreme_outliers > len(col_data) * 0.01:  # More than 1% extreme outliers
                    score -= 0.05

        return max(0.0, score)