"""
Explainability analyzer for model interpretability and feature importance analysis.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from .base_analyzer import BaseAnalyzer, AnalysisResult, ProblemType, AnalysisType


class ExplainabilityAnalyzer(BaseAnalyzer):
    """
    Advanced explainability analyzer for model interpretability.

    Provides permutation importance, partial dependence, local explanations,
    and comprehensive feature impact analysis.
    """

    def __init__(self):
        super().__init__(
            name="Explainability Analyzer",
            supported_problem_types=[
                ProblemType.BINARY_CLASSIFICATION,
                ProblemType.MULTICLASS_CLASSIFICATION,
                ProblemType.REGRESSION
            ]
        )
        self.analysis_type = AnalysisType.EXPLAINABILITY

    def analyze(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        model=None,  # Optional: actual model for better explanations
        **kwargs
    ) -> AnalysisResult:
        """
        Perform comprehensive explainability analysis.
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

        # Skip analysis if no true values (need them for explainability)
        if true_values is None:
            result.insights = ["Explainability analysis requires true values for feature importance calculation"]
            result.recommendations = ["Provide ground truth labels for comprehensive explainability analysis"]
            result.metrics = {'analysis_skipped': 1}
            result.execution_time = time.time() - start_time
            return result

        try:
            # Calculate explainability metrics
            result.metrics = self._calculate_explainability_metrics(
                data, predictions, true_values, problem_type, model
            )

            # Generate insights and recommendations
            result.insights = self._generate_insights(result.metrics, data.columns.tolist())
            result.recommendations = self._generate_recommendations(result.metrics, data.columns.tolist())

            # Create visualizations
            result.plots = self._create_plots(data, predictions, true_values, problem_type, result.metrics)

            # Detailed analysis
            result.detailed_results = self._detailed_analysis(
                data, predictions, true_values, problem_type, result.metrics
            )

        except Exception as e:
            # Fallback for explainability analysis failures
            result.insights = [f"Explainability analysis failed: {str(e)}"]
            result.recommendations = ["Check data format and ensure numeric features for explainability analysis"]
            result.metrics = {'analysis_failed': 1}

        # Update execution time
        result.execution_time = time.time() - start_time

        return result

    def _calculate_explainability_metrics(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: np.ndarray,
        problem_type: ProblemType,
        model=None
    ) -> Dict[str, Any]:
        """Calculate comprehensive explainability metrics."""
        metrics = {}

        # Prepare numeric data for analysis
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            # Try to convert categorical to numeric
            encoded_data = self._encode_categorical_features(data)
            numeric_data = encoded_data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) == 0:
            metrics['no_numeric_features'] = 1
            return metrics

        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Feature importance via permutation importance
        if model is not None:
            # Use provided model
            feature_importance = self._calculate_permutation_importance(
                model, numeric_data, true_values, problem_type
            )
        else:
            # Train a surrogate model
            feature_importance = self._calculate_surrogate_importance(
                numeric_data, true_values, problem_type
            )

        metrics['feature_importance'] = feature_importance

        # Feature correlation with target
        target_correlation = self._calculate_target_correlation(numeric_data, true_values)
        metrics['target_correlation'] = target_correlation

        # Feature interaction analysis
        feature_interactions = self._analyze_feature_interactions(numeric_data, true_values, problem_type)
        metrics['feature_interactions'] = feature_interactions

        # Prediction explanation statistics
        explanation_stats = self._calculate_explanation_statistics(
            numeric_data, predictions, true_values, feature_importance
        )
        metrics.update(explanation_stats)

        return metrics

    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for analysis."""
        encoded_data = data.copy()

        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() <= 20:  # Reasonable number of categories
                # Simple label encoding
                unique_vals = data[col].dropna().unique()
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                encoded_data[col] = data[col].map(encoding_map)

        return encoded_data

    def _calculate_permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        problem_type: ProblemType
    ) -> Dict[str, float]:
        """Calculate permutation importance using the provided model."""
        try:
            # Define scoring function
            if problem_type == ProblemType.REGRESSION:
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'accuracy'

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, scoring=scoring, n_repeats=5, random_state=42
            )

            # Create importance dictionary
            importance_dict = {}
            for i, feature in enumerate(X.columns):
                importance_dict[feature] = float(perm_importance.importances_mean[i])

            return importance_dict

        except Exception as e:
            # Fallback to correlation-based importance
            return self._calculate_correlation_importance(X, y)

    def _calculate_surrogate_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        problem_type: ProblemType
    ) -> Dict[str, float]:
        """Calculate feature importance using a surrogate model."""
        try:
            # Train a surrogate model
            if problem_type == ProblemType.REGRESSION:
                surrogate_model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                surrogate_model = RandomForestClassifier(n_estimators=50, random_state=42)

            surrogate_model.fit(X, y)

            # Get feature importance
            importance_dict = {}
            for i, feature in enumerate(X.columns):
                importance_dict[feature] = float(surrogate_model.feature_importances_[i])

            return importance_dict

        except Exception as e:
            # Final fallback to correlation-based importance
            return self._calculate_correlation_importance(X, y)

    def _calculate_correlation_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance based on correlation with target."""
        importance_dict = {}

        for feature in X.columns:
            try:
                corr = abs(np.corrcoef(X[feature].fillna(X[feature].mean()), y)[0, 1])
                importance_dict[feature] = corr if not np.isnan(corr) else 0.0
            except:
                importance_dict[feature] = 0.0

        return importance_dict

    def _calculate_target_correlation(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Calculate correlation between features and target."""
        correlations = {}

        for feature in X.columns:
            try:
                corr = np.corrcoef(X[feature].fillna(X[feature].mean()), y)[0, 1]
                correlations[feature] = float(corr) if not np.isnan(corr) else 0.0
            except:
                correlations[feature] = 0.0

        return correlations

    def _analyze_feature_interactions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        problem_type: ProblemType
    ) -> Dict[str, float]:
        """Analyze feature interactions."""
        interactions = {}

        # Limit to top features for performance
        if len(X.columns) > 10:
            # Use correlation with target to select top features
            correlations = self._calculate_target_correlation(X, y)
            top_features = sorted(correlations.keys(), key=lambda x: abs(correlations[x]), reverse=True)[:10]
            X_subset = X[top_features]
        else:
            X_subset = X

        # Calculate pairwise interactions
        for i, feat1 in enumerate(X_subset.columns):
            for j, feat2 in enumerate(X_subset.columns[i+1:], i+1):
                try:
                    # Simple interaction: correlation between product and target
                    interaction_feature = X_subset[feat1] * X_subset[feat2]
                    interaction_corr = abs(np.corrcoef(interaction_feature.fillna(0), y)[0, 1])

                    # Compare to individual correlations
                    individual_corr = max(
                        abs(np.corrcoef(X_subset[feat1].fillna(0), y)[0, 1]),
                        abs(np.corrcoef(X_subset[feat2].fillna(0), y)[0, 1])
                    )

                    # Interaction strength
                    interaction_strength = interaction_corr - individual_corr
                    if not np.isnan(interaction_strength):
                        interactions[f'{feat1}×{feat2}'] = float(max(0, interaction_strength))
                except:
                    continue

        return interactions

    def _calculate_explanation_statistics(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        true_values: np.ndarray,
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate statistics about model explanations."""
        stats = {}

        # Top feature importance concentration
        importance_values = list(feature_importance.values())
        if importance_values:
            total_importance = sum(importance_values)
            sorted_importance = sorted(importance_values, reverse=True)

            # Concentration metrics
            stats['top_feature_importance'] = sorted_importance[0] if sorted_importance else 0
            stats['top_3_importance_sum'] = sum(sorted_importance[:3]) if len(sorted_importance) >= 3 else sum(sorted_importance)
            stats['importance_concentration'] = stats['top_3_importance_sum'] / total_importance if total_importance > 0 else 0

            # Gini coefficient for importance distribution
            stats['importance_gini'] = self._calculate_gini_coefficient(importance_values)

        # Feature stability analysis
        stats['num_important_features'] = sum(1 for imp in importance_values if imp > 0.01)
        stats['feature_diversity'] = len([imp for imp in importance_values if imp > 0.01]) / len(importance_values) if importance_values else 0

        return stats

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for importance distribution."""
        if not values or all(v == 0 for v in values):
            return 0.0

        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        # Calculate Gini coefficient
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_values, 1))) / (n * sum(sorted_values))
        return float(max(0, min(1, gini)))

    def _generate_insights(self, metrics: Dict[str, Any], feature_names: List[str]) -> List[str]:
        """Generate actionable insights from explainability metrics."""
        insights = []

        if 'analysis_failed' in metrics or 'no_numeric_features' in metrics:
            insights.append("⚠️ Explainability analysis could not be completed")
            return insights

        feature_importance = metrics.get('feature_importance', {})

        if not feature_importance:
            insights.append("⚠️ No feature importance calculated")
            return insights

        # Top important features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]

        if top_features:
            top_feature_names = [f[0] for f in top_features]
            insights.append(f"🎯 Most important features: {', '.join(top_feature_names)}")

        # Importance concentration
        concentration = metrics.get('importance_concentration', 0)
        if concentration > 0.7:
            insights.append("📊 High feature importance concentration - model relies heavily on few features")
        elif concentration < 0.3:
            insights.append("🌐 Distributed feature importance - model uses many features equally")

        # Feature diversity
        diversity = metrics.get('feature_diversity', 0)
        if diversity < 0.2:
            insights.append("⚠️ Low feature diversity - consider feature engineering")
        elif diversity > 0.6:
            insights.append("✅ Good feature diversity - model uses varied information")

        # Gini coefficient insights
        gini = metrics.get('importance_gini', 0)
        if gini > 0.7:
            insights.append("📈 Highly unequal feature importance distribution")
        elif gini < 0.3:
            insights.append("⚖️ Relatively equal feature importance distribution")

        # Feature interactions
        interactions = metrics.get('feature_interactions', {})
        strong_interactions = {k: v for k, v in interactions.items() if v > 0.1}
        if strong_interactions:
            top_interaction = max(strong_interactions.items(), key=lambda x: x[1])
            insights.append(f"🔗 Strong feature interaction detected: {top_interaction[0]}")

        return insights

    def _generate_recommendations(self, metrics: Dict[str, Any], feature_names: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if 'analysis_failed' in metrics:
            recommendations.append("🔧 Ensure data contains numeric features for explainability analysis")
            recommendations.append("📊 Consider feature engineering to create numeric representations")
            return recommendations

        feature_importance = metrics.get('feature_importance', {})

        if not feature_importance:
            recommendations.append("📊 Unable to calculate feature importance - check data quality")
            return recommendations

        # Low importance features
        low_importance_features = [f for f, imp in feature_importance.items() if imp < 0.01]
        if len(low_importance_features) > len(feature_importance) * 0.5:
            recommendations.append("🔧 Consider removing low-importance features to simplify model")

        # High concentration
        concentration = metrics.get('importance_concentration', 0)
        if concentration > 0.8:
            recommendations.append("⚠️ Model heavily dependent on few features - consider regularization")
            recommendations.append("🎯 Investigate feature engineering to create more balanced importance")

        # Feature interactions
        interactions = metrics.get('feature_interactions', {})
        if interactions:
            strong_interactions = {k: v for k, v in interactions.items() if v > 0.1}
            if strong_interactions:
                recommendations.append("🔗 Strong feature interactions found - consider polynomial features")

        # General recommendations
        num_important = metrics.get('num_important_features', 0)
        if num_important < 3:
            recommendations.append("📈 Very few important features - may need more relevant data")
        elif num_important > len(feature_names) * 0.8:
            recommendations.append("🎯 Many features are important - model might be complex")

        if len(recommendations) == 0:
            recommendations.append("✅ Good feature importance distribution - model explainability is reasonable")

        return recommendations

    def _create_plots(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: np.ndarray,
        problem_type: ProblemType,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive explainability visualization plots."""
        plots = {}

        feature_importance = metrics.get('feature_importance', {})

        if not feature_importance:
            return plots

        # Feature importance bar plot
        plots['feature_importance'] = self._plot_feature_importance(feature_importance)

        # Feature correlation heatmap
        target_correlation = metrics.get('target_correlation', {})
        if target_correlation:
            plots['target_correlation'] = self._plot_target_correlation(target_correlation)

        # Feature interactions network (if available)
        interactions = metrics.get('feature_interactions', {})
        if interactions:
            plots['feature_interactions'] = self._plot_feature_interactions(interactions)

        # Importance distribution
        plots['importance_distribution'] = self._plot_importance_distribution(feature_importance)

        return plots

    def _plot_feature_importance(self, feature_importance: Dict[str, float]):
        """Create feature importance bar plot."""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Limit to top 15 for readability
        top_features = sorted_features[:15]

        features, importance = zip(*top_features)

        fig = go.Figure(data=[go.Bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            marker_color='steelblue'
        )])

        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(top_features) * 25),
            width=700,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def _plot_target_correlation(self, target_correlation: Dict[str, float]):
        """Create target correlation plot."""
        sorted_corr = sorted(target_correlation.items(), key=lambda x: abs(x[1]), reverse=True)

        # Limit to top 15
        top_corr = sorted_corr[:15]
        features, correlations = zip(*top_corr)

        # Color by positive/negative correlation
        colors = ['red' if corr < 0 else 'blue' for corr in correlations]

        fig = go.Figure(data=[go.Bar(
            x=list(correlations),
            y=list(features),
            orientation='h',
            marker_color=colors
        )])

        fig.update_layout(
            title='Feature-Target Correlation',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Features',
            height=max(400, len(top_corr) * 25),
            width=700,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def _plot_feature_interactions(self, interactions: Dict[str, float]):
        """Create feature interactions plot."""
        if not interactions:
            return None

        sorted_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        top_interactions = sorted_interactions[:10]  # Top 10 interactions

        if not top_interactions:
            return None

        interaction_names, interaction_strengths = zip(*top_interactions)

        fig = go.Figure(data=[go.Bar(
            x=list(interaction_strengths),
            y=list(interaction_names),
            orientation='h',
            marker_color='orange'
        )])

        fig.update_layout(
            title='Feature Interactions',
            xaxis_title='Interaction Strength',
            yaxis_title='Feature Pairs',
            height=max(400, len(top_interactions) * 30),
            width=800,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def _plot_importance_distribution(self, feature_importance: Dict[str, float]):
        """Create importance distribution histogram."""
        importance_values = list(feature_importance.values())

        fig = go.Figure(data=[go.Histogram(
            x=importance_values,
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7
        )])

        fig.update_layout(
            title='Feature Importance Distribution',
            xaxis_title='Importance Score',
            yaxis_title='Number of Features',
            width=600,
            height=400
        )

        return fig

    def _detailed_analysis(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_values: np.ndarray,
        problem_type: ProblemType,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform detailed explainability analysis."""
        detailed = {}

        feature_importance = metrics.get('feature_importance', {})

        if not feature_importance:
            detailed['analysis_status'] = 'failed'
            return detailed

        # Feature ranking and statistics
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        detailed['feature_ranking'] = []
        for rank, (feature, importance) in enumerate(sorted_features, 1):
            detailed['feature_ranking'].append({
                'rank': rank,
                'feature': feature,
                'importance': float(importance),
                'importance_percentage': float(importance / sum(feature_importance.values()) * 100) if sum(feature_importance.values()) > 0 else 0
            })

        # Importance statistics
        importance_values = list(feature_importance.values())
        detailed['importance_statistics'] = {
            'mean_importance': float(np.mean(importance_values)),
            'std_importance': float(np.std(importance_values)),
            'min_importance': float(np.min(importance_values)),
            'max_importance': float(np.max(importance_values)),
            'median_importance': float(np.median(importance_values))
        }

        # Feature groups by importance level
        detailed['feature_groups'] = {
            'high_importance': [f for f, imp in feature_importance.items() if imp > np.percentile(importance_values, 75)],
            'medium_importance': [f for f, imp in feature_importance.items() if np.percentile(importance_values, 25) < imp <= np.percentile(importance_values, 75)],
            'low_importance': [f for f, imp in feature_importance.items() if imp <= np.percentile(importance_values, 25)]
        }

        # Model complexity indicators
        detailed['complexity_indicators'] = {
            'effective_feature_count': len([imp for imp in importance_values if imp > 0.01]),
            'importance_entropy': self._calculate_entropy(importance_values),
            'feature_utilization_rate': len([imp for imp in importance_values if imp > 0.01]) / len(importance_values)
        }

        return detailed

    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of importance distribution."""
        if not values or sum(values) == 0:
            return 0.0

        # Normalize values to probabilities
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]

        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return float(entropy)