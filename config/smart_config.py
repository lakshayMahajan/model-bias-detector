"""
Smart configuration interface with AI-powered suggestions and automatic setup.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from utils.data_processor import DataProcessor, DataConfig, ColumnMapping, ColumnType
from analyzers.base_analyzer import ProblemType, AnalysisType


@dataclass
class AnalysisConfiguration:
    """Complete configuration for analysis pipeline."""

    # Data configuration
    data_config: DataConfig

    # Analysis settings
    problem_type: ProblemType
    analysis_types: List[AnalysisType]

    # Performance settings
    include_plots: bool = True
    export_results: bool = False
    export_format: str = "json"

    # Advanced settings
    confidence_threshold: float = 0.8
    max_features_for_analysis: int = 100
    sample_size_limit: Optional[int] = None


class SmartConfigInterface:
    """
    AI-powered smart configuration interface.

    Automatically analyzes data and suggests optimal configuration
    for comprehensive ML model analysis.
    """

    def __init__(self):
        self.data_processor = DataProcessor()
        self.config_templates = self._load_config_templates()

    def analyze_and_suggest_config(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[AnalysisConfiguration, List[str], Dict[str, float]]:
        """
        Analyze data and suggest optimal configuration.

        Args:
            data: Input DataFrame
            predictions: Model predictions (optional)
            true_values: Ground truth values (optional)
            user_preferences: User-specified preferences

        Returns:
            Tuple of (suggested_config, suggestions_explanation, confidence_scores)
        """

        # Step 1: Analyze columns
        column_mappings = self.data_processor.analyze_columns(data)

        # Step 2: Create data configuration
        data_config = self._create_smart_data_config(column_mappings, data, user_preferences)

        # Step 3: Detect problem type
        problem_type = self._detect_problem_type(data, predictions, true_values, data_config)

        # Step 4: Suggest analysis types
        analysis_types = self._suggest_analysis_types(data, problem_type, data_config)

        # Step 5: Create full configuration
        config = AnalysisConfiguration(
            data_config=data_config,
            problem_type=problem_type,
            analysis_types=analysis_types,
            include_plots=user_preferences.get('include_plots', True) if user_preferences else True,
            export_results=user_preferences.get('export_results', False) if user_preferences else False
        )

        # Step 6: Generate explanations and confidence scores
        explanations = self._generate_configuration_explanations(config, column_mappings, data)
        confidence_scores = self._calculate_confidence_scores(config, column_mappings, data)

        return config, explanations, confidence_scores

    def suggest_column_mappings(
        self,
        data: pd.DataFrame,
        target_column_type: ColumnType
    ) -> List[Tuple[str, float]]:
        """
        Suggest columns for a specific type.

        Args:
            data: Input DataFrame
            target_column_type: Type of column to find

        Returns:
            List of (column_name, confidence_score) tuples
        """
        column_mappings = self.data_processor.analyze_columns(data)

        suggestions = []
        for mapping in column_mappings:
            if mapping.type == target_column_type:
                suggestions.append((mapping.name, mapping.confidence))

        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions

    def validate_configuration(
        self,
        config: AnalysisConfiguration,
        data: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration against data.

        Args:
            config: Configuration to validate
            data: Data to validate against

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Validate data configuration
        data_validation_errors = self.data_processor.validate_data_config(data, config.data_config)
        issues.extend(data_validation_errors)

        # Validate sample size requirements
        if len(data) < 10:
            issues.append("Sample size too small (minimum 10 samples required)")

        # Validate feature availability for certain analysis types
        if AnalysisType.EXPLAINABILITY in config.analysis_types:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                issues.append("Explainability analysis requires numeric features")

        if AnalysisType.BIAS_FAIRNESS in config.analysis_types:
            if not config.data_config.protected_attributes:
                issues.append("Bias analysis requires protected attributes to be specified")

        # Validate problem type compatibility
        if config.problem_type == ProblemType.REGRESSION:
            if config.data_config.prediction_column:
                pred_col = config.data_config.prediction_column
                if pred_col in data.columns:
                    unique_vals = data[pred_col].nunique()
                    if unique_vals <= 10:
                        issues.append("Regression problem type may not be appropriate - consider classification")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_recommended_settings(
        self,
        data: pd.DataFrame,
        analysis_goal: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Get recommended settings based on data characteristics and analysis goal.

        Args:
            data: Input DataFrame
            analysis_goal: Type of analysis goal ("comprehensive", "quick", "bias_focused", "performance_focused")

        Returns:
            Dictionary of recommended settings
        """
        settings = {
            "include_plots": True,
            "export_results": False,
            "confidence_threshold": 0.8,
            "max_features_for_analysis": 100
        }

        # Adjust based on data size
        if len(data) > 10000:
            settings["sample_size_limit"] = 5000
            settings["max_features_for_analysis"] = 50
        elif len(data) < 100:
            settings["confidence_threshold"] = 0.6  # Lower threshold for small datasets

        # Adjust based on analysis goal
        if analysis_goal == "quick":
            settings["include_plots"] = False
            settings["max_features_for_analysis"] = 20
        elif analysis_goal == "bias_focused":
            settings["confidence_threshold"] = 0.9  # Higher threshold for bias analysis
        elif analysis_goal == "performance_focused":
            settings["export_results"] = True

        return settings

    def _create_smart_data_config(
        self,
        column_mappings: List[ColumnMapping],
        data: pd.DataFrame,
        user_preferences: Optional[Dict[str, Any]]
    ) -> DataConfig:
        """Create smart data configuration from column mappings."""

        config = DataConfig()

        # Apply user preferences first if provided
        if user_preferences:
            config.prediction_column = user_preferences.get('prediction_column')
            config.true_value_column = user_preferences.get('true_value_column')
            config.protected_attributes = user_preferences.get('protected_attributes', [])
            config.probability_columns = user_preferences.get('probability_columns', [])
            config.feature_columns = user_preferences.get('feature_columns', [])

        # Auto-fill missing configurations
        for mapping in column_mappings:
            if mapping.confidence < 0.5:
                continue  # Skip low-confidence mappings

            if mapping.type == ColumnType.PREDICTION and not config.prediction_column:
                config.prediction_column = mapping.name
            elif mapping.type == ColumnType.TRUE_VALUE and not config.true_value_column:
                config.true_value_column = mapping.name
            elif mapping.type == ColumnType.PROBABILITY:
                if mapping.name not in config.probability_columns:
                    config.probability_columns.append(mapping.name)
            elif mapping.type == ColumnType.PROTECTED_ATTRIBUTE:
                if mapping.name not in config.protected_attributes:
                    config.protected_attributes.append(mapping.name)
            elif mapping.type == ColumnType.TIMESTAMP and not config.timestamp_column:
                config.timestamp_column = mapping.name
            elif mapping.type == ColumnType.IDENTIFIER and not config.identifier_column:
                config.identifier_column = mapping.name
            elif mapping.type == ColumnType.FEATURE:
                if mapping.name not in config.feature_columns:
                    config.feature_columns.append(mapping.name)

        # If no features explicitly identified, include remaining numeric columns
        if not config.feature_columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude already assigned columns
            excluded_cols = [
                config.prediction_column,
                config.true_value_column,
                config.timestamp_column,
                config.identifier_column
            ] + config.probability_columns + config.protected_attributes

            excluded_cols = [col for col in excluded_cols if col is not None]

            config.feature_columns = [col for col in numeric_cols if col not in excluded_cols]

        return config

    def _detect_problem_type(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray],
        true_values: Optional[np.ndarray],
        data_config: DataConfig
    ) -> ProblemType:
        """Detect the ML problem type."""

        # Use predictions if available
        if predictions is not None:
            unique_predictions = len(np.unique(predictions))
            if unique_predictions == 2:
                return ProblemType.BINARY_CLASSIFICATION
            elif unique_predictions <= 20 and np.all(predictions == predictions.astype(int)):
                return ProblemType.MULTICLASS_CLASSIFICATION
            else:
                return ProblemType.REGRESSION

        # Use true values if available
        if true_values is not None:
            unique_true = len(np.unique(true_values))
            if unique_true == 2:
                return ProblemType.BINARY_CLASSIFICATION
            elif unique_true <= 20 and np.all(true_values == true_values.astype(int)):
                return ProblemType.MULTICLASS_CLASSIFICATION
            else:
                return ProblemType.REGRESSION

        # Use prediction column from data
        if data_config.prediction_column and data_config.prediction_column in data.columns:
            pred_col = data[data_config.prediction_column]
            unique_vals = pred_col.nunique()

            if unique_vals == 2:
                return ProblemType.BINARY_CLASSIFICATION
            elif unique_vals <= 20 and pred_col.dtype in ['int64', 'object']:
                return ProblemType.MULTICLASS_CLASSIFICATION
            else:
                return ProblemType.REGRESSION

        # Default fallback
        return ProblemType.REGRESSION

    def _suggest_analysis_types(
        self,
        data: pd.DataFrame,
        problem_type: ProblemType,
        data_config: DataConfig
    ) -> List[AnalysisType]:
        """Suggest appropriate analysis types."""

        analysis_types = [AnalysisType.DATA_QUALITY]  # Always include data quality

        # Performance analysis if we have predictions
        if data_config.prediction_column:
            analysis_types.append(AnalysisType.PERFORMANCE)

        # Bias analysis if we have protected attributes
        if data_config.protected_attributes:
            analysis_types.append(AnalysisType.BIAS_FAIRNESS)

        # Explainability analysis if we have features
        if data_config.feature_columns or len(data.select_dtypes(include=[np.number]).columns) > 0:
            analysis_types.append(AnalysisType.EXPLAINABILITY)

        return analysis_types

    def _generate_configuration_explanations(
        self,
        config: AnalysisConfiguration,
        column_mappings: List[ColumnMapping],
        data: pd.DataFrame
    ) -> List[str]:
        """Generate human-readable explanations for configuration choices."""

        explanations = []

        # Problem type explanation
        explanations.append(f"📊 **Problem Type**: {config.problem_type.value}")

        if config.problem_type == ProblemType.BINARY_CLASSIFICATION:
            explanations.append("   → Detected binary classification (2 unique prediction values)")
        elif config.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            explanations.append("   → Detected multiclass classification (multiple discrete classes)")
        else:
            explanations.append("   → Detected regression (continuous prediction values)")

        # Data configuration explanations
        if config.data_config.prediction_column:
            explanations.append(f"🎯 **Prediction Column**: {config.data_config.prediction_column}")

        if config.data_config.true_value_column:
            explanations.append(f"✅ **True Values**: {config.data_config.true_value_column}")

        if config.data_config.protected_attributes:
            explanations.append(f"🛡️ **Protected Attributes**: {', '.join(config.data_config.protected_attributes)}")

        if config.data_config.feature_columns:
            feature_count = len(config.data_config.feature_columns)
            explanations.append(f"📈 **Features**: {feature_count} columns identified for analysis")

        # Analysis types explanation
        analysis_names = [at.value.replace('_', ' ').title() for at in config.analysis_types]
        explanations.append(f"🔍 **Analysis Types**: {', '.join(analysis_names)}")

        return explanations

    def _calculate_confidence_scores(
        self,
        config: AnalysisConfiguration,
        column_mappings: List[ColumnMapping],
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate confidence scores for configuration elements."""

        scores = {}

        # Column mapping confidence
        if config.data_config.prediction_column:
            pred_mapping = next((m for m in column_mappings
                               if m.name == config.data_config.prediction_column), None)
            scores['prediction_column'] = pred_mapping.confidence if pred_mapping else 0.5

        if config.data_config.true_value_column:
            true_mapping = next((m for m in column_mappings
                               if m.name == config.data_config.true_value_column), None)
            scores['true_value_column'] = true_mapping.confidence if true_mapping else 0.5

        # Protected attributes confidence
        if config.data_config.protected_attributes:
            protected_confidences = []
            for attr in config.data_config.protected_attributes:
                attr_mapping = next((m for m in column_mappings if m.name == attr), None)
                if attr_mapping:
                    protected_confidences.append(attr_mapping.confidence)

            scores['protected_attributes'] = np.mean(protected_confidences) if protected_confidences else 0.5

        # Overall configuration confidence
        data_completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        sample_size_score = min(1.0, len(data) / 1000)  # Normalize by 1000 samples
        feature_diversity = len(data.select_dtypes(include=[np.number]).columns) / len(data.columns)

        scores['overall'] = np.mean([
            data_completeness,
            sample_size_score,
            feature_diversity,
            np.mean(list(scores.values())) if scores else 0.5
        ])

        return scores

    def _load_config_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined configuration templates."""
        return {
            'binary_classification': {
                'analysis_types': [AnalysisType.PERFORMANCE, AnalysisType.BIAS_FAIRNESS, AnalysisType.DATA_QUALITY],
                'include_plots': True,
                'confidence_threshold': 0.8
            },
            'multiclass_classification': {
                'analysis_types': [AnalysisType.PERFORMANCE, AnalysisType.DATA_QUALITY, AnalysisType.EXPLAINABILITY],
                'include_plots': True,
                'confidence_threshold': 0.7
            },
            'regression': {
                'analysis_types': [AnalysisType.PERFORMANCE, AnalysisType.DATA_QUALITY, AnalysisType.EXPLAINABILITY],
                'include_plots': True,
                'confidence_threshold': 0.8
            },
            'bias_audit': {
                'analysis_types': [AnalysisType.BIAS_FAIRNESS, AnalysisType.DATA_QUALITY],
                'include_plots': True,
                'confidence_threshold': 0.9
            }
        }

    def apply_template(
        self,
        template_name: str,
        base_config: AnalysisConfiguration
    ) -> AnalysisConfiguration:
        """Apply a configuration template to base configuration."""

        if template_name not in self.config_templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.config_templates[template_name]

        # Create new configuration with template settings
        new_config = AnalysisConfiguration(
            data_config=base_config.data_config,  # Keep original data config
            problem_type=base_config.problem_type,  # Keep original problem type
            analysis_types=template.get('analysis_types', base_config.analysis_types),
            include_plots=template.get('include_plots', base_config.include_plots),
            export_results=base_config.export_results,  # Keep original export setting
            confidence_threshold=template.get('confidence_threshold', base_config.confidence_threshold)
        )

        return new_config