"""
Data validator for comprehensive validation framework.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    affected_rows: Optional[int] = None
    suggestion: Optional[str] = None


class DataValidator:
    """
    Comprehensive data validation framework.

    Performs extensive validation checks on input data and predictions
    to ensure data quality and compatibility with analysis requirements.
    """

    def __init__(self):
        self.validation_rules = {
            'data_completeness': self._validate_data_completeness,
            'data_types': self._validate_data_types,
            'data_ranges': self._validate_data_ranges,
            'prediction_quality': self._validate_prediction_quality,
            'consistency': self._validate_consistency,
            'compatibility': self._validate_compatibility
        }

    def validate_analysis_inputs(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        protected_attributes: Optional[List[str]] = None
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Perform comprehensive validation of analysis inputs.

        Args:
            data: Input feature data
            predictions: Model predictions
            true_values: Ground truth values (optional)
            protected_attributes: Protected attribute columns (optional)

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_issues = rule_func(data, predictions, true_values, protected_attributes)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Validation rule '{rule_name}' failed: {str(e)}"
                ))

        # Determine overall validity
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        is_valid = not has_errors

        return is_valid, issues

    def _validate_data_completeness(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        protected_attributes: Optional[List[str]]
    ) -> List[ValidationIssue]:
        """Validate data completeness and presence."""
        issues = []

        # Check if data is empty
        if data is None or data.empty:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Input data is empty or None",
                suggestion="Provide non-empty DataFrame with features"
            ))
            return issues

        # Check predictions
        if predictions is None:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Predictions cannot be None",
                suggestion="Provide model predictions as Series or numpy array"
            ))
        elif len(predictions) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Predictions array is empty",
                suggestion="Provide non-empty predictions array"
            ))

        # Check length consistency
        if predictions is not None and len(predictions) != len(data):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Length mismatch: data has {len(data)} rows, predictions has {len(predictions)} elements",
                suggestion="Ensure predictions and data have same number of samples"
            ))

        if true_values is not None and len(true_values) != len(data):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Length mismatch: data has {len(data)} rows, true values has {len(true_values)} elements",
                suggestion="Ensure true values and data have same number of samples"
            ))

        # Check missing data in features
        missing_data = data.isnull().sum()
        total_missing = missing_data.sum()

        if total_missing > 0:
            missing_percentage = (total_missing / (len(data) * len(data.columns))) * 100

            if missing_percentage > 50:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Excessive missing data: {missing_percentage:.1f}% of values are missing",
                    suggestion="Consider data imputation or using a different dataset"
                ))
            elif missing_percentage > 20:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"High missing data: {missing_percentage:.1f}% of values are missing",
                    suggestion="Consider data imputation strategies"
                ))

            # Report columns with high missing rates
            high_missing_cols = missing_data[missing_data > len(data) * 0.5]
            for col in high_missing_cols.index:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{col}' has {(high_missing_cols[col]/len(data)*100):.1f}% missing values",
                    column=col,
                    suggestion="Consider removing this column or advanced imputation"
                ))

        return issues

    def _validate_data_types(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        protected_attributes: Optional[List[str]]
    ) -> List[ValidationIssue]:
        """Validate data types and formats."""
        issues = []

        # Check for object columns that might need encoding
        object_cols = data.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                unique_vals = data[col].nunique()
                if unique_vals > 50:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' has {unique_vals} unique categorical values",
                        column=col,
                        suggestion="Consider feature engineering or encoding for high-cardinality categorical features"
                    ))

        # Check predictions data type
        if predictions is not None:
            if isinstance(predictions, pd.Series):
                pred_array = predictions.values
            else:
                pred_array = predictions

            # Check for invalid values in predictions
            if np.any(np.isnan(pred_array)):
                nan_count = np.sum(np.isnan(pred_array))
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Predictions contain {nan_count} NaN values",
                    suggestion="Fix model to avoid NaN predictions"
                ))

            if np.any(np.isinf(pred_array)):
                inf_count = np.sum(np.isinf(pred_array))
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Predictions contain {inf_count} infinite values",
                    suggestion="Fix model to avoid infinite predictions"
                ))

        # Check true values data type
        if true_values is not None:
            if isinstance(true_values, pd.Series):
                true_array = true_values.values
            else:
                true_array = true_values

            if np.any(np.isnan(true_array)):
                nan_count = np.sum(np.isnan(true_array))
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"True values contain {nan_count} NaN values",
                    suggestion="Consider removing samples with missing true values"
                ))

        return issues

    def _validate_data_ranges(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        protected_attributes: Optional[List[str]]
    ) -> List[ValidationIssue]:
        """Validate data ranges and detect outliers."""
        issues = []

        # Check numeric columns for extreme values
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            # Check for zero variance
            if col_data.var() == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{col}' has zero variance (all values are the same)",
                    column=col,
                    suggestion="Consider removing this feature as it provides no information"
                ))

            # Check for extreme outliers (beyond 5 standard deviations)
            mean_val = col_data.mean()
            std_val = col_data.std()

            if std_val > 0:
                z_scores = np.abs((col_data - mean_val) / std_val)
                extreme_outliers = np.sum(z_scores > 5)

                if extreme_outliers > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' has {extreme_outliers} extreme outliers (>5 std from mean)",
                        column=col,
                        affected_rows=extreme_outliers,
                        suggestion="Investigate these extreme values - they may be data errors"
                    ))

        return issues

    def _validate_prediction_quality(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        protected_attributes: Optional[List[str]]
    ) -> List[ValidationIssue]:
        """Validate prediction quality and characteristics."""
        issues = []

        if predictions is None:
            return issues

        if isinstance(predictions, pd.Series):
            pred_array = predictions.values
        else:
            pred_array = predictions

        # Check prediction diversity
        unique_predictions = len(np.unique(pred_array))
        total_predictions = len(pred_array)

        if unique_predictions == 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="All predictions are identical - model may not be trained properly",
                suggestion="Retrain model or check for issues in model implementation"
            ))
        elif unique_predictions < total_predictions * 0.01:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Very low prediction diversity: only {unique_predictions} unique values out of {total_predictions}",
                suggestion="Check if model is overly conservative or needs more training"
            ))

        # For classification-like predictions, check class balance
        if unique_predictions <= 20:  # Likely classification
            unique_vals, counts = np.unique(pred_array, return_counts=True)
            min_count = np.min(counts)
            max_count = np.max(counts)

            if min_count / max_count < 0.01:  # Severe imbalance
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Severe prediction class imbalance detected",
                    suggestion="Check if this reflects true class distribution or indicates model bias"
                ))

        return issues

    def _validate_consistency(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        protected_attributes: Optional[List[str]]
    ) -> List[ValidationIssue]:
        """Validate data consistency and logical constraints."""
        issues = []

        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(data)) * 100

            if duplicate_percentage > 10:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"High number of duplicate rows: {duplicate_count} ({duplicate_percentage:.1f}%)",
                    affected_rows=duplicate_count,
                    suggestion="Consider removing duplicates to avoid data leakage"
                ))
            elif duplicate_count > 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Found {duplicate_count} duplicate rows",
                    affected_rows=duplicate_count,
                    suggestion="Review duplicates to ensure they are intentional"
                ))

        # Check protected attributes if specified
        if protected_attributes:
            for attr in protected_attributes:
                if attr not in data.columns:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Protected attribute '{attr}' not found in data columns",
                        column=attr,
                        suggestion="Check column names or update protected attributes list"
                    ))
                else:
                    # Check if protected attribute has reasonable number of groups
                    unique_groups = data[attr].nunique()
                    if unique_groups > 50:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Protected attribute '{attr}' has {unique_groups} unique values",
                            column=attr,
                            suggestion="Consider grouping or binning for meaningful bias analysis"
                        ))

        return issues

    def _validate_compatibility(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]],
        protected_attributes: Optional[List[str]]
    ) -> List[ValidationIssue]:
        """Validate compatibility with analysis requirements."""
        issues = []

        # Check minimum sample size
        if len(data) < 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Sample size too small: {len(data)} samples",
                suggestion="Provide at least 10 samples for meaningful analysis"
            ))
        elif len(data) < 100:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Small sample size: {len(data)} samples",
                suggestion="Results may be more reliable with larger sample sizes"
            ))

        # Check feature count
        if len(data.columns) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No features provided in data",
                suggestion="Include feature columns for analysis"
            ))
        elif len(data.columns) > 1000:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Very high feature count: {len(data.columns)} features",
                suggestion="Consider feature selection or dimensionality reduction"
            ))

        # Check if we have numeric features for certain analyses
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No numeric features found",
                suggestion="Some analyses (explainability, correlation) may be limited without numeric features"
            ))

        return issues

    def get_validation_summary(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Get summary of validation results."""
        error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.INFO)

        return {
            'total_issues': len(issues),
            'error_count': error_count,
            'warning_count': warning_count,
            'info_count': info_count,
            'is_valid': error_count == 0,
            'severity_distribution': {
                'errors': error_count,
                'warnings': warning_count,
                'info': info_count
            }
        }

    def format_issues_for_display(self, issues: List[ValidationIssue]) -> List[str]:
        """Format validation issues for user-friendly display."""
        formatted = []

        severity_icons = {
            ValidationSeverity.ERROR: "🚨",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.INFO: "ℹ️"
        }

        for issue in issues:
            icon = severity_icons.get(issue.severity, "•")
            message = f"{icon} {issue.message}"

            if issue.column:
                message += f" (Column: {issue.column})"

            if issue.affected_rows:
                message += f" ({issue.affected_rows} rows affected)"

            if issue.suggestion:
                message += f" → {issue.suggestion}"

            formatted.append(message)

        return formatted