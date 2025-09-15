"""
Base analyzer class defining the universal interface for all ML analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class ProblemType(Enum):
    """Types of ML problems that can be analyzed."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    AUTO_DETECT = "auto_detect"


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    PERFORMANCE = "performance"
    BIAS_FAIRNESS = "bias_fairness"
    DATA_QUALITY = "data_quality"
    EXPLAINABILITY = "explainability"
    DRIFT = "drift"
    ROBUSTNESS = "robustness"


@dataclass
class AnalysisResult:
    """
    Standardized result format for all analyzer outputs.
    """
    analyzer_name: str
    analysis_type: AnalysisType
    problem_type: ProblemType

    # Core metrics and results
    metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]

    # Visualization data
    plots: Dict[str, Any]

    # Detailed analysis data
    detailed_results: Dict[str, Any]

    # Metadata
    execution_time: float
    sample_size: int
    feature_count: Optional[int] = None

    # Quality indicators
    confidence_score: float = 1.0  # 0-1 confidence in results
    data_quality_issues: List[str] = None

    def __post_init__(self):
        if self.data_quality_issues is None:
            self.data_quality_issues = []


class BaseAnalyzer(ABC):
    """
    Abstract base class for all ML model analyzers.

    Defines the universal interface that all specialized analyzers must implement.
    """

    def __init__(self, name: str, supported_problem_types: List[ProblemType]):
        self.name = name
        self.supported_problem_types = supported_problem_types
        self.analysis_type = AnalysisType.PERFORMANCE  # Default, should be overridden

    def can_analyze(self, problem_type: ProblemType) -> bool:
        """Check if this analyzer supports the given problem type."""
        return problem_type in self.supported_problem_types or ProblemType.AUTO_DETECT in self.supported_problem_types

    @abstractmethod
    def analyze(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        **kwargs
    ) -> AnalysisResult:
        """
        Perform analysis on the provided data and predictions.

        Args:
            data: Input features/data
            predictions: Model predictions
            true_values: Ground truth values (if available)
            problem_type: Type of ML problem
            **kwargs: Additional analyzer-specific parameters

        Returns:
            AnalysisResult object containing all analysis outputs
        """
        pass

    def validate_inputs(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None
    ) -> List[str]:
        """
        Validate inputs and return list of validation errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Basic validation
        if data is None or data.empty:
            errors.append("Data cannot be None or empty")

        if predictions is None:
            errors.append("Predictions cannot be None")

        if len(data) == 0:
            errors.append("Data must contain at least one sample")

        # Length matching
        if hasattr(predictions, '__len__') and len(predictions) != len(data):
            errors.append(f"Predictions length ({len(predictions)}) must match data length ({len(data)})")

        if true_values is not None and hasattr(true_values, '__len__') and len(true_values) != len(data):
            errors.append(f"True values length ({len(true_values)}) must match data length ({len(data)})")

        # Problem type validation
        if problem_type and not self.can_analyze(problem_type):
            errors.append(f"Analyzer '{self.name}' does not support problem type '{problem_type.value}'")

        return errors

    def detect_problem_type(
        self,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> ProblemType:
        """
        Automatically detect the problem type based on predictions and true values.

        Returns:
            Detected ProblemType
        """
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(true_values, pd.Series):
            true_values = true_values.values

        # Check if predictions are continuous (regression) or discrete (classification)
        unique_predictions = np.unique(predictions)

        # If predictions are floats and have many unique values, likely regression
        if len(unique_predictions) > 20 and np.issubdtype(predictions.dtype, np.floating):
            return ProblemType.REGRESSION

        # If predictions are binary (0/1 or two unique values), binary classification
        if len(unique_predictions) == 2:
            return ProblemType.BINARY_CLASSIFICATION

        # If predictions have few unique discrete values, multiclass classification
        if len(unique_predictions) <= 20:
            return ProblemType.MULTICLASS_CLASSIFICATION

        # Default to regression for continuous values
        return ProblemType.REGRESSION

    def _create_base_result(
        self,
        problem_type: ProblemType,
        execution_time: float,
        sample_size: int,
        feature_count: Optional[int] = None
    ) -> AnalysisResult:
        """
        Create a base AnalysisResult with common fields populated.
        """
        return AnalysisResult(
            analyzer_name=self.name,
            analysis_type=self.analysis_type,
            problem_type=problem_type,
            metrics={},
            insights=[],
            recommendations=[],
            plots={},
            detailed_results={},
            execution_time=execution_time,
            sample_size=sample_size,
            feature_count=feature_count,
            confidence_score=1.0,
            data_quality_issues=[]
        )