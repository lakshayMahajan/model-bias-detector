"""
Basic functionality tests for the Universal AI Model Analyzer.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyzers.analysis_engine import AnalysisEngine
from analyzers.base_analyzer import ProblemType, AnalysisType
from utils.data_processor import DataProcessor, FileFormat
from utils.data_validator import DataValidator
from config.smart_config import SmartConfigInterface


class TestDataProcessor:
    """Test data processing functionality."""

    def test_data_processor_initialization(self):
        """Test that DataProcessor initializes correctly."""
        processor = DataProcessor()
        assert processor is not None
        assert hasattr(processor, 'prediction_keywords')
        assert hasattr(processor, 'true_value_keywords')

    def test_sample_data_loading(self):
        """Test loading sample datasets."""
        processor = DataProcessor()

        # Test perfect dataset
        perfect_data_path = project_root / "demo_perfect_dataset.csv"
        if perfect_data_path.exists():
            df = processor.load_data(file_path=str(perfect_data_path))
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'prediction' in df.columns
            assert 'true_label' in df.columns

        # Test salary regression dataset
        salary_data_path = project_root / "demo_salary_regression.csv"
        if salary_data_path.exists():
            df = processor.load_data(file_path=str(salary_data_path))
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'predicted_salary' in df.columns
            assert 'actual_salary' in df.columns

    def test_column_analysis(self):
        """Test column analysis functionality."""
        processor = DataProcessor()

        # Create test data
        test_data = pd.DataFrame({
            'prediction': [0, 1, 0, 1, 0],
            'true_label': [0, 1, 1, 1, 0],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'age': [25, 30, 35, 28, 32],
            'feature_1': [1.2, 2.3, 3.4, 4.5, 5.6]
        })

        mappings = processor.analyze_columns(test_data)
        assert len(mappings) == 5

        # Check that mappings have correct structure
        for mapping in mappings:
            assert hasattr(mapping, 'name')
            assert hasattr(mapping, 'type')
            assert hasattr(mapping, 'confidence')


class TestAnalysisEngine:
    """Test analysis engine functionality."""

    def test_analysis_engine_initialization(self):
        """Test that AnalysisEngine initializes correctly."""
        engine = AnalysisEngine()
        assert engine is not None
        assert hasattr(engine, 'analyzers')

    def test_problem_type_detection(self):
        """Test automatic problem type detection."""
        engine = AnalysisEngine()

        # Binary classification
        binary_predictions = np.array([0, 1, 0, 1, 0])
        problem_type = engine._auto_detect_problem_type(binary_predictions)
        assert problem_type == ProblemType.BINARY_CLASSIFICATION

        # Regression
        regression_predictions = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
        problem_type = engine._auto_detect_problem_type(regression_predictions)
        assert problem_type == ProblemType.REGRESSION

    def test_basic_analysis_workflow(self):
        """Test basic analysis workflow."""
        engine = AnalysisEngine()

        # Create test data
        test_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'age': np.random.randint(25, 65, 100)
        })

        predictions = np.random.choice([0, 1], 100)
        true_values = np.random.choice([0, 1], 100)

        try:
            report = engine.analyze_model(
                data=test_data,
                predictions=predictions,
                true_values=true_values,
                analysis_types=[AnalysisType.PERFORMANCE, AnalysisType.DATA_QUALITY]
            )

            assert report is not None
            assert hasattr(report, 'problem_type')
            assert hasattr(report, 'overall_score')
            assert hasattr(report, 'key_insights')

        except Exception as e:
            # If analysis fails, ensure it's not due to basic setup issues
            pytest.fail(f"Basic analysis workflow failed: {str(e)}")


class TestDataValidator:
    """Test data validation functionality."""

    def test_data_validator_initialization(self):
        """Test that DataValidator initializes correctly."""
        validator = DataValidator()
        assert validator is not None
        assert hasattr(validator, 'validation_rules')

    def test_basic_validation(self):
        """Test basic data validation."""
        validator = DataValidator()

        # Valid data
        valid_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        valid_predictions = np.array([0, 1, 0, 1, 0])

        is_valid, issues = validator.validate_analysis_inputs(
            valid_data, valid_predictions
        )

        # Should have no critical errors for this simple case
        error_count = sum(1 for issue in issues if issue.severity.value == 'error')
        assert error_count == 0

    def test_invalid_data_detection(self):
        """Test detection of invalid data."""
        validator = DataValidator()

        # Empty data
        empty_data = pd.DataFrame()
        empty_predictions = np.array([])

        is_valid, issues = validator.validate_analysis_inputs(
            empty_data, empty_predictions
        )

        assert not is_valid
        assert len(issues) > 0


class TestSmartConfig:
    """Test smart configuration functionality."""

    def test_smart_config_initialization(self):
        """Test that SmartConfigInterface initializes correctly."""
        smart_config = SmartConfigInterface()
        assert smart_config is not None
        assert hasattr(smart_config, 'data_processor')

    def test_config_suggestion(self):
        """Test configuration suggestion."""
        smart_config = SmartConfigInterface()

        # Create test data
        test_data = pd.DataFrame({
            'prediction': [0, 1, 0, 1, 0],
            'true_label': [0, 1, 1, 1, 0],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'feature_1': [1.2, 2.3, 3.4, 4.5, 5.6]
        })

        try:
            config, explanations, confidence_scores = smart_config.analyze_and_suggest_config(test_data)

            assert config is not None
            assert hasattr(config, 'data_config')
            assert hasattr(config, 'problem_type')
            assert isinstance(explanations, list)
            assert isinstance(confidence_scores, dict)

        except Exception as e:
            pytest.fail(f"Smart config suggestion failed: {str(e)}")


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_classification(self):
        """Test end-to-end classification analysis."""
        # Load sample data
        processor = DataProcessor()
        perfect_data_path = project_root / "demo_perfect_dataset.csv"

        if not perfect_data_path.exists():
            pytest.skip("Sample dataset not available")

        try:
            # Load data
            df = processor.load_data(file_path=str(perfect_data_path))

            # Get predictions and true values
            predictions = df['prediction'].values
            true_values = df['true_label'].values

            # Run analysis
            engine = AnalysisEngine()
            report = engine.analyze_model(
                data=df,
                predictions=predictions,
                true_values=true_values,
                protected_attributes=['gender', 'race'],
                analysis_types=[AnalysisType.PERFORMANCE, AnalysisType.BIAS_FAIRNESS]
            )

            # Verify results
            assert report is not None
            assert report.problem_type == ProblemType.BINARY_CLASSIFICATION
            assert report.performance_analysis is not None
            assert report.bias_analysis is not None
            assert len(report.key_insights) > 0

        except Exception as e:
            pytest.fail(f"End-to-end classification test failed: {str(e)}")

    def test_end_to_end_regression(self):
        """Test end-to-end regression analysis."""
        # Load sample data
        processor = DataProcessor()
        salary_data_path = project_root / "demo_salary_regression.csv"

        if not salary_data_path.exists():
            pytest.skip("Sample dataset not available")

        try:
            # Load data
            df = processor.load_data(file_path=str(salary_data_path))

            # Get predictions and true values
            predictions = df['predicted_salary'].values
            true_values = df['actual_salary'].values

            # Run analysis
            engine = AnalysisEngine()
            report = engine.analyze_model(
                data=df,
                predictions=predictions,
                true_values=true_values,
                protected_attributes=['gender', 'race'],
                analysis_types=[AnalysisType.PERFORMANCE, AnalysisType.DATA_QUALITY]
            )

            # Verify results
            assert report is not None
            assert report.problem_type == ProblemType.REGRESSION
            assert report.performance_analysis is not None
            assert report.data_quality_analysis is not None
            assert len(report.key_insights) > 0

        except Exception as e:
            pytest.fail(f"End-to-end regression test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])