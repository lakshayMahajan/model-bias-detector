"""
Universal data processor supporting multiple formats and intelligent column detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import io
from enum import Enum


class FileFormat(Enum):
    """Supported file formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"
    TSV = "tsv"


class ColumnType(Enum):
    """Types of columns for ML analysis."""
    PREDICTION = "prediction"
    TRUE_VALUE = "true_value"
    PROBABILITY = "probability"
    FEATURE = "feature"
    PROTECTED_ATTRIBUTE = "protected_attribute"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    METADATA = "metadata"


@dataclass
class ColumnMapping:
    """Mapping between column names and their types."""
    name: str
    type: ColumnType
    confidence: float  # 0-1 confidence in the mapping
    suggestions: List[str] = None  # Alternative column names

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class DataConfig:
    """Configuration for data processing."""
    prediction_column: Optional[str] = None
    true_value_column: Optional[str] = None
    probability_columns: List[str] = None
    protected_attributes: List[str] = None
    feature_columns: List[str] = None
    timestamp_column: Optional[str] = None
    identifier_column: Optional[str] = None

    def __post_init__(self):
        if self.probability_columns is None:
            self.probability_columns = []
        if self.protected_attributes is None:
            self.protected_attributes = []
        if self.feature_columns is None:
            self.feature_columns = []


class DataProcessor:
    """
    Universal data processor with intelligent column detection and multi-format support.
    """

    def __init__(self):
        self.prediction_keywords = [
            'prediction', 'pred', 'predicted', 'predict', 'output', 'y_pred',
            'forecast', 'estimate', 'score', 'class', 'label'
        ]

        self.true_value_keywords = [
            'true', 'actual', 'real', 'ground_truth', 'target', 'y_true',
            'label', 'correct', 'reference', 'expected'
        ]

        self.probability_keywords = [
            'prob', 'probability', 'confidence', 'score', 'likelihood',
            'prob_', 'proba', 'confidence_score'
        ]

        self.protected_attribute_keywords = [
            'gender', 'race', 'ethnicity', 'age', 'religion', 'disability',
            'sexual_orientation', 'nationality', 'protected', 'sensitive',
            'demographic', 'group', 'category'
        ]

        self.timestamp_keywords = [
            'time', 'date', 'timestamp', 'created', 'updated', 'datetime',
            'period', 'when', 'at'
        ]

        self.identifier_keywords = [
            'id', 'identifier', 'key', 'index', 'row', 'record', 'unique',
            'uuid', 'guid'
        ]

    def load_data(
        self,
        file_path: Optional[str] = None,
        file_content: Optional[Union[str, bytes]] = None,
        file_format: Optional[FileFormat] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from various sources and formats.

        Args:
            file_path: Path to the file
            file_content: File content (for uploaded files)
            file_format: Explicit file format
            **kwargs: Additional parameters for pandas readers

        Returns:
            Loaded DataFrame
        """
        if file_path:
            return self._load_from_path(file_path, file_format, **kwargs)
        elif file_content:
            return self._load_from_content(file_content, file_format, **kwargs)
        else:
            raise ValueError("Either file_path or file_content must be provided")

    def _load_from_path(
        self,
        file_path: str,
        file_format: Optional[FileFormat] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from file path."""
        path = Path(file_path)

        if file_format is None:
            file_format = self._detect_format_from_extension(path.suffix)

        if file_format == FileFormat.CSV:
            return pd.read_csv(file_path, **kwargs)
        elif file_format == FileFormat.TSV:
            return pd.read_csv(file_path, sep='\t', **kwargs)
        elif file_format == FileFormat.JSON:
            return pd.read_json(file_path, **kwargs)
        elif file_format == FileFormat.PARQUET:
            return pd.read_parquet(file_path, **kwargs)
        elif file_format == FileFormat.EXCEL:
            return pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _load_from_content(
        self,
        content: Union[str, bytes],
        file_format: Optional[FileFormat] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from file content."""
        if file_format == FileFormat.CSV:
            return pd.read_csv(io.StringIO(content) if isinstance(content, str) else io.BytesIO(content), **kwargs)
        elif file_format == FileFormat.TSV:
            return pd.read_csv(io.StringIO(content) if isinstance(content, str) else io.BytesIO(content), sep='\t', **kwargs)
        elif file_format == FileFormat.JSON:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = json.loads(content.decode())
            return pd.DataFrame(data)
        elif file_format == FileFormat.EXCEL:
            return pd.read_excel(io.BytesIO(content), **kwargs)
        else:
            raise ValueError(f"Unsupported file format for content loading: {file_format}")

    def _detect_format_from_extension(self, extension: str) -> FileFormat:
        """Detect file format from extension."""
        extension = extension.lower()

        # Add dot if missing
        if not extension.startswith('.'):
            extension = '.' + extension

        format_map = {
            '.csv': FileFormat.CSV,
            '.tsv': FileFormat.TSV,
            '.txt': FileFormat.CSV,  # Assume CSV for .txt
            '.json': FileFormat.JSON,
            '.parquet': FileFormat.PARQUET,
            '.xlsx': FileFormat.EXCEL,
            '.xls': FileFormat.EXCEL
        }

        if extension in format_map:
            return format_map[extension]
        else:
            raise ValueError(f"Unknown file extension: {extension}")

    def analyze_columns(self, df: pd.DataFrame) -> List[ColumnMapping]:
        """
        Analyze DataFrame columns and suggest their types.

        Args:
            df: DataFrame to analyze

        Returns:
            List of ColumnMapping objects with suggestions
        """
        mappings = []

        for col in df.columns:
            col_lower = col.lower()
            mapping = self._classify_column(col, col_lower, df[col])
            mappings.append(mapping)

        return mappings

    def _classify_column(self, col_name: str, col_lower: str, col_data: pd.Series) -> ColumnMapping:
        """Classify a single column."""
        scores = {
            ColumnType.PREDICTION: self._score_column_type(col_lower, self.prediction_keywords),
            ColumnType.TRUE_VALUE: self._score_column_type(col_lower, self.true_value_keywords),
            ColumnType.PROBABILITY: self._score_column_type(col_lower, self.probability_keywords),
            ColumnType.PROTECTED_ATTRIBUTE: self._score_column_type(col_lower, self.protected_attribute_keywords),
            ColumnType.TIMESTAMP: self._score_column_type(col_lower, self.timestamp_keywords),
            ColumnType.IDENTIFIER: self._score_column_type(col_lower, self.identifier_keywords)
        }

        # Add data-based scoring
        scores.update(self._score_by_data_characteristics(col_data))

        # Find the best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        # If confidence is too low, classify as feature
        if confidence < 0.3:
            best_type = ColumnType.FEATURE
            confidence = 0.5

        return ColumnMapping(
            name=col_name,
            type=best_type,
            confidence=confidence,
            suggestions=self._get_alternative_suggestions(col_lower, scores)
        )

    def _score_column_type(self, col_name: str, keywords: List[str]) -> float:
        """Score how well a column name matches a type based on keywords."""
        max_score = 0.0

        for keyword in keywords:
            if keyword in col_name:
                # Exact match gets higher score
                if col_name == keyword:
                    max_score = max(max_score, 1.0)
                # Substring match gets partial score
                elif col_name.startswith(keyword) or col_name.endswith(keyword):
                    max_score = max(max_score, 0.8)
                else:
                    max_score = max(max_score, 0.6)

        return max_score

    def _score_by_data_characteristics(self, col_data: pd.Series) -> Dict[ColumnType, float]:
        """Score column type based on data characteristics."""
        scores = {}

        # Check for binary data (predictions)
        unique_values = col_data.nunique()
        if unique_values == 2 and col_data.dtype in ['int64', 'bool']:
            scores[ColumnType.PREDICTION] = 0.7
            scores[ColumnType.TRUE_VALUE] = 0.7

        # Check for probability-like data (0-1 range)
        if col_data.dtype in ['float64', 'float32']:
            if col_data.min() >= 0 and col_data.max() <= 1:
                scores[ColumnType.PROBABILITY] = 0.8

        # Check for categorical data (protected attributes)
        if col_data.dtype == 'object' and unique_values < 20:
            scores[ColumnType.PROTECTED_ATTRIBUTE] = 0.4

        # Check for datetime data
        try:
            pd.to_datetime(col_data.head(100))
            scores[ColumnType.TIMESTAMP] = 0.9
        except:
            pass

        # Check for identifier-like data
        if unique_values == len(col_data) and unique_values > 10:
            scores[ColumnType.IDENTIFIER] = 0.8

        return scores

    def _get_alternative_suggestions(self, col_name: str, scores: Dict[ColumnType, float]) -> List[str]:
        """Get alternative type suggestions for a column."""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        suggestions = []

        for col_type, score in sorted_scores[:3]:  # Top 3 suggestions
            if score > 0.3:
                suggestions.append(col_type.value)

        return suggestions

    def create_data_config(
        self,
        df: pd.DataFrame,
        column_mappings: List[ColumnMapping] = None
    ) -> DataConfig:
        """
        Create a DataConfig from analyzed columns.

        Args:
            df: DataFrame
            column_mappings: Optional pre-analyzed column mappings

        Returns:
            DataConfig object
        """
        if column_mappings is None:
            column_mappings = self.analyze_columns(df)

        config = DataConfig()

        # Extract columns by type
        for mapping in column_mappings:
            if mapping.type == ColumnType.PREDICTION and config.prediction_column is None:
                config.prediction_column = mapping.name
            elif mapping.type == ColumnType.TRUE_VALUE and config.true_value_column is None:
                config.true_value_column = mapping.name
            elif mapping.type == ColumnType.PROBABILITY:
                config.probability_columns.append(mapping.name)
            elif mapping.type == ColumnType.PROTECTED_ATTRIBUTE:
                config.protected_attributes.append(mapping.name)
            elif mapping.type == ColumnType.TIMESTAMP and config.timestamp_column is None:
                config.timestamp_column = mapping.name
            elif mapping.type == ColumnType.IDENTIFIER and config.identifier_column is None:
                config.identifier_column = mapping.name
            elif mapping.type == ColumnType.FEATURE:
                config.feature_columns.append(mapping.name)

        return config

    def validate_data_config(self, df: pd.DataFrame, config: DataConfig) -> List[str]:
        """
        Validate that the data config is consistent with the DataFrame.

        Args:
            df: DataFrame to validate against
            config: DataConfig to validate

        Returns:
            List of validation errors
        """
        errors = []

        # Check that specified columns exist
        all_specified_columns = []

        if config.prediction_column:
            all_specified_columns.append(config.prediction_column)
        if config.true_value_column:
            all_specified_columns.append(config.true_value_column)
        if config.timestamp_column:
            all_specified_columns.append(config.timestamp_column)
        if config.identifier_column:
            all_specified_columns.append(config.identifier_column)

        all_specified_columns.extend(config.probability_columns)
        all_specified_columns.extend(config.protected_attributes)
        all_specified_columns.extend(config.feature_columns)

        for col in all_specified_columns:
            if col not in df.columns:
                errors.append(f"Column '{col}' not found in DataFrame")

        # Check for required columns
        if not config.prediction_column:
            errors.append("No prediction column specified")

        return errors

    def prepare_data(
        self,
        df: pd.DataFrame,
        config: DataConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for analysis based on configuration.

        Args:
            df: Input DataFrame
            config: DataConfig specifying column roles

        Returns:
            Tuple of (processed_dataframe, metadata)
        """
        processed_df = df.copy()
        metadata = {}

        # Handle missing values
        missing_info = {}
        for col in processed_df.columns:
            missing_count = processed_df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(processed_df) * 100
                }

        metadata['missing_values'] = missing_info

        # Convert data types
        if config.prediction_column and processed_df[config.prediction_column].dtype == 'object':
            try:
                processed_df[config.prediction_column] = pd.to_numeric(processed_df[config.prediction_column])
            except:
                pass

        if config.true_value_column and processed_df[config.true_value_column].dtype == 'object':
            try:
                processed_df[config.true_value_column] = pd.to_numeric(processed_df[config.true_value_column])
            except:
                pass

        # Handle timestamp columns
        if config.timestamp_column:
            try:
                processed_df[config.timestamp_column] = pd.to_datetime(processed_df[config.timestamp_column])
                metadata['has_temporal_data'] = True
            except:
                metadata['timestamp_conversion_failed'] = True

        # Collect basic statistics
        metadata['shape'] = processed_df.shape
        metadata['dtypes'] = processed_df.dtypes.to_dict()
        metadata['config'] = config

        return processed_df, metadata