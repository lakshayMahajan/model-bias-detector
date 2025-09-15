"""
Classification analyzer for binary and multiclass classification performance.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .base_analyzer import BaseAnalyzer, AnalysisResult, ProblemType, AnalysisType


class ClassificationAnalyzer(BaseAnalyzer):
    """
    Comprehensive analyzer for classification problems.

    Supports binary and multiclass classification with advanced metrics,
    ROC curves, precision-recall analysis, and confusion matrices.
    """

    def __init__(self):
        super().__init__(
            name="Classification Analyzer",
            supported_problem_types=[
                ProblemType.BINARY_CLASSIFICATION,
                ProblemType.MULTICLASS_CLASSIFICATION
            ]
        )
        self.analysis_type = AnalysisType.PERFORMANCE

    def analyze(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        probability_scores: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> AnalysisResult:
        """
        Perform comprehensive classification analysis.
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
        if isinstance(probability_scores, pd.Series):
            probability_scores = probability_scores.values

        # Create base result
        result = self._create_base_result(
            problem_type=problem_type,
            execution_time=0,  # Will be updated at the end
            sample_size=len(data),
            feature_count=len(data.columns)
        )

        if true_values is not None:
            # Calculate core metrics
            result.metrics = self._calculate_metrics(
                true_values, predictions, probability_scores, problem_type
            )

            # Generate insights and recommendations
            result.insights = self._generate_insights(result.metrics, problem_type)
            result.recommendations = self._generate_recommendations(result.metrics, problem_type)

            # Create visualizations
            result.plots = self._create_plots(
                true_values, predictions, probability_scores, problem_type
            )

            # Detailed analysis
            result.detailed_results = self._detailed_analysis(
                true_values, predictions, probability_scores, problem_type
            )
        else:
            # No true values - prediction distribution analysis only
            result.metrics = self._prediction_distribution_metrics(predictions)
            result.insights = ["Analysis performed on predictions only (no true values provided)"]
            result.recommendations = ["Obtain true labels for comprehensive performance evaluation"]
            result.plots = self._prediction_only_plots(predictions)

        # Update execution time
        result.execution_time = time.time() - start_time

        return result

    def _calculate_metrics(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        probability_scores: Optional[np.ndarray],
        problem_type: ProblemType
    ) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(true_values, predictions)

        # Handle averaging for multiclass
        average = 'binary' if problem_type == ProblemType.BINARY_CLASSIFICATION else 'weighted'

        metrics['precision'] = precision_score(true_values, predictions, average=average, zero_division=0)
        metrics['recall'] = recall_score(true_values, predictions, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(true_values, predictions, average=average, zero_division=0)

        # Advanced metrics
        try:
            metrics['matthews_corr'] = matthews_corrcoef(true_values, predictions)
        except:
            metrics['matthews_corr'] = 0.0

        try:
            metrics['cohen_kappa'] = cohen_kappa_score(true_values, predictions)
        except:
            metrics['cohen_kappa'] = 0.0

        # Probability-based metrics
        if probability_scores is not None:
            try:
                if problem_type == ProblemType.BINARY_CLASSIFICATION:
                    metrics['roc_auc'] = roc_auc_score(true_values, probability_scores)
                    metrics['log_loss'] = log_loss(true_values, probability_scores)
                else:
                    # For multiclass, use one-vs-rest
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        true_values, probability_scores, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                pass  # Skip if probability scores are incompatible

        return metrics

    def _prediction_distribution_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for prediction distribution analysis."""
        unique_values, counts = np.unique(predictions, return_counts=True)

        metrics = {
            'unique_predictions': len(unique_values),
            'most_common_prediction': unique_values[np.argmax(counts)],
            'prediction_entropy': -np.sum((counts/len(predictions)) * np.log2(counts/len(predictions) + 1e-10))
        }

        # Class balance
        for i, (value, count) in enumerate(zip(unique_values, counts)):
            metrics[f'class_{value}_proportion'] = count / len(predictions)

        return metrics

    def _generate_insights(self, metrics: Dict[str, float], problem_type: ProblemType) -> List[str]:
        """Generate actionable insights from metrics."""
        insights = []

        # Accuracy insights
        accuracy = metrics.get('accuracy', 0)
        if accuracy < 0.5:
            insights.append("⚠️ Model accuracy is below 50% - consider model retraining or feature engineering")
        elif accuracy < 0.7:
            insights.append("📊 Moderate accuracy - room for improvement through hyperparameter tuning")
        elif accuracy > 0.9:
            insights.append("🎯 Excellent accuracy - model is performing very well")

        # Precision vs Recall trade-off
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)

        if precision > 0.8 and recall < 0.6:
            insights.append("🎯 High precision, low recall - model is conservative, missing some positive cases")
        elif precision < 0.6 and recall > 0.8:
            insights.append("📢 Low precision, high recall - model is aggressive, may have false positives")
        elif precision > 0.8 and recall > 0.8:
            insights.append("✅ Excellent precision-recall balance")

        # Matthews Correlation Coefficient
        mcc = metrics.get('matthews_corr', 0)
        if mcc < 0.3:
            insights.append("📉 Low Matthews correlation - model performance is weak")
        elif mcc > 0.7:
            insights.append("📈 Strong Matthews correlation - reliable model performance")

        # ROC AUC insights
        if 'roc_auc' in metrics:
            auc = metrics['roc_auc']
            if auc < 0.6:
                insights.append("📉 Poor discriminative ability (AUC < 0.6)")
            elif auc > 0.8:
                insights.append("🎯 Excellent discriminative ability (AUC > 0.8)")

        return insights

    def _generate_recommendations(self, metrics: Dict[str, float], problem_type: ProblemType) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)

        if accuracy < 0.7:
            recommendations.append("🔧 Consider feature engineering or collecting more training data")
            recommendations.append("⚙️ Experiment with different algorithms or ensemble methods")

        if precision < 0.7:
            recommendations.append("🎯 To improve precision: increase decision threshold or use cost-sensitive learning")

        if recall < 0.7:
            recommendations.append("📢 To improve recall: lower decision threshold or address class imbalance")

        if f1 < 0.6:
            recommendations.append("⚖️ Poor F1 score indicates need for overall model improvement")

        # Matthews correlation specific
        mcc = metrics.get('matthews_corr', 0)
        if mcc < 0.5:
            recommendations.append("📊 Low MCC suggests checking for class imbalance or data quality issues")

        if len(recommendations) == 0:
            recommendations.append("✅ Model performance is good - consider A/B testing in production")

        return recommendations

    def _create_plots(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        probability_scores: Optional[np.ndarray],
        problem_type: ProblemType
    ) -> Dict[str, Any]:
        """Create comprehensive visualization plots."""
        plots = {}

        # Confusion Matrix
        plots['confusion_matrix'] = self._plot_confusion_matrix(true_values, predictions)

        # Classification Report Bar Chart
        plots['classification_report'] = self._plot_classification_report(true_values, predictions)

        # ROC Curve (for binary classification with probabilities)
        if problem_type == ProblemType.BINARY_CLASSIFICATION and probability_scores is not None:
            plots['roc_curve'] = self._plot_roc_curve(true_values, probability_scores)
            plots['precision_recall_curve'] = self._plot_precision_recall_curve(true_values, probability_scores)

        # Prediction Distribution
        plots['prediction_distribution'] = self._plot_prediction_distribution(predictions, true_values)

        return plots

    def _prediction_only_plots(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Create plots for prediction-only analysis."""
        plots = {}
        plots['prediction_distribution'] = self._plot_prediction_distribution(predictions)
        return plots

    def _plot_confusion_matrix(self, true_values: np.ndarray, predictions: np.ndarray):
        """Create confusion matrix heatmap."""
        cm = confusion_matrix(true_values, predictions)
        labels = sorted(np.unique(np.concatenate([true_values, predictions])))

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {label}' for label in labels],
            y=[f'Actual {label}' for label in labels],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class',
            width=500,
            height=400
        )

        return fig

    def _plot_classification_report(self, true_values: np.ndarray, predictions: np.ndarray):
        """Create classification report visualization."""
        from sklearn.metrics import classification_report

        report = classification_report(true_values, predictions, output_dict=True)

        # Extract metrics for each class
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']

        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            shared_yaxis=True
        )

        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in classes]
            fig.add_trace(
                go.Bar(x=classes, y=values, name=metric, showlegend=False),
                row=1, col=i+1
            )

        fig.update_layout(
            title='Classification Report by Class',
            height=400,
            width=800
        )

        return fig

    def _plot_roc_curve(self, true_values: np.ndarray, probability_scores: np.ndarray):
        """Create ROC curve plot."""
        fpr, tpr, _ = roc_curve(true_values, probability_scores)
        auc = roc_auc_score(true_values, probability_scores)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})',
            line=dict(width=2)
        ))

        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=500,
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        return fig

    def _plot_precision_recall_curve(self, true_values: np.ndarray, probability_scores: np.ndarray):
        """Create precision-recall curve plot."""
        precision, recall, _ = precision_recall_curve(true_values, probability_scores)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(width=2)
        ))

        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=500,
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        return fig

    def _plot_prediction_distribution(self, predictions: np.ndarray, true_values: Optional[np.ndarray] = None):
        """Create prediction distribution plot."""
        if true_values is not None:
            # Compare predictions vs actual
            df = pd.DataFrame({
                'Value': np.concatenate([predictions, true_values]),
                'Type': ['Predicted'] * len(predictions) + ['Actual'] * len(true_values)
            })

            fig = px.histogram(
                df, x='Value', color='Type',
                title='Prediction vs Actual Distribution',
                barmode='overlay',
                opacity=0.7
            )
        else:
            # Predictions only
            fig = px.histogram(
                x=predictions,
                title='Prediction Distribution',
                nbins=min(20, len(np.unique(predictions)))
            )

        fig.update_layout(width=600, height=400)
        return fig

    def _detailed_analysis(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        probability_scores: Optional[np.ndarray],
        problem_type: ProblemType
    ) -> Dict[str, Any]:
        """Perform detailed analysis and return structured results."""
        detailed = {}

        # Classification report
        from sklearn.metrics import classification_report
        detailed['classification_report'] = classification_report(
            true_values, predictions, output_dict=True
        )

        # Confusion matrix details
        cm = confusion_matrix(true_values, predictions)
        detailed['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': sorted(np.unique(np.concatenate([true_values, predictions]))).tolist()
        }

        # Per-class analysis
        unique_classes = np.unique(true_values)
        detailed['per_class_analysis'] = {}

        for cls in unique_classes:
            mask = true_values == cls
            class_predictions = predictions[mask]
            class_accuracy = np.mean(class_predictions == cls)

            detailed['per_class_analysis'][str(cls)] = {
                'support': int(np.sum(mask)),
                'accuracy': float(class_accuracy),
                'most_confused_with': self._get_most_confused_class(true_values, predictions, cls)
            }

        return detailed

    def _get_most_confused_class(self, true_values: np.ndarray, predictions: np.ndarray, target_class) -> str:
        """Find the class most often confused with the target class."""
        mask = true_values == target_class
        misclassified = predictions[mask]
        misclassified = misclassified[misclassified != target_class]

        if len(misclassified) == 0:
            return "None (perfect classification)"

        unique, counts = np.unique(misclassified, return_counts=True)
        most_confused = unique[np.argmax(counts)]
        return str(most_confused)