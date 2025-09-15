"""
Visualization engine for creating interactive charts and dashboards.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


class VisualizationEngine:
    """
    Comprehensive visualization engine for creating interactive charts and dashboards.

    Provides standardized plotting functions for all types of ML analysis results.
    """

    def __init__(self):
        self.default_colors = px.colors.qualitative.Set3
        self.theme = 'plotly_white'

    def create_metrics_dashboard(
        self,
        metrics: Dict[str, float],
        title: str = "Model Metrics Dashboard"
    ) -> go.Figure:
        """Create a comprehensive metrics dashboard."""
        # Organize metrics by type
        organized_metrics = self._organize_metrics(metrics)

        # Create subplots
        subplot_titles = list(organized_metrics.keys())
        n_subplots = len(subplot_titles)

        if n_subplots == 0:
            return self._create_empty_plot("No metrics available")

        cols = min(2, n_subplots)
        rows = (n_subplots + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "indicator"}] * cols for _ in range(rows)]
        )

        for i, (category, category_metrics) in enumerate(organized_metrics.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1

            # Take the first metric for the indicator
            metric_name, metric_value = next(iter(category_metrics.items()))

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=metric_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=title,
            height=200 * rows,
            width=800,
            template=self.theme
        )

        return fig

    def create_comparison_chart(
        self,
        data: Dict[str, List[float]],
        chart_type: str = "bar",
        title: str = "Comparison Chart"
    ) -> go.Figure:
        """Create comparison charts (bar, line, radar)."""
        if not data:
            return self._create_empty_plot("No data available for comparison")

        if chart_type == "radar":
            return self._create_radar_chart(data, title)
        elif chart_type == "line":
            return self._create_line_chart(data, title)
        else:  # default to bar
            return self._create_bar_chart(data, title)

    def create_distribution_plot(
        self,
        data: Union[pd.Series, np.ndarray, List[float]],
        title: str = "Distribution Plot",
        show_stats: bool = True
    ) -> go.Figure:
        """Create distribution plot with statistics."""
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=30,
            name="Distribution",
            opacity=0.7,
            marker_color='steelblue'
        ))

        if show_stats:
            # Add statistics as annotations
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)

            fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                         annotation_text=f"Median: {median_val:.2f}")

        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            template=self.theme,
            width=700,
            height=400
        )

        return fig

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Correlation Heatmap"
    ) -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            width=600,
            height=600,
            template=self.theme
        )

        return fig

    def create_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 15
    ) -> go.Figure:
        """Create feature importance plot."""
        if not feature_importance:
            return self._create_empty_plot("No feature importance data available")

        # Sort and take top N
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]

        features, importance = zip(*top_features)

        fig = go.Figure(data=[go.Bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            marker_color='steelblue'
        )])

        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(top_features) * 25),
            width=700,
            yaxis={'categoryorder': 'total ascending'},
            template=self.theme
        )

        return fig

    def create_bias_analysis_plot(
        self,
        bias_data: Dict[str, Dict[str, float]],
        title: str = "Bias Analysis"
    ) -> go.Figure:
        """Create bias analysis visualization."""
        if not bias_data:
            return self._create_empty_plot("No bias data available")

        # Create grouped bar chart
        groups = list(bias_data.keys())
        metrics = list(next(iter(bias_data.values())).keys()) if bias_data else []

        fig = go.Figure()

        for metric in metrics:
            values = [bias_data[group].get(metric, 0) for group in groups]
            fig.add_trace(go.Bar(
                name=metric,
                x=groups,
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Protected Groups',
            yaxis_title='Metric Value',
            barmode='group',
            template=self.theme,
            width=800,
            height=500
        )

        return fig

    def create_performance_comparison(
        self,
        performance_data: Dict[str, Dict[str, float]],
        title: str = "Performance by Group"
    ) -> go.Figure:
        """Create performance comparison across groups."""
        if not performance_data:
            return self._create_empty_plot("No performance data available")

        groups = list(performance_data.keys())
        metrics = list(next(iter(performance_data.values())).keys()) if performance_data else []

        # Create subplot for each metric
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            shared_yaxis=True
        )

        for i, metric in enumerate(metrics):
            values = [performance_data[group].get(metric, 0) for group in groups]

            fig.add_trace(
                go.Bar(
                    x=groups,
                    y=values,
                    name=metric,
                    showlegend=False,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            title=title,
            height=400,
            width=200 * len(metrics),
            template=self.theme
        )

        return fig

    def create_quality_overview(
        self,
        quality_metrics: Dict[str, float],
        title: str = "Data Quality Overview"
    ) -> go.Figure:
        """Create data quality overview chart."""
        if not quality_metrics:
            return self._create_empty_plot("No quality metrics available")

        # Create pie chart for quality metrics
        metrics = list(quality_metrics.keys())
        values = list(quality_metrics.values())

        # Normalize values to percentages
        total = sum(values)
        percentages = [v/total * 100 if total > 0 else 0 for v in values]

        fig = go.Figure(data=[go.Pie(
            labels=metrics,
            values=percentages,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto'
        )])

        fig.update_layout(
            title=title,
            template=self.theme,
            width=600,
            height=500
        )

        return fig

    def display_plot_in_streamlit(
        self,
        fig: go.Figure,
        use_container_width: bool = True,
        key: Optional[str] = None
    ):
        """Display plot in Streamlit with consistent formatting."""
        st.plotly_chart(fig, use_container_width=use_container_width, key=key)

    def _organize_metrics(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Organize metrics by category."""
        organized = {
            "Performance": {},
            "Quality": {},
            "Bias": {},
            "Other": {}
        }

        for key, value in metrics.items():
            key_lower = key.lower()

            if any(perf_key in key_lower for perf_key in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'rmse', 'mae']):
                organized["Performance"][key] = value
            elif any(qual_key in key_lower for qual_key in ['completeness', 'quality', 'missing', 'duplicate']):
                organized["Quality"][key] = value
            elif any(bias_key in key_lower for bias_key in ['bias', 'fairness', 'disparate', 'demographic']):
                organized["Bias"][key] = value
            else:
                organized["Other"][key] = value

        # Remove empty categories
        return {k: v for k, v in organized.items() if v}

    def _create_radar_chart(self, data: Dict[str, List[float]], title: str) -> go.Figure:
        """Create radar chart."""
        fig = go.Figure()

        categories = list(data.keys())

        # Assume all lists have same length, take first element for each category
        values = [data[cat][0] if data[cat] else 0 for cat in categories]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Metrics'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            template=self.theme
        )

        return fig

    def _create_line_chart(self, data: Dict[str, List[float]], title: str) -> go.Figure:
        """Create line chart."""
        fig = go.Figure()

        for name, values in data.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=name
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Index',
            yaxis_title='Value',
            template=self.theme
        )

        return fig

    def _create_bar_chart(self, data: Dict[str, List[float]], title: str) -> go.Figure:
        """Create bar chart."""
        fig = go.Figure()

        categories = list(data.keys())
        # Take first value from each list
        values = [data[cat][0] if data[cat] else 0 for cat in categories]

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color='steelblue'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Categories',
            yaxis_title='Value',
            template=self.theme
        )

        return fig

    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message."""
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font_size=16
        )

        fig.update_layout(
            template=self.theme,
            width=600,
            height=400
        )

        return fig