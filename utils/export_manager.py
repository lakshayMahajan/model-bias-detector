"""
Export manager for multi-format export and reporting capabilities.
"""

import json
import csv
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from dataclasses import asdict
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

if TYPE_CHECKING:
    from analyzers.analysis_engine import AnalysisReport


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    ZIP = "zip"


class ExportManager:
    """
    Comprehensive export manager for analysis results and reports.

    Supports multiple formats including JSON, CSV, HTML, PDF, and ZIP packages.
    """

    def __init__(self):
        self.supported_formats = [fmt.value for fmt in ExportFormat]

    def export_analysis_report(
        self,
        report: 'AnalysisReport',
        format: ExportFormat,
        output_path: Optional[str] = None,
        include_plots: bool = True,
        **kwargs
    ) -> str:
        """
        Export comprehensive analysis report.

        Args:
            report: AnalysisReport to export
            format: Export format
            output_path: Output file path (auto-generated if None)
            include_plots: Whether to include plot visualizations
            **kwargs: Additional format-specific options

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self._generate_output_path(report, format)

        if format == ExportFormat.JSON:
            return self._export_json(report, output_path, include_plots)
        elif format == ExportFormat.CSV:
            return self._export_csv(report, output_path)
        elif format == ExportFormat.HTML:
            return self._export_html(report, output_path, include_plots)
        elif format == ExportFormat.PDF:
            return self._export_pdf(report, output_path, include_plots)
        elif format == ExportFormat.ZIP:
            return self._export_zip_package(report, output_path, include_plots)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_metrics_only(
        self,
        metrics: Dict[str, Any],
        format: ExportFormat,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Export only metrics data."""
        if output_path is None:
            output_path = f"metrics_export.{format.value}"

        if format == ExportFormat.JSON:
            return self._export_metrics_json(metrics, output_path)
        elif format == ExportFormat.CSV:
            return self._export_metrics_csv(metrics, output_path)
        else:
            raise ValueError(f"Metrics export not supported for format: {format}")

    def export_plots(
        self,
        plots: Dict[str, go.Figure],
        format: str = "html",
        output_dir: str = "plots",
        **kwargs
    ) -> List[str]:
        """
        Export plot visualizations.

        Args:
            plots: Dictionary of plot names and figures
            format: Plot export format (html, png, jpg, svg, pdf)
            output_dir: Output directory for plots

        Returns:
            List of exported file paths
        """
        Path(output_dir).mkdir(exist_ok=True)
        exported_files = []

        for plot_name, fig in plots.items():
            if fig is None:
                continue

            safe_name = self._sanitize_filename(plot_name)
            output_path = Path(output_dir) / f"{safe_name}.{format}"

            try:
                if format == "html":
                    fig.write_html(str(output_path))
                elif format == "png":
                    fig.write_image(str(output_path), format="png", width=800, height=600)
                elif format == "jpg":
                    fig.write_image(str(output_path), format="jpeg", width=800, height=600)
                elif format == "svg":
                    fig.write_image(str(output_path), format="svg")
                elif format == "pdf":
                    fig.write_image(str(output_path), format="pdf", width=800, height=600)
                else:
                    # Default to HTML
                    fig.write_html(str(output_path.with_suffix('.html')))

                exported_files.append(str(output_path))

            except Exception as e:
                print(f"Warning: Failed to export plot {plot_name}: {str(e)}")
                continue

        return exported_files

    def _generate_output_path(self, report: 'AnalysisReport', format: ExportFormat) -> str:
        """Generate output path based on report and format."""
        timestamp = report.analysis_timestamp.replace(" ", "_").replace(":", "-")
        problem_type = report.problem_type.value
        return f"analysis_report_{problem_type}_{timestamp}.{format.value}"

    def _export_json(self, report: 'AnalysisReport', output_path: str, include_plots: bool) -> str:
        """Export report as JSON."""
        # Convert report to dictionary
        report_dict = self._report_to_dict(report, include_plots)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)

        return output_path

    def _export_csv(self, report: 'AnalysisReport', output_path: str) -> str:
        """Export report metrics as CSV."""
        # Collect all metrics from different analyses
        all_metrics = {}

        if report.performance_analysis:
            for key, value in report.performance_analysis.metrics.items():
                all_metrics[f"performance_{key}"] = value

        if report.bias_analysis:
            for key, value in report.bias_analysis.metrics.items():
                all_metrics[f"bias_{key}"] = value

        if report.data_quality_analysis:
            for key, value in report.data_quality_analysis.metrics.items():
                all_metrics[f"quality_{key}"] = value

        if report.explainability_analysis:
            for key, value in report.explainability_analysis.metrics.items():
                if isinstance(value, dict):
                    # Flatten nested dictionaries
                    for subkey, subvalue in value.items():
                        all_metrics[f"explainability_{key}_{subkey}"] = subvalue
                else:
                    all_metrics[f"explainability_{key}"] = value

        # Add summary information
        all_metrics.update({
            'overall_score': report.overall_score,
            'sample_size': report.sample_size,
            'feature_count': report.feature_count,
            'problem_type': report.problem_type.value,
            'execution_time': report.total_execution_time
        })

        # Convert to DataFrame and save
        df = pd.DataFrame([all_metrics])
        df.to_csv(output_path, index=False)

        return output_path

    def _export_html(self, report: 'AnalysisReport', output_path: str, include_plots: bool) -> str:
        """Export report as HTML."""
        html_content = self._generate_html_report(report, include_plots)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _export_pdf(self, report: 'AnalysisReport', output_path: str, include_plots: bool) -> str:
        """Export report as PDF (requires additional dependencies)."""
        try:
            import weasyprint
            from weasyprint import HTML, CSS

            # Generate HTML content
            html_content = self._generate_html_report(report, include_plots)

            # Convert to PDF
            HTML(string=html_content).write_pdf(output_path)

        except ImportError:
            # Fallback: export as HTML with PDF extension
            print("Warning: PDF export requires weasyprint. Exporting as HTML instead.")
            html_path = output_path.replace('.pdf', '.html')
            return self._export_html(report, html_path, include_plots)

        return output_path

    def _export_zip_package(self, report: 'AnalysisReport', output_path: str, include_plots: bool) -> str:
        """Export complete analysis package as ZIP."""
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export main report as JSON
            json_path = temp_path / "analysis_report.json"
            self._export_json(report, str(json_path), include_plots=False)

            # Export metrics as CSV
            csv_path = temp_path / "metrics.csv"
            self._export_csv(report, str(csv_path))

            # Export HTML report
            html_path = temp_path / "report.html"
            self._export_html(report, str(html_path), include_plots=True)

            # Export plots separately
            if include_plots:
                plots_dir = temp_path / "plots"
                plots_dir.mkdir()

                all_plots = {}
                for analysis in [report.performance_analysis, report.bias_analysis,
                               report.data_quality_analysis, report.explainability_analysis]:
                    if analysis and hasattr(analysis, 'plots'):
                        all_plots.update(analysis.plots)

                if all_plots:
                    self.export_plots(all_plots, format="html", output_dir=str(plots_dir))

            # Create ZIP file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_path)
                        zipf.write(file_path, arcname)

        return output_path

    def _export_metrics_json(self, metrics: Dict[str, Any], output_path: str) -> str:
        """Export metrics dictionary as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        return output_path

    def _export_metrics_csv(self, metrics: Dict[str, Any], output_path: str) -> str:
        """Export metrics dictionary as CSV."""
        # Flatten nested dictionaries
        flattened = self._flatten_dict(metrics)
        df = pd.DataFrame([flattened])
        df.to_csv(output_path, index=False)
        return output_path

    def _report_to_dict(self, report: 'AnalysisReport', include_plots: bool) -> Dict[str, Any]:
        """Convert AnalysisReport to dictionary."""
        report_dict = {
            'metadata': {
                'problem_type': report.problem_type.value,
                'sample_size': report.sample_size,
                'feature_count': report.feature_count,
                'overall_score': report.overall_score,
                'analysis_timestamp': report.analysis_timestamp,
                'total_execution_time': report.total_execution_time
            },
            'summary': {
                'key_insights': report.key_insights,
                'priority_recommendations': report.priority_recommendations
            },
            'analyses': {}
        }

        # Add individual analysis results
        for analysis_name, analysis in [
            ('performance', report.performance_analysis),
            ('bias', report.bias_analysis),
            ('data_quality', report.data_quality_analysis),
            ('explainability', report.explainability_analysis)
        ]:
            if analysis:
                analysis_dict = {
                    'analyzer_name': analysis.analyzer_name,
                    'analysis_type': analysis.analysis_type.value,
                    'metrics': analysis.metrics,
                    'insights': analysis.insights,
                    'recommendations': analysis.recommendations,
                    'execution_time': analysis.execution_time,
                    'confidence_score': analysis.confidence_score,
                    'detailed_results': analysis.detailed_results
                }

                if include_plots and hasattr(analysis, 'plots'):
                    # Convert plots to HTML strings for JSON export
                    analysis_dict['plots'] = {}
                    for plot_name, fig in analysis.plots.items():
                        if fig is not None:
                            try:
                                analysis_dict['plots'][plot_name] = fig.to_html()
                            except Exception:
                                analysis_dict['plots'][plot_name] = "Plot export failed"

                report_dict['analyses'][analysis_name] = analysis_dict

        return report_dict

    def _generate_html_report(self, report: 'AnalysisReport', include_plots: bool) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Model Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .insight {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                .recommendation {{ background-color: #d1edff; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }}
                .score {{ font-size: 24px; font-weight: bold; color: #28a745; }}
                ul {{ padding-left: 20px; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Model Analysis Report</h1>
                <p><strong>Problem Type:</strong> {report.problem_type.value}</p>
                <p><strong>Sample Size:</strong> {report.sample_size:,}</p>
                <p><strong>Features:</strong> {report.feature_count}</p>
                <p><strong>Analysis Date:</strong> {report.analysis_timestamp}</p>
                <p><strong>Overall Score:</strong> <span class="score">{report.overall_score:.3f}</span></p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <h3>Key Insights</h3>
                <ul>
        """

        for insight in report.key_insights:
            html += f"<li class='insight'>{insight}</li>"

        html += """
                </ul>
                <h3>Priority Recommendations</h3>
                <ul>
        """

        for rec in report.priority_recommendations:
            html += f"<li class='recommendation'>{rec}</li>"

        html += "</ul></div>"

        # Add detailed analysis sections
        analyses = [
            ('Performance Analysis', report.performance_analysis),
            ('Bias & Fairness Analysis', report.bias_analysis),
            ('Data Quality Analysis', report.data_quality_analysis),
            ('Explainability Analysis', report.explainability_analysis)
        ]

        for section_name, analysis in analyses:
            if analysis:
                html += f"""
                <div class="section">
                    <h2>{section_name}</h2>
                    <p><strong>Analyzer:</strong> {analysis.analyzer_name}</p>
                    <p><strong>Execution Time:</strong> {analysis.execution_time:.2f} seconds</p>
                    <p><strong>Confidence Score:</strong> {analysis.confidence_score:.3f}</p>

                    <h3>Key Metrics</h3>
                """

                for metric, value in analysis.metrics.items():
                    if isinstance(value, (int, float)):
                        html += f"<div class='metric'><strong>{metric}:</strong> {value:.3f}</div>"
                    else:
                        html += f"<div class='metric'><strong>{metric}:</strong> {str(value)}</div>"

                html += "<h3>Insights</h3><ul>"
                for insight in analysis.insights:
                    html += f"<li class='insight'>{insight}</li>"

                html += "</ul><h3>Recommendations</h3><ul>"
                for rec in analysis.recommendations:
                    html += f"<li class='recommendation'>{rec}</li>"

                html += "</ul>"

                # Add plots if requested
                if include_plots and hasattr(analysis, 'plots'):
                    html += "<h3>Visualizations</h3>"
                    for plot_name, fig in analysis.plots.items():
                        if fig is not None:
                            try:
                                plot_html = fig.to_html(include_plotlyjs='inline', div_id=f"plot_{plot_name}")
                                html += f"<div class='plot-container'><h4>{plot_name}</h4>{plot_html}</div>"
                            except Exception:
                                html += f"<p>Plot '{plot_name}' could not be rendered.</p>"

                html += "</div>"

        html += """
            <div class="section">
                <h2>Technical Details</h2>
                <p><strong>Total Execution Time:</strong> {:.2f} seconds</p>
                <p><strong>Report Generated:</strong> {}</p>
            </div>
        </body>
        </html>
        """.format(report.total_execution_time, report.analysis_timestamp)

        return html

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        import re
        # Replace invalid characters with underscores
        return re.sub(r'[<>:"/\\|?*]', '_', filename)