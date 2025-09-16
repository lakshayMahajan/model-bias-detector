"""
Universal AI Model Prediction Analyzer
Streamlit Web Application

A comprehensive tool for analyzing ML model predictions across all domains.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Optional, List, Dict, Any

# Import our analyzer modules
from analyzers.analysis_engine import AnalysisEngine, AnalysisReport
from analyzers.base_analyzer import ProblemType, AnalysisType
from utils.data_processor import DataProcessor, FileFormat
from utils.export_manager import ExportManager, ExportFormat
from utils.data_validator import DataValidator
from config.smart_config import SmartConfigInterface


# Page configuration
st.set_page_config(
    page_title="Universal AI Model Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #d1edff;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_config' not in st.session_state:
    st.session_state.data_config = None


def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header">🤖 Universal AI Model Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive analysis for all AI model predictions**")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode:",
        ["🚀 Quick Analysis", "⚙️ Advanced Configuration", "ℹ️ About"],
        help="Choose your analysis approach"
    )

    if mode == "🚀 Quick Analysis":
        quick_analysis_mode()
    elif mode == "⚙️ Advanced Configuration":
        advanced_configuration_mode()
    elif mode == "ℹ️ About":
        about_mode()


def quick_analysis_mode():
    """Quick analysis mode with minimal configuration."""

    st.markdown('<h2 class="section-header">🚀 Quick Analysis</h2>', unsafe_allow_html=True)
    st.write("Upload your data and get instant comprehensive analysis with smart auto-detection.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'json', 'xlsx', 'parquet'],
        help="Supported formats: CSV, JSON, Excel, Parquet"
    )

    if uploaded_file is not None:
        try:
            # Load data
            data_processor = DataProcessor()

            # Detect file format
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                file_format = FileFormat.CSV
            elif file_extension == 'json':
                file_format = FileFormat.JSON
            elif file_extension in ['xlsx', 'xls']:
                file_format = FileFormat.EXCEL
            elif file_extension == 'parquet':
                file_format = FileFormat.PARQUET
            else:
                file_format = FileFormat.CSV

            # Load the data
            df = data_processor.load_data(
                file_content=uploaded_file.getvalue(),
                file_format=file_format
            )

            st.session_state.uploaded_data = df

            # Display data preview
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

            # Smart configuration
            smart_config = SmartConfigInterface()

            with st.spinner("🧠 Analyzing data structure and suggesting configuration..."):
                try:
                    config, explanations, confidence_scores = smart_config.analyze_and_suggest_config(df)
                    st.session_state.data_config = config

                    # Display configuration suggestions
                    st.markdown('<h3 class="section-header">🤖 Smart Configuration</h3>', unsafe_allow_html=True)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write("**Configuration Suggestions:**")
                        for explanation in explanations:
                            st.write(f"• {explanation}")

                    with col2:
                        st.metric("Overall Confidence", f"{confidence_scores.get('overall', 0):.0%}")

                    # Column assignments
                    with st.expander("🎯 Column Assignments", expanded=False):
                        if config.data_config.prediction_column:
                            st.write(f"**Predictions:** {config.data_config.prediction_column}")
                        if config.data_config.true_value_column:
                            st.write(f"**True Values:** {config.data_config.true_value_column}")
                        if config.data_config.protected_attributes:
                            st.write(f"**Protected Attributes:** {', '.join(config.data_config.protected_attributes)}")
                        if config.data_config.feature_columns:
                            st.write(f"**Features:** {len(config.data_config.feature_columns)} columns")

                    # Run analysis button
                    if st.button("🔍 Run Comprehensive Analysis", type="primary"):
                        run_analysis(df, config)

                except Exception as e:
                    st.error(f"Configuration analysis failed: {str(e)}")
                    st.write("Please try the Advanced Configuration mode for manual setup.")

        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            st.write("Please check your file format and try again.")


def advanced_configuration_mode():
    """Advanced configuration mode with manual controls."""

    st.markdown('<h2 class="section-header">⚙️ Advanced Configuration</h2>', unsafe_allow_html=True)
    st.write("Fine-tune your analysis with detailed configuration options.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'json', 'xlsx', 'parquet'],
        help="Supported formats: CSV, JSON, Excel, Parquet",
        key="advanced_upload"
    )

    if uploaded_file is not None:
        try:
            # Load data
            data_processor = DataProcessor()

            # Detect file format
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                file_format = FileFormat.CSV
            elif file_extension == 'json':
                file_format = FileFormat.JSON
            elif file_extension in ['xlsx', 'xls']:
                file_format = FileFormat.EXCEL
            elif file_extension == 'parquet':
                file_format = FileFormat.PARQUET
            else:
                file_format = FileFormat.CSV

            # Load the data
            df = data_processor.load_data(
                file_content=uploaded_file.getvalue(),
                file_format=file_format
            )

            st.session_state.uploaded_data = df

            # Data preview
            st.success(f"✅ Data loaded! Shape: {df.shape}")

            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)

            # Manual configuration
            st.markdown('<h3 class="section-header">🎛️ Manual Configuration</h3>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Core Columns")

                prediction_column = st.selectbox(
                    "Prediction Column",
                    options=[None] + list(df.columns),
                    help="Column containing model predictions"
                )

                true_value_column = st.selectbox(
                    "True Values Column (optional)",
                    options=[None] + list(df.columns),
                    help="Column containing ground truth values"
                )

                probability_columns = st.multiselect(
                    "Probability Columns (optional)",
                    options=list(df.columns),
                    help="Columns containing prediction probabilities"
                )

            with col2:
                st.subheader("Analysis Settings")

                problem_type = st.selectbox(
                    "Problem Type",
                    options=["Auto-detect"] + [pt.value for pt in ProblemType if pt != ProblemType.AUTO_DETECT],
                    help="Type of machine learning problem"
                )

                protected_attributes = st.multiselect(
                    "Protected Attributes",
                    options=list(df.columns),
                    help="Columns for bias analysis (gender, race, age, etc.)"
                )

                analysis_types = st.multiselect(
                    "Analysis Types",
                    options=[at.value for at in AnalysisType],
                    default=[at.value for at in AnalysisType],
                    help="Types of analysis to perform"
                )

            # Advanced options
            with st.expander("🔧 Advanced Options"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    include_plots = st.checkbox("Include Visualizations", value=True)

                with col2:
                    export_results = st.checkbox("Export Results", value=False)

                with col3:
                    if export_results:
                        export_format = st.selectbox(
                            "Export Format",
                            options=[fmt.value for fmt in ExportFormat]
                        )

            # Validate and run analysis
            if prediction_column:
                if st.button("🔍 Run Analysis", type="primary"):
                    # Create configuration
                    from utils.data_processor import DataConfig

                    data_config = DataConfig(
                        prediction_column=prediction_column,
                        true_value_column=true_value_column,
                        probability_columns=probability_columns,
                        protected_attributes=protected_attributes
                    )

                    # Convert problem type
                    if problem_type == "Auto-detect":
                        selected_problem_type = None
                    else:
                        selected_problem_type = ProblemType(problem_type)

                    # Convert analysis types
                    selected_analysis_types = [AnalysisType(at) for at in analysis_types]

                    # Create full configuration
                    from config.smart_config import AnalysisConfiguration

                    config = AnalysisConfiguration(
                        data_config=data_config,
                        problem_type=selected_problem_type or ProblemType.AUTO_DETECT,
                        analysis_types=selected_analysis_types,
                        include_plots=include_plots,
                        export_results=export_results
                    )

                    # Run analysis
                    run_analysis(df, config)
            else:
                st.warning("⚠️ Please select a prediction column to proceed.")

        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")


def run_analysis(df: pd.DataFrame, config):
    """Run the comprehensive analysis."""

    st.markdown('<h2 class="section-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)

    # Validate data first
    with st.spinner("🔍 Validating data..."):
        validator = DataValidator()

        # Extract predictions and true values
        predictions = None
        true_values = None

        if config.data_config.prediction_column and config.data_config.prediction_column in df.columns:
            predictions = df[config.data_config.prediction_column].values

        if config.data_config.true_value_column and config.data_config.true_value_column in df.columns:
            true_values = df[config.data_config.true_value_column].values

        is_valid, validation_issues = validator.validate_analysis_inputs(
            df, predictions, true_values, config.data_config.protected_attributes
        )

        if validation_issues:
            st.warning("⚠️ Data validation found some issues:")
            for issue in validator.format_issues_for_display(validation_issues):
                st.write(f"• {issue}")

        if not is_valid:
            st.error("❌ Critical validation errors found. Please fix these issues before proceeding.")
            return

    # Run analysis
    with st.spinner("🧮 Running comprehensive analysis..."):
        try:
            engine = AnalysisEngine()

            # Prepare analysis parameters
            analysis_params = {
                'data': df,
                'predictions': predictions,
                'true_values': true_values,
                'problem_type': config.problem_type if config.problem_type != ProblemType.AUTO_DETECT else None,
                'analysis_types': config.analysis_types,
                'protected_attributes': config.data_config.protected_attributes
            }

            # Add probability scores if available
            if config.data_config.probability_columns:
                prob_cols = [col for col in config.data_config.probability_columns if col in df.columns]
                if prob_cols:
                    analysis_params['probability_scores'] = df[prob_cols[0]].values

            # Run analysis
            report = engine.analyze_model(**analysis_params)

            st.session_state.analysis_results = report

            # Display results
            display_analysis_results(report, config)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.write("Please check your data and configuration.")


def display_analysis_results(report: AnalysisReport, config):
    """Display the analysis results."""

    # Executive Summary
    st.markdown('<h3 class="section-header">📋 Executive Summary</h3>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Score", f"{report.overall_score:.3f}")

    with col2:
        st.metric("Problem Type", report.problem_type.value.replace('_', ' ').title())

    with col3:
        st.metric("Sample Size", f"{report.sample_size:,}")

    with col4:
        st.metric("Execution Time", f"{report.total_execution_time:.1f}s")

    # Key Insights
    if report.key_insights:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.write("**🔍 Key Insights:**")
        for insight in report.key_insights:
            st.write(f"• {insight}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Priority Recommendations
    if report.priority_recommendations:
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.write("**💡 Priority Recommendations:**")
        for rec in report.priority_recommendations:
            st.write(f"• {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Analysis Results
    st.markdown('<h3 class="section-header">📊 Detailed Analysis</h3>', unsafe_allow_html=True)

    # Create tabs for different analysis types
    analysis_tabs = []
    analysis_data = []

    if report.performance_analysis:
        analysis_tabs.append("🎯 Performance")
        analysis_data.append(report.performance_analysis)

    if report.bias_analysis:
        analysis_tabs.append("⚖️ Bias & Fairness")
        analysis_data.append(report.bias_analysis)

    if report.data_quality_analysis:
        analysis_tabs.append("🔍 Data Quality")
        analysis_data.append(report.data_quality_analysis)

    if report.explainability_analysis:
        analysis_tabs.append("🧠 Explainability")
        analysis_data.append(report.explainability_analysis)

    if analysis_tabs:
        tabs = st.tabs(analysis_tabs)

        for tab, analysis_result in zip(tabs, analysis_data):
            with tab:
                display_analysis_section(analysis_result, config.include_plots)

    # Export Results
    if config.export_results:
        st.markdown('<h3 class="section-header">📤 Export Results</h3>', unsafe_allow_html=True)

        export_manager = ExportManager()

        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=[fmt.value for fmt in ExportFormat],
                key="export_format_select"
            )

        with col2:
            if st.button("📥 Download Results"):
                try:
                    output_path = export_manager.export_analysis_report(
                        report,
                        ExportFormat(export_format),
                        include_plots=config.include_plots
                    )

                    # Read and provide download
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label=f"Download {export_format.upper()} Report",
                            data=f.read(),
                            file_name=output_path.split('/')[-1],
                            mime=f"application/{export_format}"
                        )

                except Exception as e:
                    st.error(f"Export failed: {str(e)}")


def display_analysis_section(analysis_result, include_plots: bool):
    """Display a single analysis section."""

    # Metrics
    if analysis_result.metrics:
        st.subheader("📊 Key Metrics")

        # Display metrics in columns
        metrics_items = list(analysis_result.metrics.items())

        # Filter out complex metrics for main display
        simple_metrics = [(k, v) for k, v in metrics_items if isinstance(v, (int, float))]

        if simple_metrics:
            cols = st.columns(min(3, len(simple_metrics)))

            for i, (metric, value) in enumerate(simple_metrics[:9]):  # Limit to 9 metrics
                with cols[i % 3]:
                    if isinstance(value, float):
                        st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
                    else:
                        st.metric(metric.replace('_', ' ').title(), f"{value}")

        # Show all metrics in expander
        with st.expander("All Metrics"):
            for metric, value in metrics_items:
                if isinstance(value, dict):
                    st.write(f"**{metric.replace('_', ' ').title()}:**")
                    st.json(value)
                else:
                    st.write(f"**{metric.replace('_', ' ').title()}:** {value}")

    # Insights
    if analysis_result.insights:
        st.subheader("💡 Insights")
        for insight in analysis_result.insights:
            st.write(f"• {insight}")

    # Recommendations
    if analysis_result.recommendations:
        st.subheader("🎯 Recommendations")
        for rec in analysis_result.recommendations:
            st.write(f"• {rec}")

    # Plots
    if include_plots and hasattr(analysis_result, 'plots') and analysis_result.plots:
        st.subheader("📈 Visualizations")

        for plot_name, fig in analysis_result.plots.items():
            if fig is not None:
                st.write(f"**{plot_name.replace('_', ' ').title()}**")
                st.plotly_chart(fig, use_container_width=True)

    # Technical Details
    if analysis_result.detailed_results:
        with st.expander("🔧 Technical Details"):
            st.json(analysis_result.detailed_results)


def about_mode():
    """About page with tool information."""

    st.markdown('<h2 class="section-header">ℹ️ About Universal AI Model Analyzer</h2>', unsafe_allow_html=True)

    st.write("""
    **Universal AI Model Analyzer** is a comprehensive tool for analyzing machine learning model predictions
    across all domains and problem types.

    ### 🎯 Key Features

    - **Universal Analysis**: Works with any ML model predictions (classification, regression, NLP, computer vision)
    - **Automated Detection**: Smart column detection and problem type identification
    - **Comprehensive Metrics**: Performance, bias, data quality, and explainability analysis
    - **Interactive Visualizations**: Rich plots and charts for insights
    - **Export Capabilities**: Multiple format support (JSON, CSV, HTML, PDF, ZIP)
    - **Bias Detection**: Advanced fairness analysis across protected attributes

    ### 🔧 Supported Analysis Types

    #### 🎯 Performance Analysis
    - Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Regression metrics (R², RMSE, MAE, residual analysis)
    - Advanced statistical tests and diagnostics

    #### ⚖️ Bias & Fairness Analysis
    - Demographic parity and equalized odds
    - Intersectional bias detection
    - Disparate impact analysis
    - Group performance comparisons

    #### 🔍 Data Quality Analysis
    - Missing data analysis
    - Outlier detection
    - Data consistency checks
    - Feature correlation analysis

    #### 🧠 Explainability Analysis
    - Feature importance analysis
    - Permutation importance
    - Local explanations (LIME-style)
    - Feature interaction detection

    ### 📊 Supported Data Formats

    - CSV files
    - JSON files
    - Excel files (.xlsx, .xls)
    - Parquet files
    - TSV files

    ### 🚀 Getting Started

    1. **Quick Analysis**: Upload your data and get instant analysis with auto-detection
    2. **Advanced Configuration**: Fine-tune settings for specialized analysis
    3. **Review Results**: Explore comprehensive insights and recommendations
    4. **Export**: Download results in your preferred format

    ### 💡 Tips for Best Results

    - Ensure your data has clear column names
    - Include both predictions and true values when available
    - Specify protected attributes for bias analysis
    - Use meaningful feature names for better explainability

    ### 🏗️ Architecture

    Built with modern, modular architecture:
    - **Analyzer Modules**: Specialized analyzers for different analysis types
    - **Smart Configuration**: AI-powered automatic setup
    - **Universal Data Processor**: Multi-format data loading and preprocessing
    - **Visualization Engine**: Interactive charts and dashboards
    - **Export Manager**: Multi-format result export

    ---

    **Built for data scientists, ML engineers, and researchers who need comprehensive model analysis.**
    """)

    # Technical specifications
    with st.expander("🔧 Technical Specifications"):
        st.write("""
        **Supported Problem Types:**
        - Binary Classification
        - Multiclass Classification
        - Regression
        - Auto-detection for unknown types

        **Analysis Engines:**
        - Classification Analyzer (scikit-learn based)
        - Regression Analyzer (statistical analysis)
        - Bias Analyzer (fairness metrics)
        - Data Quality Analyzer (comprehensive validation)
        - Explainability Analyzer (feature importance)

        **Visualization:**
        - Plotly-based interactive charts
        - Confusion matrices, ROC curves, residual plots
        - Distribution analysis, correlation heatmaps
        - Bias comparison charts

        **Export Formats:**
        - JSON (structured data)
        - CSV (metrics and data)
        - HTML (full interactive report)
        - PDF (static report)
        - ZIP (complete package)
        """)


if __name__ == "__main__":
    main()