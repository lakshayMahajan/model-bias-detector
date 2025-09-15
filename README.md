# Universal AI Model Prediction Analyzer 🤖⚖️

A comprehensive, dynamic tool for analyzing machine learning model predictions across all domains. Built with a modular architecture, this universal analyzer can handle classification, regression, bias detection, data quality assessment, and explainability analysis.

## 🌟 Key Features

### Universal Analysis Capabilities
- 🔍 **Multi-Problem Support**: Binary/multiclass classification, regression, and complex prediction tasks
- 🤖 **Auto-Detection**: Intelligent detection of problem types and optimal analysis approaches
- 🧩 **Modular Architecture**: Plugin-based analyzer system for extensibility
- 📊 **Comprehensive Metrics**: Performance, bias, data quality, and explainability analysis

### Advanced Analysis Types
- **🎯 Performance Analysis**: ROC curves, confusion matrices, regression metrics, residual analysis
- **⚖️ Bias & Fairness**: Demographic parity, equalized odds, intersectional analysis using Fairlearn
- **🔍 Data Quality**: Missing values, outliers, correlations, distribution analysis
- **💡 Explainability**: Feature importance, partial dependence, local explanations, LIME-style analysis
- **📈 Visualization**: Interactive charts, dashboards, and comprehensive reporting

### Smart Configuration
- 🧠 **AI-Powered Suggestions**: Smart configuration recommendations based on data analysis
- 📋 **Analysis Presets**: Pre-configured templates for common ML scenarios
- 🎛️ **Flexible Setup**: Support for any data format and column structure
- 🔧 **Advanced Options**: Custom thresholds, analysis parameters, and export formats

### Data & Export Support
- 📁 **Multi-Format Input**: CSV, JSON, Parquet, Excel file support
- 💾 **Comprehensive Export**: JSON, CSV, HTML, PDF reports and complete analysis packages
- 🌐 **Web Interface**: Clean, intuitive Streamlit interface with multiple analysis modes
- 🔍 **Real-time Validation**: Robust data validation with clear error handling

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/model-bias-detector.git
cd model-bias-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

### Using the Application

The application provides four main modes:

#### 1. 🚀 Quick Analysis
- Upload your data and get instant analysis recommendations
- Auto-detection of problem type and optimal analyzers
- One-click analysis with smart defaults

#### 2. ⚙️ Advanced Configuration
- Full control over analysis parameters
- Custom analyzer selection and configuration
- Detailed settings for bias thresholds, metrics, and outputs

#### 3. 📋 Analysis Presets
- Pre-configured templates for common scenarios:
  - Comprehensive Classification/Regression Analysis
  - Bias and Fairness Audit
  - Model Performance Monitoring
  - Data Quality Assessment
  - Explainability Analysis
  - Production Readiness Validation

#### 4. ℹ️ About & Help
- Information about available analyzers
- Comprehensive documentation and usage guides

## 🔧 System Architecture

### Core Components

```
Universal AI Model Prediction Analyzer
├── 🏗️ Analyzers/
│   ├── BaseAnalyzer (Abstract base class)
│   ├── ClassificationAnalyzer (Binary/multiclass performance)
│   ├── RegressionAnalyzer (Regression metrics & residuals)
│   ├── BiasAnalyzer (Fairness & bias detection)
│   ├── DataQualityAnalyzer (Data validation & quality)
│   ├── ExplainabilityAnalyzer (Feature importance & explanations)
│   └── AnalysisEngine (Orchestration & auto-detection)
├── 🛠️ Utils/
│   ├── DataProcessor (Universal data loading & preprocessing)
│   ├── VisualizationEngine (Interactive charts & dashboards)
│   ├── ExportManager (Multi-format export & reporting)
│   └── DataValidator (Comprehensive validation framework)
├── ⚙️ Config/
│   ├── SmartConfigInterface (AI-powered configuration)
│   └── AnalysisPresets (Pre-configured templates)
└── 🧪 Tests/
    ├── Unit tests for all components
    ├── Integration tests
    └── Synthetic test datasets
```

### Supported Analysis Types

| Analysis Type | Supported Problems | Key Features |
|--------------|-------------------|--------------|
| **Performance** | All types | ROC/PR curves, confusion matrices, regression metrics, residual analysis |
| **Bias & Fairness** | Classification, Regression | Demographic parity, equalized odds, intersectional analysis |
| **Data Quality** | All types | Missing values, outliers, correlations, distribution analysis |
| **Explainability** | All types | Feature importance, partial dependence, local explanations |

### Problem Type Support

- **Binary Classification**: ROC curves, precision-recall, fairness metrics
- **Multiclass Classification**: Multi-class performance, confusion matrices, class-specific bias
- **Regression**: R², MAE, RMSE, residual analysis, prediction quality
- **Auto-Detection**: Intelligent problem type identification from data

## 📊 Analysis Presets

### Available Templates

1. **Comprehensive Classification Analysis**
   - Complete performance, bias, and data quality analysis
   - Use cases: Hiring algorithms, credit scoring, medical diagnosis

2. **Comprehensive Regression Analysis**
   - Performance metrics, residual analysis, bias detection
   - Use cases: Price prediction, risk assessment, scoring systems

3. **Bias and Fairness Audit**
   - Focused bias analysis with intersectional evaluation
   - Use cases: Algorithmic auditing, compliance verification

4. **Model Performance Monitoring**
   - Performance tracking and degradation detection
   - Use cases: Production monitoring, A/B testing

5. **Data Quality Assessment**
   - Comprehensive data validation and quality checks
   - Use cases: Data pipeline monitoring, preprocessing validation

6. **Explainability Analysis**
   - Feature importance and model interpretability
   - Use cases: Model debugging, regulatory compliance

7. **Quick Model Assessment**
   - Fast, high-level model evaluation
   - Use cases: Prototype testing, initial evaluation

8. **Research and Development Analysis**
   - Comprehensive analysis with experimental features
   - Use cases: Academic research, method development

9. **Production Readiness Validation**
   - Complete validation for production deployment
   - Use cases: Pre-deployment checks, risk assessment

## 🔍 Data Requirements

### Supported File Formats
- CSV, JSON, Parquet, Excel files
- Automatic format detection
- Flexible column naming and structure

### Required Data Structure
```csv
# Minimum requirements (auto-detected):
prediction_col,true_label_col[,protected_attr_col,feature_cols...]
1,1,Male,25.5
0,0,Female,30.2
1,0,Male,22.1
```

### Column Types (Auto-Detected)
- **Predictions**: Binary, categorical, or continuous values
- **True Labels**: Ground truth values (optional for some analyses)
- **Protected Attributes**: Categorical sensitive attributes (for bias analysis)
- **Features**: Any additional columns (for explainability analysis)
- **Probabilities**: Prediction confidence scores (optional)

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Web interface)
- **Data Processing**: Pandas, NumPy
- **ML Analysis**: Scikit-learn, Fairlearn
- **Visualizations**: Plotly (Interactive charts)
- **Export**: ReportLab, FPDF (PDF generation)
- **Testing**: Pytest (Comprehensive test suite)

## 🔬 Advanced Features

### Smart Configuration
- AI-powered analysis recommendations
- Confidence scoring for suggestions
- Context-aware parameter tuning

### Explainability Analysis
- Permutation importance
- Partial dependence plots
- Local explanations (LIME-style)
- Feature interaction detection
- Model complexity assessment

### Bias Analysis
- Multiple fairness criteria
- Intersectional bias detection
- Group-specific performance metrics
- Statistical significance testing

### Export & Reporting
- Multi-format exports (JSON, CSV, HTML, PDF)
- Complete analysis packages
- Interactive visualizations
- Executive summaries

## 🧪 Testing & Validation

The system includes comprehensive testing:
- **75 Unit Tests** covering all components
- **Integration Tests** for end-to-end workflows
- **Synthetic Datasets** for reliable testing
- **Error Handling** for robust operation

Run tests with:
```bash
python -m pytest tests/ -v
```

## 🤝 Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add new analyzer types
- Extend analysis capabilities
- Improve visualization options
- Add export formats

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📈 Use Cases

### Industry Applications
- **Healthcare**: Medical diagnosis bias detection
- **Finance**: Credit scoring fairness analysis
- **HR**: Hiring algorithm audit
- **Criminal Justice**: Risk assessment validation
- **Marketing**: Customer segmentation analysis

### Analysis Scenarios
- **Model Validation**: Pre-deployment comprehensive analysis
- **Bias Auditing**: Systematic fairness evaluation
- **Performance Monitoring**: Ongoing model health checks
- **Research**: Academic ML research and development
- **Compliance**: Regulatory requirement validation

## 📚 Documentation

### Analysis Types Explained

**Performance Analysis**
- Classification: Accuracy, precision, recall, F1-score, ROC-AUC
- Regression: R², MAE, RMSE, residual analysis
- Advanced: Learning curves, validation curves

**Bias & Fairness Analysis**
- Demographic Parity: Equal positive prediction rates
- Equalized Odds: Equal TPR and FPR across groups
- Intersectional: Multi-attribute bias analysis

**Data Quality Analysis**
- Missing value patterns and impact
- Outlier detection and assessment
- Feature correlation analysis
- Distribution comparisons

**Explainability Analysis**
- Global: Overall feature importance
- Local: Instance-specific explanations
- Interactions: Feature relationship analysis

## 🔒 Privacy & Ethics

- **Data Privacy**: All analysis runs locally, no data transmitted
- **Bias Detection**: Proactive identification of unfair outcomes
- **Interpretability**: Clear explanations for all metrics
- **Best Practices**: Guidance for responsible AI development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: Web application framework
- **Fairlearn**: Bias and fairness metrics
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities
- **Open Source Community**: For inspiring this universal approach

## 🆘 Support

If you encounter issues or have questions:
1. Check the [Issues](https://github.com/yourusername/model-bias-detector/issues) page
2. Review the built-in help documentation
3. Create detailed issue reports with sample data
4. Join our community discussions

---

**⚠️ Important Note**: This tool aids in identifying potential bias and quality issues but should be part of a comprehensive fairness strategy. Always consult domain experts and consider the broader societal context of your ML systems.

**🎯 Mission**: Democratizing access to comprehensive AI model analysis, making it easier for everyone to build fair, reliable, and explainable machine learning systems.