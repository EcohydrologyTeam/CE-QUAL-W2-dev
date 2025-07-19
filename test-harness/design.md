CE-QUAL-W2 Test Harness Architecture Design

  Based on my analysis of the codebase, here's a comprehensive design for a test harness to compare multiple CE-QUAL-W2 model versions:

  System Architecture Overview

  ┌─────────────────────────────────────────────────────────┐
  │                Test Harness Controller                  │
  │  - Version Management                                   │
  │  - Test Suite Orchestration                             │
  │  - Report Generation Coordination                       │
  └─────────────────────────────────────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
  ┌─────────────┐    ┌───────────────────┐    ┌─────────────────┐
  │   Model     │    │   Comparison      │    │   Reporting     │
  │  Runner     │    │    Engine         │    │    System       │
  │             │    │                   │    │                 │
  │ - Execute   │    │ - File Parsing    │    │ - Plot Gen      │
  │   Models    │    │ - Data Comparison │    │ - Tables        │
  │ - Manage    │    │ - Metrics Calc    │    │ - HTML Reports  │
  │   I/O       │    │ - Regression      │    │ - Summaries     │
  │ - Version   │    │   Detection       │    │ - Dashboards    │
  │   Control   │    │                   │    │                 │
  └─────────────┘    └───────────────────┘    └─────────────────┘

  Core Components

  1. Configuration Management

  # test_config.yaml structure
  test_suite:
    name: "CE-QUAL-W2 Version Comparison"
    versions:
      - id: "v4.5"
        executable: "CE-QUAL-W2/v4.5/executables/model/w2_v45_64.exe"
        baseline: true
      - id: "v5.0"
        executable: "CE-QUAL-W2/v5.0/executables/w2_v5beta.exe"

    test_cases:
      - name: "Long Lake"
        input_dir: "examples/Long Lake"
        control_file: "w2_con.csv"
        runtime_hours: 0.5

    comparison_settings:
      tolerance:
        relative: 0.01  # 1% relative difference
        absolute: 1e-6  # Absolute tolerance for near-zero values
      critical_files:
        - "*.opt"  # Time series outputs
        - "*.csv"  # Balance and summary files
      ignore_patterns:
        - "*log*"
        - "*.err"
        - "*.wrn"

  2. Model Execution Engine

  - Parallel Execution: Run multiple versions simultaneously
  - Isolated Environments: Separate working directories per version/test
  - Resource Management: Memory and CPU monitoring
  - Error Handling: Capture and categorize model failures
  - Performance Metrics: Track execution time and resource usage

  3. Data Comparison Engine

  File Type Handlers:
  - .opt Files: Time-series data with Julian day timestamps
  - .csv Files: Tabular data (mass balance, flow balance, etc.)
  - .out Files: Log and diagnostic data
  - Binary Files: Handle specialized formats where applicable

  Comparison Algorithms:
  - Statistical Metrics: RMSE, MAE, correlation coefficients
  - Temporal Analysis: Time-series alignment and difference analysis
  - Threshold Detection: Configurable tolerance levels
  - Pattern Recognition: Identify systematic vs random differences

  4. Regression Detection System

  Multi-level Analysis:
  Level 1: Critical Failures
  - Model crashes/non-convergence
  - Missing output files
  - Corrupted data

  Level 2: Significant Changes
  - Results outside tolerance bands
  - Mass balance violations
  - Energy conservation issues

  Level 3: Minor Variations
  - Small numerical differences
  - Timing variations
  - Precision differences

  5. Reporting Framework

  Report Types:
  - Executive Summary: High-level pass/fail status
  - Detailed Comparison: File-by-file analysis with statistics
  - Regression Report: Specific issues identified with severity
  - Performance Report: Runtime and resource comparison
  - Visual Dashboard: Interactive plots and charts

  Output Formats:
  - HTML Dashboard: Interactive web-based reports
  - PDF Reports: Printable detailed analysis
  - JSON/CSV: Machine-readable results for CI/CD
  - Plots: Time-series comparisons, difference plots, statistical summaries

  Implementation Strategy

  Phase 1: Core Infrastructure (2-3 weeks)

  1. Project Setup: Directory structure, configuration management
  2. Model Runner: Execute different versions with proper isolation
  3. Basic File Comparison: Simple diff capabilities for .opt and .csv files
  4. Basic Reporting: Text-based summary reports

  Phase 2: Advanced Comparison (2-3 weeks)

  1. Statistical Analysis: Implement robust comparison metrics
  2. File Format Parsers: Handle all CE-QUAL-W2 output formats
  3. Regression Detection: Automated threshold-based detection
  4. Plot Generation: Basic time-series and difference plots

  Phase 3: Comprehensive Reporting (2-3 weeks)

  1. HTML Dashboard: Interactive web-based interface
  2. Advanced Visualizations: Heat maps, statistical distributions
  3. Performance Analysis: Resource usage and timing comparisons
  4. Configuration Templates: Easy setup for new test cases

  Phase 4: Production Features (1-2 weeks)

  1. CI/CD Integration: Automated testing capabilities
  2. Parallel Processing: Optimize for multiple cores/machines
  3. Database Backend: Store historical results for trend analysis
  4. Email/Slack Notifications: Alert system for critical failures

  Directory Structure

  ce_qual_w2_test_harness/
  ├── config/
  │   ├── test_suites/          # Different test configurations
  │   └── tolerance_profiles/   # Tolerance settings per application
  ├── src/
  │   ├── core/                 # Main test harness logic
  │   ├── runners/              # Model execution handlers
  │   ├── comparisons/          # Data comparison algorithms
  │   ├── reporting/            # Report generation
  │   └── utils/                # Utilities and helpers
  ├── tests/                    # Unit tests for the harness
  ├── templates/                # Report templates
  ├── results/                  # Test run results organized by date/version
  └── docs/                     # Documentation and user guides

  Technology Stack Recommendations

  - Python 3.9+: Main development language
  - pandas/numpy: Data manipulation and analysis
  - matplotlib/plotly: Visualization and plotting
  - Jinja2: HTML report templating
  - pytest: Testing framework
  - PyYAML: Configuration file handling
  - multiprocessing: Parallel execution
  - psutil: System resource monitoring

  Key Benefits

  1. Automated Regression Detection: Catch breaking changes early
  2. Comprehensive Coverage: Test all 8 model applications across versions
  3. Scalable Architecture: Easy to add new versions and test cases
  4. Rich Reporting: Clear visualization of differences and trends
  5. CI/CD Ready: Integration with automated build systems
  6. Performance Monitoring: Track computational efficiency changes

  This architecture provides a robust foundation for systematically comparing CE-QUAL-W2 versions while being flexible enough to
  accommodate future model versions and applications.

⏺ Update Todos
  ⎿  ☒ Explore CE-QUAL-W2 codebase structure to understand existing model versions and output formats
     ☒ Analyze existing model outputs and data formats for comparison
     ☒ Design test harness architecture for multi-version comparison
     ☐ Implement core comparison engine
     ☐ Develop reporting system (plots, tables, summaries)
     ☐ Create regression detection algorithms
     ☐ Build automated test suite runner


The design addresses:

  1. Multi-version comparison (8 model applications across versions)
  2. Regression detection (detecting broken functionality or significant changes)
  3. Comprehensive reporting (plots, tables, reports)
  4. Scalable architecture with clear phases
  5. Technology recommendations and implementation strategy

  The design covers:
  - System architecture with clear component separation
  - Configuration management for flexible test setup
  - Multi-level regression detection (critical failures, significant changes, minor variations)
  - Rich reporting framework with multiple output formats
  - Phased implementation strategy with realistic timelines
  - Technology stack recommendations

  Key Features of the Design:
  - Multi-version Support: Handles v4.5, v5.0, and future versions
  - Automated Regression Detection: 3-level analysis (critical failures, significant changes, minor variations)
  - Rich Reporting: HTML dashboards, plots, tables, and PDF reports
  - Scalable Architecture: Easy to add new test cases and versions
  - Performance Monitoring: Track computational efficiency changes

  Timeline Estimate: 8-11 weeks total (matching your intuitive estimates)
  - Phase 1: Core Infrastructure (2-3 weeks)
  - Phase 2: Advanced Comparison (2-3 weeks)
  - Phase 3: Comprehensive Reporting (2-3 weeks)
  - Phase 4: Production Features (1-2 weeks)

  The system will automatically run all 8 model applications with different versions, compare outputs (.opt, .csv, .out files), detect
  regressions using configurable tolerances, and generate comprehensive reports to help you evaluate whether changes broke
  functionality or significantly altered outputs.
