#!/usr/bin/env python3
"""
CE-QUAL-W2 Test Harness Backend
Comprehensive model version comparison and regression testing system
"""

import os
import sys
import yaml
import json
import logging
import pandas as pd
import numpy as np
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import hashlib
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Represents a CE-QUAL-W2 model version"""
    id: str
    executable: str
    name: str
    baseline: bool = False
    
@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    input_dir: str
    control_file: str
    runtime_hours: float
    expected_files: List[str] = None
    
@dataclass
class ComparisonResult:
    """Results from comparing two model runs"""
    test_case: str
    baseline_version: str
    test_version: str
    status: str  # PASS, WARN, FAIL
    rmse: float
    mae: float
    correlation: float
    max_difference_pct: float
    runtime_baseline: float
    runtime_test: float
    memory_baseline: int
    memory_test: int
    issues: List[str]
    timestamp: datetime
    
@dataclass
class TestSuiteConfig:
    """Configuration for a test suite"""
    name: str
    versions: List[ModelVersion]
    test_cases: List[TestCase]
    tolerance_relative: float = 0.01
    tolerance_absolute: float = 1e-6
    critical_files: List[str] = None
    ignore_patterns: List[str] = None

class FileComparator:
    """Handles comparison of different file types"""
    
    def __init__(self, tolerance_rel: float = 0.01, tolerance_abs: float = 1e-6):
        self.tolerance_rel = tolerance_rel
        self.tolerance_abs = tolerance_abs
        
    def compare_opt_files(self, file1: Path, file2: Path) -> Dict[str, float]:
        """Compare CE-QUAL-W2 .opt time series files"""
        try:
            # Read time series data - assuming space-separated format
            df1 = pd.read_csv(file1, sep=r'\s+', header=0)
            df2 = pd.read_csv(file2, sep=r'\s+', header=0)
            
            # Align by time column (assuming first column is time)
            time_col = df1.columns[0]
            merged = pd.merge(df1, df2, on=time_col, suffixes=('_1', '_2'))
            
            # Calculate statistics for numerical columns
            stats = {}
            for col in df1.columns[1:]:  # Skip time column
                if col in df2.columns:
                    col1, col2 = f"{col}_1", f"{col}_2"
                    if col1 in merged.columns and col2 in merged.columns:
                        vals1, vals2 = merged[col1], merged[col2]
                        
                        # Remove NaN values
                        mask = ~(np.isnan(vals1) | np.isnan(vals2))
                        if mask.sum() > 0:
                            vals1, vals2 = vals1[mask], vals2[mask]
                            
                            rmse = np.sqrt(np.mean((vals1 - vals2) ** 2))
                            mae = np.mean(np.abs(vals1 - vals2))
                            corr = np.corrcoef(vals1, vals2)[0, 1] if len(vals1) > 1 else 1.0
                            max_diff = np.max(np.abs((vals1 - vals2) / np.maximum(np.abs(vals1), self.tolerance_abs))) * 100
                            
                            stats[col] = {
                                'rmse': rmse,
                                'mae': mae,
                                'correlation': corr,
                                'max_diff_pct': max_diff
                            }
            
            # Aggregate statistics
            if stats:
                rmse_avg = np.mean([s['rmse'] for s in stats.values()])
                mae_avg = np.mean([s['mae'] for s in stats.values()])
                corr_avg = np.mean([s['correlation'] for s in stats.values() if not np.isnan(s['correlation'])])
                max_diff = np.max([s['max_diff_pct'] for s in stats.values()])
                
                return {
                    'rmse': rmse_avg,
                    'mae': mae_avg,
                    'correlation': corr_avg,
                    'max_diff_pct': max_diff,
                    'details': stats
                }
                
        except Exception as e:
            logger.warning(f"Error comparing {file1} and {file2}: {e}")
            
        return {'rmse': float('inf'), 'mae': float('inf'), 'correlation': 0.0, 'max_diff_pct': 100.0}
    
    def compare_csv_files(self, file1: Path, file2: Path) -> Dict[str, float]:
        """Compare CSV files (balance files, etc.)"""
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            
            # Compare common columns
            common_cols = set(df1.columns) & set(df2.columns)
            numeric_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(df1[col])]
            
            if not numeric_cols:
                return {'rmse': 0.0, 'mae': 0.0, 'correlation': 1.0, 'max_diff_pct': 0.0}
            
            stats = []
            for col in numeric_cols:
                vals1, vals2 = df1[col].dropna(), df2[col].dropna()
                if len(vals1) == len(vals2) and len(vals1) > 0:
                    rmse = np.sqrt(np.mean((vals1 - vals2) ** 2))
                    mae = np.mean(np.abs(vals1 - vals2))
                    corr = np.corrcoef(vals1, vals2)[0, 1] if len(vals1) > 1 else 1.0
                    max_diff = np.max(np.abs((vals1 - vals2) / np.maximum(np.abs(vals1), self.tolerance_abs))) * 100
                    
                    stats.append({
                        'rmse': rmse,
                        'mae': mae,
                        'correlation': corr,
                        'max_diff_pct': max_diff
                    })
            
            if stats:
                return {
                    'rmse': np.mean([s['rmse'] for s in stats]),
                    'mae': np.mean([s['mae'] for s in stats]),
                    'correlation': np.mean([s['correlation'] for s in stats if not np.isnan(s['correlation'])]),
                    'max_diff_pct': np.max([s['max_diff_pct'] for s in stats])
                }
                
        except Exception as e:
            logger.warning(f"Error comparing CSV files {file1} and {file2}: {e}")
            
        return {'rmse': float('inf'), 'mae': float('inf'), 'correlation': 0.0, 'max_diff_pct': 100.0}

class ModelRunner:
    """Handles execution of CE-QUAL-W2 model versions"""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
    def run_model(self, version: ModelVersion, test_case: TestCase, run_id: str) -> Dict[str, Any]:
        """Execute a single model run"""
        start_time = datetime.now()
        
        # Create isolated run directory
        run_dir = self.work_dir / f"{run_id}_{version.id}_{test_case.name.replace(' ', '_')}"
        run_dir.mkdir(exist_ok=True)
        
        try:
            # Copy input files
            input_path = Path(test_case.input_dir)
            if input_path.exists():
                for file in input_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, run_dir / file.name)
            
            # Execute model
            executable = Path(version.executable)
            if not executable.exists():
                raise FileNotFoundError(f"Executable not found: {executable}")
            
            # Monitor system resources
            process = psutil.Popen([str(executable)], cwd=run_dir, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            max_memory = 0
            try:
                while process.is_running():
                    try:
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        max_memory = max(max_memory, memory_mb)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                
                # Wait for completion with timeout
                timeout_seconds = test_case.runtime_hours * 3600 * 2  # 2x expected runtime
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                
            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError("Model execution timed out")
            
            end_time = datetime.now()
            runtime_minutes = (end_time - start_time).total_seconds() / 60
            
            # Check for successful completion
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Model execution failed: {error_msg}")
            
            # Verify expected output files exist
            output_files = list(run_dir.glob("*.opt")) + list(run_dir.glob("*.csv"))
            if not output_files:
                raise RuntimeError("No output files generated")
            
            return {
                'status': 'success',
                'runtime_minutes': runtime_minutes,
                'max_memory_mb': int(max_memory),
                'output_dir': run_dir,
                'output_files': [f.name for f in output_files],
                'stdout': stdout.decode() if stdout else "",
                'stderr': stderr.decode() if stderr else ""
            }
            
        except Exception as e:
            end_time = datetime.now()
            runtime_minutes = (end_time - start_time).total_seconds() / 60
            
            logger.error(f"Model run failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'runtime_minutes': runtime_minutes,
                'max_memory_mb': 0,
                'output_dir': run_dir
            }

class TestHarness:
    """Main test harness controller"""
    
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.comparator = FileComparator(
            tolerance_rel=self.config.tolerance_relative,
            tolerance_abs=self.config.tolerance_absolute
        )
        self.work_dir = Path("test_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = ModelRunner(self.work_dir)
        
    def load_config(self, config_file: str) -> TestSuiteConfig:
        """Load test configuration from YAML file"""
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse configuration
        versions = [ModelVersion(**v) for v in data['test_suite']['versions']]
        test_cases = [TestCase(**tc) for tc in data['test_suite']['test_cases']]
        
        comparison_settings = data['test_suite'].get('comparison_settings', {})
        
        return TestSuiteConfig(
            name=data['test_suite']['name'],
            versions=versions,
            test_cases=test_cases,
            tolerance_relative=comparison_settings.get('tolerance', {}).get('relative', 0.01),
            tolerance_absolute=comparison_settings.get('tolerance', {}).get('absolute', 1e-6),
            critical_files=comparison_settings.get('critical_files', ['*.opt', '*.csv']),
            ignore_patterns=comparison_settings.get('ignore_patterns', [])
        )
    
    def run_single_test(self, baseline_version: ModelVersion, test_version: ModelVersion, 
                       test_case: TestCase) -> ComparisonResult:
        """Run a single test case comparison"""
        logger.info(f"Running test: {test_case.name} ({baseline_version.id} vs {test_version.id})")
        
        run_id = hashlib.md5(f"{test_case.name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # Run baseline version
        baseline_result = self.runner.run_model(baseline_version, test_case, f"{run_id}_baseline")
        
        # Run test version
        test_result = self.runner.run_model(test_version, test_case, f"{run_id}_test")
        
        # Compare results
        if baseline_result['status'] != 'success':
            return ComparisonResult(
                test_case=test_case.name,
                baseline_version=baseline_version.id,
                test_version=test_version.id,
                status='FAIL',
                rmse=float('inf'),
                mae=float('inf'),
                correlation=0.0,
                max_difference_pct=100.0,
                runtime_baseline=baseline_result['runtime_minutes'],
                runtime_test=test_result.get('runtime_minutes', 0),
                memory_baseline=baseline_result['max_memory_mb'],
                memory_test=test_result.get('max_memory_mb', 0),
                issues=[f"Baseline failed: {baseline_result.get('error', 'Unknown error')}"],
                timestamp=datetime.now()
            )
        
        if test_result['status'] != 'success':
            return ComparisonResult(
                test_case=test_case.name,
                baseline_version=baseline_version.id,
                test_version=test_version.id,
                status='FAIL',
                rmse=float('inf'),
                mae=float('inf'),
                correlation=0.0,
                max_difference_pct=100.0,
                runtime_baseline=baseline_result['runtime_minutes'],
                runtime_test=test_result['runtime_minutes'],
                memory_baseline=baseline_result['max_memory_mb'],
                memory_test=test_result['max_memory_mb'],
                issues=[f"Test version failed: {test_result.get('error', 'Unknown error')}"],
                timestamp=datetime.now()
            )
        
        # Compare output files
        baseline_dir = baseline_result['output_dir']
        test_dir = test_result['output_dir']
        
        comparison_stats = []
        issues = []
        
        for pattern in self.config.critical_files:
            baseline_files = list(baseline_dir.glob(pattern))
            test_files = list(test_dir.glob(pattern))
            
            # Compare matching files
            for bf in baseline_files:
                tf = test_dir / bf.name
                if tf.exists():
                    if bf.suffix == '.opt':
                        stats = self.comparator.compare_opt_files(bf, tf)
                    elif bf.suffix == '.csv':
                        stats = self.comparator.compare_csv_files(bf, tf)
                    else:
                        continue
                    
                    comparison_stats.append(stats)
                    
                    # Check tolerances
                    if stats['max_diff_pct'] > self.config.tolerance_relative * 100:
                        issues.append(f"{bf.name}: Large difference ({stats['max_diff_pct']:.2f}%)")
                else:
                    issues.append(f"Missing file in test version: {bf.name}")
        
        # Aggregate statistics
        if comparison_stats:
            avg_rmse = np.mean([s['rmse'] for s in comparison_stats if np.isfinite(s['rmse'])])
            avg_mae = np.mean([s['mae'] for s in comparison_stats if np.isfinite(s['mae'])])
            avg_corr = np.mean([s['correlation'] for s in comparison_stats if np.isfinite(s['correlation'])])
            max_diff_pct = np.max([s['max_diff_pct'] for s in comparison_stats])
        else:
            avg_rmse = avg_mae = float('inf')
            avg_corr = 0.0
            max_diff_pct = 100.0
            issues.append("No comparable files found")
        
        # Determine overall status
        if issues:
            if max_diff_pct > self.config.tolerance_relative * 100 * 10:  # 10x tolerance = fail
                status = 'FAIL'
            else:
                status = 'WARN'
        else:
            status = 'PASS'
        
        return ComparisonResult(
            test_case=test_case.name,
            baseline_version=baseline_version.id,
            test_version=test_version.id,
            status=status,
            rmse=avg_rmse,
            mae=avg_mae,
            correlation=avg_corr,
            max_difference_pct=max_diff_pct,
            runtime_baseline=baseline_result['runtime_minutes'],
            runtime_test=test_result['runtime_minutes'],
            memory_baseline=baseline_result['max_memory_mb'],
            memory_test=test_result['max_memory_mb'],
            issues=issues,
            timestamp=datetime.now()
        )
    
    def run_test_suite(self, max_workers: int = 4) -> List[ComparisonResult]:
        """Run complete test suite with parallel execution"""
        logger.info(f"Starting test suite: {self.config.name}")
        
        baseline_version = next(v for v in self.config.versions if v.baseline)
        test_versions = [v for v in self.config.versions if not v.baseline]
        
        results = []
        
        # Create test execution plan
        test_plan = []
        for test_version in test_versions:
            for test_case in self.config.test_cases:
                test_plan.append((baseline_version, test_version, test_case))
        
        # Execute tests in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {
                executor.submit(self.run_single_test, baseline, test, case): (baseline, test, case)
                for baseline, test, case in test_plan
            }
            
            for future in as_completed(future_to_test):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {result.test_case} - {result.status}")
                except Exception as e:
                    baseline, test, case = future_to_test[future]
                    logger.error(f"Test failed: {case.name} ({baseline.id} vs {test.id}): {e}")
        
        return results
    
    def generate_report(self, results: List[ComparisonResult], output_dir: str = "reports"):
        """Generate comprehensive test reports"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        json_data = {
            'test_suite': self.config.name,
            'timestamp': timestamp,
            'summary': self._generate_summary(results),
            'results': [asdict(r) for r in results]
        }
        
        json_file = output_path / f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_file = output_path / f"test_report_{timestamp}.html"
        self._generate_html_report(results, html_file)
        
        # Generate plots
        self._generate_plots(results, output_path / "plots")
        
        logger.info(f"Reports generated in {output_path}")
        
    def _generate_summary(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total = len(results)
        passed = sum(1 for r in results if r.status == 'PASS')
        warnings = sum(1 for r in results if r.status == 'WARN')
        failed = sum(1 for r in results if r.status == 'FAIL')
        
        avg_runtime_improvement = np.mean([
            (r.runtime_baseline - r.runtime_test) / r.runtime_baseline * 100
            for r in results if r.runtime_baseline > 0
        ])
        
        return {
            'total_tests': total,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'pass_rate': passed / total * 100 if total > 0 else 0,
            'avg_runtime_improvement_pct': avg_runtime_improvement,
            'total_runtime_minutes': sum(r.runtime_test for r in results)
        }
    
    def _generate_html_report(self, results: List[ComparisonResult], output_file: Path):
        """Generate HTML dashboard report"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CE-QUAL-W2 Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { display: flex; gap: 20px; margin: 20px 0; }
                .metric { background: #e9e9e9; padding: 15px; border-radius: 5px; text-align: center; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .pass { color: green; font-weight: bold; }
                .warn { color: orange; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ config_name }}</h1>
                <p>Generated: {{ timestamp }}</p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Status</th>
                        <th>RMSE</th>
                        <th>Max Diff (%)</th>
                        <th>Runtime (min)</th>
                        <th>Memory (MB)</th>
                        <th>Issues</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in results %}
                    <tr>
                        <td>{{ result.test_case }}</td>
                        <td><span class="{{ result.status.lower() }}">{{ result.status }}</span></td>
                        <td>{{ "%.4f"|format(result.rmse) if result.rmse != float('inf') else 'N/A' }}</td>
                        <td>{{ "%.2f"|format(result.max_difference_pct) if result.max_difference_pct != 100 else 'N/A' }}</td>
                        <td>{{ "%.1f"|format(result.runtime_test) }}</td>
                        <td>{{ result.memory_test }}</td>
                        <td>{{ result.issues|join(', ') if result.issues else 'None' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """
        
        template_obj = Template(template)
        html_content = template_obj.render(
            config_name=self.config.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary=self._generate_summary(results),
            results=results
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_plots(self, results: List[ComparisonResult], plots_dir: Path):
        """Generate visualization plots"""
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Test Results Summary Pie Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        colors = {'PASS': 'green', 'WARN': 'orange', 'FAIL': 'red'}
        ax.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%',
               colors=[colors.get(k, 'gray') for k in status_counts.keys()])
        ax.set_title('Test Results Summary')
        plt.savefig(plots_dir / 'test_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Runtime Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        test_names = [r.test_case for r in results]
        baseline_times = [r.runtime_baseline for r in results]
        test_times = [r.runtime_test for r in results]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        ax.bar(x - width/2, baseline_times, width, label='Baseline', alpha=0.7)
        ax.bar(x + width/2, test_times, width, label='Test Version', alpha=0.7)
        
        ax.set_xlabel('Test Cases')
        ax.set_ylabel('Runtime (minutes)')
        ax.set_title('Runtime Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Statistical Metrics Heatmap
        metrics_data = []
        for result in results:
            if result.rmse != float('inf'):
                metrics_data.append([
                    result.rmse,
                    result.mae,
                    result.correlation,
                    result.max_difference_pct
                ])
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data, 
                                    columns=['RMSE', 'MAE', 'Correlation', 'Max Diff %'],
                                    index=[r.test_case for r in results if r.rmse != float('inf')])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlBu_r', ax=ax)
            ax.set_title('Statistical Comparison Metrics')
            plt.tight_layout()
            plt.savefig(plots_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Memory Usage Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        baseline_memory = [r.memory_baseline for r in results]
        test_memory = [r.memory_test for r in results]
        
        ax.scatter(baseline_memory, test_memory, alpha=0.7)
        
        # Add diagonal line for reference
        max_mem = max(max(baseline_memory), max(test_memory))
        ax.plot([0, max_mem], [0, max_mem], 'r--', alpha=0.5, label='Equal Memory Usage')
        
        ax.set_xlabel('Baseline Memory (MB)')
        ax.set_ylabel('Test Version Memory (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots generated in {plots_dir}")

class ConfigGenerator:
    """Utility class to generate test configuration files"""
    
    @staticmethod
    def create_sample_config(output_file: str = "sample_config.yaml"):
        """Create a sample configuration file"""
        config = {
            'test_suite': {
                'name': 'CE-QUAL-W2 Version Comparison',
                'versions': [
                    {
                        'id': 'v4.5',
                        'name': 'CE-QUAL-W2 v4.5 Stable',
                        'executable': 'CE-QUAL-W2/v4.5/executables/model/w2_v45_64.exe',
                        'baseline': True
                    },
                    {
                        'id': 'v5.0beta',
                        'name': 'CE-QUAL-W2 v5.0 Beta',
                        'executable': 'CE-QUAL-W2/v5.0/executables/w2_v5beta.exe',
                        'baseline': False
                    }
                ],
                'test_cases': [
                    {
                        'name': 'Long Lake',
                        'input_dir': 'examples/Long Lake',
                        'control_file': 'w2_con.csv',
                        'runtime_hours': 0.5
                    },
                    {
                        'name': 'Reservoir Ice',
                        'input_dir': 'examples/Reservoir',
                        'control_file': 'w2_con.csv',
                        'runtime_hours': 0.3
                    }
                ],
                'comparison_settings': {
                    'tolerance': {
                        'relative': 0.01,
                        'absolute': 1e-6
                    },
                    'critical_files': ['*.opt', '*.csv'],
                    'ignore_patterns': ['*log*', '*.err', '*.wrn']
                }
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Sample configuration created: {output_file}")

def main():
    """Main entry point for the test harness"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CE-QUAL-W2 Test Harness')
    parser.add_argument('command', choices=['run', 'generate-config', 'validate-config'],
                      help='Command to execute')
    parser.add_argument('--config', '-c', default='test_config.yaml',
                      help='Test configuration file')
    parser.add_argument('--output', '-o', default='reports',
                      help='Output directory for reports')
    parser.add_argument('--workers', '-w', type=int, default=4,
                      help='Number of parallel workers')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command == 'generate-config':
        ConfigGenerator.create_sample_config(args.config)
        return
    
    if args.command == 'validate-config':
        try:
            harness = TestHarness(args.config)
            logger.info("Configuration is valid")
            logger.info(f"Test suite: {harness.config.name}")
            logger.info(f"Versions: {len(harness.config.versions)}")
            logger.info(f"Test cases: {len(harness.config.test_cases)}")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
        return
    
    if args.command == 'run':
        try:
            # Initialize test harness
            harness = TestHarness(args.config)
            
            # Run test suite
            logger.info("Starting test execution...")
            results = harness.run_test_suite(max_workers=args.workers)
            
            # Generate reports
            logger.info("Generating reports...")
            harness.generate_report(results, args.output)
            
            # Print summary
            summary = harness._generate_summary(results)
            logger.info(f"Test suite completed:")
            logger.info(f"  Total tests: {summary['total_tests']}")
            logger.info(f"  Passed: {summary['passed']}")
            logger.info(f"  Warnings: {summary['warnings']}")
            logger.info(f"  Failed: {summary['failed']}")
            logger.info(f"  Pass rate: {summary['pass_rate']:.1f}%")
            
            # Exit with error code if any tests failed
            if summary['failed'] > 0:
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()
>
            
            <div class="summary">
                <div class="metric">
                    <h3>{{ summary.total_tests }}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="metric">
                    <h3>{{ summary.passed }}</h3>
                    <p>Passed</p>
                </div>
                <div class="metric">
                    <h3>{{ summary.warnings }}</h3>
                    <p>Warnings</p>
                </div>
                <div class="metric">
                    <h3>{{ summary.failed }}</h3>
                    <p>Failed</p>
                </div>
            </div