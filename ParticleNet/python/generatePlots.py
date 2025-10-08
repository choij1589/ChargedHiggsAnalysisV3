#!/usr/bin/env python
"""
Master visualization orchestration script for ParticleNet training results.

This script automatically detects all available training results and generates
comprehensive visualization plots for both binary and multi-class models.

Features:
- Auto-detection of completed training results
- Parallel generation of individual model plots
- Comparison analysis between methodologies
- Organized output directory structure
- Summary reports and logs

Usage:
    python generatePlots.py --channel Run1E2Mu --fold 3 --parallel
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    """Setup Python path to ensure imports work correctly."""
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

def find_available_results(channel, fold):
    """Find all available training results for visualization."""
    base_path = Path("/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results")

    signal_points = ["MHc160_MA85", "MHc130_MA90", "MHc100_MA95"]
    backgrounds = ["nonprompt", "diboson", "ttZ"]

    available_binary = {}
    available_multiclass = {}

    # Find binary results
    binary_path = base_path / "binary" / channel
    if binary_path.exists():
        for signal_dir in binary_path.glob("TTToHcToWAToMuMu-*"):
            signal = signal_dir.name.replace("TTToHcToWAToMuMu-", "")
            if signal in signal_points:
                pilot_dir = signal_dir / "pilot"
                if pilot_dir.exists():
                    available_binary[signal] = {}
                    for bg in backgrounds:
                        pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-binary-{bg}_performance.json"
                        perf_file = pilot_dir / pattern
                        if perf_file.exists():
                            available_binary[signal][bg] = True

    # Find multiclass results
    multiclass_path = base_path / "multiclass" / channel
    if multiclass_path.exists():
        for signal_dir in multiclass_path.glob("TTToHcToWAToMuMu-*"):
            signal = signal_dir.name.replace("TTToHcToWAToMuMu-", "")
            if signal in signal_points:
                pilot_dir = signal_dir / "pilot"
                pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-3bg_performance.json"
                perf_file = pilot_dir / pattern
                if perf_file.exists():
                    available_multiclass[signal] = True

    return available_binary, available_multiclass, signal_points, backgrounds

def run_binary_visualization(signal, background, channel, fold, output_base):
    """Run binary visualization for a specific signal-background combination."""
    script_path = Path(__file__).parent / "visualizeBinary.py"
    output_dir = Path(output_base) / "binary" / f"{signal}_vs_{background}"

    cmd = [
        "python", str(script_path),
        "--signal", signal,
        "--background", background,
        "--channel", channel,
        "--fold", str(fold),
        "--output", str(output_dir)
    ]

    try:
        logging.info(f"Starting binary visualization: {signal} vs {background}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"Completed binary visualization: {signal} vs {background}")
        return {"success": True, "signal": signal, "background": background, "output": str(output_dir)}
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed binary visualization {signal} vs {background}: {e}")
        return {"success": False, "signal": signal, "background": background, "error": str(e)}

def run_multiclass_visualization(signal, channel, fold, output_base):
    """Run multi-class visualization for a specific signal point."""
    script_path = Path(__file__).parent / "visualizeMultiClass.py"
    output_dir = Path(output_base) / "multiclass" / signal

    cmd = [
        "python", str(script_path),
        "--signal", signal,
        "--channel", channel,
        "--fold", str(fold),
        "--output", str(output_dir)
    ]

    try:
        logging.info(f"Starting multi-class visualization: {signal}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"Completed multi-class visualization: {signal}")
        return {"success": True, "signal": signal, "output": str(output_dir)}
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed multi-class visualization {signal}: {e}")
        return {"success": False, "signal": signal, "error": str(e)}

def run_comparison_analysis(channel, fold, output_base):
    """Run comparison analysis between binary and multi-class approaches."""
    script_path = Path(__file__).parent / "visualizeComparison.py"
    output_dir = Path(output_base) / "comparison"

    cmd = [
        "python", str(script_path),
        "--channel", channel,
        "--fold", str(fold),
        "--output", str(output_dir)
    ]

    try:
        logging.info("Starting comparison analysis...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("Completed comparison analysis")
        return {"success": True, "output": str(output_dir)}
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed comparison analysis: {e}")
        return {"success": False, "error": str(e)}

def generate_plots_parallel(available_binary, available_multiclass, channel, fold, output_base, max_workers=4):
    """Generate all plots in parallel using ProcessPoolExecutor."""
    tasks = []

    # Prepare binary visualization tasks
    for signal, bg_dict in available_binary.items():
        for bg in bg_dict.keys():
            tasks.append(("binary", signal, bg, channel, fold, output_base))

    # Prepare multiclass visualization tasks
    for signal in available_multiclass.keys():
        tasks.append(("multiclass", signal, None, channel, fold, output_base))

    results = []
    failed_tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for task in tasks:
            task_type, signal, bg, channel, fold, output_base = task
            if task_type == "binary":
                future = executor.submit(run_binary_visualization, signal, bg, channel, fold, output_base)
            else:  # multiclass
                future = executor.submit(run_multiclass_visualization, signal, channel, fold, output_base)
            future_to_task[future] = task

        # Collect results
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                if not result["success"]:
                    failed_tasks.append(task)
            except Exception as e:
                logging.error(f"Task {task} generated an exception: {e}")
                failed_tasks.append(task)

    return results, failed_tasks

def generate_plots_sequential(available_binary, available_multiclass, channel, fold, output_base):
    """Generate all plots sequentially."""
    results = []
    failed_tasks = []

    # Generate binary plots
    for signal, bg_dict in available_binary.items():
        for bg in bg_dict.keys():
            result = run_binary_visualization(signal, bg, channel, fold, output_base)
            results.append(result)
            if not result["success"]:
                failed_tasks.append(("binary", signal, bg))

    # Generate multiclass plots
    for signal in available_multiclass.keys():
        result = run_multiclass_visualization(signal, channel, fold, output_base)
        results.append(result)
        if not result["success"]:
            failed_tasks.append(("multiclass", signal, None))

    return results, failed_tasks

def create_index_html(output_base, results, channel, fold):
    """Create an HTML index file for easy navigation of results."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ParticleNet Visualization Results - {channel} Fold {fold}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .model-card {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; background: #f9f9f9; }}
        .model-card h3 {{ margin-top: 0; color: #555; }}
        .plot-links {{ margin: 10px 0; }}
        .plot-links a {{ display: inline-block; margin: 5px 10px 5px 0; padding: 5px 10px;
                        background: #007bff; color: white; text-decoration: none; border-radius: 3px; }}
        .plot-links a:hover {{ background: #0056b3; }}
        .summary {{ background: #e7f3ff; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>ParticleNet Visualization Results</h1>
    <div class="summary">
        <h2>Analysis Summary</h2>
        <p><strong>Channel:</strong> {channel}</p>
        <p><strong>Fold:</strong> {fold}</p>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

    # Binary classification section
    binary_results = [r for r in results if r.get("background")]
    if binary_results:
        html_content += """
    <div class="section">
        <h2>Binary Classification Results</h2>
        <div class="model-grid">
"""
        for result in binary_results:
            if result["success"]:
                signal = result["signal"]
                background = result["background"]
                rel_path = f"binary/{signal}_vs_{background}"
                html_content += f"""
            <div class="model-card">
                <h3>{signal} vs {background}</h3>
                <div class="plot-links">
                    <a href="{rel_path}/training_curves.png">Training Curves</a>
                    <a href="{rel_path}/roc_curve.png">ROC Curve</a>
                    <a href="{rel_path}/confusion_matrix.png">Confusion Matrix</a>
                    <a href="{rel_path}/score_distributions.png">Score Distributions</a>
                    <a href="{rel_path}/summary_report.txt">Summary Report</a>
                </div>
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    # Multi-class classification section
    multiclass_results = [r for r in results if not r.get("background") and r.get("signal")]
    if multiclass_results:
        html_content += """
    <div class="section">
        <h2>Multi-class Classification Results</h2>
        <div class="model-grid">
"""
        for result in multiclass_results:
            if result["success"]:
                signal = result["signal"]
                rel_path = f"multiclass/{signal}"
                html_content += f"""
            <div class="model-card">
                <h3>Multi-class {signal}</h3>
                <div class="plot-links">
                    <a href="{rel_path}/training_curves.png">Training Curves</a>
                    <a href="{rel_path}/overall_confusion_matrix.png">Overall Confusion Matrix</a>
                    <a href="{rel_path}/signal_roc_curve.png">Signal ROC</a>
                    <a href="{rel_path}/classification_report.txt">Classification Report</a>
                    <a href="{rel_path}/summary_report.txt">Summary Report</a>
                </div>
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    # Comparison section
    html_content += """
    <div class="section">
        <h2>Methodology Comparison</h2>
        <div class="plot-links">
            <a href="comparison/binary_vs_multiclass_performance.png">Performance Comparison</a>
            <a href="comparison/signal_point_analysis.png">Signal Point Analysis</a>
            <a href="comparison/background_separation_analysis.png">Background Separation</a>
            <a href="comparison/training_efficiency_comparison.png">Training Efficiency</a>
            <a href="comparison/comparison_summary.txt">Summary Report</a>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Write HTML file
    with open(Path(output_base) / "index.html", 'w') as f:
        f.write(html_content)

    logging.info(f"HTML index created: {Path(output_base) / 'index.html'}")

def generate_execution_summary(results, failed_tasks, output_base, start_time, end_time):
    """Generate a summary of the execution."""
    summary = {
        "execution_info": {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "total_tasks": len(results),
            "successful_tasks": len([r for r in results if r["success"]]),
            "failed_tasks": len(failed_tasks)
        },
        "results": results,
        "failed_tasks": failed_tasks
    }

    # Save summary as JSON
    with open(Path(output_base) / "execution_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save readable summary
    with open(Path(output_base) / "execution_summary.txt", 'w') as f:
        f.write("ParticleNet Visualization Execution Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Execution Time: {summary['execution_info']['duration_seconds']:.1f} seconds\n")
        f.write(f"Total Tasks: {summary['execution_info']['total_tasks']}\n")
        f.write(f"Successful: {summary['execution_info']['successful_tasks']}\n")
        f.write(f"Failed: {summary['execution_info']['failed_tasks']}\n\n")

        if failed_tasks:
            f.write("Failed Tasks:\n")
            for task in failed_tasks:
                if len(task) == 3 and task[2]:  # binary task
                    f.write(f"  Binary: {task[1]} vs {task[2]}\n")
                else:  # multiclass task
                    f.write(f"  Multi-class: {task[1]}\n")

    logging.info(f"Execution summary saved to: {Path(output_base) / 'execution_summary.txt'}")

def main():
    parser = argparse.ArgumentParser(description='Generate all ParticleNet visualization plots')
    parser.add_argument('--channel', default='Run1E2Mu',
                        help='Analysis channel (default: Run1E2Mu)')
    parser.add_argument('--fold', type=int, default=3,
                        help='Cross-validation fold (default: 3)')
    parser.add_argument('--output',
                        help='Output base directory (default: auto-generated)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run visualizations in parallel')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip comparison analysis')

    args = parser.parse_args()

    # Setup paths
    setup_paths()

    # Create output directory
    if args.output is None:
        base_output = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/plots"
        args.output = base_output

    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()

    try:
        # Find available results
        logging.info(f"Searching for training results in channel {args.channel}, fold {args.fold}")
        available_binary, available_multiclass, signal_points, backgrounds = find_available_results(args.channel, args.fold)

        total_binary = sum(len(bg_dict) for bg_dict in available_binary.values())
        total_multiclass = len(available_multiclass)

        logging.info(f"Found {total_binary} binary models and {total_multiclass} multi-class models")

        if total_binary == 0 and total_multiclass == 0:
            logging.warning("No training results found to visualize!")
            return

        # Generate individual model plots
        if args.parallel:
            logging.info(f"Generating plots in parallel with {args.max_workers} workers...")
            results, failed_tasks = generate_plots_parallel(
                available_binary, available_multiclass, args.channel, args.fold, args.output, args.max_workers)
        else:
            logging.info("Generating plots sequentially...")
            results, failed_tasks = generate_plots_sequential(
                available_binary, available_multiclass, args.channel, args.fold, args.output)

        # Generate comparison analysis
        if not args.skip_comparison and (total_binary > 0 and total_multiclass > 0):
            logging.info("Generating comparison analysis...")
            comparison_result = run_comparison_analysis(args.channel, args.fold, args.output)
            results.append(comparison_result)
            if not comparison_result["success"]:
                failed_tasks.append(("comparison", None, None))

        end_time = time.time()

        # Generate summary and index
        logging.info("Creating HTML index...")
        create_index_html(args.output, results, args.channel, args.fold)

        logging.info("Generating execution summary...")
        generate_execution_summary(results, failed_tasks, args.output, start_time, end_time)

        # Final report
        successful = len([r for r in results if r["success"]])
        total = len(results)
        logging.info(f"Visualization complete: {successful}/{total} tasks successful")
        logging.info(f"Total execution time: {end_time - start_time:.1f} seconds")
        logging.info(f"All results saved to: {args.output}")
        logging.info(f"Open {Path(args.output) / 'index.html'} to browse results")

        if failed_tasks:
            logging.warning(f"{len(failed_tasks)} tasks failed. See execution_summary.txt for details.")

    except Exception as e:
        logging.error(f"Error during plot generation: {e}")
        raise

if __name__ == "__main__":
    main()