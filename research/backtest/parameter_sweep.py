#!/usr/bin/env python3
"""
Parameter Sweep Script for Phase 1 Optimization

Systematically tests different ENTRY_Z and EXIT_Z combinations to find optimal parameters.
Tests all combinations in a grid search and reports best performing configurations.

Usage:
    poetry run python research/backtest/parameter_sweep.py
    poetry run python research/backtest/parameter_sweep.py --quick  # Fewer combinations for fast test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import time
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParameterSweep:
    """Runs parameter sweep for ENTRY_Z and EXIT_Z optimization."""

    def __init__(
        self,
        entry_z_values: List[float],
        exit_z_values: List[float],
        coint_pvalue_values: List[float],
        output_dir: Path = Path("results/parameter_sweep"),
        run_name_prefix: str = "sweep",
        dry_run: bool = False
    ):
        self.entry_z_values = entry_z_values
        self.exit_z_values = exit_z_values
        self.coint_pvalue_values = coint_pvalue_values
        self.output_dir = output_dir
        self.run_name_prefix = run_name_prefix
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.start_time = None
        self.total_combinations = (
            len(entry_z_values) * len(exit_z_values) * len(coint_pvalue_values)
        )

    def run_backtest(
        self,
        entry_z: float,
        exit_z: float,
        coint_pvalue: float,
        combination_num: int
    ) -> Tuple[bool, Dict]:
        """
        Run a single backtest with specified parameters.

        Returns (success: bool, metrics: Dict)
        """
        run_name = f"{self.run_name_prefix}_entry{entry_z:.1f}_exit{exit_z:.2f}_pval{coint_pvalue:.2f}"

        logger.info(
            f"[{combination_num}/{self.total_combinations}] "
            f"Testing: ENTRY_Z={entry_z}, EXIT_Z={exit_z}, COINT_PVALUE={coint_pvalue}"
        )

        if self.dry_run:
            logger.info(f"  DRY RUN: Would run backtest with name '{run_name}'")
            return True, {
                'entry_z': entry_z,
                'exit_z': exit_z,
                'coint_pvalue': coint_pvalue,
                'gross_pnl': 0.01,
                'net_pnl': 0.005,
                'sharpe': 1.0,
                'trades': 1000,
                'cost_gross_ratio': 0.5
            }

        # Build command
        cmd = [
            "poetry", "run", "python", "research/backtest/run_simulation.py",
            "--run-name", run_name,
            "--entry-z", str(entry_z),
            "--exit-z", str(exit_z),
            "--coint-pvalue", str(coint_pvalue)
        ]

        logger.info(f"  Running: {' '.join(cmd)}")

        try:
            # Run backtest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=14400  # 4 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"  ❌ FAILED: {result.stderr}")
                return False, {}

            # Extract metrics from results
            metrics_file = Path(f"results/{run_name}/window_metrics.csv")
            if not metrics_file.exists():
                logger.error(f"  ❌ Metrics file not found: {metrics_file}")
                return False, {}

            # Load and aggregate metrics
            df = pd.read_csv(metrics_file)

            # Calculate overall metrics
            total_gross = df['gross_pnl'].sum()
            total_costs = df['total_costs'].sum()
            total_net = df['net_pnl'].sum()
            total_trades = df['trades'].sum()

            # Sharpe calculation (annualized)
            returns = df['net_pnl']
            sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0.0

            cost_gross_ratio = total_costs / total_gross if total_gross > 0 else 999.0

            metrics = {
                'entry_z': entry_z,
                'exit_z': exit_z,
                'coint_pvalue': coint_pvalue,
                'gross_pnl': total_gross,
                'net_pnl': total_net,
                'total_costs': total_costs,
                'cost_gross_ratio': cost_gross_ratio,
                'sharpe': sharpe,
                'trades': int(total_trades),
                'run_name': run_name
            }

            logger.info(
                f"  ✅ SUCCESS: Net={total_net:.4f} ({total_net*100:.2f}%), "
                f"Gross={total_gross:.4f}, Sharpe={sharpe:.2f}, "
                f"Trades={int(total_trades)}, Cost/Gross={cost_gross_ratio:.1%}"
            )

            return True, metrics

        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ TIMEOUT: Backtest exceeded 4 hour limit")
            return False, {}
        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            return False, {}

    def run_sweep(self):
        """Run parameter sweep across all combinations."""
        logger.info("=" * 80)
        logger.info("PARAMETER SWEEP STARTING")
        logger.info("=" * 80)
        logger.info(f"Entry Z values: {self.entry_z_values}")
        logger.info(f"Exit Z values: {self.exit_z_values}")
        logger.info(f"Coint P-value values: {self.coint_pvalue_values}")
        logger.info(f"Total combinations: {self.total_combinations}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 80)

        self.start_time = time.time()
        combination_num = 0

        for coint_pvalue in self.coint_pvalue_values:
            for entry_z in self.entry_z_values:
                for exit_z in self.exit_z_values:
                    combination_num += 1

                    success, metrics = self.run_backtest(
                        entry_z=entry_z,
                        exit_z=exit_z,
                        coint_pvalue=coint_pvalue,
                        combination_num=combination_num
                    )

                    if success:
                        self.results.append(metrics)

                    # Save intermediate results
                    self._save_results()

                    # Estimate remaining time
                    elapsed = time.time() - self.start_time
                    avg_time_per_run = elapsed / combination_num
                    remaining = (self.total_combinations - combination_num) * avg_time_per_run

                    logger.info(
                        f"  Progress: {combination_num}/{self.total_combinations} "
                        f"({combination_num/self.total_combinations*100:.1f}%) | "
                        f"Elapsed: {elapsed/3600:.1f}h | "
                        f"Remaining: {remaining/3600:.1f}h"
                    )
                    logger.info("")

        logger.info("=" * 80)
        logger.info("PARAMETER SWEEP COMPLETE")
        logger.info("=" * 80)

        self._generate_report()

    def _save_results(self):
        """Save intermediate results to CSV and JSON."""
        if not self.results:
            return

        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "sweep_results.csv"
        df.to_csv(csv_path, index=False)

        # Save as JSON
        json_path = self.output_dir / "sweep_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.debug(f"Saved intermediate results to {csv_path} and {json_path}")

    def _generate_report(self):
        """Generate final report with best configurations."""
        if not self.results:
            logger.error("No results to report!")
            return

        df = pd.DataFrame(self.results)

        # Sort by different metrics
        by_net = df.sort_values('net_pnl', ascending=False)
        by_sharpe = df.sort_values('sharpe', ascending=False)
        by_cost_gross = df[df['cost_gross_ratio'] < 0.5].sort_values('net_pnl', ascending=False)

        report_path = self.output_dir / "sweep_report.md"

        with open(report_path, 'w') as f:
            f.write("# Parameter Sweep Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Combinations Tested:** {len(self.results)}/{self.total_combinations}\n")
            f.write(f"**Total Time:** {(time.time() - self.start_time)/3600:.2f} hours\n\n")

            f.write("---\n\n")

            # Best by Net P&L
            f.write("## Top 5 Configurations by Net P&L\n\n")
            f.write("| Rank | ENTRY_Z | EXIT_Z | COINT_P | Net P&L | Gross P&L | Cost/Gross | Sharpe | Trades |\n")
            f.write("|------|---------|--------|---------|---------|-----------|------------|--------|--------|\n")

            for i, row in by_net.head(5).iterrows():
                f.write(
                    f"| {i+1} | {row['entry_z']:.1f} | {row['exit_z']:.2f} | "
                    f"{row['coint_pvalue']:.2f} | {row['net_pnl']*100:.2f}% | "
                    f"{row['gross_pnl']*100:.2f}% | {row['cost_gross_ratio']:.1%} | "
                    f"{row['sharpe']:.2f} | {row['trades']:.0f} |\n"
                )

            f.write("\n")

            # Best by Sharpe
            f.write("## Top 5 Configurations by Sharpe Ratio\n\n")
            f.write("| Rank | ENTRY_Z | EXIT_Z | COINT_P | Sharpe | Net P&L | Gross P&L | Cost/Gross | Trades |\n")
            f.write("|------|---------|--------|---------|--------|---------|-----------|------------|--------|\n")

            for i, row in by_sharpe.head(5).iterrows():
                f.write(
                    f"| {i+1} | {row['entry_z']:.1f} | {row['exit_z']:.2f} | "
                    f"{row['coint_pvalue']:.2f} | {row['sharpe']:.2f} | "
                    f"{row['net_pnl']*100:.2f}% | {row['gross_pnl']*100:.2f}% | "
                    f"{row['cost_gross_ratio']:.1%} | {row['trades']:.0f} |\n"
                )

            f.write("\n")

            # Best with acceptable cost/gross
            f.write("## Top 5 Configurations with Cost/Gross < 50%\n\n")
            if len(by_cost_gross) > 0:
                f.write("| Rank | ENTRY_Z | EXIT_Z | COINT_P | Net P&L | Cost/Gross | Sharpe | Trades |\n")
                f.write("|------|---------|--------|---------|---------|------------|--------|--------|\n")

                for i, row in by_cost_gross.head(5).iterrows():
                    f.write(
                        f"| {i+1} | {row['entry_z']:.1f} | {row['exit_z']:.2f} | "
                        f"{row['coint_pvalue']:.2f} | {row['net_pnl']*100:.2f}% | "
                        f"{row['cost_gross_ratio']:.1%} | {row['sharpe']:.2f} | "
                        f"{row['trades']:.0f} |\n"
                    )
            else:
                f.write("*No configurations achieved Cost/Gross < 50%*\n")

            f.write("\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Best Net P&L:** {df['net_pnl'].max()*100:.2f}%\n")
            f.write(f"- **Worst Net P&L:** {df['net_pnl'].min()*100:.2f}%\n")
            f.write(f"- **Median Net P&L:** {df['net_pnl'].median()*100:.2f}%\n")
            f.write(f"- **Best Sharpe:** {df['sharpe'].max():.2f}\n")
            f.write(f"- **Configurations with Net > 0.0%:** {(df['net_pnl'] > 0).sum()}/{len(df)}\n")
            f.write(f"- **Configurations with Cost/Gross < 50%:** {(df['cost_gross_ratio'] < 0.5).sum()}/{len(df)}\n")

            f.write("\n---\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            best_net = by_net.iloc[0]
            best_sharpe = by_sharpe.iloc[0]

            f.write("### Recommended Configuration (Best Net P&L)\n\n")
            f.write("```python\n")
            f.write(f"ENTRY_Z: float = {best_net['entry_z']:.1f}\n")
            f.write(f"EXIT_Z: float = {best_net['exit_z']:.2f}\n")
            f.write(f"COINT_PVALUE_THRESHOLD: float = {best_net['coint_pvalue']:.2f}\n")
            f.write("```\n\n")
            f.write(f"- **Expected Net P&L:** {best_net['net_pnl']*100:.2f}%\n")
            f.write(f"- **Expected Sharpe:** {best_net['sharpe']:.2f}\n")
            f.write(f"- **Cost/Gross Ratio:** {best_net['cost_gross_ratio']:.1%}\n")
            f.write(f"- **Expected Trades:** {best_net['trades']:.0f}\n\n")

            if best_net['entry_z'] != best_sharpe['entry_z'] or best_net['exit_z'] != best_sharpe['exit_z']:
                f.write("### Alternative Configuration (Best Sharpe)\n\n")
                f.write("```python\n")
                f.write(f"ENTRY_Z: float = {best_sharpe['entry_z']:.1f}\n")
                f.write(f"EXIT_Z: float = {best_sharpe['exit_z']:.2f}\n")
                f.write(f"COINT_PVALUE_THRESHOLD: float = {best_sharpe['coint_pvalue']:.2f}\n")
                f.write("```\n\n")
                f.write(f"- **Expected Net P&L:** {best_sharpe['net_pnl']*100:.2f}%\n")
                f.write(f"- **Expected Sharpe:** {best_sharpe['sharpe']:.2f}\n")
                f.write(f"- **Cost/Gross Ratio:** {best_sharpe['cost_gross_ratio']:.1%}\n")
                f.write(f"- **Expected Trades:** {best_sharpe['trades']:.0f}\n\n")

            # Decision gate
            f.write("### Decision Gate: Phase 1 Success?\n\n")
            if best_net['net_pnl'] > 0.0:
                f.write("✅ **SUCCESS** - Phase 1 achieved positive net returns!\n\n")
                f.write("**Next Steps:**\n")
                f.write("1. Apply recommended configuration to config_backtest.py\n")
                f.write("2. Proceed to Phase 2 (ML Integration)\n")
                f.write("3. Follow [PAIRS_TRADING_REDESIGN_PLAN.md](../../PAIRS_TRADING_REDESIGN_PLAN.md) Phase 2\n\n")
            else:
                f.write("❌ **FAILED** - Phase 1 did not achieve positive net returns.\n\n")
                f.write("**Next Steps:**\n")
                f.write("1. Review results to understand why all configurations failed\n")
                f.write("2. Execute Contingency Plan A (pivot to different strategy)\n")
                f.write("3. Consider alternatives: longer timeframes, different assets, trend-following\n\n")

        logger.info(f"✅ Report generated: {report_path}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP RESULTS")
        print("=" * 80)
        print(f"\nTotal combinations tested: {len(self.results)}/{self.total_combinations}")
        print(f"Total time: {(time.time() - self.start_time)/3600:.2f} hours")
        print(f"\nBest Net P&L: {df['net_pnl'].max()*100:.2f}%")
        print(f"  ENTRY_Z={by_net.iloc[0]['entry_z']:.1f}, "
              f"EXIT_Z={by_net.iloc[0]['exit_z']:.2f}, "
              f"COINT_P={by_net.iloc[0]['coint_pvalue']:.2f}")
        print(f"\nBest Sharpe: {df['sharpe'].max():.2f}")
        print(f"  ENTRY_Z={by_sharpe.iloc[0]['entry_z']:.1f}, "
              f"EXIT_Z={by_sharpe.iloc[0]['exit_z']:.2f}, "
              f"COINT_P={by_sharpe.iloc[0]['coint_pvalue']:.2f}")
        print(f"\nConfigurations with Net > 0.0%: {(df['net_pnl'] > 0).sum()}/{len(df)}")
        print(f"\nFull report: {report_path}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for Phase 1 optimization"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer combinations (4 total)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (don't actually run backtests)"
    )
    parser.add_argument(
        "--entry-z",
        type=float,
        nargs='+',
        help="ENTRY_Z values to test (default: 1.8 2.0 2.2 2.5 3.0)"
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        nargs='+',
        help="EXIT_Z values to test (default: 0.05 0.3 0.6 0.8 1.0)"
    )
    parser.add_argument(
        "--coint-pvalue",
        type=float,
        nargs='+',
        help="COINT_PVALUE_THRESHOLD values to test (default: 0.10)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/parameter_sweep"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Determine parameter values
    if args.quick:
        entry_z_values = [2.0, 2.5]
        exit_z_values = [0.6, 0.8]
        coint_pvalue_values = [0.10]
    else:
        entry_z_values = args.entry_z or [1.8, 2.0, 2.2, 2.5, 3.0]
        exit_z_values = args.exit_z or [0.05, 0.3, 0.6, 0.8, 1.0]
        coint_pvalue_values = args.coint_pvalue or [0.10]

    # Create and run sweep
    sweep = ParameterSweep(
        entry_z_values=entry_z_values,
        exit_z_values=exit_z_values,
        coint_pvalue_values=coint_pvalue_values,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )

    try:
        sweep.run_sweep()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Sweep interrupted by user")
        logger.info("Saving partial results...")
        sweep._save_results()
        sweep._generate_report()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
