"""
Backtest results and metrics endpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.backtest import config_backtest as cfg

logger = logging.getLogger("api.backtest")
router = APIRouter()


def _get_all_run_dirs() -> List[Path]:
    """Get all run directories sorted by modification time (newest first)."""
    results_dir = cfg.RESULTS_DIR
    if not results_dir.exists():
        return []

    runs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    return sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)


def _load_manifest(run_dir: Path) -> Optional[Dict]:
    """Load manifest.json from a run directory."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load manifest from {run_dir}: {e}")
        return None


def _load_metrics(run_dir: Path) -> Optional[Dict]:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    try:
        with metrics_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics from {run_dir}: {e}")
        return None


@router.get("/backtest/runs")
async def list_runs() -> Dict:
    """
    List all available backtest runs.

    Returns run IDs and timestamps.
    """
    runs = _get_all_run_dirs()

    run_list = []
    for run_dir in runs:
        manifest = _load_manifest(run_dir)

        run_info = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "timestamp": manifest.get("environment", {}).get("timestamp") if manifest else None,
        }
        run_list.append(run_info)

    return {
        "total_runs": len(run_list),
        "runs": run_list,
    }


@router.get("/backtest/latest")
async def get_latest_run() -> Dict:
    """
    Get the latest backtest run with full metrics.
    """
    runs = _get_all_run_dirs()
    if not runs:
        return {"run_id": None, "message": "No backtest runs found"}

    latest_run = runs[0]
    manifest = _load_manifest(latest_run)
    metrics = _load_metrics(latest_run)

    return {
        "run_id": latest_run.name,
        "manifest": manifest,
        "metrics": metrics.get("metrics") if metrics else None,
        "scenario_specs": metrics.get("scenario_specs") if metrics else None,
    }


@router.get("/backtest/run/{run_id}")
async def get_run(run_id: str) -> Dict:
    """
    Get detailed information about a specific backtest run.

    Parameters:
        run_id: The run identifier (e.g., "run_20231213_143022")
    """
    run_dir = cfg.RESULTS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    manifest = _load_manifest(run_dir)
    metrics = _load_metrics(run_dir)

    # List available plots
    plots_dir = run_dir / "plots"
    plots = []
    if plots_dir.exists():
        plots = [p.name for p in plots_dir.iterdir() if p.suffix in {".png", ".jpg", ".jpeg"}]

    return {
        "run_id": run_id,
        "manifest": manifest,
        "metrics": metrics,
        "plots": plots,
        "plots_url": f"/static/{run_id}/plots",
    }


@router.get("/backtest/run/{run_id}/plot/{plot_name}")
async def get_plot(run_id: str, plot_name: str):
    """
    Get a specific plot image from a backtest run.

    Parameters:
        run_id: The run identifier
        plot_name: Name of the plot file (e.g., "equity_curves.png")
    """
    plot_path = cfg.RESULTS_DIR / run_id / "plots" / plot_name

    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found in run '{run_id}'")

    return FileResponse(plot_path)


@router.get("/backtest/compare")
async def compare_runs(run_ids: str) -> Dict:
    """
    Compare metrics across multiple runs.

    Parameters:
        run_ids: Comma-separated list of run IDs to compare
    """
    run_id_list = [rid.strip() for rid in run_ids.split(",")]

    comparison = []
    for run_id in run_id_list:
        run_dir = cfg.RESULTS_DIR / run_id
        if not run_dir.exists():
            continue

        metrics = _load_metrics(run_dir)
        if metrics:
            comparison.append({
                "run_id": run_id,
                "metrics": metrics.get("metrics"),
            })

    return {
        "total_runs": len(comparison),
        "comparison": comparison,
    }
