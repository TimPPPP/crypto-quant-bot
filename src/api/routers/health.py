"""
Health check endpoints.
"""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter
import requests

from src.backtest import config_backtest as cfg

logger = logging.getLogger("api.health")
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint.

    Returns system status and connectivity to dependencies.
    """
    questdb_status = "unknown"

    # Check QuestDB connectivity
    try:
        resp = requests.get("http://localhost:9000", timeout=2)
        questdb_status = "online" if resp.status_code == 200 else "offline"
    except Exception:
        questdb_status = "offline"

    return {
        "status": "online",
        "version": cfg.BACKTEST_VERSION,
        "questdb_status": questdb_status,
        "data_snapshot": cfg.DATA_SNAPSHOT_ID,
    }


@router.get("/ping")
async def ping() -> Dict:
    """Simple ping endpoint."""
    return {"message": "pong"}
