"""
FastAPI application for crypto quant bot dashboard and API.

Provides endpoints for:
- Backtest results and metrics
- Bot status and health
- Configuration management
- Performance visualization data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import backtest, health, config
from src.backtest import config_backtest as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Quant Bot API",
    description="REST API for crypto pairs trading bot - backtest results, metrics, and configuration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(backtest.router, prefix="/api", tags=["backtest"])
app.include_router(config.router, prefix="/api", tags=["config"])

# Serve static files (plots, etc.)
try:
    static_path = cfg.RESULTS_DIR
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard HTML."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crypto Quant Bot Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                background: white;
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .header p {
                color: #666;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .card h2 {
                color: #333;
                margin-bottom: 15px;
                font-size: 20px;
            }
            .metric {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }
            .metric:last-child {
                border-bottom: none;
            }
            .metric-label {
                color: #666;
                font-weight: 500;
            }
            .metric-value {
                color: #333;
                font-weight: 600;
            }
            .positive { color: #10b981; }
            .negative { color: #ef4444; }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                margin: 5px;
                transition: background 0.2s;
            }
            .btn:hover {
                background: #5568d3;
            }
            .links {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .status-badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
            }
            .status-online {
                background: #d1fae5;
                color: #065f46;
            }
            .loading {
                text-align: center;
                color: #666;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Crypto Quant Bot Dashboard</h1>
                <p>Kalman Filter Pairs Trading • Real-time monitoring & backtest analytics</p>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>System Status</h2>
                    <div id="status-content" class="loading">Loading...</div>
                </div>

                <div class="card">
                    <h2>Latest Backtest Results</h2>
                    <div id="backtest-content" class="loading">Loading...</div>
                </div>

                <div class="card">
                    <h2>Configuration</h2>
                    <div id="config-content" class="loading">Loading...</div>
                </div>
            </div>

            <div class="card">
                <h2>Quick Links</h2>
                <div class="links">
                    <a href="/docs" class="btn">📖 API Documentation</a>
                    <a href="/api/backtest/runs" class="btn">📊 View All Runs</a>
                    <a href="/api/health" class="btn">💚 Health Check</a>
                    <a href="http://localhost:3000" target="_blank" class="btn">📈 Grafana</a>
                    <a href="http://localhost:8888" target="_blank" class="btn">🔬 Jupyter Lab</a>
                    <a href="http://localhost:9000" target="_blank" class="btn">🗄️ QuestDB Console</a>
                </div>
            </div>
        </div>

        <script>
            // Fetch health status
            fetch('/api/health')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('status-content').innerHTML = `
                        <div class="metric">
                            <span class="metric-label">Status</span>
                            <span class="status-badge status-online">${data.status}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Version</span>
                            <span class="metric-value">${data.version}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">QuestDB</span>
                            <span class="metric-value">${data.questdb_status}</span>
                        </div>
                    `;
                })
                .catch(() => {
                    document.getElementById('status-content').innerHTML =
                        '<p style="color: #ef4444;">Failed to connect to API</p>';
                });

            // Fetch latest backtest
            fetch('/api/backtest/latest')
                .then(res => res.json())
                .then(data => {
                    if (data.run_id) {
                        const metrics = data.metrics?.base_case || {};
                        document.getElementById('backtest-content').innerHTML = `
                            <div class="metric">
                                <span class="metric-label">Run ID</span>
                                <span class="metric-value">${data.run_id}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Sharpe Ratio</span>
                                <span class="metric-value ${metrics.sharpe > 0 ? 'positive' : 'negative'}">
                                    ${metrics.sharpe?.toFixed(2) || 'N/A'}
                                </span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Total Return</span>
                                <span class="metric-value ${metrics.total_return > 0 ? 'positive' : 'negative'}">
                                    ${((metrics.total_return || 0) * 100).toFixed(2)}%
                                </span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Max Drawdown</span>
                                <span class="metric-value negative">
                                    ${((metrics.max_drawdown || 0) * 100).toFixed(2)}%
                                </span>
                            </div>
                        `;
                    } else {
                        document.getElementById('backtest-content').innerHTML =
                            '<p style="color: #666;">No backtest runs found</p>';
                    }
                })
                .catch(() => {
                    document.getElementById('backtest-content').innerHTML =
                        '<p style="color: #666;">No backtest data available</p>';
                });

            // Fetch config
            fetch('/api/config')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('config-content').innerHTML = `
                        <div class="metric">
                            <span class="metric-label">Entry Z-Score</span>
                            <span class="metric-value">${data.ENTRY_Z}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Stop Loss Z</span>
                            <span class="metric-value">${data.STOP_LOSS_Z}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Min Profit Hurdle</span>
                            <span class="metric-value">${(data.MIN_PROFIT_HURDLE * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Real Funding</span>
                            <span class="metric-value">${data.USE_REAL_FUNDING ? 'Enabled ✓' : 'Disabled'}</span>
                        </div>
                    `;
                })
                .catch(() => {
                    document.getElementById('config-content').innerHTML =
                        '<p style="color: #666;">Config unavailable</p>';
                });
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
