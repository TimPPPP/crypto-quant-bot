# Crypto Quant Bot API

RESTful API and web dashboard for monitoring backtest results, bot performance, and system configuration.

## 🚀 Quick Start

### Using Docker Compose

```bash
# Start all services (includes API, QuestDB, Grafana, Jupyter)
docker-compose up -d

# Start only the API service
docker-compose up -d api

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Local Development

```bash
# Install dependencies
poetry install

# Run the API server
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
poetry run python -m src.api.main
```

## 📊 Access Points

Once running, access:

- **Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/api/health

Other services:
- **QuestDB Console**: http://localhost:9000
- **Grafana**: http://localhost:3000
- **Jupyter Lab**: http://localhost:8888

## 📡 API Endpoints

### Health & Status

#### `GET /api/health`
System health check and dependency status.

**Response:**
```json
{
  "status": "online",
  "version": "1.2.0",
  "questdb_status": "online",
  "data_snapshot": "hyperliquid_2025_01_snapshot"
}
```

#### `GET /api/ping`
Simple ping/pong endpoint.

---

### Backtest Results

#### `GET /api/backtest/runs`
List all available backtest runs.

**Response:**
```json
{
  "total_runs": 3,
  "runs": [
    {
      "run_id": "run_20231213_143022",
      "path": "/app/results/run_20231213_143022",
      "timestamp": "20231213_143022"
    }
  ]
}
```

#### `GET /api/backtest/latest`
Get the most recent backtest run with full metrics.

**Response:**
```json
{
  "run_id": "run_20231213_143022",
  "manifest": { ... },
  "metrics": {
    "optimistic": {
      "total_return": 0.156,
      "sharpe": 2.34,
      "max_drawdown": 0.08,
      ...
    },
    "base_case": { ... },
    "stress": { ... }
  },
  "scenario_specs": [ ... ]
}
```

#### `GET /api/backtest/run/{run_id}`
Get detailed information about a specific run.

**Parameters:**
- `run_id`: Run identifier (e.g., "run_20231213_143022")

**Response:**
```json
{
  "run_id": "run_20231213_143022",
  "manifest": { ... },
  "metrics": { ... },
  "plots": ["equity_curves.png", "drawdowns.png", "returns_hist.png"],
  "plots_url": "/static/run_20231213_143022/plots"
}
```

#### `GET /api/backtest/run/{run_id}/plot/{plot_name}`
Get a specific plot image.

**Parameters:**
- `run_id`: Run identifier
- `plot_name`: Plot filename (e.g., "equity_curves.png")

**Returns:** Image file (PNG)

#### `GET /api/backtest/compare?run_ids={ids}`
Compare metrics across multiple runs.

**Parameters:**
- `run_ids`: Comma-separated run IDs (e.g., "run_001,run_002,run_003")

**Response:**
```json
{
  "total_runs": 3,
  "comparison": [
    {
      "run_id": "run_001",
      "metrics": { ... }
    },
    ...
  ]
}
```

---

### Configuration

#### `GET /api/config`
Get all backtest configuration parameters.

**Response:**
```json
{
  "ENTRY_Z": 2.3,
  "EXIT_Z": 0.6,
  "STOP_LOSS_Z": 4.0,
  "MIN_PROFIT_HURDLE": 0.003,
  "USE_REAL_FUNDING": true,
  ...
}
```

#### `GET /api/config/strategy`
Get strategy-specific parameters grouped by category.

**Response:**
```json
{
  "kalman": {
    "delta": 1e-6,
    "r": 0.01
  },
  "thresholds": {
    "entry_z": 2.3,
    "exit_z": 0.6,
    "stop_loss_z": 4.0
  },
  "filters": {
    "expected_revert_mult": 0.75,
    "min_profit_hurdle": 0.003,
    "min_half_life_bars": 40,
    "max_trades_per_pair": 20
  },
  "costs": {
    "fee_rate": 0.0005,
    "slippage_model": "fixed",
    "slippage_bps": 2.0
  },
  "funding": {
    "use_real_funding": true,
    "funding_drag_base_daily": 0.0001,
    "funding_drag_stress_daily": 0.0003
  }
}
```

#### `GET /api/config/paths`
Get configured file paths.

**Response:**
```json
{
  "project_root": "/app",
  "data_dir": "/app/data",
  "results_dir": "/app/results",
  "raw_parquet": "/app/data/raw_downloads/crypto_prices_1m.parquet",
  "funding_parquet": "/app/data/raw_downloads/funding_rates.parquet"
}
```

---

## 🎨 Dashboard Features

The web dashboard at http://localhost:8000 provides:

1. **System Status Card**
   - API status
   - Version info
   - QuestDB connectivity

2. **Latest Backtest Results**
   - Most recent run ID
   - Key metrics (Sharpe, Return, Drawdown)
   - Quick performance overview

3. **Configuration Summary**
   - Entry/Exit Z-scores
   - Profit hurdle
   - Real funding status

4. **Quick Links**
   - API documentation
   - All backtest runs
   - Grafana dashboards
   - Jupyter notebooks
   - QuestDB console

---

## 🔧 Development

### Project Structure

```
src/api/
├── __init__.py
├── main.py              # FastAPI app + dashboard HTML
├── README.md            # This file
└── routers/
    ├── __init__.py
    ├── health.py        # Health check endpoints
    ├── backtest.py      # Backtest results endpoints
    └── config.py        # Configuration endpoints
```

### Adding New Endpoints

1. Create a new router in `src/api/routers/`
2. Define your endpoints using FastAPI decorators
3. Import and include in `src/api/main.py`:

```python
from src.api.routers import my_router

app.include_router(my_router.router, prefix="/api", tags=["my_tag"])
```

### CORS Configuration

CORS is currently configured to allow all origins for development. For production, update in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 🐳 Docker Configuration

The API service in `docker-compose.yml`:

```yaml
api:
  build: .
  container_name: crypto_api
  command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  ports:
    - "8000:8000"
  volumes:
    - ./:/app
  depends_on:
    - db
  environment:
    - PYTHONUNBUFFERED=1
    - QUESTDB_HOST=db
    - PYTHONPATH=/app
  restart: unless-stopped
```

---

## 📝 Usage Examples

### Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/api/health")
print(response.json())

# Get latest backtest
response = requests.get("http://localhost:8000/api/backtest/latest")
metrics = response.json()["metrics"]["base_case"]
print(f"Sharpe Ratio: {metrics['sharpe']}")

# Get configuration
response = requests.get("http://localhost:8000/api/config/strategy")
config = response.json()
print(f"Entry Z: {config['thresholds']['entry_z']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Latest backtest
curl http://localhost:8000/api/backtest/latest

# List all runs
curl http://localhost:8000/api/backtest/runs

# Compare runs
curl "http://localhost:8000/api/backtest/compare?run_ids=run_001,run_002"

# Download plot
curl http://localhost:8000/api/backtest/run/run_20231213_143022/plot/equity_curves.png \
  --output equity_curves.png
```

### JavaScript (Fetch API)

```javascript
// Get latest backtest
fetch('http://localhost:8000/api/backtest/latest')
  .then(res => res.json())
  .then(data => {
    const sharpe = data.metrics.base_case.sharpe;
    console.log(`Sharpe Ratio: ${sharpe.toFixed(2)}`);
  });

// Get configuration
fetch('http://localhost:8000/api/config/strategy')
  .then(res => res.json())
  .then(config => {
    console.log('Strategy Config:', config.thresholds);
  });
```

---

## 🔐 Security Considerations

**For production deployment:**

1. **Environment Variables**: Store sensitive config in env vars, not code
2. **Authentication**: Add API key or JWT authentication
3. **CORS**: Restrict to specific origins
4. **HTTPS**: Use TLS certificates (reverse proxy with nginx/Caddy)
5. **Rate Limiting**: Add rate limiting middleware
6. **Input Validation**: All inputs are validated via Pydantic models

---

## 📈 Monitoring

Monitor API health with:

```bash
# Watch health endpoint
watch -n 5 curl -s http://localhost:8000/api/health | jq

# View logs
docker-compose logs -f api

# Check container status
docker-compose ps api
```

---

## 🐛 Troubleshooting

**API won't start:**
```bash
# Check logs
docker-compose logs api

# Rebuild container
docker-compose build api
docker-compose up -d api
```

**QuestDB connection fails:**
```bash
# Verify QuestDB is running
docker-compose ps db

# Check QuestDB logs
docker-compose logs db

# Restart QuestDB
docker-compose restart db
```

**Port 8000 already in use:**
```bash
# Find process using port
lsof -i :8000

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Host:Container
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [QuestDB Documentation](https://questdb.io/docs/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
