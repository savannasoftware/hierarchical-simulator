# 🎲 Hierarchical Data Simulator - FastAPI Application

## 🚀 Quick Start Guide

### Installation & Startup

```bash
# Navigate to API directory
cd api

# Option 1: Use startup script (recommended)
./start.sh

# Option 2: Manual start
pip install -r requirements_api.txt
python main.py
```

### Access Points

- **🖥️ Web Interface**: http://localhost:8000
- **📚 API Docs**: http://localhost:8000/docs
- **🏥 Health Check**: http://localhost:8000/api/v1/health

---

## 📊 Features Overview

### 1. **Quick Simulate** (Simple & Fast)
Generate data quickly with minimal configuration:
- Choose outcome type (Continuous/Binary/Count/Survival)
- Set number of groups
- Define group size range
- Optional: Add random seed for reproducibility

### 2. **Detailed Parameters** (Full Control)
Fine-tune every aspect:
- **Fixed Effects**: γ₀₀ (intercept), γ₁₀ (slope)
- **Random Effects**: τ₀₀, τ₁₁, τ₀₁ (variances & correlation)
- **Outcome-Specific**: σ (residual SD), dispersion parameter
- **Structure**: Groups, size ranges, predictor ranges

### 3. **Multiple Export Formats**
Download your data in:
- 📄 **CSV**: Standard comma-separated values
- 📋 **JSON**: Structured data format
- 📊 **Excel**: Microsoft Excel format (.xlsx)
- 🗄️ **Parquet**: Compressed columnar format

---

## 🎯 Outcome Types Supported

| Type | Description | Example Domains |
|------|-------------|-----------------|
| **📊 Continuous** | Numeric with decimals | Test scores, measurements, temperatures |
| **🔢 Binary** | 0 or 1 outcomes | Pass/fail, yes/no, alive/dead |
| **🔢 Count** | Non-negative integers | Event counts, frequencies, occurrences |
| **⏱️ Survival** | Time-to-event | Duration, time-to-failure, churn |

---

## 🌐 API Endpoints

### Core Simulation
```
POST /api/v1/simulate/quick
POST /api/v1/simulate/detailed
POST /api/v1/simulate/download/{format}
```

### Information
```
GET /api/v1/parameters/defaults/{outcome_type}
GET /api/v1/info/outcome-types
GET /api/v1/health
```

### Documentation
```
GET /docs          # Swagger UI
GET /redoc         # ReDoc
GET /              # Web Interface
```

---

## 💻 Example Usage

### Web Interface
1. Open http://localhost:8000
2. Select **Quick Simulate** or **Detailed Parameters**
3. Configure your parameters
4. Click **Generate Data**
5. Preview results
6. Download in your preferred format

### Python Client
```python
import requests

# Quick simulation
response = requests.post(
    "http://localhost:8000/api/v1/simulate/quick",
    json={
        "outcome_type": "continuous",
        "n_groups": 30,
        "size_range": [20, 50],
        "random_seed": 42
    }
)

data = response.json()
print(f"Generated {data['metadata']['total_observations']} observations")
```

### cURL
```bash
# Download data as CSV
curl -X POST "http://localhost:8000/api/v1/simulate/download/csv" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome_type": "binary",
    "link_function": "logit",
    "gamma_00": 0.0,
    "gamma_10": 0.5,
    "tau_00": 1.0,
    "tau_11": 0.3,
    "tau_01": 0.1,
    "n_groups": 50,
    "size_range": [10, 30],
    "random_seed": 123
  }' --output data.csv
```

---

## 🧪 Testing

### Run Test Suite
```bash
# Start the API server first
python main.py

# In another terminal, run tests
python test_client.py
```

### Test Coverage
- ✅ Health check endpoint
- ✅ Quick simulation
- ✅ Detailed simulation with all parameters
- ✅ Data download in multiple formats
- ✅ Default parameters retrieval
- ✅ Outcome types information

---

## 🔧 Configuration

### Port & Host
```python
# In main.py (bottom)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### CORS Settings
```python
# In main.py (after app initialization)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run
```bash
docker build -t hierarchical-simulator-api .
docker run -p 8000:8000 hierarchical-simulator-api
```

---

## 📦 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Using systemd (Linux)
```ini
[Unit]
Description=Hierarchical Simulator API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/api
ExecStart=/usr/bin/gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

---

## 🎨 Web Interface Features

### Design
- ✨ Modern gradient aesthetics
- 📱 Fully responsive design
- 🎯 Tab-based navigation
- ⚡ Real-time validation
- 🔄 Loading indicators
- ❌ Error handling with user-friendly messages

### User Experience
- **Quick Tab**: Get started in seconds
- **Detailed Tab**: Full parameter control
- **Info Tab**: Learn about outcome types
- **Preview**: See first 10 rows with metadata
- **Download**: One-click export in multiple formats

---

## 🔒 Thread Safety

Built on the thread-safe `SimulationParameters` class:
- ✅ Concurrent request handling
- ✅ Parameter validation with rollback
- ✅ Protected cached values
- ✅ Atomic multi-parameter operations

---

## 📈 Performance

- **Async Operations**: All endpoints are async
- **Streaming Responses**: Memory-efficient for large datasets
- **Validated Inputs**: Pydantic models ensure data integrity
- **Efficient Serialization**: Optimized for JSON/CSV/Parquet

---

## 🆘 Troubleshooting

### Port Already in Use
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

### Import Errors
```bash
cd ..
pip install -e .  # Install hierarchical-simulator library
```

### Dependencies Missing
```bash
pip install -r requirements_api.txt
```

---

## 📚 Additional Resources

- **Full Documentation**: See `api/README.md`
- **API Docs**: http://localhost:8000/docs
- **Library Docs**: https://hierarchical-simulator.readthedocs.io
- **Repository**: https://github.com/savannasoftware/hierarchical-simulator

---

## 🎉 Success!

Your FastAPI application is ready! The complete web interface allows users to:
1. **Generate** hierarchical data across multiple domains
2. **Customize** parameters for specific research needs
3. **Preview** data before downloading
4. **Export** in multiple formats (CSV, JSON, Excel, Parquet)
5. **Access** via web interface or programmatic API

**Start the server and visit http://localhost:8000 to try it out!** 🚀
