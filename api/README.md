# Hierarchical Data Simulator API

A FastAPI-based web application for generating hierarchical/multilevel data across different domains.

## Features

- 🎲 **Multiple Outcome Types**: Continuous, Binary, Count, and Survival data
- ⚡ **Quick & Detailed Modes**: Simple generation with defaults or full parameter control
- 📊 **Multiple Export Formats**: CSV, JSON, Excel, and Parquet
- 🖥️ **Web Interface**: Beautiful, responsive UI for easy interaction
- 📚 **Interactive API Docs**: Auto-generated Swagger/ReDoc documentation
- 🔒 **Thread-Safe**: Built on thread-safe SimulationParameters
- 🚀 **High Performance**: Async endpoints with streaming responses

## Installation

### 1. Install Dependencies

```bash
# Install the hierarchical-simulator library
cd ..
pip install -e .

# Install API-specific requirements
cd api
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
# Development mode (with auto-reload)
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## Usage

### Web Interface

1. Open http://localhost:8000 in your browser
2. Choose between **Quick Simulate** or **Detailed Parameters**
3. Configure your simulation parameters
4. Click **Generate Data** to create your dataset
5. Preview the data and download in your preferred format

### API Endpoints

#### Quick Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/simulate/quick" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome_type": "continuous",
    "n_groups": 30,
    "size_range": [20, 50],
    "random_seed": 42
  }'
```

#### Detailed Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/simulate/detailed" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome_type": "continuous",
    "link_function": "identity",
    "gamma_00": 2.0,
    "gamma_10": 0.5,
    "tau_00": 1.0,
    "tau_11": 0.3,
    "tau_01": 0.1,
    "sigma": 1.0,
    "n_groups": 30,
    "size_range": [20, 50],
    "predictor_range": [0.0, 1.0],
    "random_seed": 42
  }'
```

#### Download Data

```bash
# Download as CSV
curl -X POST "http://localhost:8000/api/v1/simulate/download/csv" \
  -H "Content-Type: application/json" \
  -d '{...parameters...}' \
  --output data.csv

# Download as JSON
curl -X POST "http://localhost:8000/api/v1/simulate/download/json" \
  -H "Content-Type: application/json" \
  -d '{...parameters...}' \
  --output data.json

# Download as Excel
curl -X POST "http://localhost:8000/api/v1/simulate/download/excel" \
  -H "Content-Type: application/json" \
  -d '{...parameters...}' \
  --output data.xlsx
```

#### Get Default Parameters

```bash
curl "http://localhost:8000/api/v1/parameters/defaults/continuous"
curl "http://localhost:8000/api/v1/parameters/defaults/binary"
curl "http://localhost:8000/api/v1/parameters/defaults/count"
curl "http://localhost:8000/api/v1/parameters/defaults/survival"
```

## Outcome Types

### Continuous Data
- **Description**: Numeric outcomes with decimal values
- **Examples**: Test scores, measurements, temperatures
- **Domains**: Education, Healthcare, Psychology

### Binary Data
- **Description**: Success/failure outcomes (0 or 1)
- **Examples**: Pass/fail, yes/no, alive/dead
- **Domains**: Medical trials, Customer conversion, Quality control

### Count Data
- **Description**: Non-negative integer counts
- **Examples**: Number of events, frequencies, occurrences
- **Domains**: Epidemiology, Customer transactions, Incident reporting

### Survival Data
- **Description**: Time-to-event data
- **Examples**: Duration until outcome, time until failure
- **Domains**: Clinical trials, Customer churn, Equipment reliability

## Parameters

### Fixed Effects
- **γ₀₀ (gamma_00)**: Population mean intercept
- **γ₁₀ (gamma_10)**: Population mean slope

### Random Effects
- **τ₀₀ (tau_00)**: Standard deviation of group intercepts
- **τ₁₁ (tau_11)**: Standard deviation of group slopes  
- **τ₀₁ (tau_01)**: Correlation between intercepts and slopes

### Outcome-Specific
- **σ (sigma)**: Within-group noise standard deviation (continuous)
- **dispersion**: Dispersion parameter for overdispersion (count)

### Structure
- **n_groups**: Number of groups/clusters
- **size_range**: Range for observations per group [min, max]
- **predictor_range**: Range for predictor variable values [min, max]
- **random_seed**: Random seed for reproducibility

## Python Client Example

```python
import requests
import pandas as pd

API_URL = "http://localhost:8000"

# Quick simulation
response = requests.post(
    f"{API_URL}/api/v1/simulate/quick",
    json={
        "outcome_type": "continuous",
        "n_groups": 30,
        "size_range": [20, 50],
        "random_seed": 42
    }
)

data = response.json()
print(f"Generated {data['metadata']['total_observations']} observations")
print(f"Preview:\n{pd.DataFrame(data['preview'])}")

# Download full dataset as CSV
response = requests.post(
    f"{API_URL}/api/v1/simulate/download/csv",
    json={
        "outcome_type": "binary",
        "link_function": "logit",
        "gamma_00": 0.0,
        "gamma_10": 0.5,
        "tau_00": 1.0,
        "tau_11": 0.3,
        "tau_01": 0.1,
        "n_groups": 50,
        "size_range": [10, 30],
        "predictor_range": [0.0, 1.0],
        "random_seed": 123
    }
)

with open("binary_data.csv", "wb") as f:
    f.write(response.content)
```

## Production Deployment

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements_api.txt .
COPY ../pyproject.toml ../

# Install dependencies
RUN pip install --no-cache-dir -r requirements_api.txt
RUN pip install --no-cache-dir -e ..

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t hierarchical-simulator-api .
docker run -p 8000:8000 hierarchical-simulator-api
```

### Using Gunicorn (Production)

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

## Configuration

### Environment Variables

```bash
# Set API host and port
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Enable development mode
export API_DEBUG="true"

# CORS settings (comma-separated origins)
export ALLOWED_ORIGINS="http://localhost:3000,https://example.com"
```

## Performance

- **Async Operations**: All endpoints are async for better concurrency
- **Streaming Responses**: Large datasets streamed for memory efficiency
- **Thread-Safe**: Underlying SimulationParameters uses RLock for thread safety
- **Validated Inputs**: Pydantic models ensure data integrity

## API Response Examples

### Quick Simulate Response

```json
{
  "metadata": {
    "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
    "outcome_type": "continuous",
    "n_groups": 30,
    "total_observations": 1050,
    "timestamp": "2025-11-13T10:30:00.000Z",
    "random_seed": 42,
    "parameters": {
      "size_range": [20, 50],
      "mode": "quick_simulate"
    }
  },
  "preview": [
    {"group": 0, "predictor": 0.234, "outcome": 2.145},
    {"group": 0, "predictor": 0.567, "outcome": 2.389}
  ],
  "columns": ["group", "predictor", "outcome"],
  "preview_rows": 10,
  "total_rows": 1050
}
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Import Errors

Ensure the hierarchical-simulator library is installed:

```bash
cd ..
pip install -e .
```

### CORS Issues

Update the CORS configuration in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## License

MIT License - See parent directory for details.

## Support

- **Documentation**: http://localhost:8000/docs
- **Repository**: https://github.com/savannasoftware/hierarchical-simulator
- **Issues**: https://github.com/savannasoftware/hierarchical-simulator/issues
