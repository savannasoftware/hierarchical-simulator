"""
FastAPI application for hierarchical data generation.

This API provides endpoints for generating hierarchical/multilevel data
across different domains (continuous, binary, count, survival) with
customizable parameters.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import pandas as pd
import io
import json
from datetime import datetime
import uuid

from hierarchical_simulator import (
    quick_simulate,
    simulate_continuous_data,
    simulate_binary_data,
    simulate_count_data,
    simulate_survival_data,
)
from hierarchical_simulator.core.types import OutcomeType, LinkFunction
from hierarchical_simulator.core.parameters import SimulationParameters


# Initialize FastAPI app
app = FastAPI(
    title="Hierarchical Data Simulator API",
    description="Generate realistic hierarchical/multilevel data for research and testing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class OutcomeTypeEnum(str, Enum):
    """Supported outcome types."""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    COUNT = "count"
    SURVIVAL = "survival"


class LinkFunctionEnum(str, Enum):
    """Supported link functions."""
    IDENTITY = "identity"
    LOGIT = "logit"
    PROBIT = "probit"
    LOG = "log"
    CLOGLOG = "cloglog"


class FileFormatEnum(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"


class QuickSimulationRequest(BaseModel):
    """Request model for quick simulation."""
    outcome_type: OutcomeTypeEnum = Field(..., description="Type of outcome variable")
    n_groups: int = Field(30, ge=2, le=1000, description="Number of groups")
    size_range: tuple[int, int] = Field((20, 50), description="Range for group sizes")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    @validator('size_range')
    def validate_size_range(cls, v):
        if v[0] >= v[1]:
            raise ValueError("size_range[0] must be less than size_range[1]")
        if v[0] < 1:
            raise ValueError("size_range must have positive values")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "outcome_type": "continuous",
                "n_groups": 30,
                "size_range": [20, 50],
                "random_seed": 42
            }
        }


class DetailedSimulationRequest(BaseModel):
    """Request model for detailed simulation with full parameter control."""
    outcome_type: OutcomeTypeEnum
    link_function: LinkFunctionEnum
    
    # Fixed effects
    gamma_00: float = Field(2.0, description="Population mean intercept")
    gamma_10: float = Field(0.5, description="Population mean slope")
    
    # Random effects
    tau_00: float = Field(1.0, ge=0, description="SD of group intercepts")
    tau_11: float = Field(0.3, ge=0, description="SD of group slopes")
    tau_01: float = Field(0.1, ge=-1, le=1, description="Correlation between intercepts and slopes")
    
    # Outcome-specific parameters
    sigma: Optional[float] = Field(None, ge=0, description="Within-group noise SD (continuous)")
    dispersion: Optional[float] = Field(None, ge=0, description="Dispersion parameter (count)")
    
    # Simulation structure
    n_groups: int = Field(30, ge=2, le=1000)
    size_range: tuple[int, int] = Field((20, 50))
    predictor_range: tuple[float, float] = Field((0.0, 1.0))
    random_seed: Optional[int] = Field(None)
    
    # Additional parameters for specific outcome types
    extra_params: Optional[Dict[str, Any]] = Field(None)
    
    @validator('size_range')
    def validate_size_range(cls, v):
        if v[0] >= v[1]:
            raise ValueError("size_range[0] must be less than size_range[1]")
        return v
    
    @validator('predictor_range')
    def validate_predictor_range(cls, v):
        if v[0] >= v[1]:
            raise ValueError("predictor_range[0] must be less than predictor_range[1]")
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
            }
        }


class SimulationResponse(BaseModel):
    """Response model for simulation metadata."""
    simulation_id: str
    outcome_type: str
    n_groups: int
    total_observations: int
    timestamp: str
    random_seed: Optional[int]
    parameters: Dict[str, Any]


class DataPreviewResponse(BaseModel):
    """Response with data preview."""
    metadata: SimulationResponse
    preview: List[Dict[str, Any]]
    columns: List[str]
    preview_rows: int
    total_rows: int


# ============================================================================
# Helper Functions
# ============================================================================

def convert_outcome_type(outcome_type: OutcomeTypeEnum) -> OutcomeType:
    """Convert string enum to OutcomeType."""
    mapping = {
        OutcomeTypeEnum.CONTINUOUS: OutcomeType.CONTINUOUS,
        OutcomeTypeEnum.BINARY: OutcomeType.BINARY,
        OutcomeTypeEnum.COUNT: OutcomeType.COUNT,
        OutcomeTypeEnum.SURVIVAL: OutcomeType.SURVIVAL,
    }
    return mapping[outcome_type]


def convert_link_function(link_function: LinkFunctionEnum) -> LinkFunction:
    """Convert string enum to LinkFunction."""
    mapping = {
        LinkFunctionEnum.IDENTITY: LinkFunction.IDENTITY,
        LinkFunctionEnum.LOGIT: LinkFunction.LOGIT,
        LinkFunctionEnum.PROBIT: LinkFunction.PROBIT,
        LinkFunctionEnum.LOG: LinkFunction.LOG,
        LinkFunctionEnum.CLOGLOG: LinkFunction.CLOGLOG,
    }
    return mapping[link_function]


def export_dataframe(df: pd.DataFrame, format: FileFormatEnum, filename: str) -> StreamingResponse:
    """Export DataFrame in specified format."""
    
    if format == FileFormatEnum.CSV:
        output = io.StringIO()
        df.to_csv(output, index=False)
        content = output.getvalue()
        media_type = "text/csv"
        
    elif format == FileFormatEnum.JSON:
        output = io.StringIO()
        df.to_json(output, orient="records", indent=2)
        content = output.getvalue()
        media_type = "application/json"
        
    elif format == FileFormatEnum.PARQUET:
        output = io.BytesIO()
        df.to_parquet(output, index=False)
        content = output.getvalue()
        media_type = "application/octet-stream"
        
    elif format == FileFormatEnum.EXCEL:
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        content = output.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    return StreamingResponse(
        io.BytesIO(content.encode() if isinstance(content, str) else content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - serves the web interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return {
        "name": "Hierarchical Data Simulator API",
        "version": "1.0.0",
        "description": "Generate realistic hierarchical/multilevel data",
        "documentation": "/docs",
        "endpoints": {
            "quick_simulate": "/api/v1/simulate/quick",
            "detailed_simulate": "/api/v1/simulate/detailed",
            "preview": "/api/v1/simulate/preview",
            "download": "/api/v1/simulate/download",
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "hierarchical-simulator"
    }


@app.post("/api/v1/simulate/quick", response_model=DataPreviewResponse)
async def quick_simulate_endpoint(request: QuickSimulationRequest):
    """
    Quick simulation with sensible defaults.
    
    Returns a preview of the generated data along with metadata.
    """
    try:
        # Convert outcome type
        outcome_type = convert_outcome_type(request.outcome_type)
        
        # Generate data
        data = quick_simulate(
            outcome_type=outcome_type,
            n_groups=request.n_groups,
            size_range=request.size_range,
            random_seed=request.random_seed,
        )
        
        # Create response
        simulation_id = str(uuid.uuid4())
        
        metadata = SimulationResponse(
            simulation_id=simulation_id,
            outcome_type=request.outcome_type.value,
            n_groups=request.n_groups,
            total_observations=len(data),
            timestamp=datetime.utcnow().isoformat(),
            random_seed=request.random_seed,
            parameters={
                "size_range": list(request.size_range),
                "mode": "quick_simulate"
            }
        )
        
        # Create preview (first 10 rows)
        preview_rows = min(10, len(data))
        preview = data.head(preview_rows).to_dict(orient="records")
        
        return DataPreviewResponse(
            metadata=metadata,
            preview=preview,
            columns=list(data.columns),
            preview_rows=preview_rows,
            total_rows=len(data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}")


@app.post("/api/v1/simulate/detailed", response_model=DataPreviewResponse)
async def detailed_simulate_endpoint(request: DetailedSimulationRequest):
    """
    Detailed simulation with full parameter control.
    
    Allows fine-grained control over all simulation parameters.
    """
    try:
        # Convert enums
        outcome_type = convert_outcome_type(request.outcome_type)
        link_function = convert_link_function(request.link_function)
        
        # Build parameters dictionary
        params_dict = {
            "outcome_type": outcome_type,
            "link_function": link_function,
            "gamma_00": request.gamma_00,
            "gamma_10": request.gamma_10,
            "tau_00": request.tau_00,
            "tau_11": request.tau_11,
            "tau_01": request.tau_01,
            "n_groups": request.n_groups,
            "size_range": request.size_range,
            "predictor_range": request.predictor_range,
            "random_seed": request.random_seed if request.random_seed is not None else 0,
        }
        
        # Add outcome-specific parameters
        if request.sigma is not None:
            params_dict["sigma"] = request.sigma
        if request.dispersion is not None:
            params_dict["dispersion"] = request.dispersion
        if request.extra_params:
            params_dict["extra_params"] = request.extra_params
        
        # Create parameters and generate data
        params = SimulationParameters(**params_dict)
        
        # Use appropriate simulation function based on outcome type
        if request.outcome_type == OutcomeTypeEnum.CONTINUOUS:
            data = simulate_continuous_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                sigma=request.sigma or 1.0,
                n_groups=request.n_groups,
                size_range=request.size_range,
                random_seed=request.random_seed,
            )
        elif request.outcome_type == OutcomeTypeEnum.BINARY:
            data = simulate_binary_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                n_groups=request.n_groups,
                size_range=request.size_range,
                link_function=link_function,
                random_seed=request.random_seed,
            )
        elif request.outcome_type == OutcomeTypeEnum.COUNT:
            data = simulate_count_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                dispersion=request.dispersion,
                n_groups=request.n_groups,
                size_range=request.size_range,
                random_seed=request.random_seed,
            )
        elif request.outcome_type == OutcomeTypeEnum.SURVIVAL:
            data = simulate_survival_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                n_groups=request.n_groups,
                size_range=request.size_range,
                random_seed=request.random_seed,
            )
        
        # Create response
        simulation_id = str(uuid.uuid4())
        
        metadata = SimulationResponse(
            simulation_id=simulation_id,
            outcome_type=request.outcome_type.value,
            n_groups=request.n_groups,
            total_observations=len(data),
            timestamp=datetime.utcnow().isoformat(),
            random_seed=request.random_seed,
            parameters=request.dict(exclude={'extra_params'})
        )
        
        # Create preview
        preview_rows = min(10, len(data))
        preview = data.head(preview_rows).to_dict(orient="records")
        
        return DataPreviewResponse(
            metadata=metadata,
            preview=preview,
            columns=list(data.columns),
            preview_rows=preview_rows,
            total_rows=len(data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}")


@app.post("/api/v1/simulate/download/{format}")
async def download_simulation(
    format: FileFormatEnum,
    request: DetailedSimulationRequest,
):
    """
    Generate and download data in specified format.
    
    Supported formats: csv, json, parquet, excel
    """
    try:
        # Generate data (reuse detailed simulation logic)
        outcome_type = convert_outcome_type(request.outcome_type)
        link_function = convert_link_function(request.link_function)
        
        # Use appropriate simulation function
        if request.outcome_type == OutcomeTypeEnum.CONTINUOUS:
            data = simulate_continuous_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                sigma=request.sigma or 1.0,
                n_groups=request.n_groups,
                size_range=request.size_range,
                random_seed=request.random_seed,
            )
        elif request.outcome_type == OutcomeTypeEnum.BINARY:
            data = simulate_binary_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                n_groups=request.n_groups,
                size_range=request.size_range,
                link_function=link_function,
                random_seed=request.random_seed,
            )
        elif request.outcome_type == OutcomeTypeEnum.COUNT:
            data = simulate_count_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                dispersion=request.dispersion,
                n_groups=request.n_groups,
                size_range=request.size_range,
                random_seed=request.random_seed,
            )
        elif request.outcome_type == OutcomeTypeEnum.SURVIVAL:
            data = simulate_survival_data(
                gamma_00=request.gamma_00,
                gamma_10=request.gamma_10,
                tau_00=request.tau_00,
                tau_11=request.tau_11,
                tau_01=request.tau_01,
                n_groups=request.n_groups,
                size_range=request.size_range,
                random_seed=request.random_seed,
            )
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"hierarchical_data_{request.outcome_type.value}_{timestamp}.{format.value}"
        
        # Export data
        return export_dataframe(data, format, filename)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")


@app.get("/api/v1/parameters/defaults/{outcome_type}")
async def get_default_parameters(outcome_type: OutcomeTypeEnum):
    """
    Get default parameters for a specific outcome type.
    
    Useful for understanding what parameters are available.
    """
    defaults = {
        OutcomeTypeEnum.CONTINUOUS: {
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
        },
        OutcomeTypeEnum.BINARY: {
            "link_function": "logit",
            "gamma_00": 0.0,
            "gamma_10": 0.5,
            "tau_00": 1.0,
            "tau_11": 0.3,
            "tau_01": 0.1,
            "n_groups": 30,
            "size_range": [20, 50],
            "predictor_range": [0.0, 1.0],
        },
        OutcomeTypeEnum.COUNT: {
            "link_function": "log",
            "gamma_00": 1.5,
            "gamma_10": 0.5,
            "tau_00": 0.5,
            "tau_11": 0.2,
            "tau_01": 0.0,
            "dispersion": 1.0,
            "n_groups": 30,
            "size_range": [20, 50],
            "predictor_range": [0.0, 1.0],
        },
        OutcomeTypeEnum.SURVIVAL: {
            "link_function": "log",
            "gamma_00": 2.0,
            "gamma_10": -0.5,
            "tau_00": 0.5,
            "tau_11": 0.2,
            "tau_01": 0.0,
            "n_groups": 30,
            "size_range": [20, 50],
            "predictor_range": [0.0, 1.0],
        },
    }
    
    return {
        "outcome_type": outcome_type.value,
        "default_parameters": defaults[outcome_type],
        "description": f"Default parameters for {outcome_type.value} outcome type"
    }


@app.get("/api/v1/info/outcome-types")
async def get_outcome_types():
    """Get information about supported outcome types."""
    return {
        "outcome_types": [
            {
                "type": "continuous",
                "description": "Continuous numeric outcomes (e.g., test scores, measurements)",
                "link_functions": ["identity"],
                "example_domains": ["education", "healthcare", "psychology"]
            },
            {
                "type": "binary",
                "description": "Binary outcomes (e.g., success/failure, yes/no)",
                "link_functions": ["logit", "probit", "cloglog"],
                "example_domains": ["medical trials", "customer conversion", "pass/fail"]
            },
            {
                "type": "count",
                "description": "Count data (e.g., number of events, frequencies)",
                "link_functions": ["log"],
                "example_domains": ["epidemiology", "customer transactions", "incident reporting"]
            },
            {
                "type": "survival",
                "description": "Time-to-event data (e.g., duration until outcome)",
                "link_functions": ["log"],
                "example_domains": ["clinical trials", "customer churn", "equipment failure"]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
