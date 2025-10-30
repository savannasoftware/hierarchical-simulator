# Hierarchical Data Simulator - AI Coding Agent Instructions

This is a **scientific simulation library** for generating multilevel/hierarchical data for statistical research. The codebase follows a modular factory pattern with strict separation between outcome types, link functions, and data generation strategies.

## Architecture Overview

**Core Pattern**: The library uses a **factory-based generator system** where each outcome type (continuous, binary, count, survival) has its own specialized generator implementing `AbstractOutcomeGenerator`. The `HierarchicalDataSimulator` orchestrates these through `GeneratorFactory`.

**Key Components**:

- `hierarchical_simulator/core/`: Base abstractions, parameter validation, type definitions
- `hierarchical_simulator/generators/`: Outcome-specific generators (binary.py, continuous.py, etc.)
- `hierarchical_simulator/simulation/simulator.py`: Main orchestration class
- `hierarchical_simulator/link_functions/`: Link function implementations (logit, probit, etc.)
- Root `__init__.py`: Convenience functions (`simulate_continuous_data()`, etc.) that wrap the core API

## Critical Patterns

### Parameter Configuration

All simulations use `SimulationParameters` dataclass with **cached computed values** (`_n_j_cached`, `_random_effect_cached`). Parameters are validated in `__post_init__()` and should never be modified after creation.

**CRITICAL**: Parameters are **mutable and stateful** - each simulation call modifies the internal random number generator state, breaking reproducibility.

```python
# Wrong - will not be reproducible across multiple calls
params = simulator.create_default_params(OutcomeType.BINARY, {"random_seed": 42})
data1 = simulator.simulate_data(params)  # Uses seed 42
data2 = simulator.simulate_data(params)  # Uses different random state!

# Correct - create fresh parameters for each simulation
params1 = simulator.create_default_params(OutcomeType.BINARY, {"random_seed": 42})
params2 = simulator.create_default_params(OutcomeType.BINARY, {"random_seed": 42})
data1 = simulator.simulate_data(params1)  # Uses seed 42
data2 = simulator.simulate_data(params2)  # Also uses seed 42

# Also wrong - bypasses validation
params.n_groups = 20
```

### Generator Registration

The factory auto-registers generators on first use via `_ensure_initialized()`. When adding new outcome types:

1. Implement `AbstractOutcomeGenerator`
2. Add to `OutcomeType` enum
3. Register in `GeneratorFactory._register_default_generators()`

### Data Output Schema

All generators must return DataFrames with **consistent column structure**:

- `group`: Group identifier (1, 2, 3, ...)
- `observation`: Within-group observation number
- `predictor`: Predictor variable value
- `linear_predictor`: Linear predictor (before link transformation)
- `true_beta_0`: Group-specific intercept
- `true_beta_1`: Group-specific slope
- `outcome`: Final generated outcome

## Development Workflows

### Adding New Outcome Types

1. Create generator class in `hierarchical_simulator/generators/`
2. Implement required abstract methods: `generate_outcome()`, `validate_params()`, `get_default_params()`
3. Add enum value to `OutcomeType` in `types.py`
4. Register in factory and add convenience function to root `__init__.py`

### Build System & Dependencies

**Poetry Setup**: This project uses Poetry for dependency management and packaging.

```bash
# Install dependencies (includes test dependencies in groups)
poetry install

# Install with specific groups
poetry install --with test,dev,docs

# Add new dependencies
poetry add numpy>=1.20.0
poetry add --group test pytest>=6.0
poetry add --group dev black>=22.0

# Run commands in poetry environment
poetry run pytest
poetry run python -m hierarchical_simulator
```

### Testing Strategy

**Framework**: Poetry manages test dependencies in `tool.poetry.group.test.dependencies`.

```bash
# Run tests with coverage
poetry run pytest --cov=hierarchical_simulator

# Install and run specific test groups
poetry install --with test
poetry run pytest tests/
```

**Testing Patterns for Scientific Code**:

- **Parameter Validation**: Test all boundary conditions for statistical parameters
- **Output Schema**: Verify DataFrame columns and types match specification
- **Statistical Properties**: Use hypothesis for property-based testing of distributional properties
- **Range Constraints**: Test all truncation methods preserve intended behavior
- **Reproducibility**: **CRITICAL** - Always create fresh `SimulationParameters` for each test due to stateful RNG

**Test Structure Created**:

```
tests/
├── conftest.py                    # Common fixtures and setup
├── test_convenience_functions.py  # Test main API functions
├── test_generators/
│   └── test_continuous.py        # Outcome-specific tests
└── test_simulation/
    └── test_simulator.py         # Core simulator tests
```

### Link Function Usage

**When to Use Each Link Function**:

- `LOGIT`: Binary outcomes (most common), bounded (0,1) predictions
- `PROBIT`: Binary outcomes when normal CDF assumption preferred
- `CLOGLOG`: Binary outcomes with asymmetric relationship, rare events
- `LOG`: Count outcomes, ensures positive predictions
- `IDENTITY`: Continuous outcomes, direct linear relationship

**Adding Custom Link Functions**:

1. Implement in `link_functions/implementations.py`
2. Add enum value to `LinkFunction`
3. Update generator validation to accept new link function

### Performance vs. Accuracy Trade-offs

**Critical Performance Areas**:

- **Range truncation**: `"resample"` method can be 10x slower for extreme constraints
- **Large group counts**: Memory usage scales with `n_groups * max(size_range)`
- **Random effects generation**: Cached in parameters, avoid recomputation

**When Performance Matters**:

- Monte Carlo studies (thousands of simulations)
- Interactive applications (real-time parameter changes)
- Large-scale data generation (>100k observations)

### VS Code Configuration

**Complete IDE Setup**: `.vscode/` directory includes all necessary configuration for seamless development.

**Key Features**:

- **Auto-import resolution** via `python.analysis.extraPaths` and `PYTHONPATH` settings
- **Test discovery** with pytest integration and debug configurations
- **Code formatting** with Black (line-length 88)
- **Type checking** with Pylance/Pyright (basic mode)
- **Debugging support** with proper environment variables

**Available Tasks** (Ctrl+Shift+P → "Run Task"):

- `pytest: Run all tests` - Execute full test suite
- `pytest: Run with coverage` - Generate coverage reports
- `poetry: Install dependencies` - Set up development environment
- `black: Format code` - Auto-format all Python files

**Debug Configurations**:

- `Python: Current File` - Debug any Python file
- `Python: Test Current File` - Debug specific test file
- `Python: Run All Tests` - Debug full test suite
- `Python: Run with Poetry` - Use Poetry virtual environment

### Documentation Builds

Sphinx documentation in `docs/` with autodoc. Build locally with `make html` in docs directory. Configuration supports both local and ReadtheDocs builds.

## Project-Specific Conventions

### Import Structure

- Always import from package root: `from hierarchical_simulator import simulate_binary_data`
- Internal imports use relative paths: `from ..core.types import OutcomeType`
- Circular import prevention: Use `TYPE_CHECKING` for type hints

### Statistical Accuracy

This is a **research tool** - statistical correctness is paramount:

- Always use proper link functions for outcome types (logit for binary, log for count)
- Respect parameter constraints (e.g., tau values for covariance matrices)
- Implement truncation methods correctly to preserve distributional properties

### Range Constraints

Continuous outcomes support `outcome_range` with three truncation methods:

- `"clip"`: Fast but changes distribution shape
- `"reflect"`: Moderate performance, preserves some properties
- `"resample"`: Slow but preserves distribution (use for research)

## Extension Points

### Common Extension Patterns

**Custom Validators**: Extend `ParameterValidator` in `utils/validators.py` for domain-specific constraints
**New Link Functions**: Add to `link_functions/implementations.py` with proper inverse and derivative methods  
**Custom Generators**: Implement `AbstractOutcomeGenerator` for specialized outcome types (e.g., zero-inflated, hurdle models)
**Truncation Methods**: Add new methods in continuous generator for specialized range handling

### Import Error Resolution

**Package Structure**: The package follows standard Poetry library structure with modules directly in `hierarchical_simulator/`.

**REORGANIZED**: Moved from `hierarchical_simulator/src/` to standard Poetry structure `hierarchical_simulator/` to follow Python packaging conventions.

**Import Patterns**:

- **Within package**: Use relative imports (`from ..core.types import OutcomeType`)
- **From root**: Use absolute imports (`from hierarchical_simulator.core.types import OutcomeType`)
- **Main **init**.py**: Located in root, imports from `hierarchical_simulator.*`

**Common Issues**:

- **Circular imports**: Use `TYPE_CHECKING` blocks for type hints only
- **Missing **init**.py**: Ensure all directories have proper package initialization
- **Wrong import paths**: Check if importing from `src/` subdirectory vs root package

**Debugging Import Issues**:

```python
# Add to problematic files to trace import paths
import sys
print("Python path:", sys.path)
print("Current module:", __name__)

# Test imports from project root
cd /path/to/hierachical-simulator
python -c "import sys; sys.path.insert(0, '.'); from hierarchical_simulator.core.types import OutcomeType; print('Success')"
```

## Quick Start for Development

**Poetry Environment** (Recommended):

```bash
# Set up development environment
poetry install --with test,dev,docs

# Run code in poetry environment
poetry run python -c "
from hierarchical_simulator.simulation.simulator import HierarchicalDataSimulator
from hierarchical_simulator.core.types import OutcomeType
simulator = HierarchicalDataSimulator()
params = simulator.create_default_params(OutcomeType.CONTINUOUS)
data = simulator.simulate_data(params)
print(f'Generated {len(data)} observations')
"

# Test all functionality
poetry run pytest
```

**Direct Development** (without Poetry):

```python
# Add project to path for development
import sys
sys.path.insert(0, '/path/to/hierachical-simulator')

# Use convenience functions (most common)
from __init__ import simulate_continuous_data, OutcomeType

# Use core classes directly (advanced usage)
from hierarchical_simulator.simulation.simulator import HierarchicalDataSimulator

# Test that everything works
data, sim = simulate_continuous_data(n_groups=2, size_range=(5,10))
print(f"Generated {len(data)} observations")
```

When working on this codebase, prioritize statistical validity over performance optimizations, and always maintain the consistent DataFrame output schema across all generators.
