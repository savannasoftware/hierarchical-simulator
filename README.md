# Comprehensive Documentation: Simulate Functions

## Overview

The hierarchical data simulator provides four convenient functions for simulating multilevel data across different outcome types. These functions offer an intuitive interface for generating realistic hierarchical data with proper range constraints.

## Table of Contents

1. [simulate_continuous_data()](#simulate_continuous_data)
2. [simulate_binary_data()](#simulate_binary_data)
3. [simulate_count_data()](#simulate_count_data)
4. [simulate_survival_data()](#simulate_survival_data)
5. [Common Parameters](#common-parameters)
6. [Real-World Examples](#real-world-examples)
7. [Best Practices](#best-practices)
8. [Parameter Interpretation Guide](#parameter-interpretation-guide)

---

## simulate_continuous_data()

Simulates continuous (Gaussian) hierarchical data with optional range constraints.

### Function Signature

```python
simulate_continuous_data(
    n_groups: int = 10,
    size_range: tuple = (15, 40),
    gamma: tuple = (0.0, 2.0),
    tau: tuple = (1.0, 0.5, 0.0),
    sigma: float = 1.0,
    predictor_range: tuple = (0.0, 1.0),
    outcome_range: tuple = None,
    truncation_method: str = "clip",
    random_seed: int = 0,
    **kwargs
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_groups` | int | 10 | Number of groups (e.g., schools, clinics, sites) |
| `size_range` | tuple | (15, 40) | Range of group sizes (min_size, max_size) |
| `gamma` | tuple | (0.0, 2.0) | Fixed effects (intercept, slope) |
| `tau` | tuple | (1.0, 0.5, 0.0) | Random effects (tau_00, tau_11, tau_01) |
| `sigma` | float | 1.0 | Within-group error standard deviation |
| `predictor_range` | tuple | (0.0, 1.0) | Range of predictor variable values |
| `outcome_range` | tuple | None | Valid outcome range (min, max). If None, no constraints |
| `truncation_method` | str | "clip" | How to handle out-of-bounds values: "clip", "reflect", "resample" |
| `random_seed` | int | 0 | Random seed for reproducibility |

### Returns

Tuple[pd.DataFrame, HierarchicalDataSimulator] with simulated continuous data
- `group`: Group identifier (1, 2, 3, ...)
- `observation`: Observation within group (1, 2, 3, ...)
- `predictor`: Predictor variable value
- `linear_predictor`: Linear predictor (before adding error)
- `true_beta_0`: True group-specific intercept
- `true_beta_1`: True group-specific slope
- `outcome`: Generated continuous outcome

### Examples

#### Example 1: School Test Scores (0-100 scale)

```python
from hierarchical_simulator import simulate_continuous_data

# School performance across 12 schools
school_scores, _ = simulate_continuous_data(
    n_groups=12,                    # 12 schools
    size_range=(20, 40),           # 20-40 students per school
    gamma=(75.0, 15.0),            # Mean score=75, intervention effect=+15
    tau=(8.0, 4.0, -0.3),          # School variation with negative correlation
    sigma=12.0,                    # Individual student variation
    outcome_range=(0, 100),        # Valid score range
    truncation_method="resample",  # Preserve distribution shape
    predictor_range=(0.0, 1.0),    # 0=control, 1=intervention
    random_seed=42
)

print(f"Generated {len(school_scores)} student scores")
print(f"Score range: {school_scores['outcome'].min():.1f} - {school_scores['outcome'].max():.1f}")
print(f"Mean score: {school_scores['outcome'].mean():.1f}")

# Analyze intervention effect
control_mean = school_scores[school_scores['predictor'] < 0.5]['outcome'].mean()
intervention_mean = school_scores[school_scores['predictor'] > 0.5]['outcome'].mean()
print(f"Control mean: {control_mean:.1f}")
print(f"Intervention mean: {intervention_mean:.1f}")
print(f"Effect size: {intervention_mean - control_mean:.1f} points")
```

#### Example 2: Blood Pressure Study

```python
# Multi-clinic blood pressure treatment study
bp_study, _ = simulate_continuous_data(
    n_groups=8,                     # 8 clinics
    size_range=(30, 50),           # 30-50 patients per clinic
    gamma=(140.0, -18.0),          # Baseline BP=140, treatment effect=-18 mmHg
    tau=(12.0, 5.0, -0.4),         # Clinic variation
    sigma=15.0,                    # Individual patient variation
    outcome_range=(80, 200),       # Physiological BP range
    truncation_method="reflect",   # Reflect extreme values back
    predictor_range=(0.0, 1.0),    # 0=placebo, 1=treatment
    random_seed=123
)

print(f"BP study: {len(bp_study)} patients across {bp_study['group'].nunique()} clinics")
print(f"BP range: {bp_study['outcome'].min():.1f} - {bp_study['outcome'].max():.1f} mmHg")
```

#### Example 3: Psychological Scale (No Range Constraints)

```python
# Depression scale scores (continuous, no natural bounds)
depression_study = simulate_continuous_data(
    n_groups=15,                   # 15 therapy groups
    size_range=(8, 15),           # Smaller therapy groups
    gamma=(25.0, -8.0),           # Baseline depression=25, therapy effect=-8
    tau=(6.0, 3.0, 0.2),          # Therapist variation
    sigma=10.0,                   # Individual variation
    # No outcome_range - depression scales can vary widely
    predictor_range=(0.0, 1.0),   # 0=control, 1=therapy
    random_seed=456
)
```

---

## simulate_binary_data()

Simulates binary (Bernoulli) hierarchical data using logistic, probit, or complementary log-log models.

### Function Signature

```python
simulate_binary_data(
    n_groups: int = 10,
    size_range: tuple = (15, 40),
    gamma: tuple = (0.5, 1.2),
    tau: tuple = (0.8, 0.6, -0.3),
    link_function: LinkFunction = LinkFunction.LOGIT,
    predictor_range: tuple = (0.0, 1.0),
    random_seed: int = 0,
    **kwargs
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_groups` | int | 10 | Number of groups |
| `size_range` | tuple | (15, 40) | Range of group sizes |
| `gamma` | tuple | (0.5, 1.2) | Fixed effects on link scale (intercept, slope) |
| `tau` | tuple | (0.8, 0.6, -0.3) | Random effects (tau_00, tau_11, tau_01) |
| `link_function` | LinkFunction | LOGIT | Link function: LOGIT, PROBIT, or CLOGLOG |
| `predictor_range` | tuple | (0.0, 1.0) | Range of predictor values |
| `random_seed` | int | 0 | Random seed for reproducibility |

### Returns

pandas.DataFrame with columns:
- `group`: Group identifier
- `observation`: Observation within group
- `predictor`: Predictor variable value
- `linear_predictor`: Linear predictor (on link scale)
- `true_beta_0`: True group-specific intercept
- `true_beta_1`: True group-specific slope
- `outcome`: Generated binary outcome (0 or 1)

### Examples

#### Example 1: Medical Treatment Success (Logistic Model)

```python
from hierarchical_simulator import simulate_binary_data, LinkFunction

# Treatment success across hospitals
treatment_success, _ = simulate_binary_data(
    n_groups=10,                   # 10 hospitals
    size_range=(25, 50),          # 25-50 patients per hospital
    gamma=(0.2, 1.5),             # Log-odds: modest baseline, strong treatment effect
    tau=(0.8, 0.6, -0.2),         # Hospital variation
    link_function=LinkFunction.LOGIT,  # Logistic regression
    predictor_range=(0.0, 1.0),   # 0=standard care, 1=new treatment
    random_seed=789
)

print(f"Treatment study: {len(treatment_success)} patients")
print(f"Overall success rate: {treatment_success['outcome'].mean():.1%}")

# Calculate odds ratio
from scipy.stats import contingency_table
import numpy as np

# Create 2x2 contingency table
treatment_group = (treatment_success['predictor'] > 0.5).astype(int)
outcome = treatment_success['outcome']

# Calculate success rates by group
control_success = outcome[treatment_group == 0].mean()
treatment_success_rate = outcome[treatment_group == 1].mean()

print(f"Control success rate: {control_success:.1%}")
print(f"Treatment success rate: {treatment_success_rate:.1%}")

# Calculate odds ratio
control_odds = control_success / (1 - control_success)
treatment_odds = treatment_success_rate / (1 - treatment_success_rate)
odds_ratio = treatment_odds / control_odds
print(f"Odds ratio: {odds_ratio:.2f}")
```

#### Example 2: Educational Graduation Rates (Probit Model)

```python
# High school graduation across school districts
graduation_study, _ = simulate_binary_data(
    n_groups=20,                   # 20 school districts
    size_range=(100, 200),        # Large sample sizes for graduation rates
    gamma=(-0.2, 0.8),            # Probit scale: slightly below average baseline
    tau=(0.5, 0.3, 0.1),          # District variation
    link_function=LinkFunction.PROBIT,  # Probit model (normal CDF)
    predictor_range=(0.0, 1.0),   # 0=standard program, 1=enhanced program
    random_seed=321
)

print(f"Graduation study: {len(graduation_study)} students")
print(f"Overall graduation rate: {graduation_study['outcome'].mean():.1%}")
```

#### Example 3: Equipment Failure (Complementary Log-Log)

```python
# Equipment failure analysis
failure_study, _ = simulate_binary_data(
    n_groups=8,                    # 8 manufacturing plants
    size_range=(50, 80),          # Equipment units per plant
    gamma=(-2.5, 1.2),            # Cloglog scale: low baseline failure rate
    tau=(0.6, 0.4, -0.1),         # Plant variation
    link_function=LinkFunction.CLOGLOG,  # Complementary log-log
    predictor_range=(0.0, 10.0),  # Operating hours (in thousands)
    random_seed=654
)

print(f"Failure study: {len(failure_study)} equipment units")
print(f"Overall failure rate: {failure_study['outcome'].mean():.1%}")
```

---

## simulate_count_data()

Simulates count data using Poisson or Negative-Binomial distributions with optional maximum count constraints.

### Function Signature

```python
simulate_count_data(
    n_groups: int = 8,
    size_range: tuple = (15, 40),
    gamma: tuple = (1.0, -0.5),
    tau: tuple = (0.7, 0.4, 0.2),
    dispersion: float = 1.0,
    link_function: LinkFunction = LinkFunction.LOG,
    predictor_range: tuple = (0.0, 1.0),
    max_count: int = None,
    random_seed: int = 0,
    **kwargs
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_groups` | int | 8 | Number of groups |
| `size_range` | tuple | (15, 40) | Range of group sizes |
| `gamma` | tuple | (1.0, -0.5) | Fixed effects on log-rate scale (intercept, slope) |
| `tau` | tuple | (0.7, 0.4, 0.2) | Random effects (tau_00, tau_11, tau_01) |
| `dispersion` | float | 1.0 | Dispersion parameter (1.0=Poisson, >1.0=Negative-Binomial) |
| `link_function` | LinkFunction | LOG | Link function (LOG or POISSON) |
| `predictor_range` | tuple | (0.0, 1.0) | Range of predictor values |
| `max_count` | int | None | Maximum allowed count. If None, no constraint |
| `random_seed` | int | 0 | Random seed for reproducibility |

### Returns

pandas.DataFrame with columns:
- `group`: Group identifier
- `observation`: Observation within group
- `predictor`: Predictor variable value
- `linear_predictor`: Linear predictor (log-rate scale)
- `true_beta_0`: True group-specific intercept
- `true_beta_1`: True group-specific slope
- `outcome`: Generated count outcome (non-negative integers)

### Examples

#### Example 1: Hospital Daily Admissions (Poisson)

```python
from hierarchical_simulator import simulate_count_data

# Daily hospital admissions across hospitals
hospital_admissions, _ = simulate_count_data(
    n_groups=15,                   # 15 hospitals
    size_range=(30, 40),          # 30-40 days of data per hospital
    gamma=(2.8, -0.3),            # Log-rate: baseline ~16 admissions, seasonal effect
    tau=(0.6, 0.3, 0.1),          # Hospital variation
    dispersion=1.0,               # Pure Poisson (no overdispersion)
    max_count=50,                 # Hospital capacity constraint
    predictor_range=(0.0, 1.0),   # 0=winter, 1=summer (seasonal effect)
    random_seed=987
)

print(f"Hospital admissions: {len(hospital_admissions)} hospital-days")
print(f"Admission range: {hospital_admissions['outcome'].min()} - {hospital_admissions['outcome'].max()}")
print(f"Mean daily admissions: {hospital_admissions['outcome'].mean():.1f}")
print(f"Zero admission days: {(hospital_admissions['outcome'] == 0).sum()}")
print(f"At capacity days: {(hospital_admissions['outcome'] == 50).sum()}")

# Variance-to-mean ratio (should be ~1 for Poisson)
variance_to_mean = hospital_admissions['outcome'].var() / hospital_admissions['outcome'].mean()
print(f"Variance/Mean ratio: {variance_to_mean:.2f} (1.0 = perfect Poisson)")
```

#### Example 2: Disease Cases with Overdispersion (Negative-Binomial)

```python
# Weekly disease cases across regions
disease_surveillance, _ = simulate_count_data(
    n_groups=25,                   # 25 regions
    size_range=(50, 60),          # ~1 year of weekly data per region
    gamma=(1.5, 0.4),             # Log-rate: baseline ~4 cases, increasing trend
    tau=(0.8, 0.5, 0.2),          # Regional variation
    dispersion=2.5,               # Overdispersion (clustering/outbreaks)
    max_count=100,                # Reporting/testing capacity limit
    predictor_range=(0.0, 1.0),   # 0=pre-outbreak, 1=outbreak period
    random_seed=147
)

print(f"Disease surveillance: {len(disease_surveillance)} region-weeks")
print(f"Case range: {disease_surveillance['outcome'].min()} - {disease_surveillance['outcome'].max()}")
print(f"Mean weekly cases: {disease_surveillance['outcome'].mean():.1f}")

# Variance-to-mean ratio (should be >1 for overdispersed data)
variance_to_mean = disease_surveillance['outcome'].var() / disease_surveillance['outcome'].mean()
print(f"Variance/Mean ratio: {variance_to_mean:.2f} (>1 indicates overdispersion)")
```

#### Example 3: Customer Purchases (Small Counts)

```python
# Daily customer purchases across stores
customer_purchases, _ = simulate_count_data(
    n_groups=12,                   # 12 retail stores
    size_range=(20, 30),          # 20-30 days per store
    gamma=(0.8, 0.6),             # Log-rate: low baseline, promotional effect
    tau=(0.4, 0.2, 0.0),          # Store variation
    dispersion=1.2,               # Slight overdispersion
    max_count=20,                 # Daily purchase limit per customer
    predictor_range=(0.0, 1.0),   # 0=regular day, 1=promotional day
    random_seed=258
)

print(f"Customer purchases: {len(customer_purchases)} store-days")
print(f"Purchase range: {customer_purchases['outcome'].min()} - {customer_purchases['outcome'].max()}")
print(f"Mean daily purchases: {customer_purchases['outcome'].mean():.1f}")
```

---

## simulate_survival_data()

Simulates time-to-event (survival) data with exponential survival times and optional time range constraints.

### Function Signature

```python
simulate_survival_data(
    n_groups: int = 9,
    size_range: tuple = (15, 40),
    gamma: tuple = (-2.0, 0.5),
    tau: tuple = (0.5, 0.3, 0.2),
    censoring_time: float = 10.0,
    link_function: LinkFunction = LinkFunction.LOG,
    predictor_range: tuple = (0.0, 1.0),
    time_range: tuple = None,
    random_seed: int = 0,
    **kwargs
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_groups` | int | 9 | Number of groups |
| `size_range` | tuple | (15, 40) | Range of group sizes |
| `gamma` | tuple | (-2.0, 0.5) | Fixed effects on log-hazard scale (intercept, slope) |
| `tau` | tuple | (0.5, 0.3, 0.2) | Random effects (tau_00, tau_11, tau_01) |
| `censoring_time` | float | 10.0 | Mean censoring time (exponential distribution) |
| `link_function` | LinkFunction | LOG | Link function (LOG or CLOGLOG) |
| `predictor_range` | tuple | (0.0, 1.0) | Range of predictor values |
| `time_range` | tuple | None | Valid time range (min_time, max_time). If None, no constraints |
| `random_seed` | int | 0 | Random seed for reproducibility |

### Returns

pandas.DataFrame with columns:
- `group`: Group identifier
- `observation`: Observation within group
- `predictor`: Predictor variable value
- `linear_predictor`: Linear predictor (log-hazard scale)
- `true_beta_0`: True group-specific intercept
- `true_beta_1`: True group-specific slope
- `outcome`: Observed time (either event time or censoring time)
- `event`: Event indicator (1=event occurred, 0=censored)

### Examples

#### Example 1: Cancer Clinical Trial

```python
from hierarchical_simulator import simulate_survival_data

# Multi-center cancer treatment trial
cancer_trial, _ = simulate_survival_data(
    n_groups=8,                    # 8 treatment centers
    size_range=(40, 60),          # 40-60 patients per center
    gamma=(-1.8, 0.6),            # Log-hazard: low baseline rate, treatment benefit
    tau=(0.5, 0.3, 0.1),          # Center variation
    censoring_time=730,           # 2-year follow-up (days)
    time_range=(1, 1095),         # 1 day to 3 years maximum follow-up
    predictor_range=(0.0, 1.0),   # 0=control, 1=treatment
    random_seed=369
)

print(f"Cancer trial: {len(cancer_trial)} patients across {cancer_trial['group'].nunique()} centers")
print(f"Event rate: {cancer_trial['event'].mean():.1%}")
print(f"Median survival time: {cancer_trial['outcome'].median():.1f} days")

# Calculate survival times by treatment group
control_group = cancer_trial[cancer_trial['predictor'] < 0.5]
treatment_group = cancer_trial[cancer_trial['predictor'] > 0.5]

print(f"Control group:")
print(f"  Event rate: {control_group['event'].mean():.1%}")
print(f"  Median time: {control_group['outcome'].median():.1f} days")

print(f"Treatment group:")
print(f"  Event rate: {treatment_group['event'].mean():.1%}")
print(f"  Median time: {treatment_group['outcome'].median():.1f} days")

# Convert to years for interpretation
print(f"Median survival (years): Control={control_group['outcome'].median()/365.25:.1f}, Treatment={treatment_group['outcome'].median()/365.25:.1f}")
```

#### Example 2: Equipment Failure Analysis

```python
# Industrial equipment failure study
equipment_failure, _ = simulate_survival_data(
    n_groups=12,                  # 12 manufacturing facilities
    size_range=(25, 40),          # Equipment units per facility
    gamma=(-3.2, 1.0),            # Log-hazard: very low baseline, usage effect
    tau=(0.6, 0.4, -0.2),         # Facility variation
    censoring_time=60,            # 5-year study period (months)
    time_range=(0.1, 120),        # 0.1 to 10 years operational time
    predictor_range=(0.0, 10.0),  # Usage intensity (0=light, 10=heavy)
    random_seed=741
)

print(f"Equipment study: {len(equipment_failure)} units")
print(f"Failure rate: {equipment_failure['event'].mean():.1%}")
print(f"Median operational time: {equipment_failure['outcome'].median():.1f} months")
print(f"Time range: {equipment_failure['outcome'].min():.1f} - {equipment_failure['outcome'].max():.1f} months")
```

#### Example 3: Employee Turnover Study

```python
# Employee retention across departments
employee_turnover, _ = simulate_survival_data(
    n_groups=15,                  # 15 departments
    size_range=(20, 35),          # Employees per department
    gamma=(-1.5, 0.8),            # Log-hazard: moderate baseline, workload effect
    tau=(0.4, 0.3, 0.1),          # Department variation
    censoring_time=24,            # 2-year study period (months)
    time_range=(1, 60),           # 1 month to 5 years employment
    predictor_range=(0.0, 1.0),   # 0=normal workload, 1=high workload
    random_seed=852
)

print(f"Turnover study: {len(employee_turnover)} employees")
print(f"Turnover rate: {employee_turnover['event'].mean():.1%}")
print(f"Median employment duration: {employee_turnover['outcome'].median():.1f} months")
```

---

## Common Parameters

### Fixed Effects (gamma)

The `gamma` parameter specifies the population-level effects:
- **gamma[0]**: Intercept (baseline level when predictor = 0)
- **gamma[1]**: Slope (effect of one-unit increase in predictor)

**Scale depends on outcome type:**
- **Continuous**: Natural scale of the outcome
- **Binary**: Log-odds scale (logit) or probit scale
- **Count**: Log-rate scale
- **Survival**: Log-hazard scale

### Random Effects (tau)

The `tau` parameter can be specified as:
- **2-tuple**: `(tau_00, tau_11)` - assumes no correlation (`tau_01 = 0`)
- **3-tuple**: `(tau_00, tau_11, tau_01)` - includes correlation

Where:
- **tau_00**: Standard deviation of random intercepts
- **tau_11**: Standard deviation of random slopes  
- **tau_01**: Correlation between random intercepts and slopes (-1 to 1)

### Group Structure

- **n_groups**: Number of higher-level units (schools, hospitals, etc.)
- **size_range**: Range of lower-level units per group (students per school, etc.)

### Predictor Variable

- **predictor_range**: Range of the predictor variable
- Common patterns:
  - `(0, 1)`: Binary treatment/control
  - `(-1, 1)`: Centered continuous variable
  - `(0, 10)`: Dose or intensity measure

---

## Real-World Examples

### Education Research

```python
# Multi-site educational intervention
education_study = simulate_continuous_data(
    n_groups=20,                   # 20 schools
    size_range=(80, 120),         # 80-120 students per school
    gamma=(72.0, 8.0),            # Test scores: baseline=72, intervention=+8 points
    tau=(12.0, 6.0, -0.4),        # School variation with negative correlation
    sigma=18.0,                   # Individual student variation
    outcome_range=(0, 100),       # Standard test score range
    truncation_method="resample", # Preserve distribution shape
    predictor_range=(0.0, 1.0),   # 0=control, 1=intervention
    random_seed=100
)

# Analyze by school effectiveness
school_means = education_study.groupby('group')['outcome'].mean()
print(f"School mean scores range: {school_means.min():.1f} - {school_means.max():.1f}")
```

### Healthcare Analytics

```python
# Hospital readmission rates
readmission_study = simulate_binary_data(
    n_groups=25,                   # 25 hospitals
    size_range=(200, 400),        # Large patient samples
    gamma=(-1.2, 0.6),            # Log-odds: low baseline, risk factor effect
    tau=(0.4, 0.2, 0.0),          # Hospital variation
    link_function=LinkFunction.LOGIT,
    predictor_range=(0.0, 1.0),   # 0=low risk, 1=high risk
    random_seed=200
)

# Hospital-level readmission rates
hospital_rates = readmission_study.groupby('group')['outcome'].mean()
print(f"Hospital readmission rates: {hospital_rates.min():.1%} - {hospital_rates.max():.1%}")
```

### Public Health Surveillance

```python
# Disease outbreak monitoring
outbreak_surveillance = simulate_count_data(
    n_groups=30,                   # 30 geographic regions
    size_range=(40, 60),          # Weekly observations per region
    gamma=(1.2, 1.8),             # Log-rate: baseline cases, outbreak effect
    tau=(0.7, 0.4, 0.3),          # Regional variation
    dispersion=3.0,               # High overdispersion (outbreak clustering)
    max_count=200,                # Testing/reporting capacity
    predictor_range=(0.0, 1.0),   # 0=pre-outbreak, 1=outbreak period
    random_seed=300
)
```

### Clinical Trials

```python
# Time-to-recovery clinical trial
recovery_trial = simulate_survival_data(
    n_groups=6,                    # 6 treatment sites
    size_range=(50, 80),          # Patients per site
    gamma=(-2.0, 0.7),            # Log-hazard: slow baseline recovery, treatment benefit
    tau=(0.3, 0.2, 0.1),          # Site variation
    censoring_time=180,           # 6-month follow-up
    time_range=(1, 365),          # 1 day to 1 year recovery
    predictor_range=(0.0, 1.0),   # 0=placebo, 1=active treatment
    random_seed=400
)
```

---

## Best Practices

### 1. Parameter Selection

**Start with realistic baselines:**
```python
# For test scores (0-100 scale)
gamma=(75.0, 10.0)  # 75% baseline, 10-point intervention effect

# For medical success rates
gamma=(0.2, 1.0)    # 55% baseline success rate, OR≈2.7 for treatment

# For count data (daily events)
gamma=(2.0, -0.5)   # ~7 baseline events, 40% reduction
```

**Choose appropriate random effect sizes:**
```python
# Small variation (homogeneous groups)
tau=(0.3, 0.2, 0.0)

# Moderate variation (typical)
tau=(0.8, 0.5, -0.2)

# Large variation (heterogeneous groups)
tau=(1.5, 1.0, 0.3)
```

### 2. Sample Size Considerations

**Power for detecting group-level effects:**
```python
# Smaller studies (pilot/feasibility)
n_groups=6, size_range=(15, 25)

# Typical studies
n_groups=10, size_range=(20, 40)

# Large studies (definitive trials)
n_groups=20, size_range=(50, 100)
```

### 3. Range Constraints

**Always use realistic ranges:**
```python
# Test scores
outcome_range=(0, 100), truncation_method="resample"

# Medical measurements
outcome_range=(80, 200), truncation_method="reflect"  # Blood pressure

# Count data
max_count=50  # Hospital bed capacity
```

### 4. Reproducibility

**Always set random seeds:**
```python
# For reproducible results
random_seed=42

# For multiple scenarios
for i, scenario in enumerate(scenarios):
    data = simulate_continuous_data(..., random_seed=100+i)
```

### 5. Model Interpretation

**Check your parameters make sense:**
```python
# Calculate implied probabilities for binary outcomes
import numpy as np

# For gamma=(0.5, 1.2) with logit link
baseline_prob = 1 / (1 + np.exp(-0.5))      # ~62%
treatment_prob = 1 / (1 + np.exp(-(0.5+1.2))) # ~85%
print(f"Baseline: {baseline_prob:.1%}, Treatment: {treatment_prob:.1%}")

# Calculate odds ratio
odds_ratio = np.exp(1.2)  # ~3.3
print(f"Odds ratio: {odds_ratio:.1f}")
```

---

## Parameter Interpretation Guide

### Continuous Outcomes

| Parameter | Interpretation | Typical Values |
|-----------|----------------|---------------|
| `gamma[0]` | Mean outcome when predictor=0 | Domain-specific |
| `gamma[1]` | Change in mean per unit predictor | Effect size of interest |
| `tau[0]` | SD of group means | 10-30% of gamma[0] |
| `tau[1]` | SD of group slopes | 25-75% of gamma[1] |
| `sigma` | Within-group SD | 50-150% of tau[0] |

**Example:** Test scores with gamma=(75, 10), tau=(8, 4), sigma=12
- Average score = 75 points
- Intervention effect = 10 points
- Schools vary by ±8 points in average scores
- School intervention effects vary by ±4 points
- Individual students vary by ±12 points within schools

### Binary Outcomes (Logit Link)

| Parameter | Interpretation | Conversion to Probability |
|-----------|----------------|-----------------------|
| `gamma[0]` | Log-odds at predictor=0 | `1/(1+exp(-gamma[0]))` |
| `gamma[1]` | Log-odds ratio | `exp(gamma[1])` = odds ratio |
| `tau[0]` | SD of log-odds intercepts | Affects group probability variation |
| `tau[1]` | SD of log-odds slopes | Affects group treatment effect variation |

**Example:** Treatment success with gamma=(0.2, 1.5), tau=(0.6, 0.4)
- Baseline success rate ≈ 55%
- Treatment odds ratio ≈ 4.5
- Groups vary widely in baseline and treatment effects

### Count Outcomes (Log Link)

| Parameter | Interpretation | Conversion to Rate |
|-----------|----------------|--------------------|
| `gamma[0]` | Log-rate at predictor=0 | `exp(gamma[0])` |
| `gamma[1]` | Log-rate ratio | `exp(gamma[1])` = rate ratio |
| `tau[0]` | SD of log-rate intercepts | Affects group rate variation |
| `tau[1]` | SD of log-rate slopes | Affects group effect variation |
| `dispersion` | Overdispersion | 1.0=Poisson, >1.0=more variable |

**Example:** Daily admissions with gamma=(2.5, -0.3), tau=(0.5, 0.2), dispersion=1.5
- Baseline rate ≈ 12 admissions/day
- Rate ratio ≈ 0.74 (26% reduction)
- Moderate overdispersion (clustering)

### Survival Outcomes (Log-Hazard)

| Parameter | Interpretation | Conversion to Hazard |
|-----------|----------------|--------------------|
| `gamma[0]` | Log-baseline hazard | `exp(gamma[0])` |
| `gamma[1]` | Log-hazard ratio | `exp(gamma[1])` = hazard ratio |
| `tau[0]` | SD of log-hazard intercepts | Group survival variation |
| `tau[1]` | SD of log-hazard slopes | Group treatment variation |
| `censoring_time` | Mean censoring time | Administrative/loss to follow-up |

**Example:** Cancer trial with gamma=(-2.0, 0.5), censoring_time=730
- Low baseline hazard (long survival expected)
- Hazard ratio ≈ 1.65 (65% increase in hazard)
- 2-year study follow-up

---

## Error Handling and Troubleshooting

### Common Issues

1. **TypeError with outcome_range:**
   ```python
   # Fixed in latest version - ensure you're using updated functions
   ```

2. **Unrealistic parameter combinations:**
   ```python
   # Check your gamma values make sense
   # For binary outcomes, extreme values (>5 or <-5) may cause issues
   ```

3. **All outcomes at boundaries:**
   ```python
   # Adjust sigma (continuous) or tau values if too much truncation occurs
   ```

4. **No variation in outcomes:**
   ```python
   # Increase tau values for more group-level variation
   # Increase sigma (continuous) for more individual variation
   ```

This comprehensive documentation provides everything needed to effectively use the simulation functions for research, teaching, and method development across diverse domains.