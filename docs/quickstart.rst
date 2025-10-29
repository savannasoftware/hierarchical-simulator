Quick Start Guide
=================

Welcome! This guide will get you up and running with the hierarchical data simulator in just a few minutes.

Your First Simulation
---------------------

Let's start with a simple example - simulating test scores across multiple schools:

.. code-block:: python

    from hierarchical_simulator import simulate_continuous_data

    # Generate school test score data
    data, simulator = simulate_continuous_data(
        n_groups=5,                     # 5 schools
        size_range=(20, 30),           # 20-30 students per school  
        gamma=(75.0, 10.0),            # Baseline=75, intervention effect=+10
        tau=(8.0, 4.0, -0.2),          # School variation with slight negative correlation
        sigma=12.0,                    # Individual student variation
        outcome_range=(0, 100),        # Valid test score range
        random_seed=42                 # For reproducible results
    )

    print(f"Generated data for {len(data)} students across {data['group'].nunique()} schools")
    print(f"Score range: {data['outcome'].min():.1f} - {data['outcome'].max():.1f}")
    print(f"Mean score: {data['outcome'].mean():.1f}")

.. code-block:: text

    Generated data for 124 students across 5 schools
    Score range: 42.3 - 98.7
    Mean score: 78.2

Understanding the Output
------------------------

The simulation returns a pandas DataFrame with these columns:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Description
   * - ``group``
     - Group identifier (1, 2, 3, ...)
   * - ``observation``
     - Observation within group (1, 2, 3, ...)
   * - ``predictor``
     - Predictor variable value (e.g., 0=control, 1=treatment)
   * - ``linear_predictor``
     - Linear predictor before adding error
   * - ``true_beta_0``
     - True group-specific intercept
   * - ``true_beta_1``
     - True group-specific slope
   * - ``outcome``
     - Generated outcome variable

Let's explore the data:

.. code-block:: python

    # View first few rows
    print(data.head())
    
    # Summary by group
    summary = data.groupby('group').agg({
        'outcome': ['count', 'mean', 'std'],
        'predictor': 'mean'
    }).round(2)
    print("\nSummary by school:")
    print(summary)

Basic Visualization
-------------------

Visualize your simulated data:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Distribution of outcomes
    axes[0].hist(data['outcome'], bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_title('Distribution of Test Scores')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')

    # Plot 2: Outcomes by group
    sns.boxplot(data=data, x='group', y='outcome', ax=axes[1])
    axes[1].set_title('Scores by School')
    axes[1].set_xlabel('School')
    axes[1].set_ylabel('Score')

    # Plot 3: Treatment effect
    sns.scatterplot(data=data, x='predictor', y='outcome', 
                   hue='group', alpha=0.6, ax=axes[2])
    axes[2].set_title('Treatment Effect by School')
    axes[2].set_xlabel('Treatment (0=Control, 1=Intervention)')
    axes[2].set_ylabel('Score')

    plt.tight_layout()
    plt.show()

Different Outcome Types
-----------------------

The simulator supports four types of outcomes:

Continuous Data (Gaussian)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hierarchical_simulator import simulate_continuous_data

    # Blood pressure study
    bp_data, _ = simulate_continuous_data(
        n_groups=8,                     # 8 clinics
        size_range=(30, 50),           # 30-50 patients per clinic
        gamma=(140.0, -18.0),          # Baseline=140 mmHg, treatment effect=-18
        tau=(12.0, 5.0, -0.4),         # Clinic variation
        sigma=15.0,                    # Individual variation
        outcome_range=(80, 200),       # Physiological range
        random_seed=123
    )

Binary Data (Bernoulli)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hierarchical_simulator import simulate_binary_data, LinkFunction

    # Treatment success study
    success_data, _ = simulate_binary_data(
        n_groups=10,                   # 10 hospitals
        size_range=(25, 50),          # 25-50 patients per hospital
        gamma=(0.2, 1.5),             # Log-odds scale
        tau=(0.8, 0.6, -0.2),         # Hospital variation
        link_function=LinkFunction.LOGIT,  # Logistic regression
        random_seed=456
    )

    # Calculate success rates
    control_rate = success_data[success_data['predictor'] < 0.5]['outcome'].mean()
    treatment_rate = success_data[success_data['predictor'] > 0.5]['outcome'].mean()
    print(f"Control success rate: {control_rate:.1%}")
    print(f"Treatment success rate: {treatment_rate:.1%}")

Count Data (Poisson/Negative-Binomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hierarchical_simulator import simulate_count_data

    # Hospital admissions
    admission_data, _ = simulate_count_data(
        n_groups=12,                   # 12 hospitals
        size_range=(30, 40),          # 30-40 days per hospital
        gamma=(2.8, -0.3),            # Log-rate scale
        tau=(0.6, 0.3, 0.1),          # Hospital variation
        dispersion=1.0,               # Poisson (no overdispersion)
        max_count=50,                 # Capacity constraint
        random_seed=789
    )

    print(f"Mean daily admissions: {admission_data['outcome'].mean():.1f}")
    print(f"Days at capacity: {(admission_data['outcome'] == 50).sum()}")

Survival Data (Time-to-Event)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hierarchical_simulator import simulate_survival_data

    # Clinical trial
    survival_data, _ = simulate_survival_data(
        n_groups=6,                    # 6 treatment sites
        size_range=(40, 60),          # 40-60 patients per site
        gamma=(-1.8, 0.6),            # Log-hazard scale
        tau=(0.5, 0.3, 0.1),          # Site variation
        censoring_time=365,           # 1-year follow-up
        time_range=(1, 1095),         # 1 day to 3 years
        random_seed=321
    )

    print(f"Event rate: {survival_data['event'].mean():.1%}")
    print(f"Median time: {survival_data['outcome'].median():.1f} days")

Parameter Interpretation
------------------------

Understanding the Key Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Fixed Effects (gamma)

        The ``gamma`` parameter controls population-level effects:

        - **gamma[0]**: Baseline level (intercept)
        - **gamma[1]**: Treatment/predictor effect (slope)

        **Scale depends on outcome type:**
        
        - Continuous: Natural scale (e.g., test points, mmHg)
        - Binary: Log-odds scale (logit) or probit scale  
        - Count: Log-rate scale
        - Survival: Log-hazard scale

    .. tab-item:: Random Effects (tau)

        The ``tau`` parameter controls group-level variation:

        - **tau[0]**: Standard deviation of group intercepts
        - **tau[1]**: Standard deviation of group slopes
        - **tau[2]**: Correlation between intercepts and slopes (-1 to 1)

        **Example interpretation:**
        
        .. code-block:: python

            tau=(8.0, 4.0, -0.3)
            # - Schools vary by ±8 points in average scores
            # - School treatment effects vary by ±4 points  
            # - Negative correlation: schools with higher baselines
            #   tend to have smaller treatment effects

    .. tab-item:: Error/Dispersion

        Controls within-group variation:

        - **sigma** (continuous): Within-group standard deviation
        - **dispersion** (count): 1.0=Poisson, >1.0=overdispersed

Common Patterns
~~~~~~~~~~~~~~~

.. code-block:: python

    # Small effect, low variation
    gamma=(50.0, 2.0), tau=(3.0, 1.0, 0.0), sigma=5.0

    # Large effect, high variation  
    gamma=(50.0, 15.0), tau=(12.0, 8.0, -0.5), sigma=18.0

    # No treatment effect (null simulation)
    gamma=(50.0, 0.0), tau=(8.0, 0.0, 0.0), sigma=10.0

Practical Examples
------------------

Education Research
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Reading intervention across schools
    reading_study, _ = simulate_continuous_data(
        n_groups=15,                   # 15 elementary schools
        size_range=(60, 90),          # Class sizes
        gamma=(85.0, 12.0),           # Reading scores (0-100 scale)
        tau=(15.0, 8.0, -0.4),        # School effects
        sigma=20.0,                   # Student variation
        outcome_range=(0, 100),       # Valid score range
        predictor_range=(0.0, 1.0),   # 0=standard, 1=intervention
        random_seed=2024
    )

Healthcare Research
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Medication adherence study
    adherence_study, _ = simulate_binary_data(
        n_groups=20,                   # 20 clinics
        size_range=(40, 80),          # Patients per clinic
        gamma=(-0.5, 0.8),            # Log-odds: 38% baseline, OR=2.2
        tau=(0.6, 0.4, 0.0),          # Clinic variation
        link_function=LinkFunction.LOGIT,
        random_seed=2024
    )

Next Steps
----------

Now that you've run your first simulations, explore:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 📊 Real-World Examples
        :link: examples/index
        :link-type: doc

        Detailed tutorials for education, healthcare, and clinical research
        with complete analysis workflows.

    .. grid-item-card:: 🔧 API Documentation
        :link: api/index
        :link-type: doc

        Complete parameter reference with detailed explanations
        and advanced options.

    .. grid-item-card:: 📈 Best Practices
        :link: user_guide/best_practices
        :link-type: doc

        Parameter selection guidelines, sample size considerations,
        and interpretation tips.

    .. grid-item-card:: 🎯 Parameter Guide
        :link: user_guide/parameter_guide
        :link-type: doc

        In-depth guide to choosing realistic parameters
        for your research domain.

Questions?
----------

- **Need help?** Check the :doc:`installation` guide or :doc:`api/index`
- **Found a bug?** Report it on `GitHub Issues <https://github.com/yourusername/hierarchical-simulator/issues>`_
- **Want to contribute?** See our :doc:`contributing` guide

Happy simulating! 🎲