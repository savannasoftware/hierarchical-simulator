Hierarchical Data Simulator Documentation
==========================================

.. image:: https://badge.fury.io/py/hierarchical-simulator.svg
   :target: https://badge.fury.io/py/hierarchical-simulator
   :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/hierarchical-simulator.svg
   :target: https://anaconda.org/conda-forge/hierarchical-simulator
   :alt: conda-forge version

.. image:: https://img.shields.io/pypi/pyversions/hierarchical-simulator.svg
   :target: https://pypi.org/project/hierarchical-simulator/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://readthedocs.org/projects/hierarchical-simulator/badge/?version=latest
   :target: https://hierarchical-simulator.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/yourusername/hierarchical-simulator/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yourusername/hierarchical-simulator
   :alt: Coverage

Welcome to the Hierarchical Data Simulator documentation! This package provides a comprehensive solution for simulating hierarchical (multilevel) data across different outcome types.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Quick Start
        :link: quickstart
        :link-type: doc

        Get up and running with the hierarchical simulator in minutes.
        Install the package and run your first simulation.

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Detailed installation instructions for pip, conda, and development setups.
        System requirements and troubleshooting.

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        Complete API documentation for all simulation functions.
        Parameters, examples, and return values.

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc

        Real-world examples from education, healthcare, and research.
        Step-by-step tutorials and best practices.

Features
--------

   **Four Outcome Types**
   Generate realistic data for continuous (Gaussian), binary (Bernoulli), 
   count (Poisson/Negative-Binomial), and survival (time-to-event) outcomes.

   **Multiple Link Functions**
   Support for logistic, probit, complementary log-log, and log link functions
   with proper parameter interpretation guides.

   **Range Constraints**
   Built-in support for outcome bounds with multiple truncation methods:
   clipping, reflection, and resampling.

   **Reproducible Results**
   Full control over random seeds for consistent and reproducible simulations
   across research teams.

   **Domain-Specific Defaults**
   Pre-configured parameter sets for education, healthcare, and clinical research
   based on real-world studies.

   **Performance Optimized**
   Efficient algorithms for large-scale simulations with thousands of groups
   and observations.

Quick Example
-------------

.. code-block:: python

   from hierarchical_simulator import simulate_continuous_data

   # School test scores across 12 schools
   school_data, simulator = simulate_continuous_data(
       n_groups=12,                    # 12 schools
       size_range=(20, 40),           # 20-40 students per school
       gamma=(75.0, 15.0),            # Mean score=75, intervention effect=+15
       tau=(8.0, 4.0, -0.3),          # School variation with correlation
       sigma=12.0,                    # Individual student variation
       outcome_range=(0, 100),        # Valid score range 0-100
       predictor_range=(0.0, 1.0),    # 0=control, 1=intervention
       random_seed=42
   )

   print(f"Generated {len(school_data)} student records")
   print(f"Mean score: {school_data['outcome'].mean():.1f}")

Available Simulation Functions
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Function
     - Outcome Type
     - Use Cases
   * - :func:`~hierarchical_simulator.simulate_continuous_data`
     - Continuous (Gaussian)
     - Test scores, blood pressure, psychological scales
   * - :func:`~hierarchical_simulator.simulate_binary_data`
     - Binary (Bernoulli)
     - Treatment success, graduation rates, equipment failure
   * - :func:`~hierarchical_simulator.simulate_count_data`
     - Count (Poisson/NB)
     - Hospital admissions, disease cases, customer purchases
   * - :func:`~hierarchical_simulator.simulate_survival_data`
     - Time-to-event
     - Clinical trials, equipment reliability, employee turnover

Getting Help
------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: 🐛 Found a Bug?
        :link: https://github.com/yourusername/hierarchical-simulator/issues
        :link-type: url

        Report bugs and request features on GitHub Issues.

    .. grid-item-card:: 💬 Need Help?
        :link: https://github.com/yourusername/hierarchical-simulator/discussions
        :link-type: url

        Ask questions and discuss usage on GitHub Discussions.

Citation
--------

If you use this package in your research, please cite:

.. code-block:: bibtex

   @software{hierarchical_simulator,
     title = {Hierarchical Data Simulator: A Python Package for Multilevel Data Generation},
     author = {Moses Kabungo},
     year = {2025},
     url = {https://github.com/yourusername/hierarchical-simulator},
     version = {1.0.0}
   }

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   background

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/continuous_data
   user_guide/binary_data
   user_guide/count_data
   user_guide/survival_data
   user_guide/parameter_guide
   user_guide/best_practices

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/index
   examples/education_research
   examples/healthcare_analytics
   examples/clinical_trials
   examples/power_analysis

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/utils
   api/datasets

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog
   release_notes

.. toctree::
   :maxdepth: 1
   :caption: External Links

   PyPI Package <https://pypi.org/project/hierarchical-simulator/>
   conda-forge Package <https://anaconda.org/conda-forge/hierarchical-simulator>
   GitHub Repository <https://github.com/yourusername/hierarchical-simulator>
   Issue Tracker <https://github.com/yourusername/hierarchical-simulator/issues>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`