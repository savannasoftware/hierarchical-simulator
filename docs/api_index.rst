API Reference
=============

Complete reference for all functions, classes, and utilities in the hierarchical data simulator.

Core Simulation Functions
--------------------------

The main simulation functions for different outcome types:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   hierarchical_simulator.simulate_continuous_data
   hierarchical_simulator.simulate_binary_data
   hierarchical_simulator.simulate_count_data
   hierarchical_simulator.simulate_survival_data

simulate_continuous_data
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hierarchical_simulator.simulate_continuous_data
   :no-index:

simulate_binary_data
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hierarchical_simulator.simulate_binary_data
   :no-index:

simulate_count_data
~~~~~~~~~~~~~~~~~~~

.. autofunction:: hierarchical_simulator.simulate_count_data
   :no-index:

simulate_survival_data
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hierarchical_simulator.simulate_survival_data
   :no-index:

Utility Classes and Enums
--------------------------

Supporting classes and enumerations:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   hierarchical_simulator.HierarchicalDataSimulator
   hierarchical_simulator.LinkFunction

Base Simulator Class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hierarchical_simulator.HierarchicalDataSimulator
   :members:
   :special-members: __init__
   :no-index:

Link Functions
~~~~~~~~~~~~~~

.. autoclass:: hierarchical_simulator.LinkFunction
   :members:
   :no-index:

Example Datasets
-----------------

Pre-configured example datasets for learning and testing:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   hierarchical_simulator.load_example_dataset

.. autofunction:: hierarchical_simulator.load_example_dataset
   :no-index:

Function Reference by Module
-----------------------------

Core Module
~~~~~~~~~~~

.. automodule:: hierarchical_simulator.core
   :members:
   :undoc-members:
   :show-inheritance:

Continuous Data
^^^^^^^^^^^^^^^

.. automodule:: hierarchical_simulator.core.continuous
   :members:
   :undoc-members:
   :show-inheritance:

Binary Data
^^^^^^^^^^^

.. automodule:: hierarchical_simulator.core.binary
   :members:
   :undoc-members:
   :show-inheritance:

Count Data
^^^^^^^^^^

.. automodule:: hierarchical_simulator.core.count
   :members:
   :undoc-members:
   :show-inheritance:

Survival Data
^^^^^^^^^^^^^

.. automodule:: hierarchical_simulator.core.survival
   :members:
   :undoc-members:
   :show-inheritance:

Utilities Module
~~~~~~~~~~~~~~~~

.. automodule:: hierarchical_simulator.utils
   :members:
   :undoc-members:
   :show-inheritance:

Link Functions
^^^^^^^^^^^^^^

.. automodule:: hierarchical_simulator.utils.link_functions
   :members:
   :undoc-members:
   :show-inheritance:

Distributions
^^^^^^^^^^^^^

.. automodule:: hierarchical_simulator.utils.distributions
   :members:
   :undoc-members:
   :show-inheritance:

Truncation Methods
^^^^^^^^^^^^^^^^^^

.. automodule:: hierarchical_simulator.utils.truncation
   :members:
   :undoc-members:
   :show-inheritance:

Validation
^^^^^^^^^^

.. automodule:: hierarchical_simulator.utils.validation
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
----------------------

.. automodule:: hierarchical_simulator.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main CLI
~~~~~~~~

.. automodule:: hierarchical_simulator.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

Datasets Module
---------------

.. automodule:: hierarchical_simulator.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Example Datasets
~~~~~~~~~~~~~~~~

.. automodule:: hierarchical_simulator.datasets.examples
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference Tables
----------------------

Function Comparison
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30

   * - Function
     - Outcome Type
     - Distribution
     - Common Use Cases
   * - ``simulate_continuous_data``
     - Continuous
     - Gaussian
     - Test scores, measurements, scales
   * - ``simulate_binary_data``
     - Binary
     - Bernoulli
     - Success/failure, yes/no outcomes
   * - ``simulate_count_data``
     - Count
     - Poisson/Negative-Binomial
     - Event counts, frequencies
   * - ``simulate_survival_data``
     - Time-to-event
     - Exponential
     - Time until event occurs

Parameter Patterns
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - Small Effect
     - Large Effect
   * - ``gamma`` (continuous)
     - ``(50.0, 2.0)``
     - ``(50.0, 15.0)``
   * - ``gamma`` (binary, logit)
     - ``(0.0, 0.5)``
     - ``(0.0, 2.0)``
   * - ``gamma`` (count, log)
     - ``(2.0, 0.2)``
     - ``(2.0, 1.0)``
   * - ``tau`` (low variation)
     - ``(0.3, 0.2, 0.0)``
     - ``(0.5, 0.3, 0.0)``
   * - ``tau`` (high variation)
     - ``(1.0, 0.8, -0.3)``
     - ``(2.0, 1.5, -0.5)``

Link Function Reference
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Link Function
     - Mathematical Form
     - Common Use
   * - ``LOGIT``
     - :math:`\text{logit}(p) = \log\left(\frac{p}{1-p}\right)`
     - Binary outcomes, logistic regression
   * - ``PROBIT``
     - :math:`\text{probit}(p) = \Phi^{-1}(p)`
     - Binary outcomes, normal assumption
   * - ``CLOGLOG``
     - :math:`\text{cloglog}(p) = \log(-\log(1-p))`
     - Rare events, asymmetric relationships
   * - ``LOG``
     - :math:`\log(\lambda)`
     - Count data, survival analysis

Error Handling
--------------

The package includes comprehensive error handling and validation:

Common Exceptions
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Exception
     - Description
   * - ``ValueError``
     - Invalid parameter values (negative variance, out-of-range correlations)
   * - ``TypeError``
     - Incorrect parameter types (string instead of number)
   * - ``RuntimeError``
     - Simulation convergence issues or constraint violations

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

All functions validate inputs and provide helpful error messages:

.. code-block:: python

    from hierarchical_simulator import simulate_continuous_data

    try:
        # This will raise ValueError
        data, _ = simulate_continuous_data(
            n_groups=-5,  # Invalid: negative number of groups
            tau=(1.0, 0.5, 2.0)  # Invalid: correlation > 1
        )
    except ValueError as e:
        print(f"Parameter error: {e}")

See Also
--------

- :doc:`../user_guide/parameter_guide` - Detailed parameter interpretation
- :doc:`../examples/index` - Real-world usage examples
- :doc:`../user_guide/best_practices` - Recommended practices