# Project overview
This repository optimises physical MCFOST models of the post-AGB circumbinary discs.

The optimisation framework is SMAC3.

Observational constraints include:

- spectral energy distribution (SED)
- infrared interferometry
- SPHERE polarimetric imaging
- ALMA continuum imaging

The scientific goal is to identify a single physically consistent MCFOST disc model that reproduces all observables across different wavelengths and spatial scales.

# Important scientific principles
- Do not change the physical meaning of model parameters without explicitly explaining the proposed change.
- Preserve physical units carefully and check unit conversions.
- Do not silently rescale, renormalise, reweight, or otherwise modify observational likelihood or cost terms.
- Before changing any optimisation metric or cost function, explain how the change affects the relative weighting of SED, interferometry, polarimetry, and ALMA constraints.
- Distinguish clearly between numerical optimisation choices and physical model assumptions.
- Do not modify raw observational data.
- Do not alter observational uncertainties unless explicitly requested.
- Do not change fixed MCFOST assumptions or parameter bounds without explaining the scientific consequences.
- Prefer scientifically transparent calculations over unnecessary optimisation or abstraction.

# Code modification principles
- Prefer small, modular changes over large rewrites.
- Preserve existing behaviour unless a behavioural change is explicitly requested.
- Before making a substantial change, identify the relevant files and functions and briefly explain the proposed approach.
- Avoid modifying unrelated files.
- Do not remove existing functionality simply because it appears unused without first checking how it interacts with the broader optimisation pipeline.
- When fixing a bug, identify the cause of the bug rather than only suppressing the symptom.
- When changing a cost function, preserve access to the individual observable contributions where possible so their relative magnitudes can be inspected.
- Do not launch expensive optimisation runs unless explicitly requested.

# Environment

The project is written in Python.

Main dependencies include:

- MCFOST
- SMAC3
- numpy
- scipy
- astropy
- matplotlib
- distroi

Before adding a new dependency, check whether the task can be solved using existing dependencies.

# Main code structure

modelling_optimisation/SMAC3.py
: Main SMAC3 optimisation logic and optimisation setup.

modelling_optimisation/obriy_mcfost.py
: MCFOST execution functions and combination of costs from different observables.

modelling_optimisation/obriy_sed.py
: SED-related loading, processing, modelling, and cost functions.

modelling_optimisation/obriy_alma.py
: ALMA-related loading, image processing, profile calculations, and cost functions.

modelling_optimisation/obriy_polarimetry.py
: SPHERE polarimetric-imaging related functions and cost calculations.

modelling_optimisation/obriy_interferometry.py
: Interferometric observables and associated cost calculations.

modelling_optimisation/obriy_general.py
: General utility functions and recording/logging of optimisation information.

# Run scripts and configuration

Example optimisation run scripts are located in:

modelling_optimisation/run_scripts/

Example configuration files are located in:

modelling_optimisation/config/

When investigating how an optimisation is configured, inspect both the relevant run script and its associated configuration file.

Do not assume that values in example configuration files are the values used for every optimisation run.

# Working with the optimisation pipeline

When analysing the optimisation, trace the full data flow where relevant:

SMAC3 parameter proposal
→ MCFOST model generation
→ synthetic observables
→ comparison with observations
→ individual observable costs
→ combined objective
→ objective returned to SMAC3

When investigating unexpected optimisation behaviour, inspect the individual observable cost terms before modifying their weights or normalisation.

For numerical-scale problems, distinguish between:

physical units
residual definitions
uncertainty weighting
number of data points
averaging versus summation
normalisation
explicit weighting factors

Do not assume that two cost terms should have similar numerical values simply because they represent equally important scientific constraints.



# Coding style

- Use descriptive variable names.
- Preserve existing scientific terminology.
- Add comments where mathematical transformations are not obvious.
- Avoid unnecessary abstraction.