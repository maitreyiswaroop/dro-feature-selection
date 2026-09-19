# Refactoring invariants

The repository may be reorganized, but a refactor must preserve the scientific implementation and the paper-facing experiments.

## Kernel implementation

- Preserve the variance-scaled distance calculation for both `alpha` and `theta` parameterizations.
- Preserve nearest-neighbor selection, Gaussian weighting, and normalization behavior.
- Preserve gradient flow from the estimated conditional mean to the feature-degradation parameters.
- Keep the chunked and optimized implementations numerically consistent on the same inputs.

## Paper experiments

- Keep the synthetic, UCI, and ACS entry points runnable with their existing arguments.
- Preserve the parameters encoded by the three dataset launchers in `scripts/slurm/` unless a change is explicitly documented.
- Do not silently change random seeds, preprocessing, population definitions, objectives, baselines, or result schemas.
- Compare representative outputs against the pre-refactor implementation before merging mathematical or pipeline changes.

## Refactoring practice

- Separate file moves and naming changes from behavioral changes.
- Add characterization tests before extracting mathematical code.
- Remove obsolete versions from the current tree only after confirming they are unused; Git history remains the archive.
