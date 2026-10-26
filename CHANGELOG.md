# pyReCo's Change Log

Noteworthy changes of new releases of pyReCo have been documented in this file since version
0.0.2.

## Unreleased


### Breaking Changes

* Reservoir edge direction is now consistent between the dynamics and the graph
  analysis: `weights[u, v]` is the edge `u -> v`, and the state update is
  `r(t+1) = g(r(t) @ weights + ...)`. Previously the update used `weights @ r(t)`,
  so edge `(u, v)` transmitted activity `v -> u`. Trajectories for a fixed
  reservoir matrix are now those of its transpose; results are statistically
  equivalent but not bitwise reproducible against earlier versions.
* Output normalization and output scaling were removed (`normalize_outputs`,
  `output_scaling`, `set_output_scaling`, `output_mean`, `output_std`). Ridge regression
  is linear in the targets, so dividing by the target standard deviation or scaling the
  targets does not change predictions; only the centering did. Targets are now used as
  given and predictions are in the units of `y`.
* `n_init` was removed from `fit()`. It never explored anything: the loop reused the same
  reservoir, input weights and initial state, so all initializations were identical and the
  best of `k` identical models was returned. The influence of the initial state fades with the
  echo state property and with `discard_transients`, and resampling the reservoir inside
  `fit()` would overwrite a user-supplied or pruned network. The fit history is now
  `{"train_score": float}` (plus `"res_states"` with `store_states=True`); the readout
  weights, intercept and initial state are on the layers. `examples/effects_network_initialization.py`
  was removed with it.
* Input z-scoring and `input_scaling` are now arguments of `compile()` (defaults: z-scoring
  on, scaling `1.0`). `set_normalization()` only takes `normalize_inputs`.


### Minor Improvements

* The readout now has an intercept: `y = R W + b`, fitted by centering the reservoir states
  and the targets before the ridge solve (the intercept is not penalized). Targets with a
  nonzero mean no longer have to be encoded in the reservoir states. The intercept is stored
  in `readout_layer.bias` (shape `(n_outputs,)`, `readout_layer.weights` is unchanged). Replaces the centering that output
  normalization used to provide.
* Input preprocessing is documented: per-channel z-scoring (statistics over batch and time),
  then multiplication by `input_scaling`.
* Prepared release metadata updates for `v2.1.0`.
* Confirmed automated publish flow via GitHub Release (`published`) to PyPI.


### Bugfixes

* `input_scaling` was applied in `fit()` but not in `predict()`, which made predictions
  inconsistent with training for any value other than 1. Training and inference now share
  one transform, and the preprocessing settings are frozen at fit time. `input_scaling`
  is validated (positive, finite, scalar or one value per channel).
* The training score computed inside `fit()` fed already-normalized inputs through
  `predict()` (normalizing twice) and compared them to targets in a different space.
  It now compares predictions to the targets on the original data scale.
* `AutoRC_predict` applies the same input preprocessing, and maps the fed-back
  prediction into the coordinates of the input channels.


## pyReCo-0.0.3

### Improvements

* Improvements go here


## pyReCo-0.1.0

### New Features

* New features go here

### Improvements

* Improvements go here

### Bugfixes

* Bugfixes go here


## pyReCo-1.0.0

### Breaking Changes

* Breaking changes go here


### New Features

* New features go here


### Improvements

* Improvements go here

### Bugfixes

* Bugfixes go here
