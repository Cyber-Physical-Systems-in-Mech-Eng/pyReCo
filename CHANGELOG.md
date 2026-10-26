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
* `r2` is now computed per channel and averaged with equal weights. It used to pool all
  channels, so a channel with a large mean or variance hid a failing one (perfect on a
  channel with std 50, noise on a channel with std 1: 0.9997 pooled, 0.03 per channel).
  Values for single-channel targets are unchanged.
* `EdgePruner` and `NetworkPruner` reject scores for which higher is better (`r2`) as the
  `performance_criterion` / `criterion`: candidates are ranked by the lowest value, so `r2`
  selected the worst candidate. Use a loss (`mse`, `mae`, `rmse`, `nrmse`); `r2` can still be
  tracked through `metrics`.
* Seeding was reworked. pyReCo no longer uses the global random state (`np.random.seed(...)`
  and `random.seed(...)` have no effect on it); every random draw comes from an explicit
  `numpy.random.Generator` created from a `seed` argument. `RC(seed=...)`, `AutoRC(seed=...)`
  and `ReservoirComputer(seed=...)` seed the read-in weights, the initial reservoir state and
  the choice of input and readout nodes, each from its own independent stream, so changing one
  choice (e.g. `fraction_out`) does not shift the others. Compiling twice with the same seed
  gives the same model; `seed=None` draws fresh entropy. `RandomReservoirLayer(seed=...)` now
  controls the reservoir graph completely, including the repair of isolated nodes (which used
  the global state before). `NetworkInitializer`, `NodeSelector` and `gen_ER_graph` take an
  `rng`/`seed`. The same seed value yields different draws than in earlier versions. The unused
  `seed=42` default of the `ReservoirLayer` base class was removed.
* The remaining users of the global random state were converted as well: `EdgeSelector`
  (`rng=`) and `EdgePruner` (`seed=`, every `prune()` call restarts from it, so pruning the same
  model twice proposes the same candidates), `cross_val` (`seed=`, fold shuffling),
  `utils_data.train_test_split` and the dataset generators `sine_pred`, `sine_to_cosine`,
  `sincos2`, `vector_to_vector`, `sequence_to_scalar`, `sequence_to_sequence`, `x_to_x`
  (`seed=`, the train/test split). `visualize_reservoir_network_circle(seed=...)` no longer
  overwrites the global NumPy seed.
* Input z-scoring and `input_scaling` are now arguments of `compile()` (defaults: z-scoring
  on, scaling `1.0`). `set_normalization()` only takes `normalize_inputs`.


### Minor Improvements

* New metrics `rmse` and `nrmse` (`root_mean_squared_error`,
  `normalized_root_mean_squared_error`). NRMSE is `sqrt(mean_c MSE_c / Var_c)` with the
  variance of each true channel over the evaluated data: 0 for a perfect prediction, 1 for
  predicting the mean, independent of the scale of the data (NRMSE^2 = 1 - r2). Constant
  channels are left out with a warning. All metrics now check that `y_true` and `y_pred`
  have the same shape (a mismatch such as `(n, t, 1)` against `(n, t)` used to broadcast
  silently), and `pyreco.metrics.is_loss` tells whether lower is better.
* The history returned by `fit()` gains `train_metrics`, a dict with every metric given to
  `compile()`; `train_score` remains the first one.
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

* Reservoirs with spectral radius 0 (no directed cycle) no longer produce NaN weights or
  corrupt the model. Small or sparse ER graphs often have no directed cycle (about 40% of
  5-node graphs with density 0.05) and crashed construction with a `LinAlgError`;
  `gen_ER_graph` now redraws such graphs (up to 100 attempts, then raises a clear
  `ValueError`). For graphs that have a cycle the seed gives the same graph as before.
  Rescaling a network with spectral radius 0 (e.g. `set_hp(spec_rad=...)` after pruning removed
  all cycles) now raises a `ValueError` and leaves the layer unchanged; before it multiplied the
  weights by 9e5 and then raised, leaving the corrupted weights in place.
* `remove_reservoir_edges` refreshes the `density` and `spec_rad` of the reservoir layer
  (they were stale after pruning). `EdgePruner` warns once per run when pruning drives the
  spectral radius to 0.
* The post-construction density check of `RandomReservoirLayer` was removed: it compared the
  value to itself, and the density of an ER graph is random by definition. `density` is the
  edge probability; the realized density is available as `layer.density`. The spectral radius
  check now compares against the requested value.
* `NetworkInitializer` failed for 2D shapes with the methods `random`, `ones` and `zeros`.
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
