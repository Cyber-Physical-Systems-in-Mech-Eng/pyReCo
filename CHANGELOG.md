# pyReCo's Change Log

Noteworthy changes of new releases of pyReCo have been documented in this file since version
0.0.2.

## Unreleased


### Breaking Changes

* Layers check their arguments when they are created and raise a `TypeError`/`ValueError` with a clear
  message instead of building a model that silently does something else: `nodes` an integer >= 2,
  `density` in (0, 1], `spec_rad` > 0 (0 gave a reservoir without recurrence), `leakage_rate` in
  (0, 1] (0 froze the reservoir, values above 1 made it unstable), `activation` in
  `{"tanh", "sigmoid"}` (an unknown name used to fail only at `fit()`), `fraction_input` and
  `fraction_out` in (0, 1]. Integers are accepted where a number is meant (`fraction_out=1`);
  `density=1.5` and a non-integer `nodes` were accepted or failed with a cryptic error before.
* The reservoir now starts from the zero state by default (it used to be a random normal vector).
  Other initial states can be requested with `RandomReservoirLayer(init_res_sampling="ones" |
  "random" | "random_normal")`, or set explicitly after `compile()` with
  `model.reservoir_layer.set_initial_state(r0)`; `compile()` honours the layer's setting (it ignored
  it before).
* The warm-up `discard_transients` (default 0) has a defined meaning: the target at time step j is
  paired with the reservoir state after the inputs x(0), ..., x(d + j), d being the warm-up. It
  must not exceed the number of input time steps minus the number of target time steps, otherwise
  `fit()` raises a `ValueError`. Before, a warm-up on targets as long as the inputs silently cut
  the first targets, a warm-up larger than the difference crashed or was replaced by another
  value, and for targets shorter than the inputs the warm-up was ignored and always the last
  states were used. Consequence: for a sequence-to-scalar task (or any window whose targets lie at
  the end of the input window, as in `pyreco.datasets.load`) pass `discard_transients = n_in - n_out`
  to pair the targets with the last states; the default 0 would use the first ones. Predictions
  of `predict()` cover all input steps after the warm-up, `evaluate()` compares the first
  `n_out` of them with the targets. The idea to choose the warm-up automatically is kept as a
  comment in `compile()`.
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

* New example `mwe_lorenz_reconstruction.py`: a reservoir observer that infers the third Lorenz state from
  the first two (test NRMSE 0.003 with 300 nodes), after Lu et al., Chaos 27, 041102 (2017). It uses one
  long sequence with a washout, and controls that must fail (x from z is impossible by the symmetry of the
  equations).
* New input bias: `model.compile(input_bias=b)` (also `set_input_bias`, `set_hp(input_bias=...)`,
  `get_hp`) adds a constant input `b * s` to every reservoir node, `s` being a vector of random signs
  (+1/-1, fixed at compile, `model.input_layer.bias`). Default 0, i.e. no change for existing code.
  Like the input scaling it is frozen at `fit()`. Without a bias and with a zero-mean input the
  states of a tanh reservoir are odd in the input, so even functions such as x^2 cannot be learned;
  most echo state networks of the literature have a bias. Not used by `AutoRC`.
* `InputLayer(weight_distribution=...)` chooses the read-in weights: `"normal"` (default, as before),
  `"uniform"` on [-1, 1] or `"sign"` (+1/-1, equal magnitude). The model seed keeps them reproducible
  and the bias uses its own random stream, so existing seeds give the same models as before.
* `RandomReservoirLayer(weight_distribution=...)` (and `gen_ER_graph`) can draw signed weights on
  the edges of the ER graph: `"positive"` (default, all weights equal, as before), `"sign"` (+1 or
  -1), `"uniform"` (on [-1, 1]) or `"normal"`. The graph itself is the same for every distribution,
  the weights are drawn after it and the network is scaled to the spectral radius as before. On the
  NARMA10 setup of the literature the non-negative weights were the largest single reason for the
  accuracy plateau. Along with it: `compute_density` counts negative weights (it counted only
  weights > 0), the PageRank of a node ignores the weights, the edge properties `abs_weight` and
  `sign` exist (the `EdgeAnalyzer` documentation named them), and a signed random graph whose
  cycles cancel to a spectral radius of rounding noise is redrawn like an acyclic one.

* `mwe_narma10_optimal_reservoir.py` also searches the weight distributions and the input bias. The best
  reservoir (144 nodes, signed weights) reaches a test NRMSE of 0.153 instead of 0.358, the level of a
  quadratic model on the last 20 inputs (0.150).
* New example `mwe_narma10_optimal_reservoir.py`: an optuna search over the size (up to 200 nodes),
  the density and the hyperparameters of a reservoir for NARMA10, on three independent NARMA
  sequences for training, validation and test. It reports the best reservoir, the smallest one within
  5% of it, the importances of the parameters and a note if an optimum sits at the edge of its search
  range. With the units in the nearly linear regime (small input scaling) the reservoir reaches the
  level of a linear model on the last 20 inputs (test NRMSE 0.36 vs 0.38); a quadratic model reaches
  0.15.
* `Tuner` is quiet by default: Optuna only logs warnings while it runs (the logging level of the
  caller is restored afterwards, also if a trial fails), the hyperparameter importances are printed
  only with `verbose=True`, Optuna's warnings about small studies are not shown, a single trial gives
  empty importances instead of a warning, and `save_best_model` only prints with `verbose=True`
  (it warns if there is no best model, instead of printing).
* `Tuner(seed=...)` seeds the Optuna sampler (TPE, random and grid), so a seeded tuning of a seeded
  model and data finds the same hyperparameters on every run. The samplers were unseeded before and
  every run was different. Without a seed the search is still random.
* New example `mwe_narma10_size_sweep_edge_pruning.py`: tunes the hyperparameters of reservoirs of increasing size
  for NARMA10, picks the smallest size that is within 10% of the best validation error, prunes
  its edges, and compares with a linear and a quadratic model on the lagged inputs. Every stage
  is guarded and a report lists the errors, warnings, checks that failed (reproducibility, pruner
  bookkeeping, the Tuner leaving the model alone, ...) and the amount of text printed.
* The README was brought up to date: installation first, the capabilities (pruning, benchmark data
  sets, tuning, seeds), data shapes and warm-up, initial state and reproducibility, benchmark data
  sets, edge pruning, hyperparameter tuning, an overview of the examples and how to run the tests.
  All code examples of the README are run by the test suite.
* Examples: `mwe_tuner.py` is now `mwe_hyperparameter_tuning.py`; `mwe_task_shapes.py` shows five
  input/output shapes (vector to vector, sequence to scalar, sequence to sequence, several states,
  inputs longer than targets with a warm-up) on one data set (`sine_pred`); the cross-validation
  and the drawing of the reservoir moved to `mwe_cross_validation.py`.
* New tests for the core model (`tests/test_core_model.py`) and for the warm-up and the initial
  state against an independent implementation of the reservoir update (`tests/test_warmup.py`).
* A GitHub Actions workflow (`tests.yml`) runs the test suite on every pull request and on pushes
  to `main`/`dev` (Ubuntu with Python 3.10 and 3.12, plus an informational Windows run). It does
  not build or publish anything. The test dependencies are available as `pip install -e ".[test]"`,
  and the coverage option of the pytest configuration now names the package correctly.
* `pyreco.__version__` exists (read from the installed package metadata; the version itself comes
  from the git tag of the release).
* New benchmark systems `narma5` and `narma10` in `pyreco.datasets.load`: the NARMA system
  identification tasks (random input u ~ U[0, 0.5], the target is the output of a nonlinear
  autoregressive moving-average system of order 5 or 10), which test memory and nonlinearity of a
  reservoir. Unlike the forecasting datasets, the input windows are windows of the random input
  and the target windows end with the input window (`n_out <= n_in`); `seed` draws the input
  sequence, a `washout` (default 100) removes the artificial start of the system, and sequences
  for which the NARMA10 output diverges (a few percent of the sequences of 5000 steps) are
  redrawn. See `examples/mwe_narma.py`.
* `tests/test_benchmark_accuracy.py` guards the forecast quality on Lorenz, Mackey-Glass and the
  NARMA tasks with seeded reservoirs.
* Examples were tidied up: the `testing_*` scripts that duplicated the minimal working examples
  were merged into `mwe_task_shapes.py` (vector/sequence-to-scalar/sequence tasks,
  cross-validation, reservoir drawing); the remaining ones are now called `mwe_*`;
  `mwe_chaotic_datasets.py` forecasts the Lorenz and Mackey-Glass series from `pyreco.datasets`;
  `effects_network_structure.py` works again; the broken `examples/deprecated/` folder, the
  unreferenced `predictions_model.png` files and the script `test_new_datasets.py` (replaced by
  `tests/test_datasets.py`) were removed, and `.DS_Store` / `.vscode/` are no longer tracked.
* `optuna` is now a required dependency (hyperparameter search in `pyreco.tuner`; importing the
  module failed without it). The optional extra `pyreco[benchmarks]` (tensorflow) is for the
  comparison with feed-forward networks.
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

* `fit()`, `predict()` and `evaluate()` check the data and say what is wrong: NaN or infinite values (with
  their position; `predict` used to return NaN and `fit` failed with a message of the underlying ridge
  regression), a wrong number of input channels (a broadcast error before), a sequence that is not longer
  than the warm-up (an `IndexError`), and `predict()` before `fit()`. `fit()` also checks the number of
  target channels against the readout layer: it used to train a model with other outputs than the layer
  declared without any message.
* `evaluate(x, y)` raises a `ValueError` if `y` has more time steps than there are predictions. With a
  warm-up, a target sequence that was given in full (including the first `discard_transients` steps) was
  compared with the wrong predictions without any message (e.g. an NRMSE of 1.58 instead of 0.009).
* A reservoir where `fraction_input` or `fraction_out` selects no node (e.g. `fraction_input=0.1` on 5
  nodes) is an error at `compile()` with the numbers in the message. Before, no node received the input
  (the model trained and predicted a constant, silently) or no node was read out (`fit()` failed with a
  scikit-learn message about 0 features). `fit()` also checks that the input reaches the reservoir and
  that a readout node is left, which matters after `remove_reservoir_nodes`, and says "call compile()
  first" for a model that is not compiled (it was a cryptic `einsum` error).
* `remove_reservoir_nodes` crashed for a model that was not fitted yet, and updated `fraction_out` of
  the readout layer with the wrong denominator (the old number of readout nodes); it is the fraction of
  the remaining reservoir nodes that are read out now.
* `Tuner` no longer modifies the model that is passed in. It used to set the hyperparameters of
  the trials on that very object and retrain it, so the caller's model ended up in the state of
  the last trial. Every optimization now runs on a copy; the best copy is `tuner.best_model`.
* `RidgeSK(alpha=...)` ignored its argument (the constructor set `alpha = 1.0` regardless), so every
  model compiled with e.g. `RidgeSK(alpha=0.5)` was trained with a penalty of 1.0. The argument is
  used now and validated (a finite number >= 0; `set_alpha` checks the same). The default stays
  1.0, and `model.set_hp(alpha=...)` worked before. The examples and tests that pass an explicit
  `alpha` therefore use that value now.
* `Tuner(verbose=True)` crashed with a `UnicodeEncodeError` on consoles with a legacy code page
  (Windows) because of an arrow in the trial log. The tuner now has tests.
* The README examples failed with `TypeError` (they called `evaluate(X=...)`; the argument
  is `x`). A test now runs every README example.
* `utils_data`: `vector_to_vector` raised a `NameError` for `sine_to_cosine`/`sin_to_cos2`,
  `sequence_to_scalar` rejected `sine_pred`, `x_to_x` ignored its `n_time_int`/`n_time_out`
  arguments, and unknown names raised an `UnboundLocalError`. All data functions now share one
  dispatch: the names `sine_pred`, `sine_to_cosine`, `sincos2` (and the aliases `sine_prediction`,
  `sin_to_cos2`) work everywhere and unknown names raise a `ValueError` that lists them. The
  parameter typo `n_time_int` of `x_to_x` is now `n_time_in`, the functions no longer print the
  shapes, and `available_problems()` lists the names.
* `datasets.load('mackey_glass')` integrated the delay equation with an Euler step of 1.0,
  which deviates from the true solution by up to 0.8 (the series ranges from 0.4 to 1.3). It
  now uses Heun's method with an internal step `h_int=0.1` (error below 2e-3 over the first
  600 samples) and samples every `h=1.0` time units. The generated series changes accordingly.
  The seed now uses `numpy.random.default_rng` like the rest of the library.
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
