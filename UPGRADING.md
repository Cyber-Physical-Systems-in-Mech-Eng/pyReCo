# Upgrading from pyReCo 2.x to 3.0

Most 2.x scripts need a few lines changed; some results will differ numerically even for the same
seed (see "Results change"). The full list of changes is in [CHANGELOG.md](CHANGELOG.md).

## Code that stops working

| 2.x | 3.0 |
|---|---|
| `model.fit(x, y, n_init=5)` | `model.fit(x, y)`. `n_init` never explored anything (all initializations were identical). To compare reservoirs, build several models with different `seed=`. |
| `model.fit(x, y, visualize=True)` | `model.fit(x, y)`, then `model.model_visualize()`. `fit(x, y, True)` now sets `store_states`: pass keywords. |
| `set_normalization(normalize_outputs=...)`, `set_output_scaling(...)`, `output_mean`, `output_std` | Removed. Ridge regression is linear in the targets, so scaling the targets changed nothing. Predictions are in the units of `y`. |
| z-scoring and input scaling set after compiling | Arguments of `compile(normalize_inputs=True, input_scaling=1.0)`; `set_normalization()` only takes `normalize_inputs`. |
| The history returned by `fit` | The history is `{"train_score": float, "train_metrics": {...}}` (plus `"res_states"` with `store_states=True`). Readout weights and intercept are on `model.readout_layer`. |
| `datasets.load("lorenz", h=0.01)`, `h_int=` | `dt=`, `dt_int=` (the old names still work with a `DeprecationWarning`). Unknown parameters now raise a `TypeError`. |
| `EdgePruner(..., performance_criterion="r2")` | A loss: `mse`, `mae`, `rmse`, `nrmse`. `r2` can still be tracked via `metrics`. |
| `EdgePruner(min_num_edges=...)` above the number of edges | Raises a `ValueError` (it returned the unchanged model). |
| `np.random.seed(0)` for reproducibility | No effect any more. Pass `seed=` to `RC(...)`, `RandomReservoirLayer(...)`, `EdgePruner(...)`, `cross_val(...)` and the dataset functions. |
| Layer arguments such as `leakage_rate=0` or `1.5`, `density=1.5`, an unknown `activation` | Rejected when the layer is created. A `leakage_rate` of exactly 1 warns (leaky integration is off). |
| `NetworkQuantifier`, `NodePropExtractor` from `network_prop_extractor` | They work again, but return arrays with one value per node; see the docstrings. |

## Results change

These are not errors, so check your numbers:

* **Edge direction.** `weights[u, v]` is the edge `u -> v` and the update is `r(t+1) = g(r(t) @ weights + ...)`.
  Before, the update used `weights @ r(t)`. If you build a reservoir matrix by hand
  (`reservoir_layer.set_weights`), transpose it to keep the old dynamics.
* **Initial state.** The reservoir starts from zero (it was a random normal vector). Use
  `RandomReservoirLayer(init_res_sampling="random_normal")` for the old behaviour.
* **Warm-up.** `discard_transients` has a defined meaning: the target at step `j` is paired with the state after
  the inputs `x(0), ..., x(d + j)`. For sequence-to-scalar tasks, where the target lies at the end of the input window,
  pass `discard_transients = n_in - n_out`; the default 0 pairs the targets with the first states.
  `predict()` returns all steps after the warm-up (shape `(n_batch, T - d, n_out)`).
* **Seeds.** Every random draw comes from a generator made from `seed`, each choice from its own stream.
  The same seed gives different networks than in 2.x.
* **`r2`** is averaged per channel instead of pooled over all channels (identical for one channel).
* **Inputs** are z-scored per channel by default (`compile(normalize_inputs=True)`), then multiplied with
  `input_scaling` (default 1.0).

## New, optional

Defaults reproduce the 2.x model structure; these only act when you switch them on:
`compile(input_bias=...)`, signed weights (`RandomReservoirLayer(weight_distribution="sign" | "uniform" | "normal")`,
default all positive as before), `InputLayer(weight_distribution=...)`,
`edge_analyzer.operating_point_statistics(model, x)`, and the commented quick start `examples/mwe_quickstart.py`.

## Unchanged and still experimental

`AutoRC` (closed-loop prediction) is experimental: it does not apply the input bias and its interface may move.
`model.save()` is not implemented yet.

## Requirements

Python 3.10 or newer. TensorFlow is optional (`pip install pyreco[benchmarks]`) and only needed for the comparison examples.
