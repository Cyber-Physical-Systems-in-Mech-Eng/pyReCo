"""

Showcase / tutorial script for using `pyreco.tuner.Tuner` with a Reservoir Computing (RC) model.


"""

import copy

from matplotlib import pyplot as plt

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.optimizers import RidgeSK
from pyreco.plotting import r2_scatter
from pyreco.utils_data import sequence_to_sequence

# Import your tuner class
from pyreco.tuner import Tuner


# =============================================================================
# 1) DATA
# =============================================================================
# sequence_to_sequence returns:
#   X: [n_batch, n_time, n_states]
#   y: [n_batch, n_time, n_states]
#
# Here: map sin -> cos^2 over time.
X_train, X_test, y_train, y_test = sequence_to_sequence(
    name="sin_to_cos2",
    n_states=1,
    n_batch=200,
    n_time=100,  
)

input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])


# =============================================================================
# 2) BASELINE MODEL (NO TUNING)
# =============================================================================
# Build the RC model exactly like a typical pyreco example
model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(
    RandomReservoirLayer(
        nodes= 15,
        density=0.1,
        activation="tanh",
        leakage_rate=0.1,
        fraction_input=0.4,
    )
)
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.65))

# Compile once. NOTE: `alpha` can still be tuned later via RC.set_hp(alpha=...)
# because your RC.set_hp calls self.optimizer.set_alpha(...)
model_rc.compile(
    optimizer=RidgeSK(alpha=0.5),
    metrics=["mean_squared_error"],
)

# Train baseline
model_rc.fit(X_train, y_train, visualize=False)

# Baseline eval
y_pred_base = model_rc.predict(X_test)
score_base = model_rc.evaluate(X_test, y_test)  # used as default by Tuner too (if no metric/evaluator)
print("\n=== BASELINE ===")
print("Baseline score (model.evaluate):", score_base)


# =============================================================================
# 3) DEFINE SEARCH SPACE
# =============================================================================
# The Tuner expects:
#   search_space = {
#       "param_name": ("type", ...),
#       ...
#   }
#
# Supported types in your Tuner:
#
# 1) Float:
#   ("float", low, high)                      -> uniform float in [low, high]
#   ("float", low, high, step)                -> float in [low, high] with step
#   ("float", low, high, "log")               -> log-uniform float (best for regularization like alpha)
#
# 2) Int:
#   ("int", low, high)                        -> integer in [low, high]
#   ("int", low, high, step)                  -> integer step size
#
# 3) Categorical:
#   ("categorical", [option1, option2, ...])  -> choose one option
#

search_space = {
    # float (uniform)
    "spec_rad": ("float", 0.1, 1.0),

    # float (uniform)
    "leakage_rate": ("float", 0.01, 0.5),

    # categorical
    "activation": ("categorical", ["tanh", "sigmoid"]),

    # float (uniform)
    "input_scaling": ("float", 0.5, 1.0),

    # float (uniform)
    "output_scaling": ("float", 0.1, 1.0),

    # float (log-uniform) — great for regularization hyperparameters
    "alpha": ("float", 1e-8, 1.0, "log"),
}


# =============================================================================
# 4) PICK A SAMPLER (HOW OPTUNA EXPLORES THE SPACE)
# =============================================================================
# Your Tuner supports sampler_type:
#
# 1) "tpe" (Tree-structured Parzen Estimator):
#    - Bayesian optimization-style sampler
#    - learns which regions of the search space perform well
#    - usually the best default choice for continuous HP tuning
#
# 2) "random":
#    - purely random samples
#    - good baseline, easy to debug, surprisingly strong for small budgets
#
# 3) "grid":
#    - exhaustively tries every combination from a discrete grid
#    - ONLY makes sense if your search space is discrete/small
#    - NOTE: in your Tuner, floats become a linspace grid of `num_float_steps`
#
# Try changing sampler_type below to "random" or "grid" to compare behavior.
sampler_type = "tpe"


# =============================================================================
# 5) TUNING
# =============================================================================
# For tuning we copy the model so we can compare against the baseline object.
# (If your RC carries state, you can also instantiate a fresh model instead.)
model_for_tuning = copy.deepcopy(model_rc)

tuner = Tuner(
    model=model_for_tuning,
    search_space=search_space,
    x_train=X_train,
    y_train=y_train,
    x_val=X_test,
    y_val=y_test,
    n_trials=300,              # increase for better tuning
    sampler_type=sampler_type, # "tpe" | "random" | "grid"
    direction="minimize",
    verbose=True,
)

# Optimize:
# - runs Optuna trials
# - prints importances automatically (per your Tuner.optimize)
# - stores best_model and best_score
tuner.optimize()
tuner.report()

best_model = tuner.best_model
y_pred_tuned = best_model.predict(X_test)
score_tuned = best_model.evaluate(X_test, y_test)

print("\n=== TUNED ===")
print("Tuned score (model.evaluate):", score_tuned)

# Improvement (assuming "minimize")

improvement = float(score_base[0]) - float(score_tuned[0])
rel = improvement / max(abs(float(score_base[0])), 1e-12) * 100.0
  
print("Improvement:", improvement, "Relative improvement:", rel,"%")
# =============================================================================
# 6) VISUALIZATION
# =============================================================================
# Scatter plots (overall fit quality)
print("\nPlotting R2 scatter (baseline)...")
r2_scatter(y_true=y_test, y_pred=y_pred_base)

print("Plotting R2 scatter (tuned)...")
r2_scatter(y_true=y_test, y_pred=y_pred_tuned)

# Plot a single long sequence (seq2seq)
idx = 0
t = range(X_test.shape[1])

x_seq = X_test[idx, :, 0]
y_true_seq = y_test[idx, :, 0]
y_base_seq = y_pred_base[idx, :, 0]
y_tuned_seq = y_pred_tuned[idx, :, 0]

plt.figure(figsize=(12, 4))
plt.plot(t, x_seq, alpha=0.3, label="input (sin)")
plt.plot(t, y_true_seq, label="true target (cos²)")
plt.plot(t, y_base_seq, label="baseline pred")
plt.plot(t, y_tuned_seq, label="tuned pred")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.legend()
plt.title(f"sequence-to-sequence (sampler={sampler_type}, n_time={X_test.shape[1]}): baseline vs tuned")
plt.tight_layout()
plt.show()