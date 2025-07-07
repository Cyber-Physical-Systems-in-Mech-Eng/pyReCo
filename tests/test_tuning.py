import numpy as np
import matplotlib.pyplot as plt

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.tuner import Tuner
from pyreco.utils_data import sequence_to_sequence


# === LOAD DATA ===
X_train, X_test, y_train, y_test = sequence_to_sequence(
    name="sin_to_cos2",  # FIXED name
    n_batch=30,
    n_states=1,
    n_time=500
)

input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

# === INITIALIZE RC MODEL (before tuning) ===
model = RC()
model.add(InputLayer(input_shape=input_shape))
model.add(RandomReservoirLayer(nodes=200, activation='tanh', fraction_input=0.5))
model.add(ReadoutLayer(output_shape))
model.compile(optimizer='ridge', metrics=['mse'])

# Save weights for later comparison
W_res_initial = model.reservoir_layer.weights.copy()

# === SETUP TUNER ===
search_space = {
    "spec_rad": ("float", 0.1, 1.0),
    "leakage_rate": ("float", 0.05, 1.0),
    "activation": ("categorical", ["tanh", "sigmoid"]),
    "input_scaling": ("float", 0.1, 1.0),
}

tuner = Tuner(
    model=model,
    search_space=search_space,
    x_train=X_train,
    y_train=y_train,
    x_val=X_train,  # train-as-val for now
    y_val=y_train,
    metric='mse',
    n_trials=100,
    sampler_type='tpe',
    verbose=True
)

# === RUN TUNING ===
study = tuner.optimize()
W_res_tuned = tuner.model.reservoir_layer.weights.copy()

# === UNTUNED MODEL (fresh copy) ===
model_untrained = RC()
model_untrained.add(InputLayer(input_shape=input_shape))
model_untrained.add(RandomReservoirLayer(nodes=200, activation='tanh', fraction_input=0.5))
model_untrained.add(ReadoutLayer(output_shape))
model_untrained.compile(optimizer='ridge', metrics=['mse'])
model_untrained.fit(X_train, y_train)
y_pred_untuned = model_untrained.predict(X_train)

# === TUNED MODEL PREDICTION ===
y_pred_tuned = tuner.model.predict(X_train)

# === PLOT RESULTS ===
plt.figure(figsize=(10, 4))
plt.plot(y_train[0, :, 0], label='Ground Truth', linewidth=2)
plt.plot(y_pred_untuned[0, :, 0], label='Untuned Prediction', linestyle='--')
plt.plot(y_pred_tuned[0, :, 0], label='Tuned Prediction', linestyle=':')
plt.legend()
plt.title("RC Output vs Ground Truth")
plt.xlabel("Time step")
plt.ylabel("Output value")
plt.tight_layout()
plt.show()
