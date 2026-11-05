"""
Minimal working example: k-fold cross-validation and drawing the reservoir network.

Cross-validation fits the model on k different training parts of the data and scores it on
the remaining part each time. The variation of the scores shows how much the result depends on
the choice of the training data (bias and variance of the model). It needs a data set with many
samples (n_batch), here windows of a sine signal.
"""

from pyreco.cross_validation import cross_val
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.utils_data import sequence_to_sequence

X_train, X_test, y_train, y_test = sequence_to_sequence(
    "sine_pred", n_batch=100, n_states=1, n_time=20, seed=0
)

model = RC(seed=0)
model.add(InputLayer(input_shape=X_train.shape[1:]))
model.add(RandomReservoirLayer(nodes=50, density=0.1, leakage_rate=0.3, seed=0))
model.add(ReadoutLayer(output_shape=y_train.shape[1:], fraction_out=0.65))
model.compile(metrics=["mse"])

# 5 folds; the folds are drawn with the given seed
scores, mean, std = cross_val(model, X_train, y_train, n_splits=5, metric="mse", seed=0)
print(f"MSE per fold: {scores}")
print(f"mean MSE {mean:.4f}, standard deviation {std:.4f}")

# draw the reservoir network of the fitted model (nodes receiving the input, nodes read out)
model.fit(X_train, y_train)
model.model_visualize()  # add save=True, file_name="reservoir", file_type="svg" to write a file
