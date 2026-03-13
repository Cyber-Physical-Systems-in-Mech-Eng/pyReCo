"""
COMPREHENSIVE EXAMPLE: model_visualize() Method
================================================

This example demonstrates ALL possible parameters and configurations
for the pyreco reservoir model visualization method.

Each parameter is explained with:
- Purpose and description
- Data type and valid values
- Default value (if applicable)
- Usage examples
"""

from pyreco.custom_models import AutoRC as RC
from pyreco.layers import InputLayer, ReadoutLayer, FeedbackLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK


# =============================================================================
# STEP 1: Prepare Sample Data
# =============================================================================

# Generate training and test data for sine prediction
A_train, A_test, B_train, B_test = sequence_to_sequence(
    name="sine_pred", n_states=1, n_batch=20, n_time=501
)

X_train = A_train
X_test = A_test
y_train = A_train
y_test = A_test

print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")


# =============================================================================
# STEP 2: Build and Train the Model
# =============================================================================

# Define input/output dimensions
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

# Create the reservoir computing model
model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(FeedbackLayer(feedback_shape=output_shape))
model_rc.add(
    RandomReservoirLayer(
        nodes=50,
        density=0.8,
        spec_rad=0.8,
        activation="tanh",
        leakage_rate=0.2,
        fraction_input=0.6,
    ),
)
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.1))

# Compile the model
optim = RidgeSK(alpha=1.0)
model_rc.AutoRC_compile(
    optimizer=optim, metrics=["mean_squared_error"], discard_transients=10
)

# Train the model
model_rc.fit(X_train, y_train)

# Predict the model
y_pred = model_rc.predict(X_test)
res_states, y_pred = model_rc.AutoRC_predict(X_test, fb_scale=1.0, T_run=100)

# =============================================================================
# STEP 3: Visualize the PyReCo Reservoir
# =============================================================================

model_rc.model_visualize(
    save=True,
    file_name="reservoir_graph",
    file_type="svg",
    Node_colors={
        "CWinp": "purple",
        "CWres_inp": "orange",
        "CWres_out": "yellow",
        "CWres_both": "red",
        "CWres_internal": "grey",
        "CWout": "purple",
        "Winp": "blue",
        "Wout": "red",
        "CWres": "grey",
    },
    Edge_Weights=0.7,
)
