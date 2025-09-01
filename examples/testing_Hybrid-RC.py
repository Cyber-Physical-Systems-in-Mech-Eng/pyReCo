"""
Some testing use case for the CustomModel API of pyreco
"""

from re import X
from matplotlib import pyplot as plt
from pyreco.custom_models import AutoRC as RC
from pyreco.layers import InputLayer, ReadoutLayer, FeedbackLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK

from scipy.integrate import solve_ivp
import numpy as np
"""
Classic RC built on random networks. 
Use case: predict a sine signal into the future (n_time) samples from the past are mapped to (n_time) samples in the 
future
"""

# some testing data: predict a sine signal from a sequence of 20 function values
A_train, A_test, B_train, B_test = sequence_to_sequence(
    name="sine_pred",
    n_states=1,
    n_batch=50,
    n_time=201
)


#####testing hybrid RC model on forced Duffing oscillator data#############################
def Plot(a, b, forcing):
    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 6; fig_size[1] = 1.5
    plt.rcParams["figure.figsize"] = fig_size 
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(a[:1000], lw=0.5, color='black')
    plt.ylabel(r'$q_1(t)$')
    plt.title('damped forced oscillator')

    plt.subplot(3, 1, 2)
    plt.plot(b[:1000], lw=0.5, color='black')
    plt.ylabel(r'$q_2(t)$')

    plt.subplot(3, 1, 3)
    plt.plot(forcing[:1000], lw=0.5, color='red')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f(t)$')
    plt.tight_layout()
    plt.show()

    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 1.5; fig_size[1] = 1.5
    plt.rcParams["figure.figsize"] = fig_size 
    plt.plot(a, b, lw=0.5)
    plt.scatter(a[0], b[0], marker='o',c='r')
    plt.xlabel(r'$q_1(t)$'); plt.ylabel(r'$q_2(t)$')
    plt.show()

def one_dof_oscillator(t, q: np.ndarray,
                       c: float = 0.1,
                       k: float = 1.0,
                       f: float = 0,
                       omega: float = 1.0,
                       phi: float = 0.0,
                       beta: float=0
                       ) -> np.ndarray:
    """ ODE one-dimensional oscillator. """

    A = np.array([[0, 1], [-k, -c]])
    B = np.array([[0, 0], [-beta, 0]])
    F = np.array([0, f*np.cos(omega*t+phi)])

    return np.dot(A, q) + np.dot(B, q*q*q) + F

def SolveDuffing(q0, t_eval, c, k, f, omega, phi, beta):
    # numerical time integration
    sol_forced = solve_ivp(one_dof_oscillator, t_span=[t_eval[0], t_eval[-1]], y0=q0,\
                           t_eval=t_eval, args=(c, k, f, omega, phi, beta))

    # display of trajectories
    q = sol_forced.y.T
    t = sol_forced.t
    forcing = f*np.cos(omega*t+phi)  
    # Plot(q[:, 0], q[:, 1], forcing) 
    return q, forcing

def InputGenerator(t_eval,Trans,c,k,f,omega,phi,beta,q0):
    Q_i=[];Forcing_i=[]
    for i in range(len(f)):
        for j in range(len(omega)):
            Qs, Fs = SolveDuffing(q0, t_eval, c, k, f[i], omega[j], phi, beta)
            Q_i.append(Qs); Forcing_i.append(Fs)
    
    Q_i=np.asarray(Q_i); Forcing_i=np.asarray(Forcing_i)   
    
    #####Input##################
    Inp0=[]; Inp1=[]; Inp2=[]
    for i in range(len(f)*len(omega)):
        Inp0.append(Forcing_i[i,Trans:])
        Inp1.append(Q_i[i,Trans:,0])
        Inp2.append(Q_i[i,Trans:,1])
    
    #####Plot##################
    for i in range(len(f)*len(omega)):
        Plot(Inp1[i], Inp2[i], Inp0[i])

    return np.array([np.asarray(Inp0), np.asarray(Inp1), np.asarray(Inp2)])

# ######### Duffing Oscillator parameters####################################
c = 0.3 ###0.32    # damping
k = -1.0     # linear stiffness
f = [0.49] #[0.475, 0.485] #[0.47, 0.49]    # forced case
omega = [1.46] #[1.5, 1.53]  #[1.5, 1.53]  ####good in [1.47, 1.59] for 1st bif. pt.
phi = 0
beta=1

#### initial conditions
q0 = np.array([0.05, 0.05])

#### time integration interval
T=6000; h=0.1; Trans=500
t_eval = np.arange(start=0, stop=T, step=h)

Inps = InputGenerator(t_eval,Trans,c,k,f,omega,phi,beta,q0)
print('Input shapes: ',Inps[:,0,:].T.shape)

TPts = int((T/h)-Trans)
Batch = 25

Inps_N = np.reshape(Inps[:,0,:].T, (Batch, int(TPts/Batch),  3))
print('Input reshaped to [batch, time, states]: ',Inps_N.shape)

X_train = Inps_N[:20, :, :]  # use first 8 samples for training
X_test = Inps_N[20:, :, :]   # use last 2 samples for testing

y_train = Inps_N[:20, :, 1:]  # use first 8 samples for training, only q1 and q2 as target
y_test = Inps_N[20:, :, 1:]   # use last 2 samples for testing, only q1 and q2 as target

# print('Duffing Training data shape: ',D_train.shape)
# print('Duffing Test data shape: ',D_test.shape)
#######################################################################################

###generate y_train and y_test as previous time step of the input series
# X_train = A_train#[:, 1:, :]
# X_test = A_test#[:, 1:, :]
# ### remove last time step from X_train and X_test
# y_train = A_train#[:, :-1, :]
# y_test = A_test#[:, :-1, :]

print(f"Generated training data shapes: {X_train.shape}, {y_train.shape}")
print(f"Generated test data shapes: {X_test.shape}, {y_test.shape}")

# ##plot x and y
# plt.figure()
# for i in range(X_train.shape[0]):
#     plt.subplot(X_train.shape[0], 1, i + 1)
#     plt.plot(X_train[i, :, 0], color="k", alpha=0.3)
#     plt.plot(y_train[i, :, 0], color="b", alpha=0.3)
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.show()

# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# set the dimensions (need to be [n_batch, n_time, n_states])
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])


model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(FeedbackLayer(feedback_shape=output_shape))
model_rc.add(
    RandomReservoirLayer(
        nodes=100, density=0.5, spec_rad=0.5057, activation="tanh", leakage_rate=0.2, fraction_input=0.25),
)
model_rc.add(ReadoutLayer(output_shape, fraction_out=1.0))

# Compile the model
optim = RidgeSK(alpha=5e-7) #1.0)
model_rc.AutoRC_compile(
    optimizer=optim,
    metrics=["mean_squared_error"],
    discard_transients=10
)

# Train the model
model_rc.fit(X_train, y_train)

# print(model_rc.input_layer.weights)
# Make predictions for new data
y_pred = model_rc.predict(X_test)

###plot FFRC predictions
plt.figure()
plt.title("Single-step predictions. No feedback connections.")
for i in range(y_test.shape[0]):
    for j in range(y_test.shape[2]):
        plt.subplot(y_test.shape[0], y_test.shape[2], i * y_test.shape[2] + j + 1)
        plt.plot(X_test[i, 10:, 0], color="gray", alpha=1, label="input sequence")
        plt.plot(y_test[i, 10:, j], color="blue", lw=0.75, label="true target")
        plt.plot(y_pred[i, :, j], color="red", lw=0.75, label="predicted")
        plt.xlabel("time")
        plt.ylabel(f"amplitude of state {j}")
    plt.legend()
plt.show()

# autoRC predictions
# X_test = X_train[:-2, :, :]  # use all but last time step for prediction
# y_test = y_train[:-2, :, :]  # use all but last time step for prediction

###setting time>50 of X_test (last 2 indices ie q1 and q2) =0 so that AutoRC can use it for prediction
X_testn = X_test.copy()
X_testn[:, 500:, 1:] = 0   


feedback_indices = np.array([1,2])  # specify feedback indices here
res_states, y_pred = model_rc.AutoRC_predict(X_testn, fb_scale=1.0, T_run=2200, feedback_indices=feedback_indices)

print(f"shape of input weights: {model_rc.input_layer.weights.shape}")
print(f"shape of feedback weights: {model_rc.feedback_layer.weights.shape}")
print(f"shape of readout weights: {model_rc.readout_layer.weights.shape}")

print(f"input weights:\n{model_rc.input_layer.weights[:5,:]}\n\n"
      f"feedback weights:\n{model_rc.feedback_layer.weights[:5,:]}\n\n"
      f"readout weights:\n{model_rc.readout_layer.weights[:5,:]}")

print(f"shape of predictions on test set: {y_pred.shape}")

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=["mae"])
print(f"Test model loss: {loss_rc}")

# plot predictions vs. ground truth
# r2_scatter(y_true=y_test, y_pred=y_pred)


# plot predictions
plt.figure()
### loop over batch samples in X_test
plt.title("Autonomously running RC predictions")
for i in range(y_test.shape[0]):
    for j in range(y_test.shape[2]):
        plt.subplot(y_test.shape[0], y_test.shape[2], i * y_test.shape[2] + j + 1)
        plt.plot(X_test[i, :, 0], color="gray", alpha=1, label="input sequence")
        plt.plot(y_test[i, :, j], color="blue", lw=0.75, label="true target")
        plt.plot(y_pred[i, :, j], color="red", lw=0.75, label="predicted")
        plt.xlabel("time")
        plt.ylabel(f"amplitude of state {j}")
    plt.legend()

plt.show()


plt.figure()
for i in range(y_pred.shape[0]):
    plt.subplot(y_pred.shape[0], 1, i + 1)  
    plt.plot(y_pred[i, :, 0], y_pred[i, :, 1], color="red", lw=0.75, label="predicted")
plt.show()

"""
k-fold cross-validation of the model: estimate bias and variance of the model
"""
# from pyreco.cross_validation import cross_val

# val, mean, std_dev = cross_val(model_rc, X_train, y_train, n_splits=5, metric=["mse"])
# print(f"Cross-Validation MSE: {val}")
# print(f"Mean MSE: {mean:.3f}")
# print(f"Standard Deviation of MSE: {std_dev:.3f}")
