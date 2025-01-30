"""
Provides testing data sets.

SHAPES will always be:
inputs X: [n_batch, n_timesteps, n_states]
outputs y: [n_batch, n_timesteps, n_states]

"""
import numpy as np
from matplotlib import pyplot as plt

# TODO: add Lorenz and other test cases (with potential limits to the number of states etc.)


def gen_sine(n=10, omega=np.pi):
    # generates a sequence of sines and cosines with a given frequency. sampling is 10 points per period
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.sin(omega * t)


def gen_cos(n=10, omega=np.pi):
    # generates a sequence of sines and cosines with a given frequency. sampling is 10 points per period
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.cos(omega * t)

def gen_sincos(n=10, omega=np.pi, a_sc=1, b_sc=0.25, P_sc=3):
    # generates a sequence of a_sc*sin(omega*t)^P_sc + b_sc*cos(omega*t)^P_sc
    # using gen_sine and gen_cos functions
    sine_wave = gen_sine(n, omega)
    cos_wave = gen_cos(n, omega)
    return a_sc * sine_wave**P_sc + b_sc * cos_wave**P_sc



def split_sequence(signal, n_batch, n_time_in, n_time_out):
    # expects [n_timesteps, n_states] sequence

    # convert into inputs (function at last n_time_in points) and outputs (function at next n_time_out points)
    x, y = [], []
    for i in range(n_batch):
        idx_in_end = i + n_time_in
        idx_out_end = idx_in_end + n_time_out
        x.append(signal[i:idx_in_end, :])  # last n_timestep_in points
        y.append(signal[idx_in_end:idx_out_end, :])  # next n_timestep_out points
    x, y = np.array(x), np.array(y)

    return x, y


def train_test_split(x, y):
    # train-test split 80% sample random points from the sequence and return inputs and outputs
    n = x.shape[0]

    ratio = 0.8

    split_idx = np.max([1, int(n * ratio)])

    shuffle_idx = np.random.choice(n, size=n, replace=False)
    train_idx, test_idx = shuffle_idx[:int(n * ratio)], shuffle_idx[int(n * ratio):]

    # split data according to train/test split and return.
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def sine_pred(n_batch, n_time_in, n_time_out, n_states):
    # predict a sine signal. Single- and multi-step prediction supported


    # we will create different signal frequencies for the different states
    signal, omega = [], np.pi
    for _ in range(n_states):
        signal.append(gen_sine(n=n_batch + (n_time_in + n_time_out) + 1, omega=omega))
        omega += 0.314

    # 2D array of shape [n_timesteps, n_states]
    signal = np.array(signal).transpose()

    # split into inputs and outputs
    x, y = split_sequence(signal, n_batch, n_time_in, n_time_out)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    return X_train, X_test, y_train, y_test


def sine_to_cosine(n_batch, n_time_in, n_time_out, n_states):
    # Generate sine input and cosine output signals
    total_time = n_batch + n_time_in + n_time_out
    x, y = [], []
    omega = np.pi
    for _ in range(n_states):
        x.append(gen_sine(n=total_time, omega=omega))
        y.append(gen_cos(n=total_time, omega=omega))
        omega += 0.314

    # Convert to 2D arrays of shape [n_timesteps, n_states]
    x = np.array(x).T
    y = np.array(y).T

    # Split into sequences
    x = split_sequence(x, n_batch, n_time_in, n_time_out)
    y = split_sequence(y, n_batch, n_time_in, n_time_out)

    # Unpack the tuples returned by split_sequence
    x_input, _ = x
    _, y_output = y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_input, y_output)

    return X_train, X_test, y_train, y_test


def sincos2(n_batch, n_time_in, n_time_out, n_states):
    # Generate sine input and SinCos-2 output signals
    total_time = n_batch + n_time_in + n_time_out
    x, y = [], []
    omega = 1  # As specified in the document
    
    for _ in range(n_states):
        x.append(gen_sine(n=total_time, omega=omega))
        y.append(gen_sincos(n=total_time, omega=omega))
        omega += 0.314  # Increment omega for each state, as in the original function
    
    # Convert to 2D arrays of shape [n_timesteps, n_states]
    x = np.array(x).T
    y = np.array(y).T
    
    # Split into sequences
    x = split_sequence(x, n_batch, n_time_in, n_time_out)
    y = split_sequence(y, n_batch, n_time_in, n_time_out)
    
    # Unpack the tuples returned by split_sequence
    x_input, _ = x
    _, y_output = y
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_input, y_output)
    
    return X_train, X_test, y_train, y_test


"""
generate some training and testing data from the sine function. 

CASE 1: scalar inputs, scalar outputs
"""


def scalar_to_scalar(name, n_batch: int = 50):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    if name == 'sine_pred':
        # single-step predict a sine signal
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=1, n_time_in=1, n_time_out=1)

    elif name == 'sine_to_cosine':
        # Map a sequence of sines to a sequence of cosines
        X_train, X_test, y_train, y_test = sine_to_cosine(n_batch=n_batch, n_states=1,
                                                          n_time_in=1, n_time_out=1)
        
    elif name == 'sincos2':
        # Map a sequence of sines to a sequence of sinecosines
        X_train, X_test, y_train, y_test = sincos2(n_batch=n_batch, n_states=1,
                                                          n_time_in=1, n_time_out=1)

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')
    return X_train, X_test, y_train, y_test


"""
CASE 2: Vector to Vector
"""


def vector_to_vector(name, n_batch: int = 50, n_states: int = 2):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    if name == 'sine_pred':
        # single-step predict a vector of sine signals
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=n_states, n_time_in=1, n_time_out=1)
    
    elif name == 'sine_to_cosine':
        # Map a sequence of sines to a sequence of cosines
        X_train, X_test, y_train, y_test = sine_to_cosine(n_batch=n_batch, n_states=n_states,
                                                          n_time_in=1, n_time_out=1)
    
    elif name == 'sincos2':
        # Map a sequence of sines to a sequence of sinecosines
        X_train, X_test, y_train, y_test = sincos2(n_batch=n_batch, n_states=n_states,
                                                          n_time_in=1, n_time_out=1)

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')
    return X_train, X_test, y_train, y_test


"""
CASE 3: Sequence to sequence
"""


def sequence_to_sequence(name, n_batch: int = 50, n_states: int = 2, n_time: int = 3):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    if name == 'sine_pred':
        # multi-step predict a vector of sine signals
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=n_states,
                                                     n_time_in=n_time, n_time_out=n_time)
    elif name == 'sine_to_cosine':
        # Map a sequence of sines to a sequence of cosines
        X_train, X_test, y_train, y_test = sine_to_cosine(n_batch=n_batch, n_states=n_states,
                                                          n_time_in=n_time, n_time_out=n_time)
    
    elif name == 'sincos2':
        # Map a sequence of sines to a sequence of sinecosines
        X_train, X_test, y_train, y_test = sincos2(n_batch=n_batch, n_states=n_states,
                                                          n_time_in=n_time, n_time_out=n_time)
    

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')
    return X_train, X_test, y_train, y_test


def x_to_x(name, n_batch: int = 50, n_states_in: int = 2, n_states_out: int = 2, n_time_int: int = 1, n_time_out: int =
1):

    # make sure to have at least 1 testing sample
    n_batch = np.max([n_batch, 2])

    # full flexibility in creating input and output shapes
    n_states = np.max([n_states_in, n_states_out])

    if name == 'sine_pred':
        # single-step predict a vector of sine signals
        X_train, X_test, y_train, y_test = sine_pred(n_batch=n_batch, n_states=n_states, n_time_in=1, n_time_out=1)

    elif name == 'sine_to_cosine':
        # Map a sequence of sines to a sequence of cosines
        X_train, X_test, y_train, y_test = sine_to_cosine(n_batch=n_batch, n_states=n_states,
                                                          n_time_in=1, n_time_out=1)
    
    elif name == 'sincos2':
        # Map a sequence of sines to a sequence of cosines
        X_train, X_test, y_train, y_test = sincos2(n_batch=n_batch, n_states=n_states,
                                                          n_time_in=1, n_time_out=1)

    # cut data if input and output vector length is not the same
    X_train, X_test = X_train[:, :, :n_states_in], X_test[:, :, :n_states_in]
    y_train, y_test = y_train[:, :, :n_states_out], y_test[:, :, :n_states_out]

    print(f'shape of inputs (training): {X_train.shape}, shape of outputs (training): {y_test.shape}')

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':

    # case 1

    # n_time = X_train.shape[1]
    # plt.figure()
    # for i in range(3):
    #     plt.plot(np.arange(start=1, stop=n_time), X_train[i,:,0], label='input')
    #
    # plt.legend()
    # plt.show()

    X_train, X_test, y_train, y_test = scalar_to_scalar(name='sincos2', n_batch=50)

    #print(X_train,X_test, y_test,y_train)

    X_train, X_test, y_train, y_test = vector_to_vector(name='sincos2', n_batch=1, n_states=3)

    #print(X_train,X_test, y_test,y_train)

    X_train, X_test, y_train, y_test = sequence_to_sequence(name='sincos2', n_batch=50, n_states=4, n_time=15)

    #print(X_train,X_test, y_test,y_train)

    X_train, X_test, y_train, y_test = x_to_x(name='sincos2', n_batch=100, n_states_in=4, n_states_out=3,
                                              n_time_int=10, n_time_out=2)

