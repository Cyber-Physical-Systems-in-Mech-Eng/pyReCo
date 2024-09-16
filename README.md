# pyReCo

**A Reservoir Computing Library for Time Series Forecasting and Research**



pyReCo is built by researchers for researchers: we aim to develop new RC methods that allow for fast and efficient learning for sequential data. The main focus is time series prediction, mostly performed in an auto-regressive fashion based on learning discrete flow maps. Another core aspect that motivates the implementation of a new library is *structure-function-relationships* in functional networks. pyReCo allows to implement novel ways to generate better reservoir networks than the classical random choice.  Overview of the core capabilities of pyReCo:

- [x] **Classical reservoir computing**: using random reservoir layers and training readout-layer weights using Ridge regression
- [x] **Cross-validation**: built-in functions to k-fold cross-validate any ResPy model for performance evaluation
- [ ] **auto-regressive time stepping** through feeding the predictions into the input layer (closed-loop prediction system)
- [ ] **automated hyper-parameter tuning** for leakage rate, activation function, reservoir network properties, etc. 

**Compatability**: ResPy follows the syntax of scikit-learn (pyReCo Model-API), such that an estimator has the methods `.fit()` and `.predict()`. Any more experimental modeling can be realised by pyReCo's CustomModel-API, which follows TensorFlow's Sequential-API: a custom pyReCo model is compiled using `model.add()` functions for stacking different layers.  

#### Installation
Simply use pip, or get the source files from this repo.

```
pip install pyreco
```


## Minimal Working Examples

### Model API
The ModelAPI is the simplest way to use one of the predefined RC models for training and prediction. The Model class follows the syntax of scikit-learn, such that an estimator has the methods `.fit()`, `.predict()`, and `.evaluate()`.

The following example illustrates how to translate a sine signal to a cosine signal, i.e. a sequence-to-sequence mapping task. The input is $$x=\sin\left(\pi t\right)$$ and the output is  $$x=2\cos\left(\pi t\right)$$. Thus, The model must learn a phase shift and the amplitude scaling from input to output. Both signals have 3 periods, sampled with 100 steps per period.

```python
import numpy as np
from matplotlib import pyplot as plt
from pyreco.models import ReservoirComputer as RC

# generate 3 cycles of a sine (input) and of a cosine (output)
omega = np.pi
t = np.linspace(start=0, stop=3 * (2*np.pi/omega), num=300, endpoint=True)
x = np.sin(omega * t)
y = 2 * np.cos(omega * t)

x_train = np.expand_dims(x, axis=(0,2))  # obtain shape of [n_batch, n_time, n_states]
y_train = np.expand_dims(y, axis=(0,2))

# fit a reservoir computer with 200 nodes and make predictions on the training set
model = RC(num_nodes=200, activation='tanh')
model.fit(x_train, y_train)
y_pred = model.predict(x_train)

# evaluate some metrics (for simplicity on the train set)
metric_value = model.evaluate(X=x_train, y=y_train, metrics=['mse','mae'])
print(f'scores:{metric_value}')

# print truth and predicted sequence
plt.figure()
plt.plot(y_train[0,:,0], label='ground truth', marker='.', color='#1D3557')
plt.plot(y_pred[0,:,0], label='prediction', marker='.', color='#E63946')
plt.legend()
plt.xlabel('time')
plt.show()

```
Plotting `y_pred` and `y_train` against the time vector `t` shows how the minimal RC can learn a phase shift, i.e. translate a sine signal to a cosine signal, and also scale the output to the correct amplitude. You may want to increase the reservoir size (`num_nodes`) or change the leakage rate (`leakage_rate`) to improve the prediction quality.  
![model_predictions](https://github.com/user-attachments/assets/6db9bb21-fc93-493c-adee-3310bf6a7f4f)


**Input data shapes**: the shape of the input data is of utmost importance. ResPy is built for sequential modeling tasks. Make sure to provide the input data in the shape of `[n_batch, n_time, n_states]`, where
- `n_batch` is the number of samples, i.e. the number of individual time series samples (example above: `n_batch = 1`)
- `n_time` is the number of time steps per sample (example above: `n_time = 300`)
- `n_states` is the channel dimension, i.e. the number of contemporaneous states in the time series (example above: `n_states = 1`)



### CustomModel API

The `CustomModel` API gives the user more flexibility in defining specific properties. For example, one can specify the fraction of reservoir nodes which receive the inputs. Additionally, the user can provide custom reservoir layers and add them to the model. The same example as for the Model API can be obtained with the custom model, as shown below. 

```python
import numpy as np
from matplotlib import pyplot as plt

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer

# generate 3 cycles of a sine (input) and of a cosine (output)
omega = np.pi
t = np.linspace(start=0, stop=3 * (2*np.pi/omega), num=300, endpoint=True)
x = np.sin(omega * t)
y = 2 * np.cos(omega * t)

x_train = np.expand_dims(x, axis=(0,2))  # obtain shape of [n_batch, n_time, n_states]
y_train = np.expand_dims(y, axis=(0,2))

# set the dimensions
input_shape = (x_train.shape[1], x_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

# build a custom RC model by adding layers with properties
model = RC()
model.add(InputLayer(input_shape=input_shape))
model.add(RandomReservoirLayer(nodes=200, activation='tanh', fraction_input=0.5))
model.add(ReadoutLayer(output_shape))

# compile the model
model.compile(optimizer='ridge', metrics=['mean_squared_error'])

# train the model
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_train)
print(f'shape of predicted array: {y_pred.shape}')

# evaluate some metrics (for simplicity on the train set)
metric_value = model.evaluate(X=x_train, y=y_train, metrics=['mse','mae'])
print(f'scores:{metric_value}')
```

## Background and Supplementary Material

### Introductory material to reservoir computing and echo state networks
- [Cucchi 2022: Hands-on Reservoir Computing](https://iopscience.iop.org/article/10.1088/2634-4386/ac7db7)
- [Gallichio: An Introduction to Reservoir Computing](http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/aa2/rnn4-esn.pdf)

### Other RC libraries that you may want to have a look at
- [reservoirpy](https://github.com/reservoirpy/reservoirpy)
- [PYRCN](https://github.com/TUD-STKS/PyRCN/)
- [EchoTorch](https://github.com/nschaetti/EchoTorch)
- [list of more packages](https://github.com/topics/reservoir-computing)


