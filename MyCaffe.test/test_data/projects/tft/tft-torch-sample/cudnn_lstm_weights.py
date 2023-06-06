# See https://kaixih.github.io/keras-cudnn-rnn/
import os
cwd = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES']='0'
import keras
import tensorflow as tf
from keras import layers
from tensorflow.python.ops import array_ops
import numpy as np
import torch
from torch import nn
from tft_helper import get_cudnn_lstm_weights
import torch.nn.init as init

os.chdir(cwd)
print(os.getcwd())

seed = 1704
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

hidden_size = 64
state_size = 64
lstm_layers = 1
dropout = 0

strPathWt = "data/favorita/weights/"
strPath = "test/all/iter_0/"

# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


def _canonical_to_params_tf(weights, biases, shape, transpose_weights=False):
   """Utility function convert variable to CuDNN compatible parameter.
 
   Note that Keras weights for kernels are different from the CuDNN format. Eg.:
 
   ```
     Keras                 CuDNN
     [[0, 1, 2],  <--->  [[0, 2, 4],
      [3, 4, 5]]          [1, 3, 5]]
   ```
 
   If the input weights need to be in a unified format, then set
   `transpose_weights=True` to convert the weights.
 
   Args:
     weights: list of weights for the individual kernels and recurrent kernels.
     biases: list of biases for individual gate.
     shape: the shape for the converted variables that will be feed to CuDNN.
     transpose_weights: boolean, whether to transpose the weights.
 
   Returns:
     The converted weights that can be feed to CuDNN ops as param.
   """
   def convert(w):
     return array_ops.transpose(w) if transpose_weights else w
 
   weights = [array_ops.reshape(convert(x), shape) for x in weights]
   biases = [array_ops.reshape(x, shape) for x in biases]
   return array_ops.concat(weights + biases, axis=0)

def _canonical_to_params_py(weights, biases, shape, transpose_weights=False):
   """Utility function convert variable to CuDNN compatible parameter.
 
   Note that Keras weights for kernels are different from the CuDNN format. Eg.:
 
   ```
     Keras                 CuDNN
     [[0, 1, 2],  <--->  [[0, 2, 4],
      [3, 4, 5]]          [1, 3, 5]]
   ```
 
   If the input weights need to be in a unified format, then set
   `transpose_weights=True` to convert the weights.
 
   Args:
     weights: list of weights for the individual kernels and recurrent kernels.
     biases: list of biases for individual gate.
     shape: the shape for the converted variables that will be feed to CuDNN.
     transpose_weights: boolean, whether to transpose the weights.
 
   Returns:
     The converted weights that can be feed to CuDNN ops as param.
   """
   def convert(w):
     return np.transpose(w) if transpose_weights else w
 
   weights = [np.reshape(convert(x), shape) for x in weights]
   biases = [np.reshape(x, shape) for x in biases]
   return np.concatenate(weights + biases, axis=0)

x = np.load(strPath + "YYY.past_lstm.lstm.x.npy")
x1 = tf.convert_to_tensor(x)

lstm = layers.LSTM(hidden_size, 
                   time_major=False, 
                   return_sequences=True, 
                   kernel_initializer="glorot_uniform", 
                   recurrent_initializer="orthogonal", 
                   bias_initializer="random_uniform", 
                   dropout=0.0,
                   unit_forget_bias=False)
y1 = lstm(x1)
print("Keras Kernel Weights:", lstm.get_weights()[0])
print("Keras Recurrent Weights:", lstm.get_weights()[1])
print("Keras Biases:", lstm.get_weights()[2])


wts = lstm.get_weights()
wts0 = wts[0] # shape (64,256) Kernel weights (inputSize, 4 * hiddenSize)
wts0_0 = wts0[:, :hidden_size]
wts0_1 = wts0[:, hidden_size:hidden_size * 2]
wts0_2 = wts0[:, hidden_size * 2:hidden_size * 3]
wts0_3 = wts0[:, hidden_size * 3:]

wts1 = wts[1] # shape (64,256) Recurrent weights (hiddenSize, 4 * hiddenSize)
wts1_0 = wts1[:, :hidden_size]
wts1_1 = wts1[:, hidden_size:hidden_size * 2]
wts1_2 = wts1[:, hidden_size * 2:hidden_size * 3]
wts1_3 = wts1[:, hidden_size * 3:]

wts2 = wts[2] # shape (256) Biases (4 * hiddenSize)
b0_0 = np.zeros((hidden_size,))
b0_1 = np.zeros((hidden_size,))
b0_2 = np.zeros((hidden_size,))
b0_3 = np.zeros((hidden_size,))
b1_0 = wts2[:hidden_size]
b1_1 = wts2[hidden_size:hidden_size * 2]
b1_2 = wts2[hidden_size * 2:hidden_size * 3]
b1_3 = wts2[hidden_size * 3:hidden_size * 4]

# final cudnn weight ordering will be as follows
# (with all weight matrices transposed)
#
# wts0_0, wts0_1, wts0_2, wts0_3,
# wts1_0, wts1_1, wts1_2, wts1_3,
# b0_0,   b0_1,   b0_2,   b0_3,
# b1_0,   b1_1,   b1_2,   b1_3

params = _canonical_to_params_tf(
    weights=[
        lstm.get_weights()[0][:, :hidden_size],
        lstm.get_weights()[0][:, hidden_size:hidden_size * 2],
        lstm.get_weights()[0][:, hidden_size * 2:hidden_size * 3],
        lstm.get_weights()[0][:, hidden_size * 3:],
        lstm.get_weights()[1][:, :hidden_size],
        lstm.get_weights()[1][:, hidden_size:hidden_size * 2],
        lstm.get_weights()[1][:, hidden_size * 2:hidden_size * 3],
        lstm.get_weights()[1][:, hidden_size * 3:],
    ],
    biases=[
        tf.zeros((hidden_size,)),
        tf.zeros((hidden_size,)),
        tf.zeros((hidden_size,)),
        tf.zeros((hidden_size,)),
        lstm.get_weights()[2][:hidden_size],
        lstm.get_weights()[2][hidden_size:hidden_size * 2],
        lstm.get_weights()[2][hidden_size * 2:hidden_size * 3],
        lstm.get_weights()[2][hidden_size * 3:hidden_size * 4],
    ],
    shape=tf.constant([-1]),
    transpose_weights=True)
print("CUDNN-equivalent Params:", params)

# for Pytorch, note the wts0_3 and wts0_2 are swapped as per the
# pytorch documentation at https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
params2 = _canonical_to_params_py(
    weights=[wts0_0, wts0_1, wts0_3, wts0_2,
             wts1_0, wts1_0, wts1_3, wts1_2
    ],
    biases=[b0_0, b0_1, b0_3, b0_2,
            b1_0, b1_1, b1_3, b1_2
    ],
    shape=tf.constant([-1]),
    transpose_weights=True)
print("CUDNN-equivalent Params:", params)


lstm2 = nn.LSTM(input_size=state_size,
                hidden_size=state_size,
                num_layers=1,
                dropout=dropout,
                batch_first=True)


wts_ih_l0 = np.concatenate((wts0_0, wts0_1, wts0_3, wts0_2), axis=0)
wts_hh_l0 = np.concatenate((wts1_0, wts1_1, wts1_3, wts1_2), axis=0)
b_ih_l0 = np.concatenate((b0_0, b0_1, b0_3, b0_2), axis = 0)
b_hh_l0 = np.concatenate((b1_0, b1_1, b1_3, b1_2), axis = 0)

wts_py = [wts_ih_l0, wts_hh_l0, b_ih_l0, b_hh_l0]

idx = 0
for param in lstm2.state_dict():
    lstm2.state_dict()[param] = wts_py[idx]
    idx = idx + 1

#idx = 0
#for param in lstm2.state_dict():
#        param_val = lstm2.state_dict()[param]
#        lstm2.state_dict()[param] = wts_py[idx]
#        idx = idx+1
x2 = torch.from_numpy(x)

y2 = lstm2(x2)

params3 = get_cudnn_lstm_weights(lstm2)

print("done!")
